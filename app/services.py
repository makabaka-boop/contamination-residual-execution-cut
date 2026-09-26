"""业务逻辑：方案保存、最小割计算、采用与已采用结果查询。

事务约定：
- 方案校验通过后才写库，非法整版不会改写当前方案；
- 方案保存保证"每次被接受的内容、修订号与响应一一对应"：
  * 携带 expected_revision 时，由单条条件 UPDATE 原子完成
    "比对修订号 + 整版替换"；命中 0 行说明方案不存在
    （404 PLAN_NOT_FOUND）或读取后修订号已被他人推进
    （409 REVISION_CONFLICT），落败写入回滚，不改写现有方案；
  * 未携带 expected_revision 时保持旧语义（存在则整版替换、
    不存在则新建），但替换同样由单条
    UPDATE ... SET revision = revision + 1 原子完成，并发替换
    绝不会拿到相同的新修订号；新建依赖主键唯一约束，并发首次
    创建同一方案时落败事务回滚并以 409 REVISION_CONFLICT 拒绝，
    绝不把数据库异常暴露给调用方；
  * 成功响应取自本事务提交前持行锁重读的同一行（新建则取本次
    写入值本身），因此响应就是本次提交的内容，修订号全局唯一；
- 计算失败仅落一条 FAILED 记录，不触碰方案与已采用结果；
- 成功计算在创建时冻结计算时刻的方案修订号、方案负载与最小割结果，
  三者共同组成不可混合的快照；采用时只使用该冻结快照，绝不读取
  "当前方案"，因此计算后方案被修订也不会污染快照；修订号与方案
  内容一一对应后，计算记录可始终追溯到确定版本；
- 采用历史 adoption_events 对 computation_id 永久唯一，保证
  "每个成功计算至多采用一次"——即使该采用后来被其他计算替换，
  原计算仍不可再次采用；
- 采用事务先对 plans 行加行锁串行化并发采用：仅当当前生效采用在
  持锁前后一致时才允许写入，否则以 409 ADOPTION_CONFLICT 拒绝，
  保证并发采用的返回结果确定且与最终记录一致。任何失败/冲突都
  回滚，当前方案与原采用快照不变；
- 执行复核同样先对 plans 行加锁，再读取当前生效采用快照。校验、
  残图最小割与写入只使用该快照中的冻结方案，不读取当前方案；失败
  时回滚，不留下半条复核，也不修改方案、计算记录或采用快照。
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from . import models
from .errors import ApiError
from .flow import solve_min_cut
from .validation import validate_closed_segments


def get_plan_or_404(db, plan_id):
    plan = db.get(models.Plan, plan_id)
    if plan is None:
        raise ApiError(404, "PLAN_NOT_FOUND", f"plan {plan_id!r} does not exist")
    return plan


def _reread_plan_in_transaction(db, plan_id):
    """在本写入事务内重读方案行。

    替换路径的 UPDATE 已持有该行行锁直至提交，此处读到的即本次
    事务将要提交的修订号与负载；直接以它构造响应，杜绝"提交后被
    他人覆盖、响应却属于另一版本"。
    """
    return (
        db.query(models.Plan)
        .filter(models.Plan.plan_id == plan_id)
        .populate_existing()
        .one()
    )


def save_plan(db, plan_id, canonical_payload, expected_revision=None):
    """保存（新建或整版替换）方案；payload 必须先通过校验。

    每次被接受的写入都对应唯一的新修订号，成功响应的内容就是本次
    事务提交的内容：

    - 携带 expected_revision：仅当当前修订号与之相等时才整版替换，
      "比对修订号 + 写入"由单条条件 UPDATE 原子完成。命中 0 行时
      方案不存在（404 PLAN_NOT_FOUND）或修订号已被并发请求推进
      （409 REVISION_CONFLICT，details 中带 REVISION_MISMATCH），
      落败写入回滚，不改写现有方案；
    - 未携带 expected_revision（或显式为 null）：保持旧语义——存在
      则整版替换、否则新建。替换由单条
      ``UPDATE ... SET revision = revision + 1`` 原子完成，两个并发
      替换因此一定得到不同修订号；新建依赖主键唯一约束，并发首次
      创建同一方案时落败事务回滚并以 409 REVISION_CONFLICT
      （details 中带 PLAN_ALREADY_EXISTS）拒绝，绝不暴露数据库异常。
    """
    if expected_revision is not None:
        result = db.execute(
            update(models.Plan)
            .where(
                models.Plan.plan_id == plan_id,
                models.Plan.revision == expected_revision,
            )
            .values(
                revision=expected_revision + 1,
                payload=canonical_payload,
            )
        )
        if result.rowcount == 0:
            # 条件 UPDATE 未命中：行不存在或修订号已被他人推进。
            # 先回滚释放可能持有的行锁，再在新事务中区分两种情况。
            db.rollback()
            if db.get(models.Plan, plan_id) is None:
                raise ApiError(
                    404, "PLAN_NOT_FOUND", f"plan {plan_id!r} does not exist"
                )
            raise ApiError(
                409,
                "REVISION_CONFLICT",
                f"plan {plan_id!r} was modified after revision "
                f"{expected_revision} was read; re-read the current plan and retry",
                details=[
                    {
                        "code": "REVISION_MISMATCH",
                        "field": "expected_revision",
                        "message": (
                            f"expected revision {expected_revision} no longer "
                            "matches the current revision"
                        ),
                    }
                ],
            )
        plan = _reread_plan_in_transaction(db, plan_id)
        db.commit()
        return plan

    # 未携带修订号：旧语义的"存在则整版替换，否则新建"。
    result = db.execute(
        update(models.Plan)
        .where(models.Plan.plan_id == plan_id)
        .values(
            revision=models.Plan.revision + 1,
            payload=canonical_payload,
        )
    )
    if result.rowcount == 0:
        # 行尚不存在：尝试新建。revision=1 与本次提交的负载即响应。
        plan = models.Plan(plan_id=plan_id, revision=1, payload=canonical_payload)
        db.add(plan)
        try:
            db.commit()
        except IntegrityError:
            # 并发首次创建同一方案：主键唯一约束拒绝落败事务。
            # 回滚后报告明确冲突，绝不改写已由胜者落库的方案。
            db.rollback()
            raise ApiError(
                409,
                "REVISION_CONFLICT",
                f"plan {plan_id!r} already exists; re-read the current plan "
                "and retry with expected_revision",
                details=[
                    {
                        "code": "PLAN_ALREADY_EXISTS",
                        "field": "expected_revision",
                        "message": f"plan {plan_id!r} was created concurrently",
                    }
                ],
            )
        return plan
    plan = _reread_plan_in_transaction(db, plan_id)
    db.commit()
    return plan


def compute(db, plan_id):
    """对当前方案执行最小割计算并持久化计算记录。

    计算记录冻结计算时刻的方案修订号 (plan_revision)、方案负载
    (plan_payload) 与最小割结果 (result)；之后方案再被修订也不影响
    该记录，采用时三者始终属于同一版本。
    """
    plan = get_plan_or_404(db, plan_id)
    computation_id = uuid.uuid4().hex
    # 冻结快照输入，避免与随后的写操作共享可变引用
    plan_payload = plan.payload
    plan_revision = plan.revision
    try:
        result = solve_min_cut(plan_payload)
    except Exception as exc:  # 已校验输入不应失败；兜底记录失败
        computation = models.Computation(
            computation_id=computation_id,
            plan_id=plan.plan_id,
            plan_revision=plan_revision,
            plan_payload=plan_payload,
            status="FAILED",
            result=None,
            error={"code": "INTERNAL_ERROR", "message": str(exc)},
        )
        db.add(computation)
        db.commit()
        raise ApiError(500, "COMPUTATION_FAILED", "min-cut computation failed")
    computation = models.Computation(
        computation_id=computation_id,
        plan_id=plan.plan_id,
        plan_revision=plan_revision,
        plan_payload=plan_payload,
        status="SUCCESS",
        result=result,
        error=None,
    )
    db.add(computation)
    db.commit()
    db.refresh(computation)
    return computation


def get_computation_or_404(db, plan_id, computation_id):
    computation = db.get(models.Computation, computation_id)
    if computation is None or computation.plan_id != plan_id:
        raise ApiError(
            404,
            "COMPUTATION_NOT_FOUND",
            f"computation {computation_id!r} does not exist for plan {plan_id!r}",
        )
    return computation


def _get_adoption_row(db, plan_id):
    return db.get(models.Adoption, plan_id)


def _get_adoption_event(db, computation_id):
    return (
        db.query(models.AdoptionEvent)
        .filter(models.AdoptionEvent.computation_id == computation_id)
        .first()
    )


def adopt(db, plan_id, computation_id):
    """采用一次成功计算，保存计算时刻冻结的完整快照。

    规则：
    - 只能采用本方案状态为 SUCCESS 的计算；
    - 每个成功计算至多采用一次（adoption_events 永久唯一），即使其
      先前采用已被替换，再次采用仍返回 409 COMPUTATION_ALREADY_ADOPTED；
    - 新采用替换该方案当前生效采用；
    - 并发采用由 plans 行锁串行化；若当前生效采用在等待锁期间被他人
      改变，则本请求以 409 ADOPTION_CONFLICT 失败，绝不出现"响应成功
      但最终记录属于另一次采用"或 500。
    """
    # ---- 加锁前校验（不与当前方案/修订发生任何关联，只读计算快照）----
    get_plan_or_404(db, plan_id)
    computation = get_computation_or_404(db, plan_id, computation_id)
    if computation.status != "SUCCESS":
        raise ApiError(
            409,
            "COMPUTATION_NOT_ADOPTABLE",
            f"computation {computation_id!r} did not succeed",
        )
    if _get_adoption_event(db, computation_id) is not None:
        raise ApiError(
            409,
            "COMPUTATION_ALREADY_ADOPTED",
            f"computation {computation_id!r} has already been adopted",
        )

    # 并发采用同一方案时，以当前生效采用的 computation_id 作为一致性令牌。
    # 首次采用时令牌为 None（此时尚无生效采用）。
    before_row = _get_adoption_row(db, plan_id)
    before_token = before_row.computation_id if before_row is not None else None

    # ---- 对方案行加锁，串行化同一方案上的所有采用 ----
    # SQLite 不支持 SELECT ... FOR UPDATE，SQLAlchemy 对其退化为普通查询；
    # 并发验收在真实 PostgreSQL 上进行。
    plan = (
        db.query(models.Plan)
        .filter(models.Plan.plan_id == plan_id)
        .with_for_update()
        .populate_existing()
        .one()
    )

    # ---- 持锁后复查（顺序有意义）----
    # 1) 同一计算并发重复采用：精确报 COMPUTATION_ALREADY_ADOPTED；
    if _get_adoption_event(db, computation_id) is not None:
        db.rollback()
        raise ApiError(
            409,
            "COMPUTATION_ALREADY_ADOPTED",
            f"computation {computation_id!r} has already been adopted",
        )
    # 2) 不同计算并发首次采用：等待期间生效采用若被改变，属并发冲突。
    after_row = _get_adoption_row(db, plan_id)
    after_token = after_row.computation_id if after_row is not None else None
    if after_token != before_token:
        db.rollback()
        raise ApiError(
            409,
            "ADOPTION_CONFLICT",
            "a concurrent adoption changed the current adopted result; "
            "re-query and retry with the intended computation",
        )

    adopted_at = datetime.now(timezone.utc)
    # 快照完全来自计算记录冻结的内容：计算时刻的修订号、方案负载与
    # 最小割结果同属一个版本，绝不使用当前 plan 的修订号或负载。
    snapshot = {
        "plan_id": plan.plan_id,
        "plan_revision": computation.plan_revision,
        "computation_id": computation_id,
        "adopted_at": adopted_at.isoformat(),
        "plan": computation.plan_payload,
        "result": computation.result,
    }

    event = models.AdoptionEvent(
        plan_id=plan.plan_id,
        computation_id=computation_id,
        plan_revision=computation.plan_revision,
        snapshot=snapshot,
        adopted_at=adopted_at,
    )
    db.add(event)
    if after_row is None:
        adoption = models.Adoption(
            plan_id=plan_id,
            computation_id=computation_id,
            snapshot=snapshot,
            adopted_at=adopted_at,
        )
        db.add(adoption)
    else:
        adoption = after_row
        adoption.computation_id = computation_id
        adoption.snapshot = snapshot
        adoption.adopted_at = adopted_at

    try:
        db.commit()
    except IntegrityError:
        # 极端竞态（历史唯一约束）下的最终防线：回滚，现状不变
        db.rollback()
        raise ApiError(
            409,
            "COMPUTATION_ALREADY_ADOPTED",
            f"computation {computation_id!r} has already been adopted",
        )
    db.refresh(adoption)
    return adoption


def get_adoption(db, plan_id):
    adoption = db.get(models.Adoption, plan_id)
    if adoption is None:
        raise ApiError(
            404,
            "ADOPTION_NOT_FOUND",
            f"plan {plan_id!r} has no adopted result",
        )
    return adoption


def _combined_witness(frozen_plan, closed_ids, additional_result):
    """构造由现场关闭与新增建议合并后的隔断见证。

    源侧集合沿用剩余有向网络上同一个最小源侧裁决；冻结方案中所有跨越
    该源侧边界的管段，恰好由"现场已关闭的跨越边 + 剩余网络新增建议边"
    组成。merged_segment_ids 还会包含虽未跨越最终边界但现场确实已关闭
    的冗余管段，完整记录本次执行集合。
    """
    side = set(additional_result["source_zones"])
    segment_by_id = {seg["id"]: seg for seg in frozen_plan["segments"]}
    closed_set = set(closed_ids)

    crossing_ids = [
        seg["id"]
        for seg in frozen_plan["segments"]
        if seg["from"] in side and seg["to"] not in side
    ]
    closed_crossing_ids = sorted(closed_set & set(crossing_ids))
    merged_ids = sorted(closed_set | additional_ids)
    return {
        "source_zones": additional_result["source_zones"],
        # 严格意义上的割：冻结方案中所有源侧 -> 汇侧的管段。
        "cut_segments": sorted(crossing_ids),
        # 现场执行集合；可能包含不在最终割边界上的冗余关闭管段。
        "merged_segment_ids": merged_ids,
        "closed_segment_ids": list(closed_ids),
        "closed_segments": list(closed_ids),
        "closed_cut_segments": closed_crossing_ids,
        "new_segments": additional_result["cut_segments"],
        "additional_segments": additional_result["cut_segments"],
        "additional_cut_segments": additional_result["cut_segments"],
        "closed_cost": sum(segment_by_id[sid]["cost"] for sid in closed_crossing_ids),
        "additional_cost": additional_result["total_cost"],
        "total_cost": sum(segment_by_id[sid]["cost"] for sid in crossing_ids),
    }


def execution_review_view(review):
    """对外响应；字段中的计划与结果均来自复核记录冻结的版本。"""
    witness = review.combined_witness
    return {
        "review_id": review.review_id,
        "plan_id": review.plan_id,
        "plan_revision": review.plan_revision,
        "computation_id": review.computation_id,
        "created_at": review.created_at.isoformat(),
        "plan": review.plan_snapshot,
        "adopted_result": review.adopted_result,
        "closed_segment_ids": list(review.closed_segment_ids),
        "closed_segments": list(review.closed_segment_ids),
        "additional_cut": review.additional_result,
        "result": review.additional_result,
        "new_segments": review.additional_result["cut_segments"],
        "new_segment_ids": list(review.additional_result["cut_segments"]),
        "recommended_segments": review.additional_result["cut_segments"],
        "additional_cut_segments": list(review.additional_result["cut_segments"]),
        "new_closed_segments": list(review.additional_result["cut_segments"]),
        "additional_segments": list(review.additional_result["cut_segments"]),
        "additional_cost": review.additional_cost,
        "incremental_cost": review.additional_cost,
        "combined_cut_witness": witness,
        # 兼容直观命名：现场已关闭、新增建议、追加费用、合并见证。
        "merged_cut_witness": witness,
        "combined_witness": witness,
        "witness": witness,
    }


def create_execution_review(db, plan_id, raw_body):
    """对当前已采用结果新增一次执行复核。

    复核只能引用该次采用冻结的方案：先对 plans 行加锁，再读取当前
    adoption，使复核与采用替换在 PostgreSQL 上串行化；随后校验、算法、
    持久化全部复用同一个 frozen_plan。任何校验/算法/写入错误都在事务内
    回滚，不留下半条 execution_reviews 记录，也不修改原方案、计算记录
    或采用快照。
    """
    get_plan_or_404(db, plan_id)

    # 与 adopt 使用同一把 plans 行锁，确保读到的采用快照在本事务内稳定。
    (
        db.query(models.Plan)
        .filter(models.Plan.plan_id == plan_id)
        .with_for_update()
        .one()
    )
    adoption = _get_adoption_row(db, plan_id)
    if adoption is None:
        db.rollback()
        raise ApiError(
            404,
            "ADOPTION_NOT_FOUND",
            f"plan {plan_id!r} has no adopted result to review",
        )

    snapshot = adoption.snapshot
    computation_id = snapshot["computation_id"]
    plan_revision = snapshot["plan_revision"]
    frozen_plan = snapshot["plan"]
    adopted_result = snapshot["result"]

    # 校验未知/重复 ID 时只查采用时冻结的管段，不查当前已修订方案。
    try:
        closed_segment_ids = validate_closed_segments(frozen_plan, raw_body)
    except ApiError:
        db.rollback()
        raise

    try:
        additional_result = solve_min_cut(frozen_plan, set(closed_segment_ids))
        witness = _combined_witness(
            frozen_plan, closed_segment_ids, additional_result
        )
    except Exception:
        db.rollback()
        raise ApiError(
            500, "COMPUTATION_FAILED", "execution review computation failed"
        )

    review_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    review = models.ExecutionReview(
        review_id=review_id,
        plan_id=plan_id,
        computation_id=computation_id,
        plan_revision=plan_revision,
        plan_snapshot=frozen_plan,
        adopted_result=adopted_result,
        closed_segment_ids=closed_segment_ids,
        additional_result=additional_result,
        additional_cost=additional_result["total_cost"],
        combined_witness=witness,
        created_at=created_at,
    )
    db.add(review)
    try:
        db.commit()
    except Exception:
        # 数据库写入失败时必须原子回滚；原方案/计算/采用均不得被改动。
        db.rollback()
        raise ApiError(
            500,
            "EXECUTION_REVIEW_WRITE_FAILED",
            "failed to persist the execution review",
        )
    db.refresh(review)
    return review


def get_execution_review_or_404(db, plan_id, review_id):
    review = db.get(models.ExecutionReview, review_id)
    if review is None or review.plan_id != plan_id:
        raise ApiError(
            404,
            "EXECUTION_REVIEW_NOT_FOUND",
            f"execution review {review_id!r} does not exist for plan {plan_id!r}",
        )
    return review
