"""执行复核接口、冻结版本、持久化原子性与并发读取集成测试。"""

import threading

from sqlalchemy.exc import SQLAlchemyError

from app.db import SessionLocal
from app.models import (
    AdoptionEvent,
    Computation,
    ExecutionReview,
    Plan,
)

VALID_PLAN = {
    "zones": ["SRC1", "SRC2", "MID", "SAFE1", "SAFE2"],
    "segments": [
        {"id": "p1", "from": "SRC1", "to": "MID", "cost": 4},
        {"id": "p2", "from": "SRC2", "to": "MID", "cost": 6},
        {"id": "p3", "from": "MID", "to": "SAFE1", "cost": 5},
        {"id": "p4", "from": "MID", "to": "SAFE2", "cost": 7},
    ],
    "sources": ["SRC1", "SRC2"],
    "protections": ["SAFE1", "SAFE2"],
}
REVISED_PLAN = {
    **VALID_PLAN,
    "segments": [
        {"id": "p1", "from": "SRC1", "to": "MID", "cost": 4},
        {"id": "p2", "from": "SRC2", "to": "MID", "cost": 6},
        {"id": "p3", "from": "MID", "to": "SAFE1", "cost": 5},
        {"id": "p4", "from": "MID", "to": "SAFE2", "cost": 1},
    ],
}
ZERO_PLAN = {
    "zones": ["S", "A", "B", "T"],
    "segments": [
        {"id": "free-a", "from": "S", "to": "A", "cost": 0},
        {"id": "a-t", "from": "A", "to": "T", "cost": 5},
        {"id": "free-b", "from": "S", "to": "B", "cost": 0},
        {"id": "b-t", "from": "B", "to": "T", "cost": 0},
    ],
    "sources": ["S"],
    "protections": ["T"],
}


def put(client, pid, plan, revision=None):
    body = dict(plan)
    if revision is not None:
        body["expected_revision"] = revision
    return client.put(f"/plans/{pid}", json=body)


def compute(client, pid):
    return client.post(f"/plans/{pid}/computations").json()


def adopt(client, pid, cid):
    return client.post(f"/plans/{pid}/adopt", json={"computation_id": cid})


def review(client, pid, closed):
    return client.post(
        f"/plans/{pid}/execution-reviews",
        json={"closed_segment_ids": closed},
    )


def get_review(client, pid, rid):
    return client.get(f"/plans/{pid}/execution-reviews/{rid}")


def count_reviews(pid):
    db = SessionLocal()
    try:
        return (
            db.query(ExecutionReview)
            .filter(ExecutionReview.plan_id == pid)
            .count()
        )
    finally:
        db.close()


def prepare_adopted(client, pid="rv", plan=VALID_PLAN):
    put(client, pid, plan)
    computation = compute(client, pid)
    response = adopt(client, pid, computation["computation_id"])
    assert response.status_code == 200
    return computation, response.json()


def test_non_recommended_segment_closed_first_uses_residual_frozen_graph(client):
    pid = "field-close"
    _, adopted = prepare_adopted(client, pid)
    assert adopted["result"]["cut_segments"] == ["p1", "p2"]

    response = review(client, pid, ["p3"])
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["closed_segment_ids"] == ["p3"]
    assert body["closed_segments"] == ["p3"]
    assert body["additional_cut_segments"] == ["p4"]
    assert body["additional_cost"] == 7
    assert body["additional_cut"] == {
        "source_zones": ["MID", "SRC1", "SRC2"],
        "cut_segments": ["p4"],
        "total_cost": 7,
    }

    witness = body["combined_cut_witness"]
    assert witness == body["merged_cut_witness"]
    assert witness["source_zones"] == ["MID", "SRC1", "SRC2"]
    assert witness["cut_segments"] == ["p3", "p4"]
    assert witness["merged_segment_ids"] == ["p3", "p4"]
    assert witness["closed_cut_segments"] == ["p3"]
    assert witness["additional_cut_segments"] == ["p4"]
    assert witness["closed_cost"] == 5
    assert witness["additional_cost"] == 7
    assert witness["total_cost"] == 12

    # 复核记录冻结的是采用时第 1 版及其计算。
    assert body["plan_revision"] == 1
    assert body["plan"] == VALID_PLAN
    assert body["adopted_result"] == adopted["result"]
    assert body["computation_id"] == adopted["computation_id"]

    fetched = get_review(client, pid, body["review_id"])
    assert fetched.status_code == 200
    assert fetched.json() == body

    alias = client.post(
        f"/plans/{pid}/adoption/reviews",
        json={"closed_segments": ["p3"]},
    )
    assert alias.status_code == 200
    assert alias.json()["plan_revision"] == body["plan_revision"]
    assert alias.json()["additional_cut_segments"] == body["additional_cut_segments"]
    assert (
        client.get(f"/plans/{pid}/reviews/{alias.json()['review_id']}").json()
        == alias.json()
    )


def test_closing_all_adopted_cut_segments_requires_no_addition(client):
    pid = "already-cut"
    prepare_adopted(client, pid)

    response = review(client, pid, ["p2", "p1"])
    assert response.status_code == 200
    body = response.json()
    assert body["closed_segment_ids"] == ["p1", "p2"]
    assert body["additional_cut_segments"] == []
    assert body["additional_cost"] == 0
    assert body["additional_cut"] == {
        "source_zones": ["SRC1", "SRC2"],
        "cut_segments": [],
        "total_cost": 0,
    }
    witness = body["combined_cut_witness"]
    assert witness["cut_segments"] == ["p1", "p2"]
    assert witness["merged_segment_ids"] == ["p1", "p2"]
    assert witness["total_cost"] == 10


def test_zero_cost_additional_edge_is_selected_by_same_tie_break(client):
    pid = "zero-cost"
    prepare_adopted(client, pid, ZERO_PLAN)

    # 非建议的高费用边先关闭后，仍需追加零费用入口边以完成最小割。
    response = review(client, pid, ["a-t"])
    assert response.status_code == 200
    body = response.json()
    assert body["additional_cut"] == {
        "source_zones": ["S"],
        "cut_segments": ["free-a", "free-b"],
        "total_cost": 0,
    }
    assert body["additional_cost"] == 0
    assert body["combined_cut_witness"]["cut_segments"] == ["a-t", "free-a", "free-b"]


def test_revision_after_adoption_does_not_mix_same_named_segment(client):
    pid = "frozen-review"
    computation_v1, adopted_v1 = prepare_adopted(client, pid, VALID_PLAN)
    assert adopted_v1["plan_revision"] == 1

    response = put(client, pid, REVISED_PLAN, revision=1)
    assert response.status_code == 200
    assert response.json()["revision"] == 2
    computation_v2 = compute(client, pid)
    assert computation_v2["plan_revision"] == 2
    assert computation_v2["result"]["cut_segments"] == ["p3", "p4"]
    assert computation_v2["result"]["total_cost"] == 6

    # 当前第 2 版会选择 p3+p4=6；但生效采用冻结的是第 1 版，p4 仍为 7，
    # 因此关闭 p3 后追加建议只能是第 1 版剩余网络上的 p4=7。
    response = review(client, pid, ["p3"])
    assert response.status_code == 200
    body = response.json()
    assert body["computation_id"] == computation_v1["computation_id"]
    assert body["plan_revision"] == 1
    assert body["plan"] == VALID_PLAN
    assert body["additional_cut_segments"] == ["p4"]
    assert body["additional_cost"] == 7

    # 复核不得修改原方案、计算记录或采用快照。
    assert client.get(f"/plans/{pid}").json()["revision"] == 2
    assert client.get(f"/plans/{pid}").json()["plan"] == REVISED_PLAN
    assert client.get(f"/plans/{pid}/adoption").json() == adopted_v1
    old_computation = client.get(
        f"/plans/{pid}/computations/{computation_v1['computation_id']}"
    ).json()
    assert old_computation["plan_revision"] == 1
    assert old_computation["plan_payload"] == VALID_PLAN
    assert old_computation["result"] == adopted_v1["result"]


def test_review_can_be_read_from_new_session_after_restart_like_reset(client):
    pid = "persisted-review"
    _, adopted = prepare_adopted(client, pid)
    created = review(client, pid, ["p3"]).json()

    # 模拟服务重启后的新会话；SQLite 测试库为进程内共享持久连接，
    # PostgreSQL 下同一 ID 同样可在独立连接/重启后读取。
    db = SessionLocal()
    try:
        row = db.get(ExecutionReview, created["review_id"])
        assert row is not None
        assert row.plan_id == pid
        assert row.plan_revision == 1
        assert row.plan_snapshot == VALID_PLAN
        assert row.adopted_result == adopted["result"]
        assert row.closed_segment_ids == ["p3"]
        assert row.additional_cost == 7
        assert row.combined_witness["cut_segments"] == ["p3", "p4"]
    finally:
        db.close()

    fetched = get_review(client, pid, created["review_id"])
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_unknown_duplicate_and_malformed_segments_leave_no_review(client):
    pid = "invalid-review"
    _, _ = prepare_adopted(client, pid)

    response = review(client, pid, ["ghost", "p1", "p1", "bad id"])
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    codes = {detail["code"] for detail in error["details"]}
    assert codes == {"UNKNOWN_SEGMENT", "DUPLICATE_SEGMENT_ID", "INVALID_SEGMENT_ID"}
    assert count_reviews(pid) == 0

    response = review(client, pid, "p1")
    assert response.status_code == 422
    codes = {detail["code"] for detail in response.json()["error"]["details"]}
    assert codes == {"INVALID_CLOSED_SEGMENTS_FIELD"}
    assert count_reviews(pid) == 0

    response = client.post(
        f"/plans/{pid}/execution-reviews", json={"closed_segment_ids": None}
    )
    assert response.status_code == 422
    assert count_reviews(pid) == 0

    # 失败不影响采用快照；随后的合法复核仍可成功。
    assert review(client, pid, []).status_code == 200
    assert count_reviews(pid) == 1


def test_closed_segments_alias_is_accepted(client):
    pid = "field-alias"
    prepare_adopted(client, pid)
    response = client.post(
        f"/plans/{pid}/execution-reviews",
        json={"closed_segments": ["p1", "p2"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["closed_segment_ids"] == ["p1", "p2"]
    assert body["additional_cut_segments"] == []
    assert body["additional_cost"] == 0


def test_review_without_adoption_is_404_and_creates_nothing(client):
    pid = "no-adoption"
    put(client, pid, VALID_PLAN)
    computation = compute(client, pid)
    assert computation["status"] == "SUCCESS"

    response = review(client, pid, ["p1"])
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ADOPTION_NOT_FOUND"
    assert count_reviews(pid) == 0
    assert client.get(f"/plans/{pid}/adoption").status_code == 404


def test_unknown_review_and_other_plan_are_404(client):
    pid = "review-404"
    prepare_adopted(client, pid)
    created = review(client, pid, []).json()

    response = get_review(client, pid, "0" * 32)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "EXECUTION_REVIEW_NOT_FOUND"

    put(client, "other-plan", VALID_PLAN)
    other_cid = compute(client, "other-plan")["computation_id"]
    assert adopt(client, "other-plan", other_cid).status_code == 200
    response = get_review(client, "other-plan", created["review_id"])
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "EXECUTION_REVIEW_NOT_FOUND"


def test_database_write_failure_is_atomic(client, monkeypatch):
    pid = "write-fail"
    _, adopted = prepare_adopted(client, pid)

    from app import services

    db = SessionLocal()

    def failing_commit(*args, **kwargs):
        raise SQLAlchemyError("simulated review write failure")

    monkeypatch.setattr(db, "commit", failing_commit)
    try:
        services.create_execution_review(db, pid, {"closed_segment_ids": ["p3"]})
        raise AssertionError("expected execution review write failure")
    except Exception as exc:
        assert exc.status_code == 500
        assert exc.code == "EXECUTION_REVIEW_WRITE_FAILED"
    finally:
        monkeypatch.undo()
        db.close()

    # SQLAlchemy 在失败后已 rollback；失败事务未留下半条复核记录。
    assert count_reviews(pid) == 0
    assert client.get(f"/plans/{pid}").json()["plan"] == VALID_PLAN
    assert client.get(f"/plans/{pid}/adoption").json() == adopted

    check_db = SessionLocal()
    try:
        assert check_db.query(Plan).filter(Plan.plan_id == pid).count() == 1
        assert (
            check_db.query(Computation).filter(Computation.plan_id == pid).count() >= 1
        )
        assert (
            check_db.query(AdoptionEvent)
            .filter(AdoptionEvent.plan_id == pid)
            .count()
            == 1
        )
    finally:
        check_db.close()

    response = review(client, pid, ["p3"])
    assert response.status_code == 200
    assert count_reviews(pid) == 1


def test_concurrent_reads_return_same_frozen_review(client):
    pid = "concurrent-read"
    prepare_adopted(client, pid)
    created = review(client, pid, ["p3"]).json()
    expected = get_review(client, pid, created["review_id"]).json()
    barrier = threading.Barrier(8)
    results = {}
    errors = []

    def reader(index):
        try:
            barrier.wait(timeout=10)
            db = SessionLocal()
            try:
                from app import services

                row = services.get_execution_review_or_404(
                    db, pid, created["review_id"]
                )
                results[index] = services.execution_review_view(row)
            finally:
                db.close()
        except Exception as exc:  # 并发读不应产生异常
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert set(results) == set(range(8))
    assert all(result == expected for result in results.values())
    assert all(result["plan"] == VALID_PLAN for result in results.values())
