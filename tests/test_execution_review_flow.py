"""执行复核算法的小图独立枚举对拍。

对每张随机小图和每个"现场已关闭管段"子集：
- 先在剩余有向网络中用 BFS 判定实际可达源侧；若已经无污染源到保护区
  路径，期望新增清单为空、追加费用为 0；
- 否则枚举所有合法源侧集合求最低费用，并取全部最低割源侧的交集，
  与 ``solve_min_cut(plan, removed)`` 对拍；
- 最后删除"现场已关闭 + 新增建议"后验证全部污染路径确实被切断。
"""

import random
from collections import deque
from itertools import combinations

from app.flow import solve_min_cut


def physical_reach(plan, active_ids):
    adjacency = {zone: [] for zone in plan["zones"]}
    for seg in plan["segments"]:
        if seg["id"] in active_ids:
            adjacency[seg["from"]].append(seg["to"])
    seen = set(plan["sources"])
    queue = deque(plan["sources"])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def brute_force_residual_cut(plan, removed_ids):
    zones = plan["zones"]
    active = [seg for seg in plan["segments"] if seg["id"] not in removed_ids]
    active_ids = {seg["id"] for seg in active}
    reachable = physical_reach(plan, active_ids)

    # 已隔断：物理可达集合就是唯一的最小源侧见证，无需再新增边。
    if reachable.isdisjoint(plan["protections"]):
        return {
            "source_zones": sorted(reachable),
            "cut_segments": [],
            "total_cost": 0,
        }

    index = {zone: i for i, zone in enumerate(zones)}
    required_masks = [1 << index[zone] for zone in plan["sources"]]
    forbidden = {index[zone] for zone in plan["protections"]}
    cost_by_id = {seg["id"]: seg["cost"] for seg in active}

    best_cost = None
    intersection = (1 << len(zones)) - 1
    for mask in range(1 << len(zones)):
        # 合法割必须包含全部污染源并排除全部保护区。
        if any(not mask & required_mask for required_mask in required_masks):
            continue
        if any((mask >> idx) & 1 for idx in forbidden):
            continue
        side = {zone for zone, i in index.items() if (mask >> i) & 1}
        cut = {
            seg["id"]
            for seg in active
            if seg["from"] in side and seg["to"] not in side
        }
        cost = sum(cost_by_id[seg_id] for seg_id in cut)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            intersection = mask
        elif cost == best_cost:
            # Dinic 的源侧为所有最优源侧的交集。
            intersection &= mask

    side = {zone for zone, i in index.items() if (intersection >> i) & 1}
    best_cut = sorted(
        seg["id"]
        for seg in active
        if seg["from"] in side and seg["to"] not in side
    )
    return {
        "source_zones": sorted(side),
        "cut_segments": best_cut,
        "total_cost": best_cost,
    }


def assert_combined_removal_disconnects(plan, removed_ids, result):
    removed = set(removed_ids) | set(result["cut_segments"])
    all_ids = {seg["id"] for seg in plan["segments"]}
    reachable = physical_reach(plan, all_ids - removed)
    assert reachable.isdisjoint(plan["protections"])


def make_plan(rng, n, edge_count):
    zones = [f"Z{i}" for i in range(n)]
    segments = []
    used = set()
    while len(segments) < edge_count:
        frm, to = rng.sample(range(n), 2)
        # 同一对方向只生成一条边，降低组合数但仍覆盖分支选择；
        # 平行管段由单独 API 用例覆盖。
        key = (frm, to)
        if key in used:
            continue
        used.add(key)
        roll = rng.random()
        if roll < 0.25:
            cost = 0
        elif roll < 0.9:
            cost = rng.randint(1, 6)
        else:
            cost = 10
        index = len(segments)
        segments.append(
            {
                "id": f"e{index}",
                "from": f"Z{frm}",
                "to": f"Z{to}",
                "cost": cost,
            }
        )
    source = 0
    protection = n - 1
    return {
        "zones": zones,
        "segments": segments,
        "sources": [f"Z{source}"],
        "protections": [f"Z{protection}"],
    }


def test_small_graph_all_removed_subsets_against_enumeration():
    rng = random.Random(0x0E1E1E1E)
    case_no = 0
    for n, edge_count in ((3, 6), (4, 9), (5, 11)):
        plan = make_plan(rng, n=n, edge_count=edge_count)
        original = solve_min_cut(plan)
        all_ids = {seg["id"] for seg in plan["segments"]}
        assert physical_reach(
            plan, all_ids - set(original["cut_segments"])
        ).isdisjoint(plan["protections"])

        ids = [seg["id"] for seg in plan["segments"]]
        for size in range(len(ids) + 1):
            for chosen in combinations(ids, size):
                removed = set(chosen)
                result = solve_min_cut(plan, removed)
                expected = brute_force_residual_cut(plan, removed)
                assert result == expected, (n, chosen)
                assert_combined_removal_disconnects(plan, removed, result)

                # 显式覆盖：关闭一条不在原方案建议中的管段时，结果只能
                # 来自剩余网络，而不是简单沿用原 cut。
                non_suggested = [
                    sid for sid in chosen if sid not in original["cut_segments"]
                ]
                if non_suggested:
                    assert set(result["cut_segments"]).isdisjoint(removed)
                case_no += 1
    assert case_no == (1 << 6) + (1 << 9) + (1 << 11)


def test_zero_cost_edge_only_added_when_needed():
    plan = {
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
    assert solve_min_cut(plan)["cut_segments"] == ["free-a", "free-b"]

    # 先关闭非原建议的高费用边 a-t，另一路径仍需通过最小源侧裁决
    # 追加两条零费用入口边 free-a/free-b。
    result = solve_min_cut(plan, {"a-t"})
    assert result == {
        "source_zones": ["S"],
        "cut_segments": ["free-a", "free-b"],
        "total_cost": 0,
    }

    # 两条到 T 的建议边均已现场关闭：不得为了走形式追加零费用边。
    result = solve_min_cut(plan, {"b-t", "free-a"})
    assert result == {
        "source_zones": ["B", "S"],
        "cut_segments": [],
        "total_cost": 0,
    }
    assert_combined_removal_disconnects(plan, {"b-t", "free-a"}, result)
