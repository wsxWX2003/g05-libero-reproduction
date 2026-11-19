#!/usr/bin/env python3
"""铰接物体任务 vs 其他任务的失败率对比（Fisher 精确检验）。

用法：python scripts/articulated_stats.py evidence/results/summary_libero_*.json
注意：铰接分组是在看过失败分布之后确定的（事后分组），p 值偏乐观，仅作探索性参考。
"""
import json
import sys

from scipy.stats import fisher_exact

# 涉及抽屉、微波炉、灶台旋钮等铰接物体的 7 个任务：(suite, task_id)
ARTICULATED = {
    ("libero_spatial", 4),  # bowl in the top drawer -> plate
    ("libero_goal", 0),     # open the middle drawer
    ("libero_goal", 3),     # open the top drawer and put the bowl inside
    ("libero_goal", 7),     # turn on the stove
    ("libero_10", 2),       # turn on the stove and put the moka pot on it
    ("libero_10", 3),       # bowl in the bottom drawer and close it
    ("libero_10", 9),       # mug in the microwave and close it
}


def main(paths):
    fail = {True: 0, False: 0}
    total = {True: 0, False: 0}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for t in d["per_task"]:
            key = (t["suite"], t["task_id"]) in ARTICULATED
            total[key] += t["total"]
            fail[key] += t["total"] - t["successes"]
    for key, name in ((True, "铰接物体任务"), (False, "其他任务")):
        print(f"{name}: 失败 {fail[key]}/{total[key]} = {fail[key] / total[key]:.1%}")
    table = [[fail[True], total[True] - fail[True]], [fail[False], total[False] - fail[False]]]
    odds, p = fisher_exact(table)
    print(f"Fisher 精确检验: OR = {odds:.2f}, p = {p:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
