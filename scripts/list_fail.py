#!/usr/bin/env python3
"""列出 summary.json 中未全部成功的任务。

用法：python scripts/list_fail.py evidence/results/summary_libero_*.json
"""
import json
import sys


def main(paths):
    total_fail = 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for t in d["per_task"]:
            n_fail = t["total"] - t["successes"]
            if n_fail:
                total_fail += n_fail
                print(f"{t['suite']:<15} task {t['task_id']}  失败 {n_fail}/{t['total']}  {t['description']}")
    print(f"共 {total_fail} 次失败")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
