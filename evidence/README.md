# 运行证明材料

## 复现录屏

录屏链接：[百度网盘](https://pan.baidu.com/s/1QIPjQF_tS3U34DCs5e2FuQ?pwd=mwfk)（提取码 `mwfk`）

录制时间：2026-11-19 17:47（北京时间），同一台云服务器（NVIDIA A10 24 GB，8 vCPU）。录屏内容依次为：

1. `git log -1 --oneline` → GalaxeaVLA commit `89f2322`，未修改代码
2. `sha256sum -c model.pt.sha256` → `model.pt: OK`
3. 现场运行官方评测脚本 `scripts/run/eval_libero.sh`：libero_goal，10 个任务 × 4 次，4 路并行，保存 rollout 视频。右侧终端同步显示 `nvidia-smi`
4. 现场评测结果：**39/40（97.5%）**。唯一一次失败是 task 3「open the top drawer and put the bowl inside」，与全量评测中“抽屉开口处取放更易失败”的观察一致
5. 展示此前完成的全量评测（2000 个 episode）：各 suite 耗时、SUITE SUMMARY、`summary.json` 的修改时间与数值、视频文件数（2000）
6. 在 Notebook 中播放本次现场评测的 rollout 视频，以及全量评测中的一个失败案例

录屏终端的完整文字输出见 [`logs/record_2026-11-19_terminal.txt`](logs/record_2026-11-19_terminal.txt)，可与录屏逐行对照。

> 录屏选择每个任务 4 次而不是 1 次，原因是官方评测脚本在 `--num_trials` 小于 `--num_parallel` 时成功率统计有误（初始化时给每个并行环境都分配了 trial，成功数会超过 `num_trials`，成功率可超过 100%）。全量评测使用 `num_trials=50 ≥ num_parallel=4`，不受影响，每个 suite 的 total 均为 500，视频文件共 2000 个。

## 文件

| 文件 | 内容 |
|---|---|
| `results/summary_libero_{spatial,object,goal,10}.json` | 全量评测结果（官方脚本输出的 `summary.json`，含每个任务的成功数） |
| `results/summary_record_libero_goal.json` | 录屏时现场评测的结果（39/40） |
| `logs/full_libero_*.log` | 全量评测各 suite 的汇总输出 |
| `logs/full_timing.txt` | 各 suite 退出码与耗时 |
| `logs/gpu_full.csv` | 全量评测期间每 5 秒记录的显存与 GPU 利用率 |
| `logs/gpu_smoke.csv` | 冒烟测试（1 路环境）期间的显存与 GPU 利用率 |
| `logs/record_2026-11-19_terminal.txt` | 录屏时终端的完整输出 |

快速核对：

```bash
python - <<'EOF'
import json, glob
for f in sorted(glob.glob("results/summary_libero_*.json")):
    d = json.load(open(f)); print(f, d["grand_successes"], "/", d["grand_total"])
EOF
# 合计 1974 / 2000 = 98.70%
python ../scripts/articulated_stats.py results/summary_libero_*.json
```
