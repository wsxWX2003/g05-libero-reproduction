#!/bin/bash
# G0.5 LIBERO 仿真复现 —— 云服务器环境搭建脚本（按阶段运行）
#
# 用法（建议先 tmux new -s g05，断线不中断）：
#   bash setup_g05_libero.sh env      # 阶段1：系统依赖 + 仓库 + Python 环境（无卡模式即可）
#   bash setup_g05_libero.sh ckpt     # 阶段2：只下载 LIBERO 需要的权重（无卡模式即可）
#   bash setup_g05_libero.sh libero   # 阶段3：安装 LIBERO 模拟器并打兼容补丁（无卡模式即可）
#   bash setup_g05_libero.sh smoke    # 阶段4：冒烟测试（需要 GPU）
#
# 所有输出自动追加到 $WORK/logs/ 下的日志文件。

set -eo pipefail
STAGE=${1:?请指定阶段: env | ckpt | libero | smoke}
WORK=${WORK:-/mnt/workspace}           # 本次在阿里云 PAI-DSW 上运行；其他平台改成自己的数据盘路径
mkdir -p "$WORK/logs"
exec > >(tee -a "$WORK/logs/${STAGE}_$(date +%m%d_%H%M).log") 2>&1

# 国内网络环境：PyPI / Python / HF 走镜像
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$WORK/hf_cache
export UV_PYTHON_INSTALL_DIR=$WORK/uv_python   # uv 管理的 Python 也放数据盘，避免停机重置系统盘后 .venv 失效
export UV_CACHE_DIR=$WORK/uv_cache
export MUJOCO_GL=egl                    # 无显示器的服务器用 EGL 离屏渲染
export PYOPENGL_PLATFORM=egl

REPO=$WORK/GalaxeaVLA
echo "===== 阶段: $STAGE | $(date) ====="

case "$STAGE" in
env)
    apt-get update && apt-get install -y iproute2 ffmpeg libegl1 libgl1 tmux git cmake build-essential \
        libegl1-mesa-dev libgl1-mesa-dev libglvnd-dev
    command -v uv >/dev/null || pip install uv
    cd "$WORK"
    [ -d GalaxeaVLA ] || git clone https://github.com/OpenGalaxea/GalaxeaVLA
    cd "$REPO"
    git log -1 --oneline                # 本次复现使用 89f2322
    uv sync --index-strategy unsafe-best-match
    source .venv/bin/activate
    python -c "import torch, g05; print('torch', torch.__version__, '| cuda可用:', torch.cuda.is_available()); print('g05 ok')"
    echo ">>> 无卡模式下 cuda可用=False 是正常的"
    ;;
ckpt)
    cd "$REPO" && source .venv/bin/activate
    # 下载完成后务必校验 sha256（见 README）。
    # 不要用 aria2 -c 接续 hf 下载了一半的 .incomplete 文件：hf/Xet 乱序写入，会留下全零空洞。
    # 每个文件模式单独下载一次：部分 CLI 版本重复传 --include 时只保留最后一个
    for PAT in "g05-libero/*" "action_tokenizer.pt" "qwen3_5_2b_base_processor/*"; do
        if command -v hf >/dev/null; then
            hf download OpenGalaxea/G05 --local-dir checkpoints --include "$PAT"
        else
            huggingface-cli download OpenGalaxea/G05 --repo-type model --local-dir checkpoints --include "$PAT"
        fi
    done
    ln -sf ../action_tokenizer.pt checkpoints/g05-libero/action_tokenizer.pt
    ln -sfn ../qwen3_5_2b_base_processor checkpoints/g05-libero/hf_processor
    ls -laL checkpoints/g05-libero
    du -sh checkpoints
    ;;
libero)
    cd "$WORK"
    [ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd "$REPO" && source .venv/bin/activate
    uv pip install -e ../LIBERO --no-deps   # --no-deps：避免 LIBERO 的旧依赖覆盖本仓库锁定的版本
    # LIBERO 顶层 libero/ 没有 __init__.py，新版 setuptools 的 editable 安装找不到它；用 .pth 把源码目录加入 sys.path
    SP=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$WORK/LIBERO" > "$SP/libero_path.pth"
    # PyTorch >= 2.6 的 torch.load 默认 weights_only=True，无法加载 LIBERO 的初始状态文件（numpy 数组）。只放开这一处。
    sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
        "$WORK/LIBERO/libero/libero/benchmark/__init__.py"
    grep -n "weights_only" "$WORK/LIBERO/libero/libero/benchmark/__init__.py"
    python -c "import libero; print('libero ok')"
    ;;
smoke)
    apt-get install -y iproute2 ffmpeg libegl1 libgl1 >/dev/null 2>&1 || { apt-get update && apt-get install -y iproute2 ffmpeg libegl1 libgl1; }
    cd "$REPO" && source .venv/bin/activate
    nvidia-smi
    export LIBERO_CONFIG_PATH=$REPO/experiments/libero
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
        --suites "libero_goal" --num_trials 1 --num_parallel 1 \
        --save_videos --output_dir outputs/smoke
    cat outputs/smoke/summary.json
    echo
    echo ">>> 冒烟测试通过后，用 scripts/run_full.sh 跑全量评测（8 vCPU 实例并行数用 4）"
    echo ">>> 注意：--num_trials 不要小于 --num_parallel，否则官方脚本的成功率统计会出错（可超过 100%）"
    ;;
*)
    echo "未知阶段: $STAGE"; exit 1 ;;
esac
echo "===== 阶段 $STAGE 完成 | $(date) ====="
