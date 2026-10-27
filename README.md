# TinyZero

## Reproduction Log — Jialin Li (Fall 2026)

Independent reproduction of R1-Zero-style RLVR (GRPO + rule-based rewards) on small Qwen2.5 base models, built on this TinyZero recipe (veRL).

- **Goal**: verify reward growth, response-length dynamics, and emergent self-verification at small scale; then run one ablation (TBD: KL coefficient / reward shaping / task transfer).
- **Status** (2026-10-26): **smoke run passed** — 1,280 steps on Qwen2.5-0.5B, zero failures, full metrics logged (see [Results](#results-smoke-run-05b)); 3B training (2×H100) in progress.

## Results: smoke run (0.5B)

**Config**: Qwen2.5-0.5B, GRPO (`adv_estimator=grpo`, no value critic, no learned reward model), 327K countdown prompts, single H100 80GB, veRL+vLLM+FlashAttention, 2h walltime → **1,280 steps, zero failures**. Log: [`logs/smoke_0.5b_1280steps.out`](logs/smoke_0.5b_1280steps.out) · Curves: [`logs/smoke_0.5b_curves.svg`](logs/smoke_0.5b_curves.svg) · Plot script: [`scripts/plot_smoke.py`](scripts/plot_smoke.py)

| Phase | reward/mean | response length | Reading |
|---|---|---|---|
| step 1–5 | 0.025 | ~101K tokens/batch | cold start: rambly, format not learned |
| step 10–20 | 0.078 | ~51K | **format clicks**: reward triples, length halves (`<answer>` compliance) |
| step 20–30 | **0.087 (peak)** | ~45K | peak |
| step 50–1280 | 0.077 plateau | ~40K stable | 0.5B capability ceiling; entropy steady ≈1.5 (no collapse) |

**Takeaways**: ①pipeline verified end-to-end (rollout → log-prob → update, 1,280 steps clean); ②GRPO format alignment happens within ~25 steps; ③capability plateau at 0.5B matches the upstream note ("0.5B fails to learn reasoning") — reward growth now targets the 3B run; ④length compresses as format locks in, no entropy collapse observed.

## Results: 24h run (3B, GRPO)

**Config**: Qwen2.5-3B, GRPO (`adv_estimator=grpo`), 2×H100, 327K countdown prompts, 24h walltime → **5,499 steps, zero failures**. Log: [`logs/grpo3b_24h.out`](logs/grpo3b_24h.out) · Curves: [`logs/grpo3b_24h_curves.svg`](logs/grpo3b_24h_curves.svg)

| Stage | reward/mean | response length | entropy |
|---|---|---|---|
| 0–500 | 0.367 | ~58K | 0.104 |
| 500–1000 | **0.428–0.430 (peak)** | ~43K | 0.002 |
| 2000–3000 | 0.410 | ~43K | 0.0014 |
| 4000–5000 | 0.329 | ~43K | 0.0024 |
| 5000–5499 | 0.284 | (val段 ~95K) | 0.0024 |

**Reading**: ①3B does learn — reward climbed 0.37→0.44 (+19%) in the first ~1K steps, **5.6× the 0.5B ceiling (0.078)**; ②entropy collapsed from 0.104 to ~0.001 by step ~600 (exploration death), after which reward decayed 0.44→0.28 and the tail length re-inflated; ③diagnosis: exploration collapse, not a pipeline bug — next run is a controlled **entropy-coefficient ablation (0.001 → 0.005)** (job `grpo3b-ent5`) to test whether reward recovers past the first 1K steps.
- **Hardware**: 1–2× NVIDIA H100 80GB (Digital Research Alliance of Canada; Fir cluster for env/smoke, Nibi cluster for 3B runs), MIG slice for smoke tests.

### Environment recipe (Alliance Fir)

- **⚠️ NCCL fix（2026-10-25 实锤）**：computecanada torch wheel 烧死内置 NCCL 2.18.3（cvmfs 绝对路径 RPATH），与 Nibi 新驱动（CUDA 13.0）不兼容 → FSDP 初始化报 `named symbol not found`。解法：训练前 `export LD_PRELOAD=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcccore/cuda12.6/nccl/2.27.7/lib/libnccl.so.2.27.7`（ABI 兼容，实测 broadcast 通过）。**Fir/Nibi 同病同治**；MIG 实例与整卡均可。
- Modules: `python/3.11` `cuda/12.2` `gcc` `opencv/4.11.0` `arrow/17.0.0`
- Python env (venv on `/scratch`): `torch 2.4.1+computecanada`, `vllm 0.6.2+computecanada`, `flash-attn 2.5.7+computecanada`, `transformers 4.46.3`, `triton 3.0.0`, `datasets<4`, `verl` (editable, this repo)
- Deviations from upstream README (version drift, 2024→2026): torch 2.4.0→2.4.1, vllm 0.6.3→0.6.2, flash-attn from source→wheelhouse 2.5.7; PyTorch/vLLM installed from the Alliance wheelhouse (PyPI nvidia-* wheels are stubbed there).
- **⚠️ vllm 0.6.2 缺 `compilation/levels.py`**：wheelhouse 的 vllm 0.6.2 不带该模块，但 veRL vendored 代码（`third_party/vllm/vllm_v_0_6_3/`）按 0.6.3 API 引用 → `ModuleNotFoundError: vllm.compilation.levels`。解法：把上游 0.6.3 的 `vllm/compilation/levels.py`（5 行常量类）补进 venv 对应目录；import 链已验证（executor/worker/rollout 全通）。**venv 自包含清单**（清 PYTHONPATH 后必备）：numpy1.26.4、scipy、pandas、pyarrow(PyPI 轮)、sympy、mpmath、tokenizers(PyPI 0.20.3，wheelhouse 坏轮)、pillow/psutil/pyzmq/nest_asyncio。
- Data: countdown task, 327,680 train / 1,024 test prompts (`examples/data_preprocess/countdown.py`).
- Curves, logs, and analysis will be documented here as the run progresses.

---

> **⚠️ Deprecation Notice:** This repo is no longer actively maintained. For running RL experiments, please directly use the latest [veRL](https://github.com/volcengine/verl) library.
> For the archived original documentation, see [OLD_README.md](./OLD_README.md).

![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own.

You can experience the Aha moment yourself for < $30.

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

> 📢: We release [Adaptive Parallel Reasoning](https://github.com/Parallel-Reasoning/APR), where we explore a new dimension in scaling reasoning models.

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```
conda activate zero
```

For the following code, if you see out-of-VRAM, try adding `critic.model.enable_gradient_checkpointing=True` to the script, and check out the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643).

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with Qwen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledgements
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
