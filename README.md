# RLVR Reproduction with TinyZero

Jialin Li · Fall 2026 · [Original TinyZero](https://github.com/Jiayi-Pan/TinyZero)

I ran the upstream TinyZero/veRL GRPO recipe with rule-based Countdown rewards on Qwen2.5 base models. My contribution here is the Alliance cluster setup, two training runs, extracted metrics, plots, and analysis. The training framework and Countdown task generator come from the upstream project; this is not a new RL algorithm. The results and queued-ablation status below are as of November 5, 2026.

## Results

Both runs used 327,680 Countdown training examples and 1,024 held-out examples, generated with [`examples/data_preprocess/countdown.py`](examples/data_preprocess/countdown.py) from the [upstream Countdown dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4). GRPO used rule-based rewards, with no learned reward model or value critic. Both jobs stopped at their cluster time limits rather than completing the configured 15 epochs.

| Model and compute | Steps shown | Mean reward | Mean entropy | Mean batch tokens |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B, 1 H100 | 1–5 | 0.027 | 1.514 | 98K |
| Qwen2.5-0.5B, 1 H100 | 1700–1749 | 0.078 | 1.596 | 40K |
| Qwen2.5-3B, 2 H100s | 1–500 | 0.368 | 0.104 | 58K |
| Qwen2.5-3B, 2 H100s | 501–1000 | 0.428 | 0.0021 | 43K |
| Qwen2.5-3B, 2 H100s | 5001–5499 | 0.284 | 0.0017 | 95K |

Values are arithmetic means over the listed step ranges from the console metrics `critic/rewards/mean`, `actor/entropy_loss`, and `global_seqlen/mean`. The last is a **batch token count**, not the mean length of one response. Reward values are training metrics, not held-out task accuracy; do not compare the two model sizes as a controlled performance study.

- **0.5B:** Reward rose early and later stayed near 0.078 without a comparable entropy decline. The run reached 1,749 logged steps; 1,280 steps was an earlier reported milestone, not the end of the archived run. [Per-step metrics](logs/smoke_0.5b_metrics.csv) · [Curves](logs/smoke_0.5b_curves.svg)
- **3B:** Reward rose in the first 1,000 steps, then declined late in the run while logged entropy remained low. This is an observation, **not evidence that entropy loss caused the decline**. An entropy-coefficient comparison (0.001 versus 0.005) is queued; no result is available yet. [Per-step metrics](logs/grpo3b_24h_metrics.csv) · [Curves](logs/grpo3b_24h_curves.svg)

Neither run demonstrates emergent self-verification. The runs are single trials at each size, without an ablation result or statistical uncertainty estimate.

## Reproduce and inspect

The experiments ran on Digital Research Alliance of Canada H100s (Fir for the 0.5B run; Nibi for the 3B run). Prepare the Countdown Parquet files with [`examples/data_preprocess/countdown.py`](examples/data_preprocess/countdown.py), then set `DATA_DIR`, `BASE_MODEL`, `N_GPUS`, `ROLLOUT_TP_SIZE`, and `EXPERIMENT_NAME` for [`scripts/train_tiny_zero.sh`](scripts/train_tiny_zero.sh). The runner sets `adv_estimator=grpo`, batch size 256, maximum response length 1024, and checkpoint frequency 100. Two GPUs and tensor parallel size 2 were used for the 3B run. The environment used Alliance PyTorch 2.4.1, vLLM 0.6.2, and FlashAttention 2.5.7; consult the [upstream installation guide](https://github.com/Jiayi-Pan/TinyZero) for the original recipe, not for these cluster-specific versions.

The CSVs preserve step, reward, batch token count, and entropy from the original logs. To redraw a published plot without the private raw console output:

```sh
python3 scripts/plot_metrics.py logs/grpo3b_24h_metrics.csv logs/grpo3b_24h_curves.svg 'GRPO run — Qwen2.5-3B, 2×H100'
```

With a local raw log, `python3 scripts/plot_metrics.py --extract RAW_LOG METRICS.csv` creates the corresponding metrics file. The raw logs contain unfiltered model generations, so they are excluded from the current repository tree; **older commits still contain them**. The publicly linked CSVs and plots, not the private logs, are the intended artifacts for review.

## Environment notes and attribution

- The Alliance PyTorch wheel's NCCL 2.18.3 RPATH conflicted with the Nibi driver during FSDP initialization. Preloading the cluster's NCCL 2.27.7 library resolved the observed error.
- The Alliance vLLM 0.6.2 wheel lacked `vllm.compilation.levels`, imported by veRL's vendored 0.6.3 bindings. Adding upstream 0.6.3's `levels.py` in the environment resolved that import. These are local compatibility workarounds, not general installation instructions.

Credit: [TinyZero](https://github.com/Jiayi-Pan/TinyZero) by Jiayi Pan and collaborators, built on [veRL](https://github.com/volcengine/verl); base models [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) by the Qwen team.
