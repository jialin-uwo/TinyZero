# TinyZero RLVR Reproduction — Jialin Li

An independent reproduction of GRPO with rule-based Countdown rewards on Qwen2.5 base models, using the [TinyZero recipe](https://github.com/Jiayi-Pan/TinyZero) and [veRL](https://github.com/volcengine/verl). The training framework is upstream code; the runs, metrics, plots, and analysis below describe my experiments. These runs do not establish emergent self-verification.

The 0.5B run logged 1,749 steps before its time limit; the 1,280-step milestone was an earlier status report. The 3B run logged 5,499 steps in 24 hours before its time limit. A 3B entropy-coefficient ablation is queued, not completed. The public CSV files below contain training metrics extracted from the original console logs. Raw logs are excluded from the current repository tree because they contain unfiltered model output; earlier commits still contain them.

## Results: 0.5B pipeline run

Qwen2.5-0.5B; GRPO without a value critic or learned reward model; 327,680 Countdown training prompts; one H100. Metrics: [`logs/smoke_0.5b_metrics.csv`](logs/smoke_0.5b_metrics.csv) · [curves](logs/smoke_0.5b_curves.svg). `global_seqlen/mean` is a batch token count, not response length per sample.

| Phase | reward/mean | global_seqlen/mean (tokens) | Reading |
|---|---|---|---|
| step 1–5 | 0.027 | ~98K tokens/batch | cold start |
| step 10–20 | 0.079 | ~50K | reward rises as batch token count falls |
| step 20–30 | 0.087 (window mean) | ~45K | early high point |
| step 50–1280 | 0.077 (window mean) | ~40K | reward plateaus; entropy varies rather than collapsing |
| step 1281–1749 | 0.077 (window mean) | ~40K | similar reward through the end of the logged run |

**Takeaways**: the training pipeline ran end to end and reward increased early, then plateaued on this task at 0.5B. The logged batch token count fell as reward rose; entropy did not collapse. This run alone does not establish emergent reasoning or self-verification.

## Results: 3B training run

Qwen2.5-3B; GRPO; two H100s; 327,680 Countdown training prompts. Metrics: [`logs/grpo3b_24h_metrics.csv`](logs/grpo3b_24h_metrics.csv) · [curves](logs/grpo3b_24h_curves.svg). Window values below are means, not peak values.

| Stage | reward/mean | global_seqlen/mean (tokens) | entropy |
|---|---|---|---|
| 1–500 | 0.367 | ~58K | 0.104 |
| 501–1000 | 0.428 | ~43K | 0.002 |
| 2000–3000 | 0.410 | ~43K | 0.0014 |
| 4000–5000 | 0.329 | ~43K | 0.0024 |
| 5000–5499 | 0.284 | ~95K | 0.0017 |

The reward trend and low entropy suggest reduced exploration, but do not establish a causal explanation. An entropy-coefficient ablation (0.001 to 0.005) is queued to test this hypothesis; no result is available yet.

## Reproduction notes

- Compute: Digital Research Alliance of Canada H100s; Fir for environment checks and the 0.5B run, Nibi for the 3B run.
- Environment: Alliance PyTorch 2.4.1, vLLM 0.6.2, FlashAttention 2.5.7, and an editable veRL installation. These versions differ from the upstream TinyZero instructions; use the Alliance wheelhouse for PyTorch and vLLM.
- NCCL compatibility: the Alliance PyTorch wheel referenced NCCL 2.18.3 via RPATH, which failed to initialize FSDP with the Nibi driver. Preloading the cluster NCCL 2.27.7 library resolved the initialization error in this environment.
- vLLM compatibility: the 0.6.2 wheel lacked `vllm.compilation.levels`, required by veRL's vendored 0.6.3 bindings. Copying the upstream 0.6.3 `levels.py` into this environment resolved the import failure. This is an environment-specific workaround, not an upstream installation guide.
- Prepare data with `python examples/data_preprocess/countdown.py --local_dir DATA_DIR`; the training entry point is [`scripts/train_tiny_zero.sh`](scripts/train_tiny_zero.sh). For general setup instructions and attribution, see the [original TinyZero project](https://github.com/Jiayi-Pan/TinyZero). For metrics extraction and plot commands, see [`scripts/plot_metrics.py`](scripts/plot_metrics.py).

This project is built on the original [TinyZero](https://github.com/Jiayi-Pan/TinyZero) by Jiayi Pan and collaborators, using [veRL](https://github.com/volcengine/verl) and [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
