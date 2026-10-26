#!/usr/bin/env python3
"""Parse a veRL console log into a 3-panel SVG (reward / response length / entropy).
No dependencies beyond the stdlib. Usage:
    python3 scripts/plot_smoke.py logs/smoke_0.5b_1280steps.out logs/smoke_0.5b_curves.svg
"""
import re, sys, statistics

FIELDS = {
    "reward":  re.compile(r"critic/rewards/mean:([-\d.]+)"),
    "seqlen":  re.compile(r"global_seqlen/mean:([\d.]+)"),
    "entropy": re.compile(r"actor/entropy_loss:([-\d.]+)"),
}
STEP = re.compile(r"step:(\d+)")

def parse(path):
    rows = {k: [] for k in ["step"] + list(FIELDS)}
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP.search(line)
            if not m:
                continue
            rows["step"].append(int(m.group(1)))
            for k, p in FIELDS.items():
                fm = p.search(line)
                rows[k].append(float(fm.group(1)) if fm else float("nan"))
    return rows

def smooth(y, w=25):
    out, n = [], len(y)
    for i in range(n):
        seg = [v for v in y[max(0, i - w // 2): i + w // 2 + 1] if v == v]
        out.append(statistics.mean(seg) if seg else float("nan"))
    return out

def panel(xs, ys, x0, y0, W, H, ylabel, color, raw_alpha="#c7d7ea"):
    xmin, xmax = min(xs), max(xs)
    ys_clean = [v for v in ys if v == v]
    ymin, ymax = min(ys_clean), max(ys_clean)
    pad = (ymax - ymin) * 0.08 or 1e-6
    ymin, ymax = ymin - pad, ymax + pad
    def X(x): return x0 + (x - xmin) / (xmax - xmin) * W
    def Y(v): return y0 + H - (v - ymin) / (ymax - ymin) * H
    pts_raw = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, ys) if v == v)
    pts_smt = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, smooth(ys)) if v == v)
    s = f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="none" stroke="#ccc"/>'
    s += f'<polyline points="{pts_raw}" fill="none" stroke="{raw_alpha}" stroke-width="0.7"/>'
    s += f'<polyline points="{pts_smt}" fill="none" stroke="{color}" stroke-width="2"/>'
    s += f'<text x="{x0+6}" y="{y0+16}" font-size="12" fill="#333">{ylabel}  [min {ymin:.3g} / max {ymax:.3g}]</text>'
    return s

def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = parse(src)
    n = len(d["step"])
    assert n > 10, f"only {n} steps parsed"
    W, H, PW, PH = 760, 720, 700, 200
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">']
    parts.append(f'<text x="30" y="24" font-size="14" font-weight="bold">GRPO smoke run — Qwen2.5-0.5B, single H100, {n} steps</text>')
    for i, (key, label, color) in enumerate([("reward", "reward / mean", "#1f6fd1"),
                                             ("seqlen", "response length (mean)", "#e07b00"),
                                             ("entropy", "entropy", "#2c8a3c")]):
        parts.append(panel(d["step"], d[key], 40, 40 + i * 230, PW, PH, label, color))
    parts.append(f'<text x="{W//2-20}" y="{H-10}" font-size="12" fill="#666">step</text>')
    parts.append("</svg>")
    with open(dst, "w") as f:
        f.write("\n".join(parts))
    r = d["reward"]
    print(f"parsed {n} steps [{d['step'][0]}..{d['step'][-1]}]")
    print(f"reward mean first5={statistics.mean(r[:5]):.4f} last50={statistics.mean(r[-50:]):.4f}")
    print(f"seqlen  mean first5={statistics.mean(d['seqlen'][:5]):.0f} last50={statistics.mean(d['seqlen'][-50:]):.0f}")
    print(f"entropy mean first5={statistics.mean(d['entropy'][:5]):.3f} last50={statistics.mean(d['entropy'][-50:]):.3f}")
    print(f"saved {dst}")

if __name__ == "__main__":
    main()
