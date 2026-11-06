#!/usr/bin/env python3
"""Extract numeric training metrics and plot them without exposing raw model output.

    python3 scripts/plot_metrics.py --extract RAW_LOG METRICS.csv
    python3 scripts/plot_metrics.py METRICS.csv CURVES.svg 'Run title'
"""
import csv
import re
import statistics
import sys

FIELDS = {
    "reward": re.compile(r"critic/rewards/mean:([-\d.]+)"),
    "seqlen": re.compile(r"global_seqlen/mean:([\d.]+)"),
    "entropy": re.compile(r"actor/entropy_loss:([-\d.]+)"),
}
STEP = re.compile(r"step:(\d+)")
COLUMNS = ("step", *FIELDS)

def extract(src, dst):
    count = 0
    with open(src, errors="replace") as log, open(dst, "w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(COLUMNS)
        for line in log:
            match = STEP.search(line)
            if not match:
                continue
            values = [pattern.search(line) for pattern in FIELDS.values()]
            if not all(values):
                continue
            writer.writerow((match.group(1), *(value.group(1) for value in values)))
            count += 1
    print(f"extracted {count} metric rows to {dst}")

def parse(path):
    rows = {key: [] for key in COLUMNS}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            for key in COLUMNS:
                rows[key].append(int(row[key]) if key == "step" else float(row[key]))
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
    if sys.argv[1] == "--extract":
        extract(sys.argv[2], sys.argv[3])
        return
    src, dst, title = sys.argv[1:4]
    d = parse(src)
    n = len(d["step"])
    assert n > 10, f"only {n} steps parsed"
    W, H, PW, PH = 760, 720, 700, 200
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    parts.append(f'<text x="30" y="24" font-size="14" font-weight="bold">{title}, {n} steps</text>')
    for i, (key, label, color) in enumerate([("reward", "reward / mean", "#1f6fd1"),
                                             ("seqlen", "global_seqlen/mean (batch tokens)", "#e07b00"),
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
