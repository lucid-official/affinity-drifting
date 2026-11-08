"""Flickr30k 실험 결과를 그림으로 만든다. 모두 저장된 결과 파일만 읽는다(Metal 불필요).

    .venv/bin/python experiments/make_figures.py   # -> results/figures/*.png

읽는 것: runs/flickr_s10000_*/{log.jsonl, caption_eval.json, samples_test.png, layout_cells.json, assignment.json},
runs/mae_flickr32/train.log, results/phi_diagnostics.txt, results/layout_cells_retrieval.json,
results/assignment_reference.json.
아직 없는 실행이나 파일은 건너뛴다.
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = Path("results/figures")
PREFIX = "flickr_s10000_"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# (실행 이름, 표시 이름, 색, 고정 MAE φ 여부). 색은 양성 고르는 방법을 따르고, MAE φ는 속 빈 표시로 구분한다.
# 모든 쌍 검증을 통과하는 세 색은 결론의 중심인 joint, pseudo K=256, pair에 쓰고 나머지는 회색으로 둔다.
METHODS = [
    ("pixel_blind", "blind", MUTED, False),
    ("pixel_pair", "pair", AQUA, False),
    ("pixel_pseudo", "pseudo K=32", INK2, False),
    ("pixel_pseudo256", "pseudo K=256", ORANGE, False),
    ("pixel_joint", "joint", BLUE, False),
    ("mae_w1000_pseudo256", "pseudo K=256 + MAE", ORANGE, True),
    ("mae_w1000_joint", "joint + MAE", BLUE, True),
]
PIXEL = [m for m in METHODS if not m[3]]

plt.rcParams.update(
    {
        "font.family": "Apple SD Gothic Neo",
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)


def group_runs() -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for run in sorted(glob.glob(f"runs/{PREFIX}*_seed*")):
        if not (Path(run) / "caption_eval.json").exists():  # 학습이나 평가가 끝나지 않은 실행
            continue
        name = re.sub(r"_seed\d+$", "", Path(run).name)[len(PREFIX) :]
        grouped[name].append(Path(run))
    return grouped


RUNS = group_runs()
PRESENT = [m for m in METHODS if m[0] in RUNS]


def evals(name: str) -> list[dict]:
    return [json.loads((r / "caption_eval.json").read_text()) for r in RUNS[name]]


def log_axis(ax, axis: str = "y") -> None:
    """로그 축 눈금을 10^-2 대신 0.01처럼 쓴다."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.4g}"))
    target.set_minor_formatter(NullFormatter())


def logs(name: str) -> list[list[dict]]:
    return [[json.loads(line) for line in (r / "log.jsonl").read_text().splitlines()] for r in RUNS[name]]


def spread(values, gap: float) -> np.ndarray:
    """라벨이 겹치지 않도록 값 순서를 지키며 최소 간격을 둔다. 겹친 쌍을 위아래로 반씩 벌려 제자리에 가깝게 둔다."""
    values = np.asarray(values, float)
    order = np.argsort(values)
    y = values[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(y)):
            overlap = gap - (y[i] - y[i - 1])
            if overlap > 1e-12:
                y[i - 1] -= overlap / 2
                y[i] += overlap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return out


def marker_style(color: str, hollow: bool, size: float = 8) -> dict:
    return {
        "marker": "o",
        "markersize": size,
        "markerfacecolor": SURFACE if hollow else color,
        "markeredgecolor": color if hollow else SURFACE,
        "markeredgewidth": 2.0,
        "color": color,
    }


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(OUT / name)


# 1. 학습 추이 -----------------------------------------------------------------------------------------


def series(name: str, key: str):
    runs = [run for run in logs(name) if all(key in row for row in run)]
    steps = np.array([row["step"] for row in runs[0]])
    values = np.array([[row[key] for row in run] for run in runs])
    return steps, values


GRAD_CLIP = 2.0  # TrainConfig.grad_clip. 로그의 grad_norm은 이보다 클 수 있으므로 클리핑 전 값이다.


def clip_line(ax) -> None:
    ax.axhline(GRAD_CLIP, color=INK2, linewidth=1.0, zorder=1)
    ax.annotate(f"클리핑 기준 {GRAD_CLIP:g}", (ax.get_xlim()[1], GRAD_CLIP), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)


def training_figure() -> None:
    panels = [
        ("loss", "loss (식 25로 drift 정규화 후)", False),
        ("raw/pixel", "drift 크기 E‖V‖²/C (정규화 전, 로그 축)", True),
        ("grad_norm", "생성기 기울기 노름 (클리핑 전)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, log) in zip(axes, panels):
        ends = []
        for name, label, color, _ in PIXEL:
            if name not in RUNS:
                continue
            steps, values = series(name, key)
            mean = values.mean(0)
            ax.fill_between(steps, values.min(0), values.max(0), color=color, alpha=0.14, linewidth=0)
            ax.plot(steps, mean, color=color, label=label, solid_capstyle="round")
            ends.append((mean[-3:].mean(), label))
        ax.set_title(title)
        ax.set_xlabel("스텝")
        ax.set_xlim(0, 10000)
        if log:
            log_axis(ax)
        if key == "raw/pixel":  # 끝값 직접 라벨: pair가 나머지와 한 자리 다르다
            ys = np.log10([v for v, _ in ends])
            for y, (_, label) in zip(10 ** spread(ys, 0.09), ends):
                ax.annotate(label, (10000, y), xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=9, color=INK2, annotation_clip=False)
        if key == "grad_norm":
            clip_line(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.04, "선: 시드 3개 평균, 띠: 시드 최소~최대. 500스텝마다 한 배치에서 잰 값이다.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(w_pad=3.5)
    save(fig, "training.png")


def training_mae_figure() -> None:
    mae = [m for m in METHODS if m[3] and m[0] in RUNS]
    if not mae:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (loss_ax, pixel_ax), (mae_ax, grad_ax) = axes
    pairs = [(m, next(p for p in PIXEL if p[2] == m[2] and p[0] in RUNS)) for m in mae]
    for (name, label, color, _), (pname, plabel, _, _) in pairs:
        for ax, key in [(pixel_ax, "raw/pixel"), (grad_ax, "grad_norm")]:
            steps, values = series(pname, key)
            ax.plot(steps, values.mean(0), color=color, alpha=0.4, label=f"{plabel} (픽셀만, 시드 3개 평균)")
            steps, values = series(name, key)
            ax.plot(steps, values.mean(0), **marker_style(color, True, 5), markevery=2, label=label)
        for ax, key in [(mae_ax, "raw/mae"), (loss_ax, "loss")]:
            steps, values = series(name, key)
            ax.plot(steps, values.mean(0), **marker_style(color, True, 5), markevery=2, label=label)
    titles = {
        loss_ax: "loss (픽셀 항 ×1000 + MAE 특징 63개 항)",
        pixel_ax: "픽셀 drift E‖V‖²/C (로그 축)",
        mae_ax: "MAE drift E‖V‖²/C, 특징 63개 합 (로그 축)",
        grad_ax: "생성기 기울기 노름, 클리핑 전 (로그 축)",
    }
    for ax, title in titles.items():
        ax.set_title(title)
        ax.set_xlim(0, 10000)
    for ax in axes[1]:
        ax.set_xlabel("스텝")
    loss_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for ax in (pixel_ax, mae_ax, grad_ax):
        log_axis(ax)
    pixel_ax.set_ylim(top=0.06)  # 0.05 눈금이 보이게
    clip_line(grad_ax)
    handles, labels = pixel_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.02, "속 빈 점: 고정 MAE φ를 붙인 실행(시드 0). 흐린 선: 같은 방법의 픽셀 전용 실행.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=3.5)
    save(fig, "training_mae.png")


# 2. 지표 비교 -----------------------------------------------------------------------------------------

METRICS = [
    ("layout_corr", "배치 상관 ↑", {"retrieval": "캡션 검색"}, None),
    ("layout_r_precision", "배치 R-precision ↑", {"retrieval": "캡션 검색", "chance": "우연"}, None),
    ("clip_score", "CLIP score ↑", {"shuffled": "캡션 섞은 실제 사진"}, "실제 사진 0.263, 캡션 검색 0.242 (축 밖)"),
    ("r_precision", "CLIP R-precision ↑", {"chance": "우연"}, "실제 사진 0.654, 캡션 검색 0.425 (축 밖)"),
    ("fd", "FD (CLIP 공간) ↓", {}, "실제 학습 사진 vs 테스트 사진 0.026 (축 밖)"),
    ("diversity", "다양성 ↑", {}, None),
]


def reference_values(metric: str) -> dict[str, float]:
    ref = evals(PRESENT[0][0])[0]
    out = {"chance": 0.01}
    if metric in ref["retrieval"]:
        out["retrieval"] = ref["retrieval"][metric]
    if metric == "clip_score":
        out["shuffled"] = ref["real"]["clip_score_shuffled"]
    return out


def metrics_figure(alpha: str = "alpha=1.0") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    rows = list(range(len(PRESENT)))[::-1]  # 표와 같은 순서로 위에서 아래
    for ax, (metric, title, refs, note) in zip(axes.flat, METRICS):
        for y, (name, label, color, hollow) in zip(rows, PRESENT):
            values = np.array([e[alpha][metric] for e in evals(name)])
            ax.plot([values.min(), values.max()], [y, y], color=color, linewidth=2, alpha=0.5,
                    solid_capstyle="round")
            ax.scatter(values, np.full(len(values), y), s=16, color=color, alpha=0.55, linewidths=0, zorder=3)
            ax.plot([values.mean()], [y], linestyle="none", zorder=4, **marker_style(color, hollow, 9))
        ref = reference_values(metric)
        for key, text in refs.items():
            ax.axvline(ref[key], color=INK2 if key != "chance" else MUTED, linewidth=1.0, zorder=1)
            ax.annotate(text, (ref[key], len(PRESENT) - 0.45), xytext=(3, 0), textcoords="offset points",
                        fontsize=8, color=INK2 if key != "chance" else MUTED, va="center")
        ax.set_title(title)
        ax.set_ylim(-0.7, len(PRESENT) - 0.1)
        ax.set_yticks(rows, [label for _, label, _, _ in PRESENT])
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if note:
            ax.set_xlabel(note, fontsize=8, color=MUTED)
    for ax in axes[:, 1:].flat:
        ax.set_yticklabels([])
    fig.text(0.5, -0.02,
             "큰 점: 시드 평균(속 빈 점은 고정 MAE φ, 시드 1개), 작은 점과 선: 시드별 값과 범위. α=1, 테스트 캡션 1,000개.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    save(fig, "metrics.png")


def tradeoff_figure(alpha: str = "alpha=1.0") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    points = []
    for name, label, color, hollow in PRESENT:
        rows = evals(name)
        x = np.array([e[alpha]["fd"] for e in rows])
        y = np.array([e[alpha]["layout_corr"] for e in rows])
        ax.scatter(x, y, s=18, color=color, alpha=0.5, linewidths=0, zorder=3)
        ax.plot([x.mean()], [y.mean()], linestyle="none", zorder=6, **marker_style(color, hollow, 10))
        points.append((x.mean(), y.mean(), label))
    retrieval = reference_values("layout_corr")["retrieval"]
    ax.axhline(retrieval, color=INK2, linewidth=1.0, zorder=1)
    left, right = ax.get_xlim()
    ax.set_xlim(left - 0.012, right + 0.012)  # 양 끝 점의 이름표가 축 안에 들어오게
    ax.annotate("캡션 검색 기준선", (ax.get_xlim()[1], retrieval), xytext=(-4, -4), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK2)
    # 이름표는 오른쪽에 두되, 그 자리에 다른 평균점이 있으면 왼쪽으로 옮긴다. 배경을 깔아 시드 점 위에서도 읽히게 한다.
    fig.canvas.draw()
    pixels = {label: ax.transData.transform((x, y)) for x, y, label in points}
    scale = fig.dpi / 72
    for x, y, label in points:
        px, py = pixels[label]
        width = (6 * len(label) + 20) * scale
        blocked = any(0 < ox - px < width and abs(oy - py) < 12 * scale for other, (ox, oy) in pixels.items()
                      if other != label)
        ax.annotate(label, (x, y), xytext=(-9 if blocked else 9, 0), textcoords="offset points",
                    ha="right" if blocked else "left", va="center", fontsize=9, color=INK2, zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
    ax.set_xlabel("FD (CLIP 공간, ← 낮을수록 실제 분포에 가까움)")
    ax.set_ylabel("배치 상관 (↑ 높을수록 프롬프트를 따름)")
    ax.set_title("조건 강도와 분포 품질")
    save(fig, "tradeoff.png")


def guidance_figure() -> None:
    """blind는 텍스트 조건을 배우지 않아 α가 무의미하므로 뺀다(축이 0까지 늘어나 나머지가 뭉친다)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (metric, title) in zip(axes, [("layout_corr", "배치 상관 ↑"), ("fd", "FD ↓")]):
        ends = []
        for name, label, color, hollow in [m for m in PRESENT if m[0] != "pixel_blind"]:
            rows = evals(name)
            values = [np.mean([e[a][metric] for e in rows]) for a in ("alpha=1.0", "alpha=2.0")]
            ax.plot([1, 2], values, **marker_style(color, hollow, 8))
            ends.append((values[1], label))
        low, high = ax.get_ylim()
        gap = 0.055 * (high - low)  # 이름표 한 줄 높이만큼
        for y, (_, label) in zip(spread([v for v, _ in ends], gap), ends):
            ax.annotate(label, (2, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9,
                        color=INK2, annotation_clip=False)
        ax.set_xticks([1, 2], ["α = 1", "α = 2"])
        ax.set_xlim(0.85, 2.15)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    fig.tight_layout(w_pad=9)
    save(fig, "guidance.png")


# 3. 칸별 배치 상관 ----------------------------------------------------------------------------------------

RAMP = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def layout_cells_figure() -> None:
    panels = []
    retrieval = Path("results/layout_cells_retrieval.json")
    if retrieval.exists():
        panels.append(("캡션 검색 기준선", np.array(json.loads(retrieval.read_text())["alpha=1.0"]).mean(0)))
    for name, label, _, _ in PRESENT:
        files = [r / "layout_cells.json" for r in RUNS[name] if (r / "layout_cells.json").exists()]
        if files and name != "pixel_blind":
            cells = np.mean([np.array(json.loads(f.read_text())["alpha=1.0"]).mean(0) for f in files], axis=0)
            panels.append((label, cells))
    if len(panels) < 2:
        return
    top = max(c.max() for _, c in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.5))
    for ax, (label, cells) in zip(axes, panels):
        image = ax.imshow(cells, cmap=RAMP, vmin=0, vmax=top)
        for (i, j), value in np.ndenumerate(cells):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > 0.6 * top else INK)
        ax.set_title(f"{label}\n평균 {cells.mean():.3f}", fontsize=9.5, loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8)
    fig.text(0.45, -0.02, "4×4 칸마다 RGB 세 채널의 상관을 평균했다. 위가 이미지 위쪽이다. 생성 방법은 시드 평균, α=1.",
             ha="center", color=MUTED, fontsize=9)
    save(fig, "layout_cells.png")


# 4. φ 진단 ---------------------------------------------------------------------------------------------


# results/phi_diagnostics.txt의 실행 -> (데이터, 생성기). 가중치는 그 생성기를 학습할 때 쓴 픽셀 항 배율이다.
GRADIENT_RUNS = {
    "runs/s3000_pixel_joint_seed0": ("shapes", "pixel"),
    "runs/s3000_mae_joint_seed0": ("shapes", "mae"),  # 픽셀 가중치 1
    "runs/flickr_s10000_pixel_joint_seed0": ("Flickr30k", "pixel"),
    "runs/flickr_s10000_mae_w1000_joint_seed0": ("Flickr30k", "mae"),  # 픽셀 가중치 1000
}


def gradient_ratios() -> dict[tuple[str, str], float]:
    """MAE 기울기 / 픽셀 기울기를 읽는다. 같은 실행을 여러 번 쟀으면 평균한다."""
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    current = None
    for line in Path("results/phi_diagnostics.txt").read_text().splitlines():
        if line.startswith("runs/"):
            current = GRADIENT_RUNS.get(line.split()[0])
        elif current and (match := re.match(r"\s+mae\s.*ratio=\s*([\d.]+)", line)):
            out[current].append(float(match.group(1)))
    return {key: float(np.mean(values)) for key, values in out.items()}


def phi_figure() -> None:
    ratios = gradient_ratios()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    bars = {
        "pixel": ("픽셀 φ만으로 학습한 joint", "#b5b3ab"),
        "mae": ("MAE φ를 붙여 학습한 joint", INK2),
    }
    notes = {("shapes", "mae"): "가중치 1", ("Flickr30k", "mae"): "가중치 1000"}
    width = 0.36
    for k, (kind, (label, color)) in enumerate(bars.items()):
        xs = np.arange(2) + (k - 0.5) * (width + 0.04)
        values = [ratios.get((data, kind), 0.0) for data in ("shapes", "Flickr30k")]
        ax.bar(xs, values, width=width, color=color, label=label)
        for x, v, data in zip(xs, values, ("shapes", "Flickr30k")):
            note = notes.get((data, kind))
            ax.annotate(f"{v:,.0f}배" + (f"\n({note})" if note else ""), (x, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1], ["shapes", "Flickr30k"])
    ax.set_ylim(0, 1.3 * max(ratios.values()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="x", visible=False)
    ax.set_title("MAE 기울기 / 픽셀 기울기")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1]
    log = Path("runs/mae_flickr32/train.log")
    rows = [re.search(r"step=(\d+) loss=([\d.]+)", line) for line in log.read_text().splitlines()]
    steps, losses = zip(*[(int(m.group(1)), float(m.group(2))) for m in rows if m])
    ax.plot(steps, losses, color=INK2)
    log_axis(ax)
    ax.set_xlim(0, max(steps))
    ax.set_xlabel("스텝")
    ax.set_title("Flickr30k MAE 재구성 loss (로그 축)")
    ax.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), xytext=(-4, 8), textcoords="offset points",
                ha="right", fontsize=9, color=INK2)
    fig.text(0.5, -0.04,
             "왼쪽: 학습된 생성기의 샘플에서 잰 이미지 공간 기울기 크기의 비율(Flickr는 두 번 잰 평균). "
             "가중치는 그 생성기를 학습할 때의 픽셀 항 배율.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(w_pad=4)
    save(fig, "phi.png")


# 5. 샘플 -----------------------------------------------------------------------------------------------

LABEL, CELL, TILE = 300, 66, 64  # caption_eval.save_grid(label_width=300, scale=2)의 배치
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def samples_figure(prompts: int = 6, per: int = 3, scale: int = 3) -> None:
    """samples_test.png에서 생성 샘플만 잘라 방법별로 붙인다. 0열(실제 Flickr 사진)은 쓰지 않는다.

    32×32 샘플을 화소 그대로 `scale`배 키운다. 캡션은 테스트 매니페스트의 앞 `prompts`줄이다.
    """
    lines = Path("data/flickr30k_32/test.jsonl").read_text(encoding="utf-8").splitlines()
    captions = [json.loads(line)["caption"] for line in lines[:prompts]]
    chosen = [m for m in PRESENT if m[0] not in ("pixel_pair", "pixel_pseudo")]
    size, pad, gap, label_width, header = 32 * scale, 4, 28, 430, 48
    block = per * size + (per - 1) * pad
    width = label_width + len(chosen) * block + (len(chosen) - 1) * gap
    canvas = Image.new("RGB", (width, header + prompts * size + (prompts - 1) * pad), SURFACE)
    draw = ImageDraw.Draw(canvas)
    head, body = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 20)
    for p, caption in enumerate(captions):
        y = header + p * (size + pad)
        text = wrap(draw, caption, body, label_width - 24)
        if len(text) > 4:
            text = [*text[:3], text[3] + " …"]
        top = y + (size - 24 * len(text)) // 2
        for i, line in enumerate(text):
            draw.text((0, top + 24 * i), line, fill=INK2, font=body)
    for k, (name, label, _, _) in enumerate(chosen):
        grid = Image.open(RUNS[name][0] / "samples_test.png")
        x = label_width + k * (block + gap)
        draw.text((x, 6), label, fill=INK, font=head)
        for p in range(prompts):
            for s in range(per):
                left, top = LABEL + (s + 1) * CELL, p * CELL  # 0열은 실제 사진
                tile = grid.crop((left, top, left + TILE, top + TILE)).resize((32, 32), Image.Resampling.NEAREST)
                canvas.paste(tile.resize((size, size), Image.Resampling.NEAREST),
                             (x + s * (size + pad), header + p * (size + pad)))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "samples.png")
    print(OUT / "samples.png")


# 6. 배정 행렬 (experiments/assignment.py) ------------------------------------------------------------------


W = r"$\tilde{w}$"  # 한글 글꼴에 결합 틸데가 없어 mathtext로 쓴다.


def assignments(name: str) -> list[dict]:
    return [json.loads((r / "assignment.json").read_text()) for r in RUNS[name] if (r / "assignment.json").exists()]


def assignment_figure() -> None:
    """affinity-drifting 브랜치 F0.3과 같은 배치: 예시 그룹 하나의 배정 행렬과 w̃."""
    path = Path("results/assignment_reference.json")
    if not path.exists():
        return
    ref = json.loads(path.read_text())
    panels = [(f"{W}: joint kernel의 캡션 가중치", np.array(ref["example_w"]), f"자기 짝 {ref['w_self']:.2f} · pair 목표는 대각선")]
    for name, label, _, _ in PRESENT:
        rows = assignments(name)
        if rows:
            seeds = f"시드 {len(rows)}개 평균" if len(rows) > 1 else "시드 1개"
            panels.append((f"{label} ({seeds})", np.mean([r["example"] for r in rows], axis=0),
                           f"자기 짝 {np.mean([r['self'] for r in rows]):.3f} · TV {np.mean([r['tv_w'] for r in rows]):.2f}"))
    norm = PowerNorm(0.5, vmin=0, vmax=max(p[1].max() for p in panels))
    cols = 4
    fig, axes = plt.subplots(-(-len(panels) // cols), cols, figsize=(12, 6.9))
    for ax, (title, matrix, note) in zip(axes.flat, panels):
        image = ax.imshow(matrix, cmap=RAMP, norm=norm, interpolation="nearest")
        ax.set_title(f"{title}\n{note}", fontsize=9.5, loc="center")
        ax.set_xticks([0, len(matrix) - 1])
        ax.set_yticks([0, len(matrix) - 1])
        ax.tick_params(length=0, labelsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("조건 k (캡션)")
    for ax in axes[-1]:
        ax.set_xlabel("positive j (학습 이미지)")
    bar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02, ticks=[0, 0.05, 0.2, 0.4, 0.6, 0.8])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8)
    fig.text(0.45, 0.0,
             f"텍스트 이웃 그룹 1개(학습 항목 {ref['members']}개, 텍스트 키 첫 주성분 순). 행마다 조건 k의 샘플 "
             f"{ref['per_prompt']}장이 픽셀 공간에서 가장 가까운 y_j의 비율. 색은 제곱근 눈금.\n"
             f"제목의 자기 짝(대각선 평균)과 TV(A, {W})는 그룹 {ref['groups']}개 평균. 우연이면 자기 짝 1/32 = 0.031, "
             f"균등 배정의 TV는 {ref['tv_uniform_w']:.2f}.",
             ha="center", va="top", color=MUTED, fontsize=9)
    save(fig, "assignment.png")


def dot_rows(ax, values_by_method: list[tuple[str, str, bool, np.ndarray]]) -> None:
    """metrics_figure와 같은 표기: 시드별 작은 점과 범위, 평균은 큰 점."""
    rows = list(range(len(values_by_method)))[::-1]
    for y, (_, color, hollow, values) in zip(rows, values_by_method):
        ax.plot([values.min(), values.max()], [y, y], color=color, linewidth=2, alpha=0.5, solid_capstyle="round")
        ax.scatter(values, np.full(len(values), y), s=16, color=color, alpha=0.55, linewidths=0, zorder=3)
        ax.plot([values.mean()], [y], linestyle="none", zorder=4, **marker_style(color, hollow, 9))
    ax.set_ylim(-0.7, len(values_by_method) - 0.1)
    ax.set_yticks(rows, [label for label, _, _, _ in values_by_method])
    ax.grid(axis="y", visible=False)


def reference_line(ax, x: float, text: str, top: float) -> None:
    ax.axvline(x, color=MUTED, linewidth=1.0, zorder=1)
    ax.annotate(text, (x, top), xytext=(3, 0), textcoords="offset points", fontsize=8, color=MUTED, va="center")


def assignment_summary_figure() -> None:
    path = Path("results/assignment_reference.json")
    if not path.exists():
        return
    ref = json.loads(path.read_text())
    chosen = [(label, color, hollow, assignments(name)) for name, label, color, hollow in PRESENT]
    chosen = [c for c in chosen if c[3]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.35]})

    ax = axes[0]
    dot_rows(ax, [(label, color, hollow, np.array([r["self"] for r in rows])) for label, color, hollow, rows in chosen])
    reference_line(ax, 1 / ref["members"], "우연 1/32", len(chosen) - 0.45)
    ax.set_title("자기 짝 이미지에 배정된 비율")
    ax.set_xlabel(f"목표: pair 1, {W} {ref['w_self']:.2f}, 실제 학습 그룹의 {W} {ref['sampler_self_mass']:.2f} (모두 축 밖)",
                  fontsize=8, color=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(4))

    ax = axes[1]
    dot_rows(ax, [(label, color, hollow, np.array([r["top3"] for r in rows])) for label, color, hollow, rows in chosen])
    reference_line(ax, 3 / ref["members"], "균등 3/32", len(chosen) - 0.45)
    ax.set_yticklabels([])
    ax.set_title("그룹 32장 중 상위 3장에 몰린 비율")
    ax.set_xlabel("조건과 상관없이 같은 몇 장으로 가는 정도", fontsize=8, color=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(4))

    ax = axes[2]
    memo = ref["memorization"]
    edges = np.array(memo["edges"])[::2]  # 1 단위 구간을 2 단위로 묶는다(실제 사진은 512장뿐이라 들쭉날쭉하다).
    centers = (edges[:-1] + edges[1:]) / 2

    def density(counts) -> np.ndarray:
        counts = np.asarray(counts, float).reshape(-1, 2).sum(1)
        return counts / counts.sum()

    for name, label, color, _ in [m for m in PRESENT if m[0] in ("pixel_pair", "pixel_joint")]:
        rows = assignments(name)
        ax.plot(centers, density(np.sum([r["nn_hist"] for r in rows], axis=0)), color=color,
                label=f"{label}: 최근접 학습 이미지까지")
        ax.plot(centers, density(np.sum([r["own_hist"] for r in rows], axis=0)), color=color, linestyle=(0, (3, 2)),
                label=f"{label}: 자기 짝 이미지까지")
    ax.plot(centers, density(memo["real_loo_hist"]), color=INK2, label="실제 사진: 최근접 다른 학습 이미지까지")
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("픽셀 공간 L2 거리 (외웠다면 0 근처에 몰린다)")
    ax.set_ylabel("비율 (폭 2 구간)")
    ax.set_title("학습 이미지 29,000장까지의 거리")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.text(0.5, -0.06,
             f"왼쪽·가운데: 텍스트 이웃 그룹 {ref['groups']}개 평균(큰 점은 시드 평균, 속 빈 점은 고정 MAE φ). "
             f"오른쪽: 학습 캡션 {memo['prompts']}개 × {memo['per_prompt']}장, 시드 3개 합. α=1.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    save(fig, "assignment_summary.png")


if __name__ == "__main__":
    training_figure()
    training_mae_figure()
    metrics_figure()
    tradeoff_figure()
    guidance_figure()
    layout_cells_figure()
    phi_figure()
    samples_figure()
    assignment_figure()
    assignment_summary_figure()
