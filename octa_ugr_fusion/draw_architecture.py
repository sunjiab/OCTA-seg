"""Draw a reproducible scientific diagram with unchanged dataset image panels."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE = "#2464A0"
TEAL = "#158477"
ORANGE = "#B66C12"
INK = "#17324A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="10451")
    parser.add_argument("--output-dir", type=Path, default=HERE / "assets")
    args = parser.parse_args()
    root = HERE.parent / "data/OCTA-500数据集/3mm"
    images = []
    for modality in ("OCTA", "OCT"):
        with Image.open(root / f"{modality}(ILM_OPL)" / f"{args.case_id}.bmp") as source:
            images.append(np.array(source.convert("L")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(24, 14), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 24), ylim=(0, 14))
    ax.axis("off")

    def text(x, y, value, size=12, color=INK, weight="normal", ha="center"):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", linespacing=1.5)

    def box(x, y, w, h, value, color=BLUE, fill="#EDF4FC", size=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                  facecolor=fill, edgecolor=color, lw=1.6))
        text(x + w / 2, y + h / 2, value, size, color)

    def arrow(points, color=BLUE, dashed=False, width=1.8):
        xs, ys = zip(*points)
        if len(points) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=width, ls="--" if dashed else "-", zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
                                    lw=width, color=color, linestyle="--" if dashed else "-", zorder=3))

    def image_panel(array, x, y, label, color):
        panel = fig.add_axes([x / 24, y / 14, 2.6 / 24, 2.6 / 14])
        panel.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        text(x + 1.3, y + 2.88, label, 14, color, "bold")
        text(x + 1.3, y - .25, "1 × 304 × 304", 11, color)

    text(12, 13.45, "UGR-Fusion | OCTA predicts, OCT refines", 25, weight="bold")
    text(12, 12.95, f"Actual input projections: OCTA-500, case {args.case_id}, 3 mm, ILM-OPL  |  Batch dimension omitted", 12)
    text(.5, 12.4, "A. Complete forward path", 15, weight="bold", ha="left")
    image_panel(images[0], .5, 9.15, "OCTA input", BLUE)
    image_panel(images[1], .5, 5.7, "OCT input", TEAL)

    box(3.8, 10.05, 4.05, 1.8, "OCTA encoder\n32 / 64 / 128 / 256 / 512 channels\n304 / 152 / 76 / 38 / 19 pixels", size=12)
    box(8.7, 10.3, 3, 1.3, "U-Net decoder\nwith skip connections", size=12)
    box(12.5, 10.3, 2.4, 1.3, "Anchor logits  zA\n1 × 304 × 304", size=12)
    arrow([(3.1, 10.7), (3.8, 10.7)])
    arrow([(7.85, 10.95), (8.7, 10.95)])
    arrow([(11.7, 10.95), (12.5, 10.95)])
    text(9.95, 12, "Strong OCTA prediction path", 12, BLUE)

    box(3.8, 6, 4.05, 1.8, "Light OCT pyramid\n8 / 16 / 32 / 64 / 128 channels\nSame five spatial scales", TEAL, "#E9F7F2", 12)
    box(8.7, 6.2, 4.0, 1.4, "5 gated residual heads\nOne head per matched scale\nSee panel B", TEAL, "#E9F7F2", 12)
    box(13.35, 6.2, 2.6, 1.4, "Upsample to 304²\nAverage 5 scales\nr = mean(up(g × d))", TEAL, "#E9F7F2", 11)
    box(16.65, 6.2, 3.7, 1.4, "Bounded correction\ndelta = s × q × 2 tanh(r / 2)\nq = 0.1 + 0.9u", ORANGE, "#FFF4E5", 12)
    arrow([(3.1, 7), (3.8, 7)], TEAL)
    arrow([(7.85, 6.9), (8.7, 6.9)], TEAL)
    arrow([(12.7, 6.9), (13.35, 6.9)], TEAL)
    arrow([(15.95, 6.9), (16.65, 6.9)], TEAL)
    arrow([(5.8, 10.05), (5.8, 8.8), (9.4, 8.8), (9.4, 7.6)], BLUE)
    text(6.9, 9.02, "5 OCTA features", 10, BLUE)

    box(12.5, 8.25, 3.7, 1.2, "pA = sigmoid(zA).detach()\nu = 4pA(1 - pA)", ORANGE, "#FFF4E5", 12)
    arrow([(13.7, 10.3), (13.7, 9.45)], ORANGE, True)
    arrow([(12.5, 8.85), (11.4, 8.85), (11.4, 7.6)], ORANGE, True)
    text(11.8, 8.15, "resize u", 10, ORANGE)
    arrow([(16.2, 8.85), (18.5, 8.85), (18.5, 7.6)], ORANGE, True)
    text(17.5, 9.1, "uncertainty weight", 10, ORANGE)

    ax.add_patch(Circle((19.5, 10.95), .3, ec=INK, fc="white", lw=1.8))
    text(19.5, 10.95, "+", 24)
    arrow([(14.9, 10.95), (19.2, 10.95)], BLUE, width=2.3)
    text(16.9, 11.28, "preserve anchor path", 11, BLUE)
    arrow([(20.35, 6.9), (20.7, 6.9), (20.7, 9.85), (19.5, 9.85), (19.5, 10.65)], TEAL)
    box(20.55, 10.05, 2.95, 1.8, "Final prediction\nsigmoid(zA + delta)\nThreshold = 0.5", BLUE, "#EDF4FC", 12)
    arrow([(19.8, 10.95), (20.55, 10.95)], BLUE)
    text(12.2, 5.55, "Zero-initialized residual heads: initial output = OCTA anchor.   s: 5-epoch ramp to 1; inference s = 1.", 11)

    ax.plot([.5, 23.5], [4.95, 4.95], color="#DCE5ED", lw=1.5)
    text(.5, 4.5, "B. Inside one gated residual head (repeated at all 5 scales)", 15, weight="bold", ha="left")
    box(.65, 2.9, 3.15, .7, "OCTA feature → 1×1 → A: 16 ch", BLUE, "#EDF4FC", 11)
    box(.65, 1.8, 3.15, .7, "OCT feature → 1×1 → O: 16 ch", TEAL, "#E9F7F2", 11)
    box(.65, .7, 3.15, .7, "Resized uncertainty: 1 ch", ORANGE, "#FFF4E5", 11)
    box(5, 1.5, 3.8, 1.5, "Concatenate\n[A, O, |A - O|, u]\n16 + 16 + 16 + 1 = 49 ch", BLUE, "#F1F5FA", 12)
    arrow([(3.8, 3.25), (4.3, 3.25), (4.3, 2.7), (5, 2.7)], BLUE)
    arrow([(3.8, 2.15), (5, 2.15)], TEAL)
    arrow([(3.8, 1.05), (4.3, 1.05), (4.3, 1.7), (5, 1.7)], ORANGE, True)
    box(9.6, 1.5, 3.4, 1.5, "3×3 Conv: 49 → 16\nGroupNorm + SiLU\nMixed features", BLUE, "#F1F5FA", 12)
    arrow([(8.8, 2.25), (9.6, 2.25)])
    box(14.1, 2.75, 3.5, .9, "1×1 Conv + sigmoid\nGate g: 0 to 1", ORANGE, "#FFF4E5", 12)
    box(14.1, .85, 3.5, .9, "1×1 Conv (zero init)\nSigned residual d", TEAL, "#E9F7F2", 12)
    arrow([(13, 2.25), (13.5, 2.25), (13.5, 3.2), (14.1, 3.2)], ORANGE)
    arrow([(13.5, 2.25), (13.5, 1.3), (14.1, 1.3)], TEAL)
    ax.add_patch(Circle((19.25, 2.25), .3, ec=TEAL, fc="white", lw=1.8))
    text(19.25, 2.25, "×", 23, TEAL)
    arrow([(17.6, 3.2), (19.25, 3.2), (19.25, 2.55)], ORANGE)
    arrow([(17.6, 1.3), (19.25, 1.3), (19.25, 1.95)], TEAL)
    box(20.25, 1.65, 3.2, 1.2, "Gated residual g × d\n1 channel per scale\nTo upsample + average", TEAL, "#E9F7F2", 11)
    arrow([(19.55, 2.25), (20.25, 2.25)], TEAL)
    text(12, .25, "Original image intensities retained; no generated vessel output. Diagram shows inference; ground truth is used only for training losses.", 10, "#536779")

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"ugr_fusion_architecture.{suffix}", dpi=180, facecolor="white")
    fig.savefig(args.output_dir / "ugr_fusion_architecture_preview.jpg", dpi=75, facecolor="white")
    plt.close(fig)
    print(args.output_dir / "ugr_fusion_architecture.png")


if __name__ == "__main__":
    main()
