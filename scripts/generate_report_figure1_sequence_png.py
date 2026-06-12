from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "figure1_sequence_transformation.png"


def configure_font() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
        Path("C:/Windows/Fonts/batang.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = ["Malgun Gothic", "Gulim", "Batang", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    selected_font = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")

    plt.rcParams.update(
        {
            "font.family": selected_font,
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def draw_sequence_row(ax, y: float, label: str, centers: list[float], values: list[str]) -> None:
    ax.text(0.55, y, label, fontsize=12.5, va="center")

    left_bracket_x = centers[0] - 1.05
    right_bracket_x = centers[-1] + 0.9
    ax.text(left_bracket_x, y, "[", fontsize=18, va="center", ha="center")
    ax.text(right_bracket_x, y, "]", fontsize=18, va="center", ha="center")

    for x, value in zip(centers, values):
        ax.text(x, y, value, fontsize=12.5, ha="center", va="center")

    comma_xs = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
    for x in comma_xs:
        ax.text(x, y, ",", fontsize=13.5, ha="center", va="center")


def main() -> None:
    configure_font()

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    ax.text(0.55, 7.7, "원본 이벤트 로그", fontsize=15, fontweight="bold", va="center")
    ax.text(0.55, 6.7, "Case 1", fontsize=13, va="center")

    box_y = 5.95
    box_w = 1.7
    box_h = 0.95
    box_xs = [2.85, 5.0, 7.15, 9.3]
    box_centers = [x + box_w / 2 for x in box_xs]
    events = ["A, t1", "B, t2", "C, t3", "D, t4"]

    for x, label in zip(box_xs, events):
        rect = Rectangle((x, box_y), box_w, box_h, facecolor="#EAF2FB", edgecolor="#4C78A8", linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_y + box_h / 2, label, ha="center", va="center", fontsize=13)

    for i in range(len(box_xs) - 1):
        start_x = box_xs[i] + box_w + 0.08
        end_x = box_xs[i + 1] - 0.08
        ax.annotate(
            "",
            xy=(end_x, box_y + box_h / 2),
            xytext=(start_x, box_y + box_h / 2),
            arrowprops=dict(arrowstyle="->", lw=2.4, color="#666666", mutation_scale=22),
            zorder=5,
        )

    ax.text(0.55, 4.9, "생성되는 시퀀스", fontsize=15, fontweight="bold", va="center")

    centers = box_centers
    row_ys = [3.95, 3.08, 2.21, 1.34]

    draw_sequence_row(ax, row_ys[0], "Activity sequence", centers, ["A", "B", "C", "D"])
    draw_sequence_row(ax, row_ys[1], "delta_prev_seconds", centers, ["0", "t2 - t1", "t3 - t2", "t4 - t3"])
    draw_sequence_row(ax, row_ys[2], "delta_start_seconds", centers, ["0", "t2 - t1", "t3 - t1", "t4 - t1"])
    draw_sequence_row(ax, row_ys[3], "delta_next_seconds", centers, ["t2 - t1", "t3 - t2", "t4 - t3", "-"])

    fig.subplots_adjust(left=0.02, right=0.995, top=0.98, bottom=0.04)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
