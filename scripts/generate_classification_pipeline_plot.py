import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, title, subtitle, facecolor, edgecolor="#2b2b2b"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=10)


def add_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color="#1f2937",
    )
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center", va="center", fontsize=9, color="#374151")


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Pipeline de Classification Multi-label",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.91,
        "De l'embedding Word2Vec pondere TF-IDF a la prediction finale",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    y = 0.55
    w = 0.16
    h = 0.22

    add_box(
        ax,
        0.04,
        y,
        w,
        h,
        "1. Entree Texte",
        "Summary uniquement\npretraite",
        facecolor="#e0f2fe",
    )
    add_box(
        ax,
        0.24,
        y,
        w,
        h,
        "2. Features",
        "Word2Vec +\nponderation TF-IDF",
        facecolor="#dbeafe",
    )
    add_box(
        ax,
        0.44,
        y,
        w,
        h,
        "3. Modele OVR",
        "LogReg binaire\npar classe",
        facecolor="#ede9fe",
    )
    add_box(
        ax,
        0.64,
        y,
        w,
        h,
        "4. Probabilites",
        "p_k(x) pour\nchaque label",
        facecolor="#fae8ff",
    )
    add_box(
        ax,
        0.84,
        y,
        0.12,
        h,
        "5. Decision",
        "y_k = 1[p_k >= tau]",
        facecolor="#fee2e2",
    )

    add_arrow(ax, 0.20, y + h / 2, 0.24, y + h / 2)
    add_arrow(ax, 0.40, y + h / 2, 0.44, y + h / 2)
    add_arrow(ax, 0.60, y + h / 2, 0.64, y + h / 2)
    add_arrow(ax, 0.80, y + h / 2, 0.84, y + h / 2)

    add_box(
        ax,
        0.23,
        0.18,
        0.23,
        0.20,
        "Validation",
        "Grid-search sur tau\nmaximisant F1 micro",
        facecolor="#dcfce7",
    )
    add_box(
        ax,
        0.54,
        0.18,
        0.23,
        0.20,
        "Evaluation Test",
        "Precision/Recall/F1\nMicro, Macro, AUC",
        facecolor="#fef9c3",
    )

    add_arrow(ax, 0.70, 0.55, 0.34, 0.38, text="calibrage seuil")
    add_arrow(ax, 0.90, 0.55, 0.66, 0.38, text="mesure finale")

    ax.text(
        0.5,
        0.06,
        "OneVsRest entraine un classifieur par label et applique un seuil global optimise sur validation.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )

    plt.tight_layout()
    plt.savefig("classification_pipeline.png", dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    main()
