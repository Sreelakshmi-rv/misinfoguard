import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

def create_architecture_diagram():

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    def draw_box(x, y, w, h, color, label, sublabel=""):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor="white",
                             linewidth=1.5,
                             alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.2 if sublabel else 0),
                label,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25,
                    sublabel,
                    ha="center", va="center",
                    fontsize=7.5, color="#cccccc")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="white",
                        lw=1.5
                    ))

    # ── Title ──
    ax.text(7, 6.6, "MisinfoGuard — System Architecture",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")

    # ── Row 1: Input ──
    draw_box(0.3, 4.8, 2.5, 1.0, "#2c3e50",
             "User Input", "News claim / text")

    draw_box(0.3, 3.2, 2.5, 1.0, "#2c3e50",
             "Network Config", "Nodes, seed strategy")

    # ── Row 2: Processing ──
    draw_box(3.5, 5.0, 2.8, 0.8, "#8e44ad",
             "NLP Classifier", "TF-IDF + XGBoost")

    draw_box(3.5, 3.0, 2.8, 0.8, "#16a085",
             "Graph Generator", "Barabasi-Albert")

    draw_box(3.5, 0.8, 2.8, 0.8, "#16a085",
             "SIR Spread Model", "S → I → R")

    # ── Row 3: RL ──
    draw_box(7.5, 3.8, 2.8, 1.2, "#c0392b",
             "Multi-Agent RL", "2 × DQN Agents\nRegion 0 | Region 1")

    draw_box(7.5, 2.0, 2.8, 1.2, "#d35400",
             "Intervention", "Flag | Counter-msg\nQuarantine | None")

    # ── Row 4: Output ──
    draw_box(11.2, 4.6, 2.5, 0.8, "#27ae60",
             "Classification", "FAKE / REAL")

    draw_box(11.2, 3.2, 2.5, 0.8, "#27ae60",
             "Spread Chart", "SIR curve")

    draw_box(11.2, 1.8, 2.5, 0.8, "#27ae60",
             "Comparison", "4-scenario results")

    # ── Arrows ──
    # Input → Processing
    draw_arrow(2.8, 5.3, 3.5, 5.4)
    draw_arrow(2.8, 3.7, 3.5, 3.4)

    # Graph → SIR
    draw_arrow(4.9, 3.0, 4.9, 1.6)

    # NLP → Output
    draw_arrow(6.3, 5.4, 11.2, 5.0)

    # SIR → Multi-Agent RL (agents observe infection state)
    draw_arrow(6.3, 1.2, 7.5, 3.8)

    # RL → Intervention
    draw_arrow(8.9, 3.8, 8.9, 3.2)

    # Intervention → Spread chart
    draw_arrow(10.3, 2.6, 11.2, 3.6)

    # RL → Comparison
    draw_arrow(10.3, 2.2, 11.2, 2.2)

    plt.tight_layout()
    plt.savefig("architecture.png", dpi=150,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Architecture diagram saved as architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()