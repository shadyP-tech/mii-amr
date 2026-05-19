from pathlib import Path

from .fallback_png import write_fallback_plot
from .statistics import ellipse_parameters


def plot_scatter_with_ellipse(points, mu, sigma, path, title, xlabel, ylabel, labels=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError:
        write_fallback_plot(
            path,
            groups=[(points, (46, 92, 170)), ([mu], (180, 40, 40))],
            ellipses=[(mu, sigma, (20, 130, 60))],
        )
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = ellipse_parameters(mu, sigma)
    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in points], [point[1] for point in points], label=labels or "samples")
    ax.scatter([mu[0]], [mu[1]], marker="x", s=100, label="mean")
    ellipse = Ellipse(
        xy=mu,
        width=params["major_axis_length_m"],
        height=params["minor_axis_length_m"],
        angle=params["orientation_deg"],
        fill=False,
        linewidth=2,
        label="95% ellipse",
    )
    ax.add_patch(ellipse)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sim_vs_real_errors(real_errors, sim_errors, real_mu, sim_mu, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_fallback_plot(
            path,
            groups=[
                (real_errors, (46, 92, 170)),
                (sim_errors, (220, 120, 40)),
                ([real_mu], (180, 40, 40)),
                ([sim_mu], (40, 130, 60)),
            ],
            ellipses=[],
        )
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in real_errors], [point[1] for point in real_errors], label="real local error")
    ax.scatter([point[0] for point in sim_errors], [point[1] for point in sim_errors], label="simulation local error")
    ax.scatter([real_mu[0]], [real_mu[1]], marker="x", s=100, label="real mean")
    ax.scatter([sim_mu[0]], [sim_mu[1]], marker="+", s=120, label="simulation mean")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Local displacement error after 30 cm command")
    ax.set_xlabel("forward error [m]")
    ax.set_ylabel("lateral error [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
