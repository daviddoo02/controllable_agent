from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPO_ROOT = REPO_ROOT / "outputs" / "diayn_ppo"


def ReadCsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ReadJson(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def LoadPpoSeries(ppoRoot: Path) -> dict[str, dict[int, list[tuple[float, float]]]]:
    series: dict[str, dict[int, list[tuple[float, float]]]] = {}

    for summaryPath in sorted(ppoRoot.rglob("summary.json")):
        summary = ReadJson(summaryPath)
        evalPath = summaryPath.parent / "eval.csv"

        if not evalPath.exists():
            continue

        task = str(summary["task"])
        seed = int(summary["seed"])
        rows = ReadCsv(evalPath)
        points = [(float(row["frame"]), float(row["episode_reward"])) for row in rows]
        series.setdefault(task, {})[seed] = points

    return series


def ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PPO controller evaluation runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ppo-root",
        default=str(DEFAULT_PPO_ROOT),
        help="Root directory to search recursively for PPO eval outputs.",
    )
    parser.add_argument(
        "--curves-output",
        default=str(REPO_ROOT / "ppo_learning_curves.png"),
        help="Path to write the PPO learning-curve figure.",
    )
    parser.add_argument(
        "--bars-output",
        default=str(REPO_ROOT / "ppo_final_reward_bars.png"),
        help="Path to write the PPO final-reward bar chart.",
    )
    return parser.parse_args()


def PlotPpoLearningCurves(
    series: dict[str, dict[int, list[tuple[float, float]]]],
    outputPath: Path,
) -> None:
    tasks = sorted(series.keys())
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(tasks),
        figsize=(7 * len(tasks), 5),
        constrained_layout=True,
        sharey=False,
    )

    if len(tasks) == 1:
        axes = [axes]

    for axis, task in zip(axes, tasks):
        for seed, points in sorted(series[task].items()):
            frames, rewards = zip(*points)
            axis.plot(frames, rewards, label=f"seed {seed}", linewidth=2)

        axis.set_title(f"PPO controller - {task}")
        axis.set_xlabel("ppo controller frame")
        axis.set_ylabel("episode reward")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.savefig(outputPath, dpi=200)
    plt.close(figure)


def PlotPpoFinalRewardBars(
    series: dict[str, dict[int, list[tuple[float, float]]]],
    outputPath: Path,
) -> None:
    tasks = sorted(series.keys())
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(tasks),
        figsize=(7 * len(tasks), 5),
        constrained_layout=True,
        sharey=True,
    )

    if len(tasks) == 1:
        axes = [axes]

    for axis, task in zip(axes, tasks):
        seeds = sorted(series[task].keys())
        finalRewards = [series[task][seed][-1][1] for seed in seeds]
        axis.bar([str(seed) for seed in seeds], finalRewards)
        axis.set_title(f"PPO final reward - {task}")
        axis.set_xlabel("seed")
        axis.set_ylabel("final episode reward")
        axis.grid(True, axis="y", alpha=0.3)

    figure.savefig(outputPath, dpi=200)
    plt.close(figure)


def main() -> None:
    args = ParseArgs()
    series = LoadPpoSeries(Path(args.ppo_root))
    PlotPpoLearningCurves(series, Path(args.curves_output))
    PlotPpoFinalRewardBars(series, Path(args.bars_output))


if __name__ == "__main__":
    main()
