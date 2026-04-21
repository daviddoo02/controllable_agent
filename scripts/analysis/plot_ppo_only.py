from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_EVAL_DIR = REPO_ROOT / "runs-eval" / "evals"


def ReadCsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ReadJson(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def ParseTaskFromRunName(runName: str) -> str:
    parts = runName.split("_")

    return "_".join(parts[2:])


def LoadPpoSeries() -> dict[str, dict[int, list[tuple[float, float]]]]:
    series: dict[str, dict[int, list[tuple[float, float]]]] = {}

    for evalDir in sorted(path for path in RUNS_EVAL_DIR.iterdir() if path.is_dir()):
        task = ParseTaskFromRunName(evalDir.name)
        manifest = ReadJson(evalDir / "manifest.json")
        seed = int(manifest["seed"])
        rows = ReadCsv(evalDir / "02_ppo_controller" / "eval.csv")
        points = [(float(row["frame"]), float(row["episode_reward"])) for row in rows]
        series.setdefault(task, {})[seed] = points

    return series


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
    series = LoadPpoSeries()
    PlotPpoLearningCurves(series, REPO_ROOT / "ppo_learning_curves.png")
    PlotPpoFinalRewardBars(series, REPO_ROOT / "ppo_final_reward_bars.png")


if __name__ == "__main__":
    main()
