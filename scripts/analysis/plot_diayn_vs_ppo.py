from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RUNS_EVAL_DIR = REPO_ROOT / "runs-eval" / "evals"
PLOTS_DIR = REPO_ROOT / "plots"


def ReadCsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ParseTaskFromRunName(runName: str) -> str:
    parts = runName.split("_")

    return "_".join(parts[2:])


def ReadSeedFromCommand(commandPath: Path) -> int:
    command = commandPath.read_text().strip()

    for part in command.split():

        if part.startswith("seed="):
            return int(part.split("=", maxsplit=1)[1])

    raise ValueError(f"Could not find seed in {commandPath}")


def ReadManifest(manifestPath: Path) -> dict[str, object]:
    with manifestPath.open() as handle:
        return json.load(handle)


def LoadSeries() -> dict[str, dict[str, dict[int, list[tuple[float, float]]]]]:
    series: dict[str, dict[str, dict[int, list[tuple[float, float]]]]] = {
        "pretrain": {},
        "ppo": {},
    }

    for runDir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        task = ParseTaskFromRunName(runDir.name)
        seed = ReadSeedFromCommand(runDir / "command.txt")
        rows = ReadCsv(runDir / "eval.csv")
        points = [(float(row["frame"]), float(row["episode_reward"])) for row in rows]
        series["pretrain"].setdefault(task, {})[seed] = points

    for evalDir in sorted(path for path in RUNS_EVAL_DIR.iterdir() if path.is_dir()):
        task = ParseTaskFromRunName(evalDir.name)
        manifest = ReadManifest(evalDir / "manifest.json")
        seed = int(manifest["seed"])
        rows = ReadCsv(evalDir / "02_ppo_controller" / "eval.csv")
        points = [(float(row["frame"]), float(row["episode_reward"])) for row in rows]
        series["ppo"].setdefault(task, {})[seed] = points

    return series


def PlotLearningCurves(
    series: dict[str, dict[str, dict[int, list[tuple[float, float]]]]],
) -> None:
    tasks = sorted(series["pretrain"].keys())
    figure, axes = plt.subplots(
        nrows=len(tasks),
        ncols=2,
        figsize=(14, 5 * len(tasks)),
        constrained_layout=True,
    )

    if len(tasks) == 1:
        axes = [axes]

    for rowIndex, task in enumerate(tasks):
        pretrainAxis, ppoAxis = axes[rowIndex]

        for seed, points in sorted(series["pretrain"][task].items()):
            frames, rewards = zip(*points)
            pretrainAxis.plot(frames, rewards, label=f"seed {seed}", linewidth=2)

        pretrainAxis.set_title(f"{task} original DIAYN eval")
        pretrainAxis.set_xlabel("pretrain frame")
        pretrainAxis.set_ylabel("episode reward")
        pretrainAxis.grid(True, alpha=0.3)
        pretrainAxis.legend()

        for seed, points in sorted(series["ppo"][task].items()):
            frames, rewards = zip(*points)
            ppoAxis.plot(frames, rewards, label=f"seed {seed}", linewidth=2)

        ppoAxis.set_title(f"{task} PPO controller eval")
        ppoAxis.set_xlabel("ppo controller frame")
        ppoAxis.set_ylabel("episode reward")
        ppoAxis.grid(True, alpha=0.3)
        ppoAxis.legend()

    figure.suptitle("DIAYN original eval vs PPO controller eval", fontsize=16)
    figure.savefig(PLOTS_DIR / "diayn_vs_ppo_learning_curves.png", dpi=200)
    plt.close(figure)


def PlotFinalRewardComparison(
    series: dict[str, dict[str, dict[int, list[tuple[float, float]]]]],
) -> None:
    tasks = sorted(series["pretrain"].keys())
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
        seeds = sorted(series["pretrain"][task].keys())
        pretrainFinal = [series["pretrain"][task][seed][-1][1] for seed in seeds]
        ppoFinal = [series["ppo"][task][seed][-1][1] for seed in seeds]
        xValues = list(range(len(seeds)))
        width = 0.36

        axis.bar(
            [value - width / 2 for value in xValues],
            pretrainFinal,
            width=width,
            label="original eval",
        )
        axis.bar(
            [value + width / 2 for value in xValues],
            ppoFinal,
            width=width,
            label="ppo controller",
        )
        axis.set_title(f"{task} final reward")
        axis.set_xlabel("seed")
        axis.set_xticks(xValues, [str(seed) for seed in seeds])
        axis.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("final episode reward")
    axes[0].legend()
    figure.savefig(PLOTS_DIR / "diayn_vs_ppo_final_reward.png", dpi=200)
    plt.close(figure)


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    series = LoadSeries()

    PlotLearningCurves(series)

    PlotFinalRewardComparison(series)


if __name__ == "__main__":
    main()
