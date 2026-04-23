from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPO_ROOT = REPO_ROOT / "outputs" / "diayn_ppo"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "ppo_reward_table.md"


def ReadSummary(summaryPath: Path) -> dict[str, object]:
    with summaryPath.open() as handle:
        return json.load(handle)


def ReadMeanEpisodeReward(evalPath: Path) -> float:
    with evalPath.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    return mean(float(row["episode_reward"]) for row in rows)


def BuildTaskSummary(ppoRoot: Path) -> dict[str, dict[str, object]]:
    byTask: dict[str, list[dict[str, float | int]]] = {}

    for summaryPath in sorted(ppoRoot.rglob("summary.json")):
        summary = ReadSummary(summaryPath)
        evalPath = summaryPath.parent / "eval.csv"

        if not evalPath.exists():
            continue

        task = str(summary["task"])
        seed = int(summary["seed"])
        finalReward = float(summary["final_reward"])
        meanEpisodeReward = ReadMeanEpisodeReward(evalPath)

        byTask.setdefault(task, []).append(
            {
                "seed": seed,
                "final_reward": finalReward,
                "mean_episode_reward": meanEpisodeReward,
            }
        )

    result: dict[str, dict[str, object]] = {}

    for task, items in sorted(byTask.items()):
        items.sort(key=lambda item: int(item["seed"]))
        finalRewards = [float(item["final_reward"]) for item in items]
        meanRewards = [float(item["mean_episode_reward"]) for item in items]
        result[task] = {
            "seeds": len(items),
            "final_reward_mean": mean(finalRewards),
            "final_reward_std": stdev(finalRewards) if len(finalRewards) > 1 else 0.0,
            "final_reward_per_seed": [float(item["final_reward"]) for item in items],
            "mean_episode_reward_mean": mean(meanRewards),
            "mean_episode_reward_std": (
                stdev(meanRewards) if len(meanRewards) > 1 else 0.0
            ),
            "mean_episode_reward_per_seed": [
                float(item["mean_episode_reward"]) for item in items
            ],
        }

    return result


def FormatList(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.1f}" for value in values) + "]"


def BuildTable(taskSummary: dict[str, dict[str, object]]) -> str:
    lines = [
        "| Run | Metric | Seeds | Mean | Std | Per-seed |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for task, summary in taskSummary.items():
        taskLabel = f"PPO / {task}"
        lines.append(
            "| "
            + " | ".join(
                [
                    taskLabel,
                    "Final reward",
                    str(summary["seeds"]),
                    f"{float(summary['final_reward_mean']):.2f}",
                    f"{float(summary['final_reward_std']):.2f}",
                    FormatList(list(summary["final_reward_per_seed"])),
                ]
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    taskLabel,
                    "Mean episode reward",
                    str(summary["seeds"]),
                    f"{float(summary['mean_episode_reward_mean']):.2f}",
                    f"{float(summary['mean_episode_reward_std']):.2f}",
                    FormatList(list(summary["mean_episode_reward_per_seed"])),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Markdown summary table for PPO controller eval runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ppo-root",
        default=str(DEFAULT_PPO_ROOT),
        help="Root directory to search recursively for PPO eval outputs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the Markdown table file to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = ParseArgs()
    outputPath = Path(args.output)
    taskSummary = BuildTaskSummary(Path(args.ppo_root))
    outputPath.write_text(BuildTable(taskSummary))
    print(outputPath.read_text())


if __name__ == "__main__":
    main()
