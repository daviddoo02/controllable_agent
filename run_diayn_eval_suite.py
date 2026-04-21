"""
Run original, PPO, and argmax DIAYN post-pretraining evaluations.

This script assumes a pretrained DIAYN snapshot already exists. It does not
run pretraining.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DOMAIN_TASKS = {
    "cheetah": ["walk", "walk_backward", "run", "run_backward"],
    "quadruped": ["stand", "walk", "run", "jump"],
    "walker": ["stand", "walk", "run", "flip"],
}


def DetectDomain(task: str) -> str:
    return task.split("_", maxsplit=1)[0]


def BuildPretrainEvalCommand(
    python_executable: str,
    snapshot: Path,
    task: str,
    seed: int,
    run_dir: Path,
    final_tests: int,
    device: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "url_benchmark.pretrain",
        "agent=diayn",
        f"task={task}",
        f"seed={seed}",
        "num_train_frames=0",
        f"load_model={snapshot}",
        f"final_tests={final_tests}",
        f"device={device}",
        "use_tb=false",
        "use_wandb=false",
        "save_video=false",
        f"hydra.run.dir={run_dir}",
    ]


def BuildPpoCommand(
    python_executable: str,
    snapshot: Path,
    task: str,
    seed: int,
    outdir: Path,
    total_frames: int,
    skill_horizon: int,
    eval_every: int,
    eval_episodes: int,
    device: str,
) -> list[str]:
    return [
        python_executable,
        "diayn_ppo_controller.py",
        "--snapshot",
        str(snapshot),
        "--task",
        task,
        "--seed",
        str(seed),
        "--outdir",
        str(outdir),
        "--total_frames",
        str(total_frames),
        "--skill_horizon",
        str(skill_horizon),
        "--eval_every",
        str(eval_every),
        "--eval_episodes",
        str(eval_episodes),
        "--device",
        device,
    ]


def RunCommand(command: list[str], cwd: Path) -> None:
    print(f"\n[Run] cwd={cwd}")
    print("[Run] " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def RunOriginalFinalizeEval(
    repo_root: Path,
    python_executable: str,
    snapshot: Path,
    task: str,
    seed: int,
    run_dir: Path,
    final_tests: int,
    device: str,
    original_ref: str,
) -> dict[str, str]:
    command = BuildPretrainEvalCommand(
        python_executable=python_executable,
        snapshot=snapshot,
        task=task,
        seed=seed,
        run_dir=run_dir,
        final_tests=final_tests,
        device=device,
    )

    with tempfile.TemporaryDirectory(prefix="diayn-original-") as tmpdir:
        worktree = Path(tmpdir)
        worktree_added = False

        add_command = [
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree),
            original_ref,
        ]

        remove_command = ["git", "worktree", "remove", "--force", str(worktree)]

        try:
            RunCommand(add_command, repo_root)
            worktree_added = True
            RunCommand(command, worktree)

        finally:

            if worktree_added:
                subprocess.run(remove_command, cwd=repo_root, check=True)

    return {
        "method": "original_finalize",
        "output_dir": str(run_dir),
        "command": " ".join(command),
        "ref": original_ref,
    }


def RunPpoController(
    repo_root: Path,
    python_executable: str,
    snapshot: Path,
    task: str,
    seed: int,
    outdir: Path,
    total_frames: int,
    skill_horizon: int,
    eval_every: int,
    eval_episodes: int,
    device: str,
) -> dict[str, str]:
    command = BuildPpoCommand(
        python_executable=python_executable,
        snapshot=snapshot,
        task=task,
        seed=seed,
        outdir=outdir,
        total_frames=total_frames,
        skill_horizon=skill_horizon,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        device=device,
    )
    RunCommand(command, repo_root)
    return {
        "method": "ppo_controller",
        "output_dir": str(outdir),
        "command": " ".join(command),
    }


def RunArgmaxFinalizeEval(
    repo_root: Path,
    python_executable: str,
    snapshot: Path,
    task: str,
    seed: int,
    run_dir: Path,
    final_tests: int,
    device: str,
) -> dict[str, str]:
    command = BuildPretrainEvalCommand(
        python_executable=python_executable,
        snapshot=snapshot,
        task=task,
        seed=seed,
        run_dir=run_dir,
        final_tests=final_tests,
        device=device,
    )
    RunCommand(command, repo_root)
    return {
        "method": "argmax_finalize",
        "output_dir": str(run_dir),
        "command": " ".join(command),
    }


def ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original finalize evals, PPO skill control, and argmax evals from a pretrained DIAYN snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Absolute or relative path to a pretrained DIAYN snapshot.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="DMC task name, for example walker_walk or cheetah_run.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--outdir",
        required=True,
        help="Root output directory for all three runs.",
    )
    parser.add_argument(
        "--original_ref",
        default="main",
        help="Git ref to use for the original finalize baseline.",
    )
    parser.add_argument("--final_tests", type=int, default=10)
    parser.add_argument("--ppo_total_frames", type=int, default=100_000)
    parser.add_argument("--ppo_skill_horizon", type=int, default=50)
    parser.add_argument("--ppo_eval_every", type=int, default=4_000)
    parser.add_argument("--ppo_eval_episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = ParseArgs()
    repo_root = Path(__file__).resolve().parent
    snapshot = Path(args.snapshot).resolve()
    outdir = Path(args.outdir).resolve()

    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")

    domain = DetectDomain(args.task)

    if domain not in DOMAIN_TASKS:
        raise ValueError(
            f"Unsupported domain '{domain}'. Supported domains: {sorted(DOMAIN_TASKS)}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    original_dir = outdir / "01_original_finalize"
    ppo_dir = outdir / "02_ppo_controller"
    argmax_dir = outdir / "03_argmax_finalize"

    original_dir.mkdir(parents=True, exist_ok=True)
    ppo_dir.mkdir(parents=True, exist_ok=True)
    argmax_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "snapshot": str(snapshot),
        "task": args.task,
        "seed": args.seed,
        "device": args.device,
        "methods": [],
    }

    manifest["methods"].append(
        RunOriginalFinalizeEval(
            repo_root=repo_root,
            python_executable=sys.executable,
            snapshot=snapshot,
            task=args.task,
            seed=args.seed,
            run_dir=original_dir,
            final_tests=args.final_tests,
            device=args.device,
            original_ref=args.original_ref,
        )
    )

    manifest["methods"].append(
        RunPpoController(
            repo_root=repo_root,
            python_executable=sys.executable,
            snapshot=snapshot,
            task=args.task,
            seed=args.seed,
            outdir=ppo_dir,
            total_frames=args.ppo_total_frames,
            skill_horizon=args.ppo_skill_horizon,
            eval_every=args.ppo_eval_every,
            eval_episodes=args.ppo_eval_episodes,
            device=args.device,
        )
    )

    manifest["methods"].append(
        RunArgmaxFinalizeEval(
            repo_root=repo_root,
            python_executable=sys.executable,
            snapshot=snapshot,
            task=args.task,
            seed=args.seed,
            run_dir=argmax_dir,
            final_tests=args.final_tests,
            device=args.device,
        )
    )

    manifest_path = outdir / "manifest.json"

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
