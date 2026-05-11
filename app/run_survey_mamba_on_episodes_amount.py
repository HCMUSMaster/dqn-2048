import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app import run_mamba_with_best_hyper_params as best


ALLOWED_MODELS = (
    "dqn",
    "double_dqn",
    "dueling_double_dqn",
    "qr_dqn",
    "h_dqn",
)
ALL_MODELS_TOKEN = "all"
DEFAULT_EPISODES_CSV = "300,500,1000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey Mamba2 models' performance vs training episode count."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALLOWED_MODELS),
        help="Comma-separated models to run: dqn,double_dqn,dueling_double_dqn,qr_dqn,h_dqn,all",
    )
    parser.add_argument(
        "--num_episodes",
        type=str,
        default=DEFAULT_EPISODES_CSV,
        help="Comma-separated training episode counts, e.g. 300,500,1000",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=5000,
        help="Maximum steps per environment episode.",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=100,
        help="Evaluation seeds per run.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="In-training greedy evaluation interval.",
    )
    return parser


def parse_models(models_csv: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in models_csv.split(','):
        model = raw_model.strip().lower()
        if not model:
            allowed = ",".join((*ALLOWED_MODELS, ALL_MODELS_TOKEN))
            raise ValueError(f"Unknown model in --models: {model}. Allowed models: {allowed}")
        if model == ALL_MODELS_TOKEN:
            return list(ALLOWED_MODELS)
        if model not in ALLOWED_MODELS:
            allowed = ",".join((*ALLOWED_MODELS, ALL_MODELS_TOKEN))
            raise ValueError(f"Unknown model in --models: {model}. Allowed models: {allowed}")
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def parse_episode_list(episodes_csv: str) -> list[int]:
    episodes: list[int] = []
    seen: set[int] = set()
    for raw_value in episodes_csv.split(','):
        value = raw_value.strip()
        if not value:
            raise ValueError('Empty value in --num_episodes list.')
        episode_count = int(value)
        if episode_count <= 0:
            raise ValueError('Episode values in --num_episodes must be positive integers.')
        if episode_count not in seen:
            episodes.append(episode_count)
            seen.add(episode_count)
    return episodes


def build_run_namespace(
    *,
    model: str,
    num_episodes: int,
    output_dir: Path,
    seed: int,
    max_steps_per_episode: int,
    num_eval_seeds: int,
    eval_every: int,
) -> SimpleNamespace:
    common = {
        'seed': seed,
        'num_episodes': num_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'num_eval_seeds': num_eval_seeds,
        'eval_every': eval_every,
        'output_dir': str(output_dir),
        'log_episodes': False,
        'print_eval_summary': True,
        'log_save': False,
    }
    # Not used programmatically here, but keep compatibility shape.
    return SimpleNamespace(**common, **{})


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def to_jsonable_summary(summary: dict) -> dict:
    return {
        "avg_return": float(summary["avg_return"]),
        "std_return": float(summary["std_return"]),
        "avg_length": float(summary["avg_length"]),
        "avg_max_tile": float(summary["avg_max_tile"]),
        "avg_illegal_action_attempts": float(summary["avg_illegal_action_attempts"]),
        "total_illegal_action_attempts": int(summary["total_illegal_action_attempts"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    selected_models = parse_models(args.models)
    episode_values = parse_episode_list(args.num_episodes)

    root_dir = Path(__file__).resolve().parents[1]
    survey_root = root_dir / 'output' / 'survey_mamba'
    survey_root.mkdir(parents=True, exist_ok=True)

    full_summary: dict[str, list[dict]] = {}

    for model in selected_models:
        slug = model if model != 'double_dqn' else 'double-dqn'
        if model == 'dueling_double_dqn':
            slug = 'dueling-double-dqn'
        if model == 'qr_dqn':
            slug = 'qr-dqn'
        if model == 'h_dqn':
            slug = 'h-dqn'

        model_root = survey_root / slug
        model_root.mkdir(parents=True, exist_ok=True)
        model_summary: list[dict] = []

        for episode_count in episode_values:
            run_output_dir = model_root / f'episodes_{episode_count}'
            run_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running model={model} episodes={episode_count}...")

            # Build command using helper builders from best hyperparams module
            if model == 'dqn':
                cmd = best.build_mamba_dqn_command(episode_count, run_output_dir)
            elif model == 'double_dqn':
                cmd = best.build_mamba_double_dqn_command(episode_count, run_output_dir)
            elif model == 'dueling_double_dqn':
                cmd = best.build_dueling_double_dqn_command(episode_count, run_output_dir)
            elif model == 'qr_dqn':
                cmd = best.build_mamba_qr_dqn_command(episode_count, run_output_dir)
            elif model == 'h_dqn':
                cmd = best.build_mamba_h_dqn_command(episode_count, run_output_dir)
            else:
                print(f"No builder for model {model}, skipping.")
                continue

            try:
                run_command(cmd, cwd=root_dir)
            except subprocess.CalledProcessError as exc:
                print(f"Run failed for model={model} episodes={episode_count}: {exc}")
                continue

            # Read eval_meta.json produced by mamba2_train_test via eval.save_eval_results
            eval_meta_path = run_output_dir / 'eval_meta.json'
            if not eval_meta_path.exists():
                print(f"Missing eval_meta.json at {eval_meta_path}, skipping summary entry.")
                continue

            meta = json.loads(eval_meta_path.read_text(encoding='utf-8'))
            summary = to_jsonable_summary(meta['summary'])

            model_summary.append(
                {
                    'num_episodes': episode_count,
                    'output_dir': str(run_output_dir.relative_to(root_dir)),
                    'summary': summary,
                }
            )

        model_summary_path = model_root / 'summary_by_episodes.json'
        model_summary_path.write_text(json.dumps(model_summary, indent=2), encoding='utf-8')
        full_summary[model] = model_summary
        print(f"Saved survey summary: {model_summary_path}")

    full_summary_path = survey_root / 'summary_all_models.json'
    full_summary_path.write_text(json.dumps(full_summary, indent=2), encoding='utf-8')
    print(f"Saved combined survey summary: {full_summary_path}")


if __name__ == '__main__':
    main()
