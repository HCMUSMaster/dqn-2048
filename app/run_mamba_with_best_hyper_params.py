import argparse
import subprocess
import sys
from pathlib import Path


ALLOWED_MODELS = (
    "dqn",
    "double_dqn",
    "mamba2_double_dqn",
    "mamba2_dueling_double_dqn",
    "mamba2_qr_dqn",
    "mamba2_h_dqn",
)
ALL_MODELS_TOKEN = "all"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run best-known hyperparameters for selected Mamba/Mamba2 2048 models."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=300,
        help="Number of training episodes for each selected model.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALLOWED_MODELS),
        help="Comma-separated models to run: dqn,double_dqn,mamba2_double_dqn,mamba2_dueling_double_dqn,mamba2_qr_dqn,mamba2_h_dqn,all",
    )
    return parser


def parse_models(models_csv: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in models_csv.split(","):
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


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def build_mamba_dqn_command(num_episodes: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "app.mamba2_train_test",
        "--algorithm",
        "dqn",
        "--num_episodes",
        str(num_episodes),
        "--buffer_size",
        "200000",
        "--batch_size",
        "64",
        "--seq_len",
        "16",
        "--gamma",
        "0.9739846568718238",
        "--lr",
        "4.1433825788519225e-05",
        "--target_sync_every",
        "500",
        "--learn_start",
        "2000",
        "--learn_every",
        "8",
        "--eps_start",
        "1.0",
        "--eps_end",
        "0.08453161403381923",
        "--eps_decay_steps",
        "5000",
        "--max_steps_per_episode",
        "5000",
        "--grad_clip",
        "14.823449461719127",
        "--hidden_dim",
        "256",
        "--mamba_layers",
        "4",
        "--mamba_state_dim",
        "64",
        "--mamba_conv_dim",
        "4",
        "--mamba_expand",
        "2",
        "--num_eval_seeds",
        "100",
        "--output_dir",
        str(output_dir),
    ]


def build_mamba_double_dqn_command(num_episodes: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "app.mamba2_train_test",
        "--algorithm",
        "double_dqn",
        "--num_episodes",
        str(num_episodes),
        "--buffer_size",
        "20000",
        "--batch_size",
        "64",
        "--seq_len",
        "4",
        "--gamma",
        "0.9560651463990533",
        "--lr",
        "0.0006598261030886547",
        "--target_sync_every",
        "100",
        "--learn_start",
        "5000",
        "--learn_every",
        "8",
        "--eps_start",
        "1.0",
        "--eps_end",
        "0.1972633080505522",
        "--eps_decay_steps",
        "50000",
        "--max_steps_per_episode",
        "5000",
        "--grad_clip",
        "16.20361877066044",
        "--hidden_dim",
        "256",
        "--mamba_layers",
        "3",
        "--mamba_state_dim",
        "128",
        "--mamba_conv_dim",
        "4",
        "--mamba_expand",
        "3",
        "--num_eval_seeds",
        "100",
        "--output_dir",
        str(output_dir),
    ]


def build_mamba2_double_dqn_command(num_episodes: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "app.mamba2_train_test",
        "--algorithm",
        "double_dqn",
        "--num_episodes",
        str(num_episodes),
        "--buffer_size",
        "200000",
        "--batch_size",
        "64",
        "--seq_len",
        "12",
        "--gamma",
        "0.958278580825883",
        "--lr",
        "5.1152218086166416e-05",
        "--target_sync_every",
        "100",
        "--learn_start",
        "5000",
        "--learn_every",
        "4",
        "--eps_start",
        "1.0",
        "--eps_end",
        "0.09461215273949933",
        "--eps_decay_steps",
        "5000",
        "--max_steps_per_episode",
        "5000",
        "--grad_clip",
        "5.6534348547385",
        "--hidden_dim",
        "256",
        "--mamba_layers",
        "2",
        "--mamba_state_dim",
        "64",
        "--mamba_conv_dim",
        "2",
        "--mamba_expand",
        "1",
        "--num_eval_seeds",
        "100",
        "--output_dir",
        str(output_dir),
    ]


def run_placeholder(model_name: str) -> None:
    print(f"TODO: best hyperparameters for {model_name} are not filled in yet. Skipping.")


def main() -> None:
    args = build_parser().parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    base_out = root_dir / "output" / "mamba"
    base_out.mkdir(parents=True, exist_ok=True)

    selected_models = parse_models(args.models)

    if "dqn" in selected_models:
        print("Running best Mamba DQN hyperparameters...")
        run_command(build_mamba_dqn_command(args.num_episodes, base_out / "dqn"), cwd=root_dir)

    if "double_dqn" in selected_models:
        print("Running best Mamba Double DQN hyperparameters...")
        run_command(build_mamba_double_dqn_command(args.num_episodes, base_out / "double-dqn"), cwd=root_dir)

    if "mamba2_double_dqn" in selected_models:
        print("Running best Mamba2 Double DQN hyperparameters...")
        run_command(build_mamba2_double_dqn_command(args.num_episodes, base_out / "mamba2-double-dqn"), cwd=root_dir)

    if "mamba2_dueling_double_dqn" in selected_models:
        run_placeholder("mamba2_dueling_double_dqn")

    if "mamba2_qr_dqn" in selected_models:
        run_placeholder("mamba2_qr_dqn")

    if "mamba2_h_dqn" in selected_models:
        run_placeholder("mamba2_h_dqn")

    print(f"Selected models: {','.join(selected_models)}")
    print(f"Completed available runs. Outputs are in: {base_out}")


if __name__ == "__main__":
    main()