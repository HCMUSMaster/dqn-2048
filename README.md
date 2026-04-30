# Survey Deep Q Network Playing 2048 game

## Set-up

Install the required packages:

```bash
uv sync
```

Test installation:

```bash
uv run pytest tests
```

## Finding the best hyperparameters

We have 2 types of networks, normal DQN:

```bash
uv run -m app.optuna_tune --storage sqlite:///optuna.db --save_best_dir ./output/optuna
```

And Mamba DQN:

```bash
uv run -m app.optuna_mamba --storage sqlite:///optuna.db --save_best_dir ./output/optuna_mamba
```

## Train and test with best hyperparameters

Adjust number at `app/run_with_best_hyper_params.py` and run:

```bash
uv run -m app.run_with_best_hyper_params
```