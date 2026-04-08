# PPO Enhancements for Sparse-Reward Game Environments

This project investigates improvements to Proximal Policy Optimization (PPO) in sparse-reward, game-like environments. We use [CleanRL](https://github.com/vwxyzjn/cleanrl)'s single-file PPO implementation as our baseline and evaluate enhancements using [MiniGrid](https://minigrid.farama.org/) environments. Experiments are tracked with [Weights & Biases](https://wandb.ai).

---

## Setup (Windows)

### 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing so `uv` is on your PATH.

### 2. Clone the repo and install dependencies

```powershell
git clone https://github.com/FabriziOki/ppo-mlbda.git
cd ppo-mlbda
uv sync
```

This creates a `.venv` and installs all locked dependencies automatically.

### 3. Set up Weights & Biases

Create a free account at [wandb.ai](https://wandb.ai), accept the team invite sent to your email, then login:

```powershell
uv run wandb login
```

Paste your API key from `wandb.ai/settings` when prompted. You only need to do this once.

---

## Setup (Linux)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installing so `uv` is on your PATH.

### 2. Install system dependencies

MiniGrid requires pygame which needs a few system libraries:

```bash
sudo apt update
sudo apt install -y python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

### 3. Clone the repo and install dependencies

```bash
git clone https://github.com/FabriziOki/ppo-mlbda.git
cd ppo-mlbda
uv sync
```

This creates a `.venv` and installs all locked dependencies automatically.

> **Note:** If you have Python 3.14 as your system default, uv will automatically use the correct version (3.11 or 3.12) as specified in `pyproject.toml`. No manual intervention needed.

### 4. Set up Weights & Biases

Create a free account at [wandb.ai](https://wandb.ai), accept the team invite sent to your email, then login:

```bash
uv run wandb login
```

Paste your API key from `wandb.ai/settings` when prompted. You only need to do this once.

---

## Running Experiments

### Sanity check (CartPole)

Run this first to verify your setup is working:

```bash
uv run python ppo.py --env-id CartPole-v1 --total-timesteps 500000 
```

You should see `episodic_return` climbing toward 500 in the W&B dashboard.

### Baseline PPO (MiniGrid)

```bash
uv run python ppo.py --num-envs 64 
```

### RND Enhanced PPO (MiniGrid)

```bash
uv run python ppo_rnd.py --num-envs 64 
```

### Running multiple seeds (for ablation study)

Always run at least 3 seeds per experiment. Name your runs clearly:

```bash
uv run python ppo.py --num-envs 64 --seed 1 --run-name "baseline-seed1"
uv run python ppo.py --num-envs 64 --seed 2 --run-name "baseline-seed2"
uv run python ppo.py --num-envs 64 --seed 3 --run-name "baseline-seed3"
```
---

## Key Hyperparameters (Examples)

| Flag | Default | Description |
|---|---|---|
| `--env-id` | `CartPole-v1` | Environment to train on |
| `--total-timesteps` | `500000` | Total training steps |
| `--num-envs` | `4` | Parallel environments (use 64 for MiniGrid) |
| `--num-steps` | `128` | Rollout length per environment per update |
| `--seed` | `1` | Random seed |
| `--track` | `False` | Enable W&B logging |
| `--capture-video` | `False` | Save mp4 videos of agent episodes |
| `--wandb-project-name` | `cleanRL` | W&B project name |
| `--wandb-entity` | `None` | W&B team name |
| `--run-name` | auto-generated | Name of the run in W&B |

---

## Monitoring Training

Open your [W&B dashboard](https://wandb.ai) in the browser while training runs. Key metrics to watch:

- **charts/episodic_return** — primary metric, should increase over time
- **charts/episodic_length** — episode duration, should decrease as agent learns
- **losses/policy_loss** — actor learning signal
- **losses/value_loss** — critic learning signal
- **losses/entropy** — exploration level

> **Tip:** In W&B, set the x-axis to `global_step` for all charts to see actual environment timesteps instead of logging steps.

---

## Dependency Management

> **Important:** Always use `uv` to manage dependencies — do not use `pip install` directly as it bypasses `pyproject.toml` and `uv.lock`, causing environment inconsistencies across the team.

```bash
uv add requests       # add a new dependency
uv remove requests    # remove a dependency
uv sync               # sync your environment with the lockfile
```
