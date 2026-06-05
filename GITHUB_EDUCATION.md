# GitHub Education Features for LightFlight

This guide explains how to use GitHub Education features to accelerate your ternary-weight neural network research.

---

## 1. GitHub Actions Workflows

### Current Workflows

#### **python-checks.yml**
Runs on every push/PR. Enforces:
- **Black formatting** — code style consistency
- **isort import ordering** — clean imports
- **flake8 linting** — PEP 8 compliance
- **mypy type checking** — optional but helpful for catching bugs
- **pytest coverage** — unit test execution

**To run locally before pushing:**
```bash
black train_baseline.py eval_fp32.py compare_metrics.py
isort train_baseline.py eval_fp32.py compare_metrics.py
flake8 train_baseline.py eval_fp32.py --max-line-length=100
mypy train_baseline.py --ignore-missing-imports
pytest tests/ -v
```

#### **model-export-verify.yml**
Automatically verifies model loading when you create export scripts (Phase 2).

Triggers on changes to:
- `export_ternary.py`
- `verify_export.py`
- `train_baseline.py`

---

## 2. GitHub Codespaces

### Quick Start

1. Go to your repository on GitHub
2. Click **Code → Codespaces → Create codespace on main**
3. Wait ~2 min for environment setup
4. VS Code opens in browser with full Python environment

### What's Preconfigured

- Ubuntu 20.04 with Python 3.8
- All `requirements.txt` dependencies installed
- VS Code extensions: Pylance, Debugger, Makefile tools
- Black formatter on save
- Flake8 linting enabled

### When to Use Codespaces

**Perfect for:**
- Debugging from school/coffee shop
- Running analysis scripts
- Writing export/verification code (Phase 2)
- Paper writing + code alongside

**Not ideal for:**
- Training (Gazebo simulation is local-only)
- Long-running processes (limited compute hours/month)

### Limitations

- 60 hours/month free compute for students
- No GPU (not applicable for CPU-bound Gazebo anyway)
- Best for 1-2 hour editing sessions, not 15-hour training runs

---

## 3. Microsoft Azure Credits

### Your Strongest Tool for Phase 2

Azure Student Pack typically includes:
- **$100 / month free compute credits** × 12 months = $1200 total
- Expires at graduation (check your pack email)

### Use Case: TTQ Lambda Sweep

**Problem:** Testing lambda ∈ {0.01, 0.1, 0.5, 1.0} sequentially:
```
lambda=0.01: wait 15 hours
lambda=0.1:  wait 15 hours  
lambda=0.5:  wait 15 hours
lambda=1.0:  wait 15 hours
Total: 2.5 days
```

**Solution with Azure:**
- Spin up 4 identical NC6 VMs ($0.90/hr each)
- Run all lambdas **in parallel**
- Total: 15 hours wall-clock (vs 60 hours sequential)
- Cost: 4 VMs × 15 hr × $0.90 = **$54** (well within monthly credits)

### Getting Started

1. Activate your Student Pack: https://azure.microsoft.com/en-us/free/students/
2. Create Azure ML Workspace
3. Use `train_baseline.py` as compute target
4. Submit parallel sweep job

**Azure ML Example:**
```python
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.environment import Environment

ws = Workspace.from_config()
env = Environment.from_pip_requirements("ppo-env", "requirements.txt")

for lambda_val in [0.01, 0.1, 0.5, 1.0]:
    config = ScriptRunConfig(
        source_directory=".",
        script="train_ternary.py",
        arguments=["--lambda", str(lambda_val)],
        environment=env,
        compute_target="gpu-cluster"
    )
    Experiment(ws, "ternary-sweep").submit(config)
```

---

## 4. Repository Privacy Settings

### Current Recommendation: PUBLIC with LICENSE

**Reasoning:**
1. Research transparency
2. Easier sharing with collaborators
3. Can still keep experimental branches private

**Structure:**
```
main (public)
├── Phase 1: FP32 baseline ✓
├── Phase 2: Ternary quantization (when complete)
└── paper/ (when accepted)

develop/ (private branch — optional)
├── Experimental ideas
├── Failed approaches
└── Unpublished results
```

### Settings

Go to **Settings → Visibility** and set:
- ✓ Public repo
- ✓ Add MIT or Apache 2.0 LICENSE
- Add CITATION.cff for paper citation

### When to Go Private

- **Pre-submission:** 1 month before paper deadline → private
- **Post-acceptance:** Switch back to public before publication

---

## 5. GitHub Advanced Security (Free for Public Repos)

### Automatically Enabled Features

1. **Dependabot**: Scans dependencies for vulnerabilities
   - Automatically creates PRs for security updates
   - Already active on your repo

2. **Code Scanning**: Detects issues before merge
   - Enable in **Settings → Security → Code scanning**
   - Free for public repos

### How It Protects Your Research

- Prevents accidentally committing API keys/credentials
- Warns about vulnerable TensorFlow/PyTorch versions
- Catches supply-chain issues before deployment

---

## 6. Parallel Hyperparameter Sweeps (Azure Example)

### Setup for TTQ Lambda Sweep

**Create `train_ternary.py`:**
```python
import argparse
from train_baseline import *

parser = argparse.ArgumentParser()
parser.add_argument("--lambda", type=float, default=0.1)
args = parser.parse_args()

# Load FP32 baseline
base_model = PPO.load("models/fp32_baseline/best_model")

# Add ternary loss with lambda
# ... training code ...
```

**Submit to Azure:**
```bash
az ml job create --file sweep.yml
```

**sweep.yml:**
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: .
command: python train_ternary.py --lambda ${{search_space.lambda}}
environment: azureml:ppo-env@latest
compute: gpu-cluster
search_space:
  lambda: choice(0.01, 0.1, 0.5, 1.0)
```

---

## Quick Checklist

- [ ] Verify workflows run on next push
- [ ] Try spinning up a Codespace (free 1-2hr session)
- [ ] Apply for Azure Student Pack if not done
- [ ] Set repo to Public + add LICENSE
- [ ] Enable Dependabot alerts (Settings → Security)

---

## Reference: Student Pack Contents

Check your student email for GitHub Pack. Typically includes:

| Tool | Value | Use Case |
|------|-------|----------|
| GitHub Copilot | ~$120/yr | Already using |
| Codespaces | 60 hrs/mo | Debugging & scripts |
| Advanced Security | Free | Dependency scanning |
| Azure Credits | $100/mo × 12 | **Hyperparameter sweeps** |
| JetBrains IDE | $165/yr | Optional IDE |
| Namecheap domain | $9/yr | Not needed |

---

## Next Steps

1. **Phase 1 completion** → evaluate Run 11 results
2. **Phase 2 prep** → set up Azure for lambda sweep
3. **Export pipeline** → use model-export-verify workflow
4. **Paper prep** → use Codespaces for writing + analysis
