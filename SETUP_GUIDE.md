# CSO Setup Guide

Quick setup guide for Critical Step Optimization (CSO).

## Prerequisites

- Python 3.10+
- pip

## Installation

### 1. Clone and Navigate

```bash
git clone https://github.com/your-org/CSO.git
cd CSO
```

### 2. Install Dependencies

This project is based on [Cognitive Kernel-Pro](https://github.com/Tencent/CognitiveKernel-Pro) but requires local installation due to CSO modifications:

```bash
# Install core dependencies
pip install playwright anthropic openai azure-identity flask flask-cors

# Install Playwright browser
playwright install chromium

# Install additional packages as needed
pip install numpy pandas tqdm rich
```

### 3. Set Python Path

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/CSO"' >> ~/.bashrc
source ~/.bashrc
```

## API Configuration

### Option 1: Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"
export OPENAI_API_TYPE="azure_ai"
```

### Option 2: OpenAI

```bash
export OPENAI_API_KEY="your_openai_key"
export LLM_URL="https://api.openai.com/v1/chat/completions"
```

### Option 3: Claude (Anthropic)

```bash
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Option 4: Custom Endpoint

```bash
export LLM_URL="http://your-custom-endpoint:8081/v1/chat/completions"
export VLM_URL="http://your-vision-endpoint:8081/v1/chat/completions"
```

## Quick Start

### 1. Run Baseline Agent

```bash
cd scripts/inference
bash run_gaia_baseline.sh
```

### 2. Resample with PRM

```bash
bash run_resample_prm.sh
```

### 3. Generate Verified CSO Data

```bash
cd ../data_processing
bash run_verify_and_generate_dpo.sh
```

## Verification

Test your installation:

```python
# test_import.py
import sys
sys.path.insert(0, '.')

from system.ckv3.agents.session import AgentSession
from system.ckv3.ck_main.agent import CKAgent

print("✓ All imports successful!")
```

```bash
python test_import.py
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Should include your CSO directory
export PYTHONPATH="/path/to/CSO:${PYTHONPATH}"
```

### Playwright Issues

```bash
# Reinstall Playwright
pip uninstall playwright
pip install playwright
playwright install chromium
```

### API Key Issues

```bash
# Verify keys are set
echo $AZURE_OPENAI_API_KEY
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

## Directory Structure

```
CSO/
├── system/ckv3/          # Core CK system
├── scripts/
│   ├── inference/        # Agent inference scripts
│   └── data_processing/  # DPO data generation
└── assets/               # Documentation assets
```

## Next Steps

- See [README.md](README.md) for detailed usage
- Check [scripts/inference/](scripts/inference/) for example scripts
- Review [scripts/data_processing/](scripts/data_processing/) for data pipeline

## Support

For issues related to:
- **CSO**: Open an issue in this repository
- **Cognitive Kernel-Pro**: Check the [original CK-Pro repo](https://github.com/Tencent/CognitiveKernel-Pro)

---

**Note**: This project requires local installation even if you're familiar with CK-Pro, as we've made modifications for CSO.
