# MMPlanner

Multimodal instruction planner that breaks down instructions into step-by-step plans with generated images.

## Features

- Generate step-by-step text plans from natural language instructions
- Create illustrative images for each step using Stable Diffusion
- Automatically select the best image from multiple candidates using OpenAI

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast package management.

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project dependencies
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Quick install (alternative)

```bash
# Install dependencies directly without creating venv first
uv pip install openai requests Pillow
```

## Configuration

1. Set your OpenAI API key in `main.py`:
   ```python
   planner = InstructionPlanner(api_key="YOUR_OPENAI_API_KEY")
   ```

2. Ensure Stable Diffusion API is running locally at `http://localhost:7860/sdapi/v1/txt2img`
   - Or modify the `sd_api_url` parameter to point to your Stable Diffusion endpoint

## Usage

```python
from instruction_planner import InstructionPlanner

planner = InstructionPlanner(api_key="YOUR_OPENAI_API_KEY")
text_plan = planner.generate_text_plan("how to cook an egg")
print(text_plan)
```

Run the example:

```bash
python main.py
```

## Project Structure

```
mmplanner/
├── stable_diffusion_generator.py  # Handles image generation via Stable Diffusion API
├── step_image_selector.py         # Generates and selects best images for steps
├── instruction_planner.py          # Main orchestration logic
├── main.py                         # Entry point and usage example
└── pyproject.toml                  # Project dependencies and configuration
```

## Dependencies

- `openai` - For GPT-based text generation and image selection
- `requests` - For HTTP communication with Stable Diffusion API
- `Pillow` - For image processing and manipulation

## License

MIT

