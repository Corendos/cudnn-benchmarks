### cuDNN benchmarks

### Installation

First, install [uv](https://github.com/astral-sh/uv) to avoid package version problems.

Then:
```sh
# Create virtualenv
uv venv venv
source venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Override cudnn backend version
uv pip install nvidia-cudnn-cu12==9.8.0.87
```

### Running

```sh
python main.py
```

This will run 3 successive benchmarks and report the timings.