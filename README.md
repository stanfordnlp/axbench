<h1 align="center"> <p>pyreax<sub> by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></p></h1>
<h3 align="center">
    <p>Understand and control your LMs with representation abstraction (ReAX)</p>
    <a href=""><strong>Read our paper »</strong></a></a>
</h3>

## Creating your validated abstractions in minutes.

Make sure you have your OpenAI API key set:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

Create your abstraction:
```python
import pyreax

ax = pyreax.create(
    model_name="google/gemma-2-2b", layer=20,
    concept="terms related to the Golden Gate Bridge")

# latent activations
ax.latent("I want to visit the Golden Gate Bridge this weekend!")

# steering
ax.steer("Can you recommend me places to visit over the weekend?")

# upload abstraction to hf
ax.upload(hf_repo="ggb_abstraction")

# download abstraction
ax = pyreax.download(hf_repo="ggb_abstraction")
```

## ReAX Workflow

### generate
Generate training dataset for representation abstractions:
```bash
python scripts/generate.py --config demo/sweep/generate.yaml
```

### train
Train and save representation abstractions with generated datasets:
```bash
python scripts/train.py --config demo/sweep/train.yaml
```

### inference

#### latent
Inference with latent activations with representation abstractions:
```bash
python scripts/inference.py --config demo/sweep/inference.yaml --mode latent
```

#### steering (not implemented yet)
Inference with model steering with representation abstractions:
```bash
python scripts/inference.py --config demo/sweep/inference.yaml --mode steer
```

### evaluate

#### latent
To evaluate inference results for latent activations:
```bash
python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode latent
```

#### steering (not implemented yet)
To evaluate inference results for steering:
```bash
python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode steer
```

### (optional) plot

We provide a plot notebook (`scripts/plots.ipynb`) for generating various figures.

