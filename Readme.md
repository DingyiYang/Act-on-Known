# KSTE
The datasets and codes from the paper:
[Are LLMs Smarter Than Chimpanzees? An Evaluation on Perspective Taking and Knowledge State Estimation](https://arxiv.org/pdf/2601.12410)

## Overview
Cognitive anthropology suggests that the distinction of human intelligence lies in the ability to infer other individuals’ knowledge states and
understand their intentions. In comparison, our closest animal relative, chimpanzees, lack the capacity to do so. With this paper, we aim to evaluate LLM performance in the area of **knowledge state tracking and estimation.** We design two tasks: 
- IKD task: If LLMs can detect when story characters, through their actions, demonstrate knowledge they should not possess. 
- KNP task: If LLMs can predict story characters’ next actions based on their own knowledge vs. objective truths they do not know. (shown in figure below)

Results reveal that most current state-of-the-art LLMs achieve near-random performance on both tasks, and are substantially inferior to humans. We argue future LLM research should place more weight on the abilities of knowledge estimation and intention understanding.

<div align="center">
<img src="./assets/Figure1.png" width="400" />
</div>



## Release :loudspeaker:
- **2026/03**: Our datasets and codes are released. 
- The complete dataset construction process is coming soon.

## Dataset
```
KSTE_data/
  ├── IKD/                      # Implausible knowledge detection
  │     ├── original/           # The stories without plot errors
  │     └── errors/             # The stories with plot errors
  └── IKD/                      # Knowledge-sensitive Next-action Prediction
```

## How to run
### Requirements
1. Create a Python 3.10 virtual environment
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Inference
1. Prompt templates are located at ```codes/prompt_templates/```

2. API Inference:
```bash
cd codes
python multi_process_infer.py \
    --prompts_dir ./prompts/ \
    --output_dir ./outputs/ \
    --model gpt-5-medium \
    --processes 10 \
```