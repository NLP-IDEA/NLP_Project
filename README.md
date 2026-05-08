# Can Machines Pass the Civil Service Exam?
## Complex Logical Inference with Fine-Tuned Diffusion Models

This project explores whether diffusion language models can handle complex Chinese logical reasoning tasks from the Civil Service Examination.  
We construct a reasoning dataset from Chinese Civil Service Examination questions and evaluate different diffusion-based models under fine-tuning and few-shot settings.

## Project Overview

Chinese logical reasoning is challenging because it requires models to understand abstract relations, compare options, and select the most logically consistent answer.  
In this project, we focus on Civil Service Examination-style multiple-choice questions and investigate whether diffusion language models can improve reasoning performance through task-specific adaptation.

The project mainly includes:

- Dataset construction from Civil Service Examination logical reasoning questions
- Fine-tuning a Qwen-0.5B diffusion model
- Few-shot evaluation of different diffusion-based models
- Comparison between fine-tuned small models and larger few-shot models
- Temperature control experiment for diffusion language model generation

## Models

We evaluate the following models:

- Fine-tuned Qwen-0.5B diffusion model
- Classical Qwen diffusion
- Classical LLaDA diffusion
- LLaDA instructed diffusion
- BERT diffusion

## Dataset

The dataset is built from Chinese Civil Service Examination logical reasoning questions from 2010 to 2020.

Each sample is formatted as a multiple-choice reasoning question:

```json
{
  "question": "汽油：汽车",
  "options": {
    "A": "食物：肠胃",
    "B": "水：喝",
    "C": "太阳：阳光",
    "D": "风：风筝"
  },
  "answer": "D"
}
```

The model is required to output only one option:

```text
A / B / C / D
```

## Few-shot Evaluation

For few-shot evaluation, we compare different models under:

- 1-shot
- 3-shot
- 5-shot

The prompt contains:

1. Task instruction
2. Several demonstration examples
3. A new test question
4. A strict output format rule

Example prompt format:

```text
下面是一些行政职业能力测试单项选择题示例。
请严格模仿示例的作答方式。
对每道新题，你必须且只能输出一个大写字母：A、B、C 或 D。

示例1
窑：陶瓷
A. 唯物主义：唯心主义
B. 整数：负整数
C. 青年：少年
D. 烤箱：面包
正确选项：D

现在开始答题。
汽油：汽车
A. 食物：肠胃
B. 水：喝
C. 太阳：阳光
D. 风：风筝

请只输出一个大写字母（A/B/C/D）：
```

## Evaluation Metrics

We use two main metrics:

- Accuracy
- F1 Score

Accuracy measures whether the predicted option is correct.  
F1 Score gives a more balanced view of model performance, especially when predictions are unevenly distributed.

## Main Results

The fine-tuned Qwen-0.5B diffusion model achieves better parameter efficiency than larger few-shot baselines.

| Model | F1 Score | Accuracy | Params | Acc/Params |
|---|---:|---:|---:|---:|
| Our Model (fine-tuned) | 0.1485 | 0.52 | 0.5B | 1.040 |
| Qwen diffusion (few-shot) | 0.1771 | 0.16 | 0.5B | 0.320 |
| LLaDA diffusion (few-shot) | 0.6165 | 0.48 | 8B | 0.060 |
| LLaDA instructed diffusion (few-shot) | 0.8791 | 0.88 | 8B | 0.110 |
| BERT diffusion (few-shot) | 0.0956 | 0.07 | 0.15B | 0.467 |

Few-shot results show that larger and instruction-tuned diffusion models benefit more from demonstrations, while weaker models may not improve consistently.

## Few-shot Results

| Few-shot setting | Model name | F1 Score | Accuracy |
|---|---|---:|---:|
| 1-shot | Classical QWen diffusion (few-shot) | 0.1169 | 0.1000 |
| 1-shot | Classical LLaDA diffusion (few-shot) | 0.5170 | 0.4000 |
| 1-shot | LLaDA instructed diffusion (few-shot) | 0.8686 | 0.8700 |
| 1-shot | BERT diffusion (few-shot) | 0.0810 | 0.0600 |
| 3-shot | Classical QWen diffusion (few-shot) | 0.1771 | 0.1600 |
| 3-shot | Classical LLaDA diffusion (few-shot) | 0.6165 | 0.4800 |
| 3-shot | LLaDA instructed diffusion (few-shot) | 0.8791 | 0.8800 |
| 3-shot | BERT diffusion (few-shot) | 0.0956 | 0.0700 |
| 5-shot | Classical QWen diffusion (few-shot) | 0.1466 | 0.1300 |
| 5-shot | Classical LLaDA diffusion (few-shot) | 0.7248 | 0.6100 |
| 5-shot | LLaDA instructed diffusion (few-shot) | 0.8809 | 0.8800 |
| 5-shot | BERT diffusion (few-shot) | 0.0000 | 0.0000 |

## Temperature Experiment

We also study the effect of temperature on the fine-tuned Qwen diffusion model.

| Temperature | Accuracy | F1 Score | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.0 | 50.00% | 0.5184 | 0.6397 | 0.5000 |
| 0.3 | 51.00% | 0.5455 | 0.6598 | 0.5100 |
| 0.5 | 52.00% | 0.5505 | 0.6557 | 0.5200 |
| 0.7 | 44.00% | 0.4528 | 0.5430 | 0.4400 |
| 1.0 | 36.00% | 0.4162 | 0.5235 | 0.3600 |

The best performance is achieved at temperature `0.5`, suggesting that moderate randomness is more suitable for this task.

## Project Structure

```text
.
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── processed/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── fewshot_eval.py
│   └── temperature_eval.py
│
├── models/
│   └── checkpoints/
│
├── results/
│   ├── fewshot_results.csv
│   ├── temperature_results.csv
│   └── figures/
│
├── report/
│   └── project_report.pdf
│
├── requirements.txt
└── README.md
```

Please adjust the folder names according to the actual project files.

## Installation

Create a new environment:

```bash
conda create -n dllm python=3.10
conda activate dllm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch with CUDA is needed, install the correct version according to your CUDA environment.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## How to Run

### 1. Fine-tune the model

```bash
python scripts/train.py \
  --train_file data/train.jsonl \
  --model_name_or_path Qwen-0.5B-diffusion \
  --output_dir models/checkpoints/qwen_diffusion_finetuned
```

### 2. Evaluate the fine-tuned model

```bash
python scripts/evaluate.py \
  --model_name_or_path models/checkpoints/qwen_diffusion_finetuned \
  --test_file data/test.jsonl
```

### 3. Run few-shot evaluation

```bash
python scripts/fewshot_eval.py \
  --train_file data/train.jsonl \
  --test_file data/test.jsonl \
  --shots 1 3 5
```

### 4. Run temperature experiment

```bash
python scripts/temperature_eval.py \
  --model_name_or_path models/checkpoints/qwen_diffusion_finetuned \
  --test_file data/test.jsonl \
  --temperatures 0.0 0.3 0.5 0.7 1.0
```

## Key Findings

- Chinese Civil Service Examination questions are difficult for diffusion language models.
- Task-specific fine-tuning improves the performance of small diffusion models.
- Few-shot prompting is not equally useful for all models.
- Larger and instruction-tuned diffusion models benefit more from demonstrations.
- The fine-tuned 0.5B model is efficient, but larger models still have a higher performance upper bound.
- A moderate temperature setting gives the best performance in our experiment.

## Limitations

This project has several limitations:

- The dataset size is relatively small.
- The task only focuses on multiple-choice logical reasoning.
- Some models are evaluated only in few-shot settings rather than full fine-tuning.
- The fine-tuned small model still has a clear performance gap compared with larger instruction-tuned models.

## Contributors

- Wang Han
- Jia Mu

## Acknowledgments

We thank the course instructor and teaching assistants for their guidance and feedback throughout this project. We also acknowledge the open-source diffusion language modeling frameworks and pretrained models used in our experiments.
