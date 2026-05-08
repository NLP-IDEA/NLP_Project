import json
import random
import re
from dataclasses import dataclass

import torch
import dllm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


# =========================================================
# 路径配置
# =========================================================
BASE_DIR = "/root/dllm-main"

TRAIN_PATH = f"{BASE_DIR}/train.jsonl"
TEST_PATH = f"{BASE_DIR}/test.jsonl"

MODEL_LIST = [
    {
        "name": "Classical QWen diffusion(few-shot)",
        "type": "diffusion",
        "path": f"{BASE_DIR}/Qwen2.5-diffusion",
        "params": "0.5B",
    },
    {
        "name": "Classical LLADA diffusion(few-shot)",
        "type": "diffusion",
        "path": f"{BASE_DIR}/LLaDA-8B-Base",
        "params": "8B",
    },
    {
        "name": "LLADA instructed diffusion(few-shot)",
        "type": "diffusion",
        "path": f"{BASE_DIR}/LLaDA-8B-Instruct",
        "params": "8B",
    },
    {
        "name": "BERT diffusion(few-shot)",
        "type": "bert",
        "path": f"{BASE_DIR}/ModernBERT-base-chat-v0.1",
        "params": "0.15B",
    },
]

K_SHOT = 5
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 通用工具函数
# =========================================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_prompt_and_label(text):
    """
    必须按最后一个“答案：”切分，
    因为题干里本身可能有“请选择正确答案：”
    """
    if "答案：" in text:
        left, right = text.rsplit("答案：", 1)
        prompt = left.strip()
        label = right.strip()
    else:
        prompt = text.strip()
        label = ""
    return prompt, label


def normalize_label(label):
    label = label.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if ch in label:
            return ch
    return label


def normalize_prompt_for_mc(prompt: str) -> str:
    """
    不要破坏题干内容。
    “请选择正确答案：”属于题目本身，不能删除。
    “[MASK]”也先保留，保证问给模型的是完整原题。
    """
    prompt = prompt.strip()

    # 仅在极端情况下，去掉末尾孤立残留的“答案：”
    if prompt.endswith("答案："):
        prompt = prompt[:-3].rstrip()

    return prompt


def extract_option(text):
    """
    只允许返回 A/B/C/D/INVALID
    """
    if not text:
        return "INVALID"

    text = text.strip().upper()

    strict_patterns = [
        r"^\s*([ABCD])\s*$",
        r"^\s*正确选项[:：]?\s*([ABCD])\s*$",
        r"^\s*答案[:：]?\s*([ABCD])\s*$",
        r"^\s*选项[:：]?\s*([ABCD])\s*$",
    ]

    for p in strict_patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)

    matches = re.findall(r"\b([ABCD])\b", text)
    if matches:
        return matches[-1]

    return "INVALID"


def sample_fewshot_examples(train_data, k, seed=42):
    rng = random.Random(seed)
    if k > len(train_data):
        raise ValueError(f"K_SHOT={k} 大于训练集大小 {len(train_data)}")
    return rng.sample(train_data, k)


def build_fewshot_prefix(demos):
    """
    仍然保持选择题风格，但 few-shot 示例统一成：
    题目 + 选项 + 正确选项
    """
    parts = []
    parts.append("下面是一些行政职业能力测试单项选择题示例。")
    parts.append("请严格模仿示例的作答方式。")
    parts.append("对每道新题，你必须且只能输出一个大写字母：A、B、C 或 D。")
    parts.append("不要输出“答案：”，不要解释，不要重复题目，不要输出任何其他文字。")
    parts.append("")

    for i, item in enumerate(demos, start=1):
        prompt, label = split_prompt_and_label(item["text"])
        prompt = normalize_prompt_for_mc(prompt)
        label = normalize_label(label)

        parts.append(f"示例{i}")
        parts.append(prompt)
        parts.append(f"正确选项：{label}")
        parts.append("")

    parts.append("现在开始答题。")
    parts.append("再次提醒：你必须且只能输出一个大写字母 A、B、C 或 D。")
    parts.append("")
    return "\n".join(parts)


def build_eval_prompt(prefix, sample_prompt):
    """
    正式评测 prompt
    """
    sample_prompt = normalize_prompt_for_mc(sample_prompt)
    return (
        prefix
        + sample_prompt
        + "\n"
        + "请只输出一个大写字母（A/B/C/D）："
    )


def build_bert_prompt(prefix, sample_prompt, mask_token):
    sample_prompt = normalize_prompt_for_mc(sample_prompt)
    return (
        prefix
        + sample_prompt
        + "\n"
        + f"请只输出一个大写字母（A/B/C/D）： {mask_token}"
    )


def prepare_test_data(test_raw):
    test_data = []
    for item in test_raw:
        prompt, label = split_prompt_and_label(item["text"])
        label = normalize_label(label)
        test_data.append({
            "prompt": prompt,
            "label": label
        })
    return test_data


# =========================================================
# Diffusion 模型评测
# =========================================================
@dataclass
class ScriptArguments:
    model_name_or_path: str
    seed: int = RANDOM_SEED
    chat_template: bool = False
    visualize: bool = False


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 10
    block_size: int = 32
    temperature: float = 0.5
    remasking: str = "low_confidence"


def evaluate_diffusion_model(model_name, model_path, prefix, test_data):
    print(f"\n开始评测 diffusion 模型: {model_name}")
    print("=" * 70)

    script_args = ScriptArguments(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = SamplerConfig()

    predictions = []
    ground_truths = []

    for idx, sample in enumerate(tqdm(test_data, desc=model_name)):
        full_prompt = build_eval_prompt(prefix, sample["prompt"])

        inputs = tokenizer([full_prompt], add_special_tokens=False)["input_ids"]

        outputs = sampler.sample(
            inputs,
            sampler_config,
            return_dict=True,
        )

        response = dllm.utils.sample_trim(
            tokenizer,
            outputs.sequences.tolist(),
            inputs
        )[0]

        pred = extract_option(response)

        predictions.append(pred)
        ground_truths.append(sample["label"])

        if idx < 3:
            print(f"\n样本 {idx + 1}")
            print(f"Raw output: {response}")
            print(f"Pred: {pred} | Gold: {sample['label']}")

    invalid_count = sum(p == "INVALID" for p in predictions)
    print(f"非法预测数: {invalid_count}/{len(predictions)}")

    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average="weighted", zero_division=0)

    return {
        "model_name": model_name,
        "f1": f1,
        "accuracy": acc,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


# =========================================================
# BERT / ModernBERT 单独评测
# =========================================================
def predict_with_bert_maskedlm(model, tokenizer, full_prompt):
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("这个 BERT/ModernBERT tokenizer 没有 mask_token_id。")

    input_ids = inputs["input_ids"]
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)

    if len(mask_positions) == 0:
        raise ValueError("输入里没有 [MASK] 位置，无法预测。")

    mask_pos = mask_positions[0, 1].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_pos, :]
        pred_id = torch.argmax(logits).item()

    pred_text = tokenizer.decode([pred_id]).strip()
    return pred_text


def evaluate_bert_model(model_name, model_path, prefix, test_data):
    print(f"\n开始评测 BERT 模型: {model_name}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    if tokenizer.mask_token is None:
        raise ValueError(f"{model_name} 的 tokenizer 没有 mask token。")

    predictions = []
    ground_truths = []

    for idx, sample in enumerate(tqdm(test_data, desc=model_name)):
        full_prompt = build_bert_prompt(prefix, sample["prompt"], tokenizer.mask_token)
        response = predict_with_bert_maskedlm(model, tokenizer, full_prompt)
        pred = extract_option(response)

        predictions.append(pred)
        ground_truths.append(sample["label"])

        if idx < 3:
            print(f"\n样本 {idx + 1}")
            print(f"Raw output: {response}")
            print(f"Pred: {pred} | Gold: {sample['label']}")

    invalid_count = sum(p == "INVALID" for p in predictions)
    print(f"非法预测数: {invalid_count}/{len(predictions)}")

    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average="weighted", zero_division=0)

    return {
        "model_name": model_name,
        "f1": f1,
        "accuracy": acc,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


# =========================================================
# 主程序
# =========================================================
def main():
    print("=" * 70)
    print("加载数据")
    print("=" * 70)

    train_raw = load_jsonl(TRAIN_PATH)
    test_raw = load_jsonl(TEST_PATH)

    print(f"训练集大小: {len(train_raw)}")
    print(f"测试集大小: {len(test_raw)}")

    demos = sample_fewshot_examples(train_raw, K_SHOT, seed=RANDOM_SEED)
    prefix = build_fewshot_prefix(demos)
    test_data = prepare_test_data(test_raw)

    print("\nFew-shot 前缀预览：")
    print("-" * 70)
    print(prefix[:1200])
    print("\n" + "-" * 70)

    results = []

    for model_info in MODEL_LIST:
        model_name = model_info["name"]
        model_type = model_info["type"]
        model_path = model_info["path"]

        try:
            if model_type == "diffusion":
                result = evaluate_diffusion_model(
                    model_name=model_name,
                    model_path=model_path,
                    prefix=prefix,
                    test_data=test_data,
                )
            elif model_type == "bert":
                result = evaluate_bert_model(
                    model_name=model_name,
                    model_path=model_path,
                    prefix=prefix,
                    test_data=test_data,
                )
            else:
                raise ValueError(f"未知模型类型: {model_type}")

            result["params"] = model_info["params"]
            results.append(result)

        except Exception as e:
            print(f"\n❌ 模型 {model_name} 运行失败: {e}")
            results.append({
                "model_name": model_name,
                "f1": None,
                "accuracy": None,
                "params": model_info["params"],
            })

    print("\n" + "=" * 90)
    print("最终结果表")
    print("=" * 90)
    print(f"{'Model name':40} {'F1 Score':12} {'Accuracy':12} {'Parameters'}")
    print("-" * 90)

    for r in results:
        f1_str = f"{r['f1']:.4f}" if r["f1"] is not None else "FAILED"
        acc_str = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "FAILED"
        print(f"{r['model_name'][:38]:40} {f1_str:12} {acc_str:12} {r['params']}")

    print("\n表里建议写成：")
    print("F1 Score (weighted)")
    print("因为这份脚本算的是 weighted F1。")


if __name__ == "__main__":
    main()