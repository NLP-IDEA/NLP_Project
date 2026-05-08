import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers
import dllm
from datasets import load_dataset  # 添加这一行

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "/root/autodl-fs/LLaDA-8B-Base/"  # 改为本地路径


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "train.jsonl"  # 改为本地文件
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = "./my_output"
    group_by_length: bool = True
    num_train_epochs: float = 10
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset (改为使用本地数据集) --------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        # 原来的: dataset = dllm.data.load_sft_dataset(...)
        # 改为:
        full_dataset = load_dataset('json', data_files='train.jsonl', split='train')
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        dataset = {
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        }
        
        # 数据预处理
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset["train"] = dataset["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=data_args.num_proc,
            desc="预处理训练集"
        )
        dataset["test"] = dataset["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["test"].column_names,
            num_proc=data_args.num_proc,
            desc="预处理验证集"
        )
        
        # 设置数据格式
        dataset["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataset["test"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("开始训练...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,
                ),
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )
    print(f"✅ 训练完成！模型保存在: {training_args.output_dir}")


if __name__ == "__main__":
    train()
