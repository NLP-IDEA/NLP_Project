"""
合并 LoRA 到基础 Qwen 模型
"""
import os
from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

print("="*50)
print("合并 LoRA 到基础 Qwen 模型...")
print("="*50)

base_path = "/root/autodl-fs/Qwen2.5-diffusion/"
lora_path = "./qwen_lora_output/checkpoint-final/"
output_path = "/root/autodl-fs/qwen_merged_chat/"

print("1. 加载基础模型...")
base_model = AutoModelForMaskedLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    base_path,
    trust_remote_code=True
)

print("2. 加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("3. 合并中...")
merged_model = model.merge_and_unload()

print("4. 保存合并后的模型...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ 合并完成！模型保存在: {output_path}")
print(f"大小:")
os.system(f"du -sh {output_path}")
