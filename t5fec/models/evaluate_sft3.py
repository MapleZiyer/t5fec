import logging
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import set_seed
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # 设置随机种子
    set_seed(42)

    # 加载训练后的模型和分词器
    model_name = "../checkpoints/Llama-3.2-1B-Instruct4"  # 你训练后保存的模型路径
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 配置模型并移动到CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式
    model.config.pad_token_id = tokenizer.eos_token_id  # 设置pad token为eos token

    # 加载验证数据集
    dataset = load_dataset('json', data_files={'validation': '../data/sft.jsonl'})
    
    # 数据预处理函数
    def preprocess_function(examples):
        data_instance = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'\n\n"
        targets = f"<answer>{examples['original']}</answer>"

        # 使用tokenizer处理输入数据
        model_inputs = tokenizer(
            data_instance,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None  
        )
        
        labels = tokenizer(
            targets,
            max_length=4096,  # 修改为与输入相同的长度
            truncation=True,
            padding='max_length',
            return_tensors=None
        )

        # 手动转换为张量并移动到正确的设备
        device = model.device
        input_ids = torch.tensor(model_inputs["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.long, device=device)
        labels_tensor = torch.tensor(labels["input_ids"], dtype=torch.long, device=device)
        
        # 更新模型输入
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        model_inputs["labels"] = labels_tensor

        # 确保返回的是张量而不是列表
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor
        }

    # 处理验证数据集
    processed_dataset = dataset['validation'].map(
        preprocess_function,
        remove_columns=dataset['validation'].column_names,
        desc="Processing validation dataset",
        batched=True
    )

    # 推理并打印结果
    logger.info("*** Starting validation ***")
    
    # 为了避免内存问题，使用一个小批次来进行推理
    for idx, batch in enumerate(processed_dataset):
        # 由于已经在预处理时将张量移动到了正确的设备，这里不需要再次移动
        input_ids = batch["input_ids"].unsqueeze(0) if batch["input_ids"].dim() == 1 else batch["input_ids"]
        attention_mask = batch["attention_mask"].unsqueeze(0) if batch["attention_mask"].dim() == 1 else batch["attention_mask"]

        with torch.no_grad():  # 禁用梯度计算
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=256,  # 生成的最大长度
                num_beams=5,  # 使用束搜索
                no_repeat_ngram_size=2,  # 避免重复n-gram
                early_stopping=True  # 提前停止生成
            )

        # 解码并打印结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        original_text = tokenizer.decode(batch['labels'][0], skip_special_tokens=True)

        logger.info(f"Generated: {generated_text}")
        logger.info(f"Original: {original_text}")

    logger.info("*** Validation complete ***")

if __name__ == "__main__":
    main()
