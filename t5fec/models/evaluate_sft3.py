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
    # 预处理时确保数据格式正确
    def preprocess_function(examples):
        data_instance = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'\n\n"
        targets = f"<answer>{examples['original']}</answer>"

        model_inputs = tokenizer(
            data_instance,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None  # 这里改成 None，确保返回 list
        )

        labels = tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )

        return {
            "input_ids": model_inputs["input_ids"],  
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }


    # 处理数据集
    processed_dataset = dataset['validation'].map(
        preprocess_function,
        remove_columns=dataset['validation'].column_names,
        desc="Processing validation dataset",
        batched=True  # 让 map() 处理 batch，防止数据被转换成 list
    )

    # 推理时确保 `input_ids` 是 tensor
    for idx, batch in enumerate(processed_dataset):
        inputs = {
            key: torch.tensor(value, dtype=torch.long).to(model.device)  
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=256,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )


        # 解码并打印结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        original_text = tokenizer.decode(batch['labels'][0], skip_special_tokens=True)

        logger.info(f"Generated: {generated_text}")
        logger.info(f"Original: {original_text}")

    logger.info("*** Validation complete ***")

if __name__ == "__main__":
    main()
