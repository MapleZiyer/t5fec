import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trl import GRPOTrainer
import wandb

from sentence_transformers import SentenceTransformer

from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution

# 设定日志格式
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 设置日志级别
datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

@dataclass
class GRPOScriptArguments:
    """GRPO 训练参数"""
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "奖励函数列表。可选: 'accuracy', 'format', 'tag_count'"
        },
    )

def main():
    checkpoint_dir = "../checkpoints/long-t5-tglobal-large-sft"
    
    # 训练参数
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/long-t5-tglobal-large-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="long-t5-tglobal-large-grpo-run",
        deepspeed="../configs/ds_config_zero3.json",
        local_rank=-1,
        ddp_find_unused_parameters=False,
        fp16=False,
    )

    # 额外 GRPO 参数
    setattr(training_args, 'reward_weights', [1.0])
    setattr(training_args, 'reward_scale', 1.0)
    setattr(training_args, 'model_init_kwargs', None)
    setattr(training_args, 'max_prompt_length', 4096)
    setattr(training_args, 'max_completion_length', 256)
    setattr(training_args, 'num_generations', 2)
    setattr(training_args, 'use_vllm', False)
    setattr(training_args, 'beta', 0.1)
    setattr(training_args, 'log_completions', False)
    setattr(training_args, 'temperature', 0.7)
    setattr(training_args, 'sync_ref_model', True)

    set_seed(42)

    # 检查是否有 checkpoint
    last_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    # 加载模型
    model_name = "google/long-t5-tglobal-large"
    model_kwargs = {"torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float32, "use_cache": False}
    
    if last_checkpoint:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint, **model_kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载数据集
    dataset = load_dataset("json", data_files={"train": "../data/train.json"})
    dataset = dataset.filter(lambda x: x["label"] == "refutes")

    def preprocess_function(examples):
        prompt = f"""You are an expert in correcting erroneous sentences. 
        Based on the following evidence, identify and correct errors in the original statement. 
        Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.
    
        Evidence: {examples['evidence']}
        Original statement: {examples['claim']}
        Corrected statement: """

        inputs = tokenizer(prompt, max_length=4096, truncation=True, padding="max_length", return_tensors=None)
        labels = tokenizer(examples["claim"], max_length=256, truncation=True, padding="max_length", return_tensors=None)

        # 处理 label 并右移（符合 T5 训练）
        labels["input_ids"] = [
            (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
        ]

        inputs["labels"] = labels["input_ids"]
        return inputs

    processed_dataset = dataset["train"].map(preprocess_function, remove_columns=["id", "evidence", "claim"], desc="Processing dataset", keep_in_memory=True)

    # 事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()

    # 奖励函数
    similarity_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    def accuracy_reward(outputs, batch):
        rewards = []
        for output, sample in zip(outputs, batch):
            output_embedding = similarity_model.encode(output, convert_to_tensor=True)
            target_embedding = similarity_model.encode(sample["claim"], convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))

            if similarity < 0.7:
                rewards.append(0.0)
                continue

            programs = program_generator.generate_program(output)
            sample_data = [{"idx": 0, "id": sample.get("id"), "claim": output, "gold": "", "predicted_programs": programs, "evidence": sample.get("evidence")}]
            prediction = program_executor.execute_on_dataset(sample_data)

            rewards.append(1.0 if prediction else 0.0)

        return torch.tensor(rewards)

    reward_funcs = [accuracy_reward]

    # GRPO 训练器
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
    )

    # 开始训练
    logger.info("*** Starting training ***")
    wandb.init(project="long-t5-tglobal-large-grpo", name=training_args.run_name)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型和 tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
