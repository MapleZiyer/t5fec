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

from trl import GRPOTrainer  # 原始 GRPOTrainer
import wandb

from sentence_transformers import SentenceTransformer

from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)

datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

@dataclass
class GRPOScriptArguments:
    """GRPO 训练参数"""
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={"help": "奖励函数列表。可选: 'accuracy', 'format', 'tag_count'"}
    )

# 定义包装器，移除不支持的 logits_to_keep 参数（可选）
class T5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, **kwargs):
        if "logits_to_keep" in kwargs:
            kwargs.pop("logits_to_keep")
        return self.model(**kwargs)

# 定义 GRPOTrainer 的子类，用猴子补丁方式避免加载 ref_model
from trl.trainer.grpo_trainer import GRPOTrainer as BaseGRPOTrainer

class GRPOSeq2SeqTrainer(BaseGRPOTrainer):
    def __init__(self, model, reward_funcs, args, train_dataset, processing_class):
        # 临时替换 AutoModelForSeq2SeqLM.from_pretrained，避免加载参考模型
        original_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained
        AutoModelForSeq2LM_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained  # 保存备用
        try:
            # 替换为 lambda，直接返回传入的 model 实例
            AutoModelForSeq2SeqLM.from_pretrained = lambda *a, **kw: model
            # 调用父类 __init__，这样 ref_model 就会被设为 model
            super().__init__(model, reward_funcs, args, train_dataset, processing_class)
        finally:
            # 恢复原来的 from_pretrained 函数
            AutoModelForSeq2SeqLM.from_pretrained = original_from_pretrained
        # 强制确保参考模型为当前模型
        self.ref_model = model

def main():
    checkpoint_dir = "../checkpoints/long-t5-tglobal-large-sft"
    # 训练参数设置
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

    # 此处暂不使用 checkpoint
    last_checkpoint = None
    # if os.path.isdir(checkpoint_dir):
    #     last_checkpoint = get_last_checkpoint(checkpoint_dir)
    # if last_checkpoint is not None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # 设置日志级别
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 定义预训练模型名称（LongT5 encoder-decoder）
    model_name = "google/long-t5-tglobal-large"
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype, "use_cache": False}
    
    # 加载模型实例
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    model = T5Wrapper(model)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载数据集并过滤
    dataset = load_dataset("json", data_files={"train": "../data/train.json"})
    dataset = dataset.filter(lambda x: x["label"] == "refutes")

    def preprocess_function(examples):
        prompt_template = """You are an expert in correcting erroneous sentences. Based on the following evidence, identify and correct errors in the original statement. Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.

        Evidence: {evidence}
        Original statement: {original_statement}
        Corrected statement: """
        if not examples['evidence'] or not examples['claim']:
            return {}
        formatted_prompt = prompt_template.format(evidence=examples['evidence'], original_statement=examples['claim'])
    
        model_inputs = tokenizer(
            formatted_prompt,
            max_length=4096,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt"
        )
        # 将 prompt 存为字符串
        model_inputs["prompt"] = formatted_prompt

        # 强制保证输入序列末尾为 EOS
        eos_token_id = tokenizer.eos_token_id
        if model_inputs["input_ids"][:, -1].item() != eos_token_id:
            model_inputs["input_ids"][:, -1] = eos_token_id

        return model_inputs

    processed_dataset = dataset["train"].map(
        preprocess_function,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["id", "evidence", "claim"]],
        desc="Processing dataset",
        keep_in_memory=True
    )

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()

    # 定义奖励函数
    similarity_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    def accuracy_reward(outputs, batch):
        rewards = []
        for output, sample in zip(outputs, batch):
            output_embedding = similarity_model.encode(output, convert_to_tensor=True)
            target_embedding = similarity_model.encode(sample["claim"], convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))
            print(f"\nOutput: {output}\nSimilarity: {similarity}\n")
            if similarity < 0.7:
                rewards.append(0.0)
                continue
            programs = program_generator.generate_program(output)
            sample_data = [{
                "idx": 0,
                "id": sample.get("id"),
                "claim": output,
                "gold": "",
                "predicted_programs": programs,
                "evidence": sample.get("evidence")
            }]
            prediction = program_executor.execute_on_dataset(sample_data)
            print(f"\nPrograms: {programs}\nPrediction: {prediction}\n")
            rewards.append(1.0 if prediction else 0.0)
        return torch.tensor(rewards)

    reward_funcs = [accuracy_reward]

    # 使用自定义的 GRPOSeq2SeqTrainer 初始化训练器
    trainer = GRPOSeq2SeqTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    logger.info("*** Starting training ***")
    wandb.init(project="long-t5-tglobal-large-grpo", name=training_args.run_name)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    tokenizer.save_pretrained(training_args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
