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

from trl import GRPOTrainer, get_peft_config
import wandb

from sentence_transformers import SentenceTransformer

from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO  # 设置基础日志级别为INFO
)
logger = logging.getLogger(__name__)

# 确保其他日志也设置正确的级别
datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

@dataclass
class GRPOScriptArguments:
    """Script arguments for the GRPO training script."""
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'tag_count'"
        },
    )

# 定义包装器，移除不支持的 logits_to_keep 参数
class T5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config  # 添加config属性
        self.warnings_issued = {"estimate_tokens": False}  # 添加warnings_issued属性
        self._model_tags = set()

    def forward(self, **kwargs):
        if "logits_to_keep" in kwargs:
            kwargs.pop("logits_to_keep")
        return self.model(**kwargs)

    def add_model_tags(self, tags):
        """添加模型标签"""
        self._model_tags.update(tags)

    def get_model_tags(self):
        """获取模型标签"""
        return list(self._model_tags)

def main():
    checkpoint_dir = "../checkpoints/long-t5-tglobal-large-sft"
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/long-t5-tglobal-large-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="long-t5-tglobal-large-run",
    )
    # 添加额外参数
    setattr(training_args, 'reward_weights', [1.0])
    setattr(training_args, 'reward_scale', 1.0)
    setattr(training_args, 'model_init_kwargs', {})
    setattr(training_args, 'max_prompt_length', 4096)
    setattr(training_args, 'max_completion_length', 256)
    setattr(training_args, 'num_generations', 4)
    setattr(training_args, 'use_vllm', False)
    setattr(training_args, 'beta', 0.1)
    setattr(training_args, 'log_completions', False)
    setattr(training_args, 'temperature', 0.7)
    setattr(training_args, 'sync_ref_model', True)

    # 设置随机种子
    set_seed(42)

    # 检查最新的 checkpoint
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # 设置日志级别
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 定义预训练模型名称
    model_pretrained_name = "google/long-t5-tglobal-large"
    # 加载模型实例
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_pretrained_name)

    # 使用包装器包装模型，避免 logits_to_keep 参数问题
    model = T5Wrapper(model)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_pretrained_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 设置聊天模板（使用 set_chat_template，如果你的 transformers 版本支持此方法）
    try:
        tokenizer.set_chat_template("content")
    except AttributeError:
        # 如果不支持，则直接设置 chat_template 属性
        tokenizer.chat_template = "content"

    # 加载数据集并过滤
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/train.json'}
    )
    dataset = dataset.filter(lambda x: x['label'] == 'refutes')

    # 数据预处理函数
    def preprocess_function(examples):
        prompt_template = """You are an expert in correcting erroneous sentences. Based on the following evidence, identify and correct errors in the original statement. Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.
    
        Evidence: {evidence}
    
        Original statement: {original_statement}
    
        Corrected statement: """
        # 检查输入有效性
        if not examples['evidence'] or not examples['claim']:
            return {}
        formatted_prompt = prompt_template.format(evidence=examples['evidence'], original_statement=examples['claim'])
    
        model_inputs = tokenizer(
            formatted_prompt,
            max_length=4096,
            truncation='only_first',
            padding='max_length',
            return_tensors="pt"
        )
    
        # 这里将 prompt 字段设为原始格式的字符串
        model_inputs["prompt"] = formatted_prompt
    
        # 确保末尾是 EOS
        eos_token_id = tokenizer.eos_token_id
        if model_inputs["input_ids"][:, -1].item() != eos_token_id:
            model_inputs["input_ids"][:, -1] = eos_token_id
    
        return model_inputs

    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=[col for col in dataset['train'].column_names if col not in ['id', 'evidence', 'claim']],
        desc="Processing dataset",
        keep_in_memory=True
    )

    # 设置模型参数相关（这里 torch_dtype 没有实际用处，可选）
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    training_args.model_init_kwargs = None

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()
    
    # 定义奖励函数，并初始化 sentence transformer 模型
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    def accuracy_reward(outputs, batch):
        rewards = []
        for output, sample in zip(outputs, batch):
            output_embedding = similarity_model.encode(output, convert_to_tensor=True)
            target_embedding = similarity_model.encode(sample['claim'], convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))
            print(f"\nOutput: {output}\nSimilarity: {similarity}\n")
            if similarity < 0.7:
                rewards.append(0.0)
                continue
            programs = program_generator.generate_program(output)
            sample_data = [{
                "idx": 0,
                "id": sample.get('id', None),
                "claim": output,
                "gold": "",
                "predicted_programs": programs,
                "evidence": sample.get('evidence', None)
            }]
            prediction = program_executor.execute_on_dataset(sample_data)
            print(f"\nPrograms: {programs}\nPrediction: {prediction}\n")
            rewards.append(1.0 if prediction else 0.0)
        return torch.tensor(rewards)

    reward_funcs = [accuracy_reward]

    # 初始化 GRPO 训练器
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    logger.info("*** Starting training ***")
    wandb.init(project="long-t5-tglobal-large-grpo", name=training_args.run_name)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    
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
