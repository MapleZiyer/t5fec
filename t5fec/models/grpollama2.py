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
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load

from rouge_score import rouge_scorer, scoring

from trl import GRPOTrainer, get_peft_config
import wandb

from sentence_transformers import SentenceTransformer

from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution
# 移除LoRA相关导入

count = 0

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

def main():
    checkpoint_dir="../checkpoints/Llama-3.2-1B-Instruct333"
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 适合单GPU的batch size
        gradient_accumulation_steps=8,  # 调整梯度累积步数
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="llama-3.2-1b-instruct-grpo-run",
    )
    # 添加reward_weights参数
    setattr(training_args, 'reward_weights', [1.0])
    setattr(training_args, 'reward_scale', 1.0)
    # 添加 model_init_kwargs 参数
    setattr(training_args, 'model_init_kwargs', None)
    # 添加 max_prompt_length 参数
    setattr(training_args, 'max_prompt_length', 4096)
    # 添加 max_completion_length 参数
    setattr(training_args, 'max_completion_length', 1024)
    # 添加 num_generations 参数
    setattr(training_args, 'num_generations', 2)  # 将生成数量设置为1
    # 添加 temperature 参数
    setattr(training_args, 'temperature', 0.7)  # 降低temperature以减少随机性
    # 添加 use_vllm 参数
    setattr(training_args, 'use_vllm', False)
    # 添加 beta 参数
    setattr(training_args, 'beta', 0.1)
    # 添加 log_completions 参数
    setattr(training_args, 'log_completions', False)
    # 添加 temperature 参数
    setattr(training_args, 'temperature', 0.7)
    # 添加 sync_ref_model 参数
    setattr(training_args, 'sync_ref_model', True)
    setattr(training_args, 'use_cache', False)
    # 添加 ref_model_sync_steps 参数
    setattr(training_args, 'ref_model_sync_steps', 1)
    # 添加 ref_model_mixup_alpha 参数
    setattr(training_args, 'ref_model_mixup_alpha', 1.0)
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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # 设置模型加载参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False  # 始终禁用缓存以配合梯度检查点
    )
    
    # 加载模型实例（若有 checkpoint 则从 checkpoint 加载，否则从预训练模型加载）
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    

    
    # 确保模型处于训练模式
    model.train()

    # 加载分词器（使用预训练模型名称）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Llama使用右侧填充

    # 加载数据集并过滤
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/train.json'}
    )
    # 只保留 label 为 refutes 的数据
    dataset = dataset.filter(lambda x: x['label'] == 'refutes')

    # 数据预处理函数
    def preprocess_function(examples):
        prompt = """
        You are a feature error correction assistant. The user provides an incorrect statement, and you need to correct it. Evidence for correcting this incorrect statement will be provided to you, and you must use the given evidence to revise the incorrect statement. Only correct the erroneous parts of the sentence while keeping the rest intact. All the information contained in the original sentence must be retained and cannot be deleted, only modified.The corrected sentence must not be exactly the same as the original sentence.The original meaning of the sentence must not be changed.The revised statement should not differ significantly in semantics and format from the original statement.
        You must first think through the reasoning process in your mind before providing the user with the answer. The reasoning process and the answer should be enclosed within the <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
        All your outputs must be wrapped in tags. The thought process should be enclosed in <think></think>, and the final result should be enclosed in <answer></answer>. First, output the reasoning process, then output the final answer.The two pairs of tags must both be output.Each tag pair can and must appear only once.Tags cannot be nested.
        User:'{original_statement}'.Evidence:'{evidence}' Assistant:
        """
        inputs = prompt.format(evidence=examples['evidence'], original_statement=examples['claim'])

        if not inputs.strip():
            inputs = "No input provided."
            logger.warning("Empty input detected, using default input")

        model_inputs = tokenizer(
            inputs,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )

        # 确保输入张量设置正确的requires_grad属性
        input_ids = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.long)
        
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        model_inputs['prompt'] = inputs

        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=[col for col in dataset['train'].column_names if col not in ['id', 'evidence', 'claim']],  # 保留必要的列
        desc="Processing dataset",
        keep_in_memory=True  # 保持数据在内存中
    )

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()
    
    # 定义奖励函数，并初始化 sentence transformer 模型
    similarity_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    similarity_model = similarity_model.to('cuda')
    similarity_model.eval()

    def accuracy_reward(prompts, completions, **kwargs):
        rewards = []
        for output, prompt in zip(completions, prompts):
            output_text = output if isinstance(output, str) else str(output).strip()

            prompt_text = prompt.split("User:'")[1].split("'.Evidence:")[0].strip()
            evidence = prompt.split("'.Evidence:'")[1].split("' Assistant:")[0].strip()

            print(f"\nOrginal:{prompt_text}\n\n")
            print(f"Model Output:\n{output_text}\n\n")

            # 检查格式是否正确 - 必须先有<think></think>，然后再有<answer></answer>
            think_start = output_text.find('<think>')
            think_end = output_text.find('</think>')
            answer_start = output_text.find('<answer>')
            answer_end = output_text.find('</answer>')
            
            # 检查所有标签是否存在且顺序正确
            if (think_start == -1 or think_end == -1 or answer_start == -1 or answer_end == -1 or
                think_start > think_end or answer_start > answer_end or
                think_end > answer_start):
                print(f"Format error: Tags missing or in wrong order\n")
                rewards.append(0.0)
                continue

            # 检查每个标签是否只出现一次
            if (output_text.count('<think>') != 1 or output_text.count('</think>') != 1 or
                output_text.count('<answer>') != 1 or output_text.count('</answer>') != 1):
                print(f"Format error: Each tag should appear exactly once\n")
                rewards.append(0.0)
                continue

            # 检查所有文本是否都在标签内
            text_before_think = output_text[:think_start].strip()
            text_between_tags = output_text[think_end + len('</think>'):answer_start].strip()
            text_after_answer = output_text[answer_end + len('</answer>'):].strip()
            if text_before_think or text_between_tags or text_after_answer:
                print(f"Format error: All text must be wrapped in tags\n")
                rewards.append(0.0)
                continue
                
            # 提取answer标签内的内容
            output_text = output_text[answer_start + len('<answer>'):answer_end].strip()
            
            if output_text.count('<') > 0 and output_text.count('>') > 0:
                print(f"Format error: More tag\n")
                rewards.append(0.0)
                continue

            if len(output_text)*1.0 / len(prompt_text) < 0.8 or len(output_text)*1.0 / len(prompt_text) > 1.5:
                print(f"Format error: Answer too short or too long\n")
                rewards.append(0.05)
                continue

            if len(output_text)*1.0 / len(prompt_text) < 0.5 or len(output_text)*1.0 / len(prompt_text) > 2:
                print(f"Format error: Answer too too short or too too long\n")
                rewards.append(0.0)
                continue

            # 标准化处理字符串，移除多余空格和换行符
            normalized_output = ' '.join(output_text.strip().split())
            normalized_prompt = ' '.join(prompt_text.strip().split())
            

            # 编码文本并确保维度正确
            output_embedding = similarity_model.encode(
                output_text, 
                convert_to_tensor=True,
                device='cuda'
            )              
            target_embedding = similarity_model.encode(
                prompt_text, 
                convert_to_tensor=True, 
                device='cuda'
            )

            # 计算余弦相似度，直接使用二维张量
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(prompt_text,output_text)
            rouge1_f1 = scores['rouge1'].fmeasure
            rouge2_f1 = scores['rouge2'].fmeasure
            rougeL_f1 = scores['rougeL'].fmeasure
            rouge_f1 = (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3
            sari = load("He-Xingwei/sari_metric")
            results_sari = sari.compute(sources=[prompt_text], predictions=[output_text], references=[[""]])
            results_sari = results_sari['sari']
            result_final = rouge1_f1*0.5*0.3 + results_sari*0.005*0.7
            if normalized_output == normalized_prompt or (rouge_f1 > 0.8 and similarity>0.8) or similarity>0.9:
                print(f"Output is exactly the same as input\n")
                rewards.append(0.05)
                continue

            print(f"Similarity: {similarity},Rouge_f1:{rouge_f1},SARI:{results_sari},Final:{result_final}\n")
            if result_final < 0.3:
                rewards.append(similarity)
                continue
            # 使用事实验证模块评估生成文本
            programs = program_generator.batch_generate_programs(output_text)
            # 执行推理程序
            print(f"\nPrograms:{programs},Type:{type(programs)}\n")
            global count
            sample_data = {
                "idx": count,
                "id": None,
                "claim": output_text,
                "gold": "",
                "predicted_programs": [programs],
                "evidence": evidence
            }
            count += 1
            prediction = program_executor.execute_on_dataset(sample_data)
            print(f"\nPrediction: {prediction}\n")
            if prediction:
                rewards.append(1.0)
            else:
                rewards.append(0.4)
            print(f"\nRewards:{rewards}\n")
        return torch.tensor(rewards, requires_grad=True).clone().detach().requires_grad_(True)

    reward_funcs = [accuracy_reward]

    # 初始化 GRPO 训练器，直接传入模型实例
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    logger.info("*** Starting training ***")
    wandb.init(project="llama-3.2-1b-instruct-grpo", name=training_args.run_name)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型
    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # 同时保存分词器
    tokenizer.save_pretrained(training_args.output_dir)

    # 完成 wandb 日志
    wandb.finish()

if __name__ == "__main__":
    main()