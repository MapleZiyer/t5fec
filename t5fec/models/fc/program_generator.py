from fc.prompts import Prompt_Loader
from fc.utils import OpenAIModel

class Reasoning_Program_Generator:
    def __init__(self):
        self.data_path = "./datasets"
        self.dataset_name = "HOVER"
        self.model_name = "gpt-3.5-turbo"
        self.save_path = "./results/programs"
        self.num_programs_per_example = 1
        self.stop_words = '# The claim is'
        self.max_new_tokens = 1024

        self.openai_api = OpenAIModel("sk-NVz2LEoGeiJ0vMTkt4nwTHestJiEoRcjs8aplkkAEjBPULme", self.model_name, self.stop_words, self.max_new_tokens)
        self.prompt_loader = Prompt_Loader()

    def batch_generate_programs(self, generated_texts):
        temperature = 0.0 if self.num_programs_per_example == 1 else 0.7

        # 确保输入是字符串类型
        if isinstance(generated_texts, list):
            generated_texts = generated_texts[0]

        full_prompt = self.prompt_loader.prompt_construction(generated_texts, self.dataset_name)

        output = self.openai_api.generate(full_prompt, temperature)
        
        # 将输出字符串按行分割，并过滤掉空行
        program_lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        return program_lines
