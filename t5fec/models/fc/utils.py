import openai
import backoff
import asyncio
from typing import Any

@backoff.on_exception(backoff.expo, openai.OpenAIError)
@backoff.on_exception(backoff.expo, (openai.OpenAIError, TimeoutError), max_tries=3)
def completions_with_backoff(**kwargs):
    try:
        # 确保提供 'model' 和 'prompt' 参数
        if 'model' not in kwargs or 'prompt' not in kwargs:
            raise ValueError("Missing required arguments: 'model' and 'prompt' are required")
        return openai.Completion.create(**kwargs)  # 使用 Completion.create 而非 completions.create
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, (openai.OpenAIError, TimeoutError), max_tries=3)
def chat_completions_with_backoff(**kwargs):
    try:
        # 确保提供 'model' 和 'messages' 参数
        if 'model' not in kwargs or 'messages' not in kwargs:
            raise ValueError("Missing required arguments: 'model' and 'messages' are required")
        return openai.ChatCompletion.create(**kwargs)  # 使用 ChatCompletion.create 而非 completions.create
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        raise

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    try:
        async_responses = [
            openai.ChatCompletion.acreate(  # 使用新的接口 acreate
                model=model,
                messages=x,  # 'messages' 参数
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop_words
            )
            for x in messages_list
        ]
        print(f"Sending {len(messages_list)} requests to OpenAI API...")
        responses = await asyncio.gather(*async_responses)
        print("Successfully received all API responses")
        return responses
    except Exception as e:
        print(f"Error in batch API requests: {str(e)}")
        raise

async def dispatch_openai_prompt_requests(
    prompt_list: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(  # 使用新的接口 acreate
            model=model,
            prompt=x,  # 'prompt' 参数
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_words
        )
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        openai.api_base = "https://api.openai.com/v1"  # 如果有其他API域名，可以修改
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature=0.0):
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],  # 需要传递 'messages'
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text

    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,  # 需要传递 'prompt'
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, temperature=0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")

    def batch_chat_generate(self, messages_list, temperature=0.0):
        open_ai_messages_list = [
            [{"role": "user", "content": message}] for message in messages_list
        ]
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]

    def batch_prompt_generate(self, prompt_list, temperature=0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature=0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,  # 需要传递 'suffix'
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text
