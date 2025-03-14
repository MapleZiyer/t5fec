import backoff
import openai
import os
import asyncio
from typing import Any, List

# 配置 OpenAI API 基础 URL
openai.api_base = "https://api.bianxie.ai/v1"

# 异常重试机制
@backoff.on_exception(backoff.expo, openai.OpenAIError)
@backoff.on_exception(backoff.expo, (openai.OpenAIError, TimeoutError), max_tries=3)
def completions_with_backoff(**kwargs):
    try:
        # 确保提供 'model' 和 'prompt' 参数
        if 'model' not in kwargs or 'prompt' not in kwargs:
            raise ValueError("Missing required arguments: 'model' and 'prompt' are required")
        return openai.Completion.create(**kwargs)
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, (openai.OpenAIError, TimeoutError), max_tries=3)
def chat_completions_with_backoff(**kwargs):
    try:
        # 确保提供 'model' 和 'messages' 参数
        if 'model' not in kwargs or 'messages' not in kwargs:
            raise ValueError("Missing required arguments: 'model' and 'messages' are required")
        return openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        raise

async def dispatch_openai_chat_requests(
    messages_list: List[List[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[str]:
    try:
        async_responses = [
            openai.ChatCompletion.acreate(  # 使用新的接口
                model=model,
                messages=x,
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
        return [response['choices'][0]['message']['content'].strip() for response in responses]
    except Exception as e:
        print(f"Error in batch API requests: {str(e)}")
        raise

async def dispatch_openai_prompt_requests(
    prompt_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[str]:
    try:
        async_responses = [
            openai.Completion.acreate(  # 使用新的接口
                model=model,
                prompt=x,  # prompt 参数
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop_words
            )
            for x in prompt_list
        ]
        responses = await asyncio.gather(*async_responses)
        return [response['choices'][0]['text'].strip() for response in responses]
    except Exception as e:
        print(f"Error in batch API requests: {str(e)}")
        raise

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        openai.api_base = "https://api.bianxie.ai/v1"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # 用于 chat-gpt 和 gpt-4
    def chat_generate(self, input_string: str, temperature: float = 0.0) -> str:
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],  # 需要传递 'messages'
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words
        )
        return response['choices'][0]['message']['content'].strip()

    # 用于 text/code-davinci
    def prompt_generate(self, input_string: str, temperature: float = 0.0) -> str:
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
        return response['choices'][0]['text'].strip()

    def generate(self, input_string: str, temperature: float = 0.0) -> str:
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")

    def batch_chat_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        open_ai_messages_list = [
            [{"role": "user", "content": message}] for message in messages_list
        ]
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return predictions

    def batch_prompt_generate(self, prompt_list: List[str], temperature: float = 0.0) -> List[str]:
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return predictions

    def batch_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string: str, suffix: str, temperature: float = 0.0) -> str:
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
        return response['choices'][0]['text'].strip()
