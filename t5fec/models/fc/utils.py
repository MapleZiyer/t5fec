import backoff  # for exponential backoff
import openai
from openai.error import APIError, RateLimitError, APIConnectionError, AuthenticationError, ServiceUnavailableError
import os
import asyncio
import logging
from typing import Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量获取API密钥，如果不存在则使用默认值
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-NVz2LEoGeiJ0vMTkt4nwTHestJiEoRcjs8aplkkAEjBPULme")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.bianxie.ai/v1")

openai.api_base = BASE_URL
openai.api_key = API_KEY

# 定义重试策略的常量
MAX_RETRIES = 10000
INITIAL_WAIT = 1  # 初始等待时间（秒）
MAX_WAIT = 60     # 最大等待时间（秒）

@backoff.on_exception(
    backoff.expo, 
    (APIError, RateLimitError, APIConnectionError, TimeoutError, ServiceUnavailableError),
    max_tries=MAX_RETRIES,
    factor=2,
    base=INITIAL_WAIT,
    max_value=MAX_WAIT
)
def completions_with_backoff(**kwargs):
    try:
        logger.info(f"Sending completion request with model: {kwargs.get('model', 'unknown')}")
        return openai.Completion.create(**kwargs, timeout=30)
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection error: {str(e)}")
        raise
    except AuthenticationError as e:
        logger.error(f"Authentication error: {str(e)}")
        raise
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise

@backoff.on_exception(
    backoff.expo, 
    (APIError, RateLimitError, APIConnectionError, TimeoutError, ServiceUnavailableError),
    max_tries=MAX_RETRIES,
    factor=2,
    base=INITIAL_WAIT,
    max_value=MAX_WAIT
)
def chat_completions_with_backoff(**kwargs):
    try:
        logger.info(f"Sending chat completion request with model: {kwargs.get('model', 'unknown')}")
        return openai.ChatCompletion.create(**kwargs, timeout=30)
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection error: {str(e)}")
        raise
    except AuthenticationError as e:
        logger.error(f"Authentication error: {str(e)}")
        raise
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    try:
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop = stop_words,
                timeout=30
            )
            for x in messages_list
        ]
        logger.info(f"Sending {len(messages_list)} requests to OpenAI API...")
        
        # 使用gather_with_concurrency限制并发请求数量
        responses = []
        for i in range(0, len(async_responses), 5):  # 每批5个请求
            batch = async_responses[i:i+5]
            batch_responses = await asyncio.gather(*batch, return_exceptions=True)
            responses.extend(batch_responses)
        
        # 检查是否有异常
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Error in request {i}: {str(resp)}")
                raise resp
        
        logger.info("Successfully received all API responses")
        return responses
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded in batch requests: {str(e)}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection error in batch requests: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in batch API requests: {str(e)}")
        raise

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    try:
        async_responses = [
            openai.Completion.acreate(
                model=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
                stop = stop_words,
                timeout=30
            )
            for x in messages_list
        ]
        logger.info(f"Sending {len(messages_list)} prompt requests to OpenAI API...")
        
        # 使用gather_with_concurrency限制并发请求数量
        responses = []
        for i in range(0, len(async_responses), 5):  # 每批5个请求
            batch = async_responses[i:i+5]
            batch_responses = await asyncio.gather(*batch, return_exceptions=True)
            responses.extend(batch_responses)
        
        # 检查是否有异常
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Error in prompt request {i}: {str(resp)}")
                raise resp
        
        logger.info("Successfully received all prompt API responses")
        return responses
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded in batch prompt requests: {str(e)}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection error in batch prompt requests: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in batch prompt API requests: {str(e)}")
        raise

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_base = BASE_URL
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        # 使用带有重试逻辑的函数
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x.choices[0].message.content.strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text