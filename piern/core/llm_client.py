"""
LLM 客户端，支持多种 API 提供商。

支持的提供商：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- SiliconFlow (开源模型)
- 本地 LLM (通过 OpenAI 兼容接口)
"""

import os
import time
import logging
from typing import Optional, Dict
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的 LLM 客户端接口。"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        初始化 LLM 客户端。

        Args:
            provider: API 提供商 ("openai", "anthropic", "siliconflow", "local")
            model: 模型名称
            api_key: API 密钥（如果为 None，从环境变量读取）
            base_url: API 基础 URL（用于本地部署或代理）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # 获取 API 密钥
        if api_key is None:
            api_key = self._get_api_key_from_env()
        self.api_key = api_key

        # 设置 API 端点
        self.base_url = base_url or self._get_default_base_url()

        # 读取自定义headers（如果存在）
        self.custom_headers = self._get_custom_headers()

        logger.info(f"初始化 LLM 客户端: provider={provider}, model={model}, base_url={self.base_url}")

    def _get_api_key_from_env(self) -> str:
        """从环境变量获取 API 密钥。"""
        env_vars = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],  # 支持AUTH_TOKEN
            "siliconflow": ["SILICONFLOW_API_KEY"],
            "local": None,  # 本地部署可能不需要密钥
        }

        env_var_list = env_vars.get(self.provider)
        if env_var_list is None:
            return ""

        # 尝试多个环境变量
        if isinstance(env_var_list, list):
            for env_var in env_var_list:
                api_key = os.getenv(env_var)
                if api_key:
                    return api_key

            # 所有环境变量都未找到
            raise ValueError(
                f"未找到 API 密钥。请设置环境变量 {' 或 '.join(env_var_list)} "
                f"或在初始化时传入 api_key 参数。"
            )

        return ""

    def _get_default_base_url(self) -> str:
        """获取默认的 API 基础 URL。"""
        # 优先从环境变量读取（支持代理）
        env_base_url = {
            "openai": "OPENAI_BASE_URL",
            "anthropic": "ANTHROPIC_BASE_URL",
            "siliconflow": "SILICONFLOW_BASE_URL",
        }.get(self.provider)

        if env_base_url:
            base_url = os.getenv(env_base_url)
            if base_url:
                return base_url

        # 默认URL
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "siliconflow": "https://api.siliconflow.cn/v1",
            "local": "http://localhost:8000/v1",
        }
        return default_urls.get(self.provider, "https://api.openai.com/v1")

    def _get_custom_headers(self) -> Dict[str, str]:
        """从环境变量读取自定义headers。"""
        custom_headers = {}

        # 读取ANTHROPIC_CUSTOM_HEADERS（格式：Key1: Value1, Key2: Value2）
        headers_str = os.getenv("ANTHROPIC_CUSTOM_HEADERS", "")
        if headers_str and self.provider == "anthropic":
            for pair in headers_str.split(","):
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    custom_headers[key.strip()] = value.strip()

        return custom_headers

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        生成文本。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数（0-1，越高越随机）
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider in ["openai", "siliconflow", "local"]:
                    return self._generate_openai_compatible(
                        prompt, system_prompt, temperature, max_tokens
                    )
                elif self.provider == "anthropic":
                    return self._generate_anthropic(
                        prompt, system_prompt, temperature, max_tokens
                    )
                else:
                    raise ValueError(f"不支持的提供商: {self.provider}")

            except Exception as e:
                logger.warning(
                    f"LLM 生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise RuntimeError(f"LLM 生成失败，已重试 {self.max_retries} 次") from e

    def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """使用 OpenAI 兼容接口生成文本。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """使用 Anthropic API 生成文本。"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # 添加自定义headers
        headers.update(self.custom_headers)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return result["content"][0]["text"].strip()

    def batch_generate(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        show_progress: bool = True,
    ) -> list[str]:
        """
        批量生成文本。

        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            show_progress: 是否显示进度条

        Returns:
            生成的文本列表
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            prompts = tqdm(prompts, desc="LLM 生成")

        for prompt in prompts:
            text = self.generate(prompt, system_prompt, temperature, max_tokens)
            results.append(text)

        return results


def test_llm_client():
    """测试 LLM 客户端。"""
    # 测试 OpenAI
    try:
        client = LLMClient(provider="openai", model="gpt-3.5-turbo")
        response = client.generate("你好，请用一句话介绍自己。")
        print(f"OpenAI 响应: {response}")
    except Exception as e:
        print(f"OpenAI 测试失败: {e}")

    # 测试 SiliconFlow
    try:
        client = LLMClient(
            provider="siliconflow",
            model="Qwen/Qwen2.5-7B-Instruct"
        )
        response = client.generate("你好，请用一句话介绍自己。")
        print(f"SiliconFlow 响应: {response}")
    except Exception as e:
        print(f"SiliconFlow 测试失败: {e}")


if __name__ == "__main__":
    test_llm_client()
