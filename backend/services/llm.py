"""
统一的大语言模型接口
支持: DeepSeek / OpenAI / Ollama / Mock (离线演示用)
"""
import json
from typing import List, Dict, Any, Generator

from backend.config import settings


class BaseLLM:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """本地占位模型 —— 从最后一条消息中抽取内容返回，方便无 API Key 时演示"""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        return (
            f"【Mock 演示回答】\n\n"
            f"📝 系统提示: {system_msg[:80]}...\n\n"
            f"🙋 你的问题: {user_msg[:200]}\n\n"
            f"> 当前使用的是 **Mock LLM**（无真实 API Key 时的兜底）。\n"
            f"> 请在 `.env` 中配置 `DEEPSEEK_API_KEY` 或 `OLLAMA` 后端以启用真实回答。"
        )

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        yield self.chat(messages)


class DeepSeekLLM(BaseLLM):
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装 openai: pip install openai") from e
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self.model = settings.deepseek_model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        for chunk in stream:
            # 部分中转站会在结尾发送 choices 为空、仅带 usage 的块
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OpenAILLM(BaseLLM):
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装 openai: pip install openai") from e
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        for chunk in stream:
            # 部分中转站会在结尾发送 choices 为空、仅带 usage 的块
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaLLM(BaseLLM):
    """通过 HTTP 调用本地 Ollama"""

    def __init__(self):
        self.base = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def _post(self, payload: dict) -> dict:
        import httpx
        resp = httpx.post(f"{self.base}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        data = self._post({"model": self.model, "messages": messages, "stream": False})
        return data.get("message", {}).get("content", "")

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        import httpx
        with httpx.stream(
            "POST",
            f"{self.base}/api/chat",
            json={"model": self.model, "messages": messages, "stream": True},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    continue


def get_llm_client(provider: str | None = None) -> BaseLLM:
    """根据配置返回对应 LLM 客户端"""
    provider = provider or settings.llm_provider
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            print("[WARN] 未配置 DEEPSEEK_API_KEY，降级为 Mock LLM")
            return MockLLM()
        return DeepSeekLLM()
    if provider == "openai":
        if not settings.openai_api_key:
            print("[WARN] 未配置 OPENAI_API_KEY，降级为 Mock LLM")
            return MockLLM()
        return OpenAILLM()
    if provider == "ollama":
        return OllamaLLM()
    return MockLLM()
