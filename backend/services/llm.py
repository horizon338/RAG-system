"""
大语言模型 —— 基于 LangChain 的 ChatModel 抽象
支持: OpenAI 兼容接口 / DeepSeek / Ollama / Mock
统一返回 BaseChatModel，供 LCEL 链式编排使用
"""
import json
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from backend.config import settings


class MockChatModel(BaseChatModel):
    """离线下使用的占位模型，保证整条链路可跑通"""

    @property
    def _llm_type(self) -> str:
        return "mock"

    @staticmethod
    def _build_text(messages: List[BaseMessage]) -> str:
        user_msg = ""
        for m in reversed(messages):
            if m.type == "human":
                user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break
        return (
            "【Mock 演示回答】\n\n"
            f"你的问题片段：{user_msg[:120]}\n\n"
            "> 当前使用的是 **Mock LLM**（无真实 API Key 时的兜底）。\n"
            "> 请在 `.env` 中配置 `OPENAI_API_KEY` 或改用 Ollama 后端以启用真实回答。"
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._build_text(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._build_text(messages)
        step = 24
        for i in range(0, len(text), step):
            piece = text[i : i + step]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


class ChatOllamaHTTP(BaseChatModel):
    """
    Ollama 本地模型（HTTP /api/chat）
    —— 官方 langchain-ollama 未安装时使用，接口行为保持一致
    """

    model_name: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "ollama-http"

    def _payload(self, messages: List[BaseMessage], stream: bool) -> dict:
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": {"human": "user", "ai": "assistant", "system": "system"}.get(
                        m.type, "user"
                    ),
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                }
                for m in messages
            ],
            "stream": stream,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import httpx

        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=self._payload(messages, stream=False),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import httpx

        with httpx.stream(
            "POST",
            f"{self.base_url.rstrip('/')}/api/chat",
            json=self._payload(messages, stream=True),
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = obj.get("message", {}).get("content", "")
                if not piece:
                    continue
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                if run_manager:
                    run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk


def _openai_compatible(api_key: str, base_url: str, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
    )


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """根据配置返回 LangChain ChatModel"""
    provider = provider or settings.llm_provider

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            print("[WARN] 未配置 DEEPSEEK_API_KEY，降级为 Mock ChatModel")
            return MockChatModel()
        return _openai_compatible(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            print("[WARN] 未配置 OPENAI_API_KEY，降级为 Mock ChatModel")
            return MockChatModel()
        return _openai_compatible(
            settings.openai_api_key,
            settings.openai_base_url,
            settings.openai_model,
        )

    if provider == "ollama":
        return ChatOllamaHTTP(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    return MockChatModel()
