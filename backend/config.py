"""
全局配置文件
通过 .env 文件或环境变量覆盖默认值
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# ===== HuggingFace 国内镜像 (解决被墙问题) =====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ===== 路径 =====
BASE_DIR = Path(__file__).resolve().parent.parent  # RAG问答系统/
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_index"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== Embedding 配置 =====
    # 可选: BAAI/bge-m3 (全能) | BAAI/bge-small-zh-v1.5 (中文小模型, 快)
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"  # cpu / cuda

    # ===== 文档分块配置 =====
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ===== 检索配置 =====
    top_k: int = 4

    # ===== 大模型配置 =====
    # 可选后端: deepseek | openai | ollama | mock
    llm_provider: str = "mock"  # 默认 mock 方便无 API Key 时演示

    # DeepSeek (推荐)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Ollama (本地)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # ===== 服务器配置 =====
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


settings = Settings()
