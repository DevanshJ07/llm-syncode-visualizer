"""
Application-wide configuration loaded from environment variables.
Pydantic Settings validates and coerces all values at startup.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    app_name: str = "LLM Syncode Visualizer API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS — set to your Next.js dev URL in .env
    cors_origins: list[str] = ["http://localhost:3000"]

    # Model
    # TinyLlama is CPU-compatible and ≈ 2.2 GB RAM in fp32.
    # Switch to meta-llama/Meta-Llama-3-3B once GPU is available.
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cpu"    # "cpu" | "cuda" | "mps"
    max_new_tokens: int = 64
    default_top_k: int = 10

    # Storage — experiments are stored as JSON files under this directory
    experiments_dir: str = "logs/experiments"

    # Feature flags
    syncode_enabled: bool = False  # Syncode not yet implemented
    model_loaded: bool = False     # Flipped to True after model warm-up


# Singleton — import `settings` everywhere, never instantiate Settings directly.
settings = Settings()
