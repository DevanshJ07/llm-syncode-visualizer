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

    # CORS – set to your Next.js dev URL in .env
    cors_origins: list[str] = ["http://localhost:3000"]

    # Model
    model_name: str = "meta-llama/Meta-Llama-3-3B"
    device: str = "cuda"          # "cuda" | "cpu" | "mps"
    max_new_tokens: int = 512
    default_top_k: int = 50

    # Storage
    # Experiments are stored as JSON files under this directory.
    experiments_dir: str = "logs/experiments"

    # Feature flags
    syncode_enabled: bool = True   # Disable to run without Syncode dependency
    model_loaded: bool = False     # Flipped to True after model warm-up


# Singleton – import `settings` everywhere, never instantiate Settings directly.
settings = Settings()
