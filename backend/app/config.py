from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Create a .env file in the backend directory to override defaults.
    See .env.example for all available options.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 180.0
    ollama_max_retries: int = 3
    ollama_max_tokens: int = 8192  # Ollama handles this better
    
    # vLLM Configuration
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "llama3.2:3b"
    vllm_timeout: float = 180.0
    vllm_max_retries: int = 3
    vllm_max_tokens: int = 2048  # Lower default for smaller context models

    # File storage
    data_dir: str = "./data"

    # OCR
    tesseract_config: str = "--psm 6"
    
    # ELA (Error Level Analysis) Configuration
    ela_min_size: int = 150  # Minimum image size (width/height) to analyze
    ela_allow_non_jpeg: bool = False  # Allow ELA on non-JPEG formats (PNG, etc.)
    ela_scale_for_heatmap: int = 77  # Scale factor for heatmap visualization
    ela_quality: int = 95  # JPEG quality for re-saving
    
    def get_llm_config(self, provider: str) -> dict:
        """Get LLM configuration for the specified provider."""
        if provider == "vllm":
            return {
                "provider": "vllm",
                "base_url": self.vllm_base_url,
                "model": self.vllm_model,
                "timeout": self.vllm_timeout,
                "max_retries": self.vllm_max_retries,
                "max_tokens": self.vllm_max_tokens,
            }
        else:
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "timeout": self.ollama_timeout,
                "max_retries": self.ollama_max_retries,
                "max_tokens": self.ollama_max_tokens,
            }


settings = Settings()