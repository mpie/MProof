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

    # Ollama LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_timeout: float = 60.0
    ollama_max_retries: int = 3

    # File storage
    data_dir: str = "./data"

    # OCR
    tesseract_config: str = "--psm 6"


settings = Settings()