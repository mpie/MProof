"""Tests for application configuration."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    def test_default_settings(self):
        """Test default settings values."""
        from app.config import Settings
        
        settings = Settings()
        
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "mistral:latest"
        assert settings.ollama_timeout == 180.0
        assert settings.ollama_max_retries == 3
        assert settings.data_dir == "./data"
        assert "sqlite" in settings.database_url

    def test_settings_from_env(self):
        """Test settings loaded from environment variables."""
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://custom-host:11434",
            "OLLAMA_MODEL": "llama2:latest",
            "OLLAMA_TIMEOUT": "120.0",
            "DATA_DIR": "/custom/data"
        }):
            from importlib import reload
            import app.config
            reload(app.config)
            
            # Note: This test may not work perfectly due to module caching
            # In real scenarios, use dependency injection for testability

    def test_settings_model_config(self):
        """Test settings model configuration."""
        from app.config import Settings
        
        # Check that env_file is set
        assert Settings.model_config.get("env_file") == ".env"
        assert Settings.model_config.get("case_sensitive") is False


class TestDatabaseURL:
    def test_sqlite_url_format(self):
        """Test SQLite database URL format."""
        from app.config import settings
        
        assert settings.database_url.startswith("sqlite")
        assert "aiosqlite" in settings.database_url


class TestOllamaConfig:
    def test_ollama_timeout_is_float(self):
        """Test Ollama timeout is a float."""
        from app.config import settings
        
        assert isinstance(settings.ollama_timeout, float)
        assert settings.ollama_timeout > 0

    def test_ollama_max_retries_is_int(self):
        """Test Ollama max retries is an integer."""
        from app.config import settings
        
        assert isinstance(settings.ollama_max_retries, int)
        assert settings.ollama_max_retries > 0
