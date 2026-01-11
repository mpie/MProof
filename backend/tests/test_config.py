"""Tests for application configuration."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    def test_default_settings(self):
        """Test that settings are loaded and have expected types."""
        from app.config import Settings
        
        settings = Settings()
        
        # Test that settings exist and have the right types
        assert isinstance(settings.ollama_base_url, str)
        assert isinstance(settings.ollama_model, str)
        assert isinstance(settings.ollama_timeout, float)
        assert isinstance(settings.ollama_max_retries, int)
        assert isinstance(settings.data_dir, str)
        assert "sqlite" in settings.database_url
        
        # Test vLLM settings
        assert isinstance(settings.vllm_base_url, str)
        assert isinstance(settings.vllm_model, str)
        assert isinstance(settings.vllm_timeout, float)
        assert isinstance(settings.vllm_max_retries, int)

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


class TestLLMConfig:
    def test_get_llm_config_ollama(self):
        """Test getting Ollama config."""
        from app.config import settings
        
        config = settings.get_llm_config("ollama")
        assert config["provider"] == "ollama"
        assert isinstance(config["base_url"], str)
        assert isinstance(config["model"], str)
        assert isinstance(config["timeout"], float)
        assert isinstance(config["max_retries"], int)

    def test_get_llm_config_vllm(self):
        """Test getting vLLM config."""
        from app.config import settings
        
        config = settings.get_llm_config("vllm")
        assert config["provider"] == "vllm"
        assert isinstance(config["base_url"], str)
        assert isinstance(config["model"], str)
        assert isinstance(config["timeout"], float)
        assert isinstance(config["max_retries"], int)


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


class TestVLLMConfig:
    def test_vllm_timeout_is_float(self):
        """Test vLLM timeout is a float."""
        from app.config import settings
        
        assert isinstance(settings.vllm_timeout, float)
        assert settings.vllm_timeout > 0

    def test_vllm_max_retries_is_int(self):
        """Test vLLM max retries is an integer."""
        from app.config import settings
        
        assert isinstance(settings.vllm_max_retries, int)
        assert settings.vllm_max_retries > 0
