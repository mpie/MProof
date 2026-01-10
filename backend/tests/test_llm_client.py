import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.services.llm_client import LLMClient, LLMClientError


class TestLLMClient:
    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    @pytest.mark.asyncio
    async def test_generate_json_success(self, llm_client):
        """Test successful JSON generation."""
        mock_response = '{"test": "response_from_assistant"}'

        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = (mock_response, "curl command")

            schema = {
                "type": "object",
                "properties": {
                    "test": {"type": "string"}
                }
            }

            result, raw = await llm_client.generate_json_with_raw("test prompt", schema)

            assert result == {"test": "response_from_assistant"}

    @pytest.mark.asyncio
    async def test_generate_json_with_json_repair(self, llm_client):
        """Test JSON repair when initial parsing fails."""
        malformed_response = 'Here is the result: {"test": "value", "number": 42} Hope this helps!'

        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = (malformed_response, "curl command")

            result, raw = await llm_client.generate_json_with_raw("test prompt")

            assert result == {"test": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_generate_json_schema_validation_failure(self, llm_client):
        """Test schema validation failure."""
        invalid_response = '{"unexpected_field": "value"}'

        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = (invalid_response, "curl command")

            schema = {
                "type": "object",
                "properties": {"expected_field": {"type": "string"}},
                "required": ["expected_field"]
            }

            with pytest.raises(LLMClientError, match="Response does not match expected schema"):
                await llm_client.generate_json_with_raw("test prompt", schema)

    @pytest.mark.asyncio
    async def test_generate_json_timeout_retry(self, llm_client):
        """Test timeout and retry logic."""
        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.side_effect = [
                asyncio.TimeoutError(),
                asyncio.TimeoutError(),
                ('{"success": true}', "curl command")
            ]

            result, raw = await llm_client.generate_json_with_raw("test prompt")

            assert result == {"success": True}
            assert mock_make_request.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_json_max_retries_exceeded(self, llm_client):
        """Test max retries exceeded."""
        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.side_effect = asyncio.TimeoutError()

            with pytest.raises(LLMClientError, match="Max retries exceeded"):
                await llm_client.generate_json_with_raw("test prompt")

    @pytest.mark.asyncio
    async def test_generate_json_http_error(self, llm_client):
        """Test HTTP error handling."""
        import httpx

        with patch.object(llm_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=None,
                response=httpx.Response(404)
            )

            with pytest.raises(httpx.HTTPStatusError):
                await llm_client.generate_json_with_raw("test prompt")

    @pytest.mark.asyncio
    async def test_check_health_success(self, llm_client):
        """Test successful health check."""
        mock_response = {"models": [{"name": "mistral:latest"}]}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response_obj = AsyncMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response_obj
            mock_client_class.return_value = mock_client

            result = await llm_client.check_health()

            assert result is True

    @pytest.mark.asyncio
    async def test_check_health_model_not_found(self, llm_client):
        """Test health check when model is not available."""
        mock_response = {"models": [{"name": "other-model"}]}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response_obj = AsyncMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response_obj
            mock_client_class.return_value = mock_client

            result = await llm_client.check_health()

            assert result is False

    @pytest.mark.asyncio
    async def test_check_health_connection_error(self, llm_client):
        """Test health check connection error."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            result = await llm_client.check_health()

            assert result is False

    def test_repair_json_multiple_objects(self, llm_client):
        """Test repairing and merging multiple JSON objects."""
        text = '''{ "data": { "iban": "NL89 RABO 0340 3410 92 EUR" }, "evidence": { "iban": [ {"page": 0, "start": 56, "end": 73, "quote": "IBAN / Rekeningnummer"} ] } } { "data": { "iban": "NL37 RABO 1520 4812 76 EUR" }, "evidence": { "iban": [ {"page": 1, "start": 56, "end": 73, "quote": "IBAN / Rekeningnummer"} ] } }'''
        result = llm_client._repair_json(text)

        # Should merge the data and evidence
        assert result is not None
        assert "data" in result
        assert "evidence" in result
        assert result["data"]["iban"] == "NL89 RABO 0340 3410 92 EUR"  # First value wins
        assert len(result["evidence"]["iban"]) == 2  # Both evidence arrays combined

    def test_extract_json_objects_multiple(self, llm_client):
        """Test extracting multiple JSON objects from text."""
        text = '''{ "data": { "iban": "NL89" } } some text { "data": { "iban": "NL37" } }'''
        objects = llm_client._extract_json_objects(text)

        assert len(objects) == 2
        assert objects[0]["data"]["iban"] == "NL89"
        assert objects[1]["data"]["iban"] == "NL37"

    def test_merge_json_objects_metadata(self, llm_client):
        """Test merging JSON objects for metadata extraction."""
        objects = [
            {
                "data": {"iban": "NL89 RABO 0340 3410 92 EUR"},
                "evidence": {"iban": [{"page": 0, "quote": "IBAN1"}]}
            },
            {
                "data": {"rekeninghouder": "Jan Jansen"},
                "evidence": {"rekeninghouder": [{"page": 1, "quote": "Name"}]}
            }
        ]

        result = llm_client._merge_json_objects(objects)

        assert result is not None
        assert result["data"]["iban"] == "NL89 RABO 0340 3410 92 EUR"
        assert result["data"]["rekeninghouder"] == "Jan Jansen"
        assert len(result["evidence"]["iban"]) == 1
        assert len(result["evidence"]["rekeninghouder"]) == 1