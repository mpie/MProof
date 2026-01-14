import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
import httpx
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: Optional provider override. If not specified, uses the active provider.
        """
        self._refresh_config(provider)
    
    def _refresh_config(self, provider: Optional[str] = None) -> None:
        """Refresh configuration from settings."""
        use_provider = provider or "ollama"  # Default to ollama
        config = settings.get_llm_config(use_provider)
        
        self.provider = config["provider"]
        self.base_url = config["base_url"].rstrip('/')
        self.model = config["model"]
        self.timeout = config["timeout"]
        self.max_retries = config["max_retries"]
        self.max_tokens = config.get("max_tokens", 2048)
        
        logger.info(f"LLMClient configured with provider: {self.provider}, base_url: {self.base_url}, model: {self.model}, max_tokens: {self.max_tokens}")

    def _generate_curl_command(self, url: str, payload: Dict[str, Any]) -> str:
        """Generate curl command equivalent of the HTTP request."""
        json_payload = json.dumps(payload, indent=2)
        return f"curl -X POST '{url}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{json_payload}'"

    def _get_api_url(self) -> str:
        """Get the API endpoint URL based on the provider."""
        if self.provider == "vllm":
            return f"{self.base_url}/v1/chat/completions"
        else:  # ollama
            return f"{self.base_url}/api/chat"

    def _estimate_input_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate input tokens from messages (conservative estimate: ~3.5 chars per token for safety)."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Conservative estimate: ~3.5 chars per token (instead of 4) to account for whitespace, punctuation, etc.
        # Add overhead for message structure (role, JSON formatting, etc.) - roughly 15 tokens per message
        estimated_tokens = int((total_chars / 3.5) + (len(messages) * 15))
        # Add 20% buffer for safety (JSON schemas, special characters, etc.)
        estimated_tokens = int(estimated_tokens * 1.2)
        return estimated_tokens

    def _calculate_safe_max_tokens(self, messages: List[Dict[str, str]], context_length: int = 4096) -> int:
        """Calculate safe max_tokens based on input length and context limit."""
        input_tokens = self._estimate_input_tokens(messages)
        safety_margin = 200  # Increased safety margin for vLLM
        available_tokens = context_length - input_tokens - safety_margin
        
        # Ensure we have at least some tokens available, but cap at configured max
        if available_tokens < 100:
            # If we're really tight on space, use a minimal amount
            safe_max = max(50, available_tokens // 2) if available_tokens > 0 else 50
            logger.warning(
                f"Very limited token space! Using {safe_max} max_tokens "
                f"(input: ~{input_tokens} tokens, available: {available_tokens}, context: {context_length})"
            )
        else:
            # Use the minimum of configured max_tokens and available tokens
            safe_max = min(self.max_tokens, available_tokens)
            
            if safe_max < self.max_tokens:
                logger.warning(
                    f"Reducing max_tokens from {self.max_tokens} to {safe_max} "
                    f"(input: ~{input_tokens} tokens, available: {available_tokens}, context: {context_length})"
                )
        
        return safe_max

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize messages for the provider.
        For vLLM: Convert system messages to user messages since vLLM requires roles to alternate user/assistant.
        For Ollama: Keep messages as-is (supports system messages).
        """
        if self.provider == "vllm":
            # vLLM requires roles to alternate user/assistant/user/assistant
            # System messages break this pattern, so merge them into the first user message
            normalized = []
            system_content_parts = []
            
            for msg in messages:
                if msg["role"] == "system":
                    # Collect system messages to merge into first user message
                    system_content_parts.append(msg["content"])
                else:
                    # If we have collected system content and this is the first non-system message
                    if system_content_parts and not normalized:
                        # Merge all system content into this message if it's a user message
                        if msg["role"] == "user":
                            combined_content = "\n\n".join(system_content_parts) + "\n\n" + msg["content"]
                            normalized.append({"role": "user", "content": combined_content})
                            system_content_parts = []  # Clear after merging
                        else:
                            # If first non-system is not user, create a user message with system content
                            combined_content = "\n\n".join(system_content_parts)
                            normalized.append({"role": "user", "content": combined_content})
                            normalized.append(msg)  # Then add the original message
                            system_content_parts = []
                    else:
                        # No pending system content, just add the message
                        normalized.append(msg)
            
            # If we have system content but no user message to merge into, create a user message
            if system_content_parts and not normalized:
                combined_content = "\n\n".join(system_content_parts)
                normalized.append({"role": "user", "content": combined_content})
            
            return normalized
        else:
            # Ollama supports system messages, return as-is
            return messages

    def _build_request_payload(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
        """Build the request payload based on the provider."""
        # Normalize messages for the provider (e.g., convert system to user for vLLM)
        normalized_messages = self._normalize_messages(messages)
        
        if self.provider == "vllm":
            # Calculate available tokens based on actual model context
            # Use 4096 as conservative default - the model will reject if we exceed
            MODEL_CONTEXT = 4096
            SAFETY_MARGIN = 100  # Buffer for tokenization differences
            
            input_tokens = self._estimate_input_tokens(normalized_messages)
            available_tokens = MODEL_CONTEXT - input_tokens - SAFETY_MARGIN
            
            # Ensure we don't exceed available space
            if available_tokens <= 0:
                safe_max_tokens = 50
            else:
                safe_max_tokens = min(available_tokens, 2048)
                safe_max_tokens = max(safe_max_tokens, 50)
            
            logger.info(f"vLLM request: input ~{input_tokens} tokens, available ~{available_tokens}, max_tokens={safe_max_tokens}")
            # OpenAI-compatible format for vLLM
            return {
                "model": self.model,
                "messages": normalized_messages,
                "temperature": temperature,
                "max_tokens": safe_max_tokens,
                "top_p": 0.9,
                "stop": ["\n\n\n", "```", "---"]  # Stop sequences to prevent infinite generation
            }
        else:  # ollama
            return {
                "model": self.model,
                "messages": normalized_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9
                }
            }

    def _extract_response_content(self, data: Dict[str, Any]) -> str:
        """Extract the response content based on the provider."""
        if self.provider == "vllm":
            # OpenAI-compatible format
            if "choices" not in data or not data["choices"]:
                raise LLMClientError("Invalid response format from vLLM")
            choice = data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                raise LLMClientError("Invalid response format from vLLM")
            return choice["message"]["content"]
        else:  # ollama
            if "message" not in data or "content" not in data["message"]:
                raise LLMClientError("Invalid response format from Ollama")
            return data["message"]["content"]

    async def _make_request(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Tuple[str, str, float]:
        """Make a request to the LLM API with retries. Returns (response_text, curl_command, duration_seconds)."""
        url = self._get_api_url()
        payload = self._build_request_payload(messages, temperature)

        # Generate curl command
        curl_command = self._generate_curl_command(url, payload)

        # Log detailed request information
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        prompt_length = len(user_message)
        doc_length = sum(len(msg.get("content", "")) for msg in messages)
        request_id = id(payload)  # Simple request ID
        
        logger.info(f"LLM Request: provider={self.provider}, model={self.model}, request_id={request_id}, prompt_length={prompt_length}, doc_length={doc_length}")
        logger.debug(f"LLM Request (curl equivalent):\n{curl_command}")
        logger.debug(f"LLM Request payload content: {payload}")

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    duration = time.time() - start_time
                    latency_ms = int(duration * 1000)
                    status_code = response.status_code

                    data = response.json()
                    response_content = self._extract_response_content(data)
                    
                    # Check if response was truncated (Ollama-specific)
                    if self.provider == "ollama":
                        done = data.get("done", True)
                        if not done:
                            logger.warning(f"LLM response may be incomplete (done=false). Response length: {len(response_content)} chars")
                    
                    # Check for incomplete JSON in response
                    # But also check if we have multiple complete JSON objects (which is valid for merging)
                    open_braces = response_content.count('{')
                    close_braces = response_content.count('}')
                    open_brackets = response_content.count('[')
                    close_brackets = response_content.count(']')
                    
                    # If braces/brackets are balanced or only off by 1 (common with multiple objects), check for multiple objects
                    brace_diff = open_braces - close_braces
                    bracket_diff = open_brackets - close_brackets
                    
                    # Check for multiple JSON objects by looking for patterns like "}\n{" or "}\r\n{" or "} } {"
                    json_object_patterns = [
                        response_content.count('}\n{'),
                        response_content.count('}\r\n{'),
                        response_content.count('} } {'),
                        response_content.count('}\n\n{'),
                    ]
                    json_object_count = sum(json_object_patterns) + (1 if response_content.strip().startswith('{') else 0)
                    
                    if abs(brace_diff) <= 1 and bracket_diff == 0:
                        # Likely multiple JSON objects
                        if json_object_count > 1:
                            logger.debug(f"LLM response contains {json_object_count} separate JSON objects (will be merged automatically)")
                        # Don't warn - this is valid and will be handled by repair logic
                    elif brace_diff > 1 or bracket_diff > 0:
                        # Check if we have multiple objects despite unbalanced braces (extra closing braces)
                        if json_object_count > 1:
                            logger.debug(f"LLM response contains {json_object_count} separate JSON objects with extra closing braces (will be merged automatically)")
                        else:
                            logger.warning(f"LLM response appears truncated (unbalanced braces/brackets: {open_braces}/{close_braces} braces, {open_brackets}/{close_brackets} brackets). Response length: {len(response_content)} chars")
                    
                    logger.info(f"LLM Response: provider={self.provider}, model={self.model}, request_id={id(payload)}, latency_ms={latency_ms}, status_code={status_code}, response_length={len(response_content)}")
                    logger.debug(f"LLM Response content: {response_content[:500]}{'...' if len(response_content) > 500 else ''}")

                    return response_content, curl_command, duration

            except httpx.TimeoutException:
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise LLMClientError("LLM request timed out")
                await asyncio.sleep(1)

            except httpx.HTTPStatusError as e:
                # Try to get error details from response body
                try:
                    error_body = e.response.text
                    logger.error(f"LLM HTTP error {e.response.status_code} response body: {error_body}")
                except Exception:
                    error_body = "Could not read response body"
                
                if e.response.status_code == 404:
                    raise LLMClientError(f"Model '{self.model}' not found on {self.provider} server")
                elif e.response.status_code == 400:
                    # 400 errors are usually client errors that won't be fixed by retrying
                    raise LLMClientError(f"LLM request rejected (400): {error_body}")
                logger.warning(f"LLM HTTP error {e.response.status_code} (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise LLMClientError(f"LLM request failed: {e}")
                await asyncio.sleep(1)

        raise LLMClientError("Max retries exceeded")

    async def _make_single_request_no_retry(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Tuple[str, str, float]:
        """Make a single request without retries (for batch processing). Returns (response_text, curl_command, duration_seconds)."""
        url = self._get_api_url()
        payload = self._build_request_payload(messages, temperature)
        curl_command = self._generate_curl_command(url, payload)

        start_time = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_body = response.text
                logger.error(f"LLM HTTP error {e.response.status_code} response body: {error_body}")
                if e.response.status_code == 400:
                    raise LLMClientError(f"LLM request rejected (400): {error_body}")
                raise
            data = response.json()
            response_content = self._extract_response_content(data)
            duration = time.time() - start_time
            return response_content, curl_command, duration

    async def make_batch_requests(
        self,
        requests: List[Dict[str, Any]],
        temperature: float = 0.1
    ) -> List[Tuple[str, str, float]]:
        """
        Make multiple requests in parallel (optimized for vLLM).
        
        Each request dict should have:
        - messages: List[Dict[str, str]] - The chat messages
        
        Returns list of (response_text, curl_command, duration_seconds) tuples in the same order.
        
        For vLLM: All requests are sent in parallel.
        For Ollama: Requests are sent sequentially (Ollama handles one at a time).
        """
        if not requests:
            return []
        
        if self.provider == "vllm":
            # vLLM can handle parallel requests efficiently
            logger.info(f"Making {len(requests)} parallel requests to vLLM")
            
            async def make_single(req: Dict[str, Any]) -> Tuple[str, str, float]:
                messages = req.get("messages", [])
                temp = req.get("temperature", temperature)
                return await self._make_single_request_no_retry(messages, temp)
            
            # Run all requests in parallel
            results = await asyncio.gather(*[make_single(req) for req in requests], return_exceptions=True)
            
            # Process results, re-raising exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch request {i} failed: {result}")
                    raise LLMClientError(f"Batch request {i} failed: {result}")
                processed_results.append(result)
            
            return processed_results
        else:
            # Ollama: process sequentially
            logger.info(f"Making {len(requests)} sequential requests to Ollama")
            results = []
            for req in requests:
                messages = req.get("messages", [])
                temp = req.get("temperature", temperature)
                result = await self._make_request(messages, temp)
                results.append(result)
            return results

    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair malformed JSON responses, including multiple JSON objects."""
        # Remove markdown code blocks
        text = text.strip()
        
        # Remove instruction text that the LLM sometimes echoes back
        # This handles cases where the LLM includes schema instructions in its response
        import re
        instruction_patterns = [
            r'"evidence"\s+field\s+MUST\s+be\s+an?\s+[A-Z]+[^"]*',  # "evidence" field MUST be an OBJECT/ARRAY...
            r'field\s+MUST\s+be\s+an?\s+[A-Z]+[^"]*',  # field MUST be an...
            r'CRITICAL\s+RULES:.*?(?=\{|\Z)',  # CRITICAL RULES: ...
            r'Example\s+structure.*?(?=\{|\Z)',  # Example structure...
            r'IMPORTANT:.*?(?=\{|\Z)',  # IMPORTANT: ...
            r'NOTE:.*?(?=\{|\Z)',  # NOTE: ...
        ]
        
        original_text = text
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        if text != original_text:
            logger.info("Removed instruction text from LLM response")
            text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Check for truncated JSON (missing closing braces/brackets or unclosed strings)
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        # Detect if we're in the middle of a string by checking for unclosed quotes
        # Count unescaped double quotes
        import re
        # Remove escaped quotes first
        unescaped_text = re.sub(r'\\.', '', text)
        quote_count = unescaped_text.count('"')
        is_in_string = quote_count % 2 != 0
        
        if open_braces > close_braces or open_brackets > close_brackets or is_in_string:
            logger.warning(f"Detected truncated JSON: braces {open_braces}/{close_braces}, brackets {open_brackets}/{close_brackets}, in_string={is_in_string}. Attempting repair.")
            
            # If we're in the middle of a string, close it first
            if is_in_string:
                # Find the last unescaped quote position
                last_quote_pos = -1
                i = len(text) - 1
                while i >= 0:
                    if text[i] == '"' and (i == 0 or text[i-1] != '\\'):
                        last_quote_pos = i
                        break
                    i -= 1
                
                # If we found an opening quote, we need to close the string
                if last_quote_pos >= 0:
                    # Check what's after the last quote
                    remaining = text[last_quote_pos+1:]
                    
                    # If remaining is empty or doesn't end with a structure char, close the string
                    # This handles cases like: "datum_overeenkomst": "  (truncated mid-value)
                    # and cases like: "datum_overeenkomst": "some value  (truncated mid-string)
                    if not remaining.rstrip().endswith(('}', ']', ',', ':')):
                        # Close the string with null value if it's a key-value truncation
                        # Check if this looks like a truncated value (preceded by colon)
                        before_quote = text[:last_quote_pos].rstrip()
                        if before_quote.endswith(':'):
                            # The value was truncated - use null instead
                            text = text[:last_quote_pos] + 'null'
                            logger.info("Replaced truncated string value with null")
                        else:
                            # Just close the string
                            text = text + '"'
                            logger.info("Added closing quote for truncated string")
            
            # Add missing closing brackets first, then braces
            text = text + (']' * (open_brackets - close_brackets))
            text = text + ('}' * (open_braces - close_braces))
            logger.info(f"Added {open_brackets - close_brackets} brackets and {open_braces - close_braces} braces to repair truncated JSON")

        # First, try to parse directly (might just have extra whitespace)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to fix common issues before extracting
        import re
        # Remove trailing commas
        fixed_text = re.sub(r',\s*}', '}', text)
        fixed_text = re.sub(r',\s*]', ']', fixed_text)
        # Try parsing again
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass

        # Try to find and merge multiple JSON objects
        json_objects = self._extract_json_objects(text)

        if len(json_objects) == 1:
            # Single JSON object
            logger.info("Successfully extracted single JSON object")
            return json_objects[0]
        elif len(json_objects) > 1:
            # Multiple JSON objects - try to merge them
            logger.info(f"Found {len(json_objects)} separate JSON objects, attempting merge")
            logger.debug(f"JSON objects keys: {[list(obj.keys()) for obj in json_objects]}")
            merged = self._merge_json_objects(json_objects)
            if merged and (merged.get("data") or merged.get("evidence")):
                logger.info(f"JSON objects successfully merged into single object with keys: {list(merged.keys())}")
                logger.debug(f"Merged result: {merged}")
                return merged
            else:
                logger.warning("Merging produced empty or invalid result, trying direct combination")
                # Try direct combination as fallback
                if len(json_objects) == 2:
                    obj1, obj2 = json_objects[0], json_objects[1]
                    if "data" in obj1 and "evidence" in obj2:
                        return {"data": obj1["data"], "evidence": obj2["evidence"]}
                    elif "evidence" in obj1 and "data" in obj2:
                        return {"data": obj2["data"], "evidence": obj1["evidence"]}
                    # If both have data, merge them
                    elif "data" in obj1 and "data" in obj2:
                        merged_data = {}
                        if isinstance(obj1["data"], dict) and isinstance(obj2["data"], dict):
                            merged_data.update(obj1["data"])
                            merged_data.update({k: v for k, v in obj2["data"].items() if v is not None})
                        return {"data": merged_data}
                # Last resort: return first object or combine all data keys
                logger.warning("Using first JSON object as fallback")
                if json_objects[0]:
                    return json_objects[0]
                # If first is empty, try to combine all
                combined = {}
                for obj in json_objects:
                    if "data" in obj and isinstance(obj["data"], dict):
                        if "data" not in combined:
                            combined["data"] = {}
                        combined["data"].update({k: v for k, v in obj["data"].items() if v is not None})
                return combined if combined else json_objects[0] if json_objects else {}
        else:
            # No JSON objects found - try aggressive extraction
            logger.warning(f"Failed to extract JSON objects, trying aggressive extraction. Text preview: {text[:200]}")
            import re
            # Try to find JSON object even if surrounded by text
            # Look for { ... } pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    # Try to fix common issues
                    fixed_json = re.sub(r',\s*}', '}', json_str)
                    fixed_json = re.sub(r',\s*]', ']', fixed_json)
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
            
            # Last resort: try to extract just the JSON part by finding first { and last }
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = text[first_brace:last_brace + 1]
                try:
                    fixed_json = re.sub(r',\s*}', '}', json_str)
                    fixed_json = re.sub(r',\s*]', ']', fixed_json)
                    return json.loads(fixed_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Even aggressive extraction failed: {e}. JSON string: {json_str[:200]}")
            
            logger.error(f"All JSON repair attempts failed. Full text length: {len(text)} chars, preview: {text[:500]}")
            return None

    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract all valid JSON objects from text."""
        objects = []
        remaining_text = text.strip()
        max_iterations = 100  # Prevent infinite loops
        iteration = 0

        while remaining_text and iteration < max_iterations:
            iteration += 1
            
            # Find next JSON object
            start_idx = remaining_text.find("{")
            if start_idx == -1:
                break

            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(remaining_text)):
                char = remaining_text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break

            if brace_count != 0:
                # Unmatched braces - try to parse what we have anyway
                logger.warning(f"Unmatched braces in JSON extraction, attempting to parse anyway")
                end_idx = len(remaining_text) - 1

            json_str = remaining_text[start_idx:end_idx + 1]

            # Try to parse this JSON object
            try:
                obj = json.loads(json_str)
                objects.append(obj)
                logger.debug(f"Successfully parsed JSON object {len(objects)}")
            except json.JSONDecodeError as e:
                # Try to fix common issues
                import re
                fixed_json = re.sub(r',\s*}', '}', json_str)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                try:
                    obj = json.loads(fixed_json)
                    objects.append(obj)
                    logger.debug(f"Successfully parsed JSON object {len(objects)} after repair")
                except json.JSONDecodeError:
                    # Skip this malformed object but log it
                    logger.warning(f"Failed to parse JSON object: {json_str[:100]}... Error: {e}")
                    # Still try to continue to next object

            # Move to next part of text
            remaining_text = remaining_text[end_idx + 1:].strip()

            # Stop if no more text or no more JSON indicators
            if not remaining_text or "{" not in remaining_text:
                break

        if iteration >= max_iterations:
            logger.warning(f"Reached max iterations ({max_iterations}) while extracting JSON objects")

        logger.info(f"Extracted {len(objects)} JSON objects from text")
        return objects

    def _merge_json_objects(self, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Merge multiple JSON objects into a single cohesive response."""
        if not objects:
            logger.warning("No objects to merge")
            return None

        logger.debug(f"Merging {len(objects)} JSON objects")

        # For metadata extraction, we expect objects with "data" and "evidence" keys
        # Merge data and evidence from all objects
        merged_data = {}
        merged_evidence = {}

        for idx, obj in enumerate(objects):
            logger.debug(f"Processing object {idx + 1}: {list(obj.keys())}")
            
            # Merge data
            if "data" in obj and isinstance(obj["data"], dict):
                for key, value in obj["data"].items():
                    # Skip null/None values - they indicate the field wasn't found in this chunk
                    if value is None:
                        logger.debug(f"Skipping null value for key '{key}'")
                        continue
                    
                    if key not in merged_data:
                        merged_data[key] = value
                        logger.debug(f"Added data key '{key}': {value}")
                    # If we have multiple values for the same key, keep the first non-empty one
                    elif not merged_data[key] and value:
                        merged_data[key] = value
                        logger.debug(f"Updated data key '{key}' with non-empty value: {value}")
                    else:
                        logger.debug(f"Skipping duplicate data key '{key}' (keeping first value: {merged_data[key]})")

            # Merge evidence
            if "evidence" in obj and isinstance(obj["evidence"], dict):
                for key, value in obj["evidence"].items():
                    # Skip null or empty evidence arrays
                    if value is None or (isinstance(value, list) and len(value) == 0):
                        logger.debug(f"Skipping empty evidence for key '{key}'")
                        continue
                    
                    if key not in merged_evidence:
                        merged_evidence[key] = value
                        logger.debug(f"Added evidence key '{key}' with {len(value) if isinstance(value, list) else 1} items")
                    elif isinstance(value, list) and isinstance(merged_evidence[key], list):
                        # Combine evidence arrays
                        before_len = len(merged_evidence[key])
                        merged_evidence[key].extend(value)
                        logger.debug(f"Extended evidence key '{key}': {before_len} -> {len(merged_evidence[key])} items")
                    else:
                        logger.debug(f"Skipping evidence key '{key}' (type mismatch)")

        if merged_data or merged_evidence:
            result = {}
            if merged_data:
                result["data"] = merged_data
            if merged_evidence:
                result["evidence"] = merged_evidence
            logger.info(f"Successfully merged JSON objects. Data keys: {list(merged_data.keys())}, Evidence keys: {list(merged_evidence.keys())}")
            return result
        
        # Special case: if we have exactly 2 objects, try to combine them directly
        # This handles the case where LLM returns {"data": {...}} and {"evidence": {...}} separately
        if len(objects) == 2:
            obj1, obj2 = objects[0], objects[1]
            combined = {}
            
            # Check if one has data and the other has evidence
            if "data" in obj1 and "evidence" in obj2:
                combined["data"] = obj1["data"]
                combined["evidence"] = obj2["evidence"]
                logger.info("Combined two objects: first has data, second has evidence")
            elif "evidence" in obj1 and "data" in obj2:
                combined["data"] = obj2["data"]
                combined["evidence"] = obj1["evidence"]
                logger.info("Combined two objects: first has evidence, second has data")
            else:
                # Try to merge all keys from both objects
                combined = {**obj1, **obj2}
                logger.info(f"Combined two objects by merging all keys: {list(combined.keys())}")
            
            if combined:
                return combined

        # Fallback: return the first object if merging fails
        logger.warning("Merging produced empty result, returning first object as fallback")
        return objects[0]

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.3) -> Tuple[str, float]:
        """Generate plain text response from LLM. Returns (response_text, duration_seconds)."""
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Temporarily override max_tokens if provided
        original_max_tokens = self.max_tokens
        if max_tokens:
            self.max_tokens = max_tokens
        
        try:
            response_text, _, duration = await self._make_request(messages, temperature=temperature)
            return response_text.strip(), duration
        finally:
            # Restore original max_tokens
            if max_tokens:
                self.max_tokens = original_max_tokens

    async def generate_json(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate JSON response from LLM with validation."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds ONLY with valid JSON. Do not include any explanatory text, markdown formatting, code blocks, or multiple JSON objects. Always respond with exactly ONE complete JSON object that matches the requested schema. If extracting data from multiple pages, combine all results into a single JSON response."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if schema:
            messages[0]["content"] += f"\n\nExpected JSON schema: {json.dumps(schema, indent=2)}"

        logger.info(f"Sending JSON request to LLM (model: {self.model})")
        try:
            response_text, curl_command, duration = await self._make_request(messages)
            logger.info(f"Received LLM response ({len(response_text)} chars)")

            # Try to parse JSON
            try:
                result = json.loads(response_text)
                logger.info("JSON parsing successful")
            except json.JSONDecodeError:
                logger.warning(f"Initial JSON parse failed, attempting repair: {response_text[:200]}...")
                result = self._repair_json(response_text)

                if result is None:
                    raise LLMClientError(f"Failed to parse JSON response: {response_text[:500]}")

            if schema and not self._validate_schema(result, schema):
                raise LLMClientError(f"Response does not match expected schema: {result}")

            return result

        except Exception as e:
            logger.error(f"LLM JSON generation failed: {e}")
            raise

    async def generate_json_with_raw(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str, str, float]:
        """Generate JSON response from LLM, returning parsed JSON, raw response text, curl command, and duration in seconds."""
        messages = [
            {
                "role": "system",
                "content": "You are a document analysis assistant. Extract information from the provided document text and respond with valid JSON only. Always extract REAL values from the document - never use placeholder text. Respond with a single complete JSON object, no explanations or markdown. IMPORTANT: Only extract the fields that are explicitly listed in the prompt - do not add fields that are not mentioned."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if schema:
            # Extract field names from schema for clarity
            field_names = []
            if isinstance(schema, dict) and "properties" in schema:
                data_props = schema.get("properties", {}).get("data", {}).get("properties", {})
                field_names = list(data_props.keys())
            
            schema_note = f"\n\nExpected JSON schema: {json.dumps(schema, indent=2)}"
            if field_names:
                schema_note += f"\n\nREQUIRED FIELDS to extract: {', '.join(field_names)}"
            messages[0]["content"] += schema_note

        response_text, curl_command, duration = await self._make_request(messages)

        # Pre-process: remove any instruction text that the LLM might have echoed
        import re
        cleaned_response = response_text
        instruction_patterns = [
            r'"evidence"\s+field\s+MUST\s+be\s+[^"]*',  # "evidence" field MUST be...
            r'field\s+MUST\s+be\s+[^"]*(?=\s*"evidence"|$)',  # field MUST be...
        ]
        for pattern in instruction_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE)
        
        if cleaned_response != response_text:
            logger.info("Pre-cleaned instruction text from LLM response")
            # Try to find and keep only valid JSON structure
            # Look for the pattern: {"data": {...}} or {"data": {...}, "evidence": {...}}
            json_start = cleaned_response.find('{')
            if json_start >= 0:
                cleaned_response = cleaned_response[json_start:]

        # Try to parse JSON
        result = None  # Initialize to avoid "referenced before assignment" errors
        try:
            result = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Try with original response if cleaned version failed
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Check if response has preamble text before the JSON
                # Try to find and extract the JSON object
                first_brace = response_text.find('{')
                if first_brace > 0:
                    # There's text before the JSON - try to extract just the JSON part
                    logger.info(f"Found preamble text before JSON (starts at char {first_brace}), extracting JSON")
                    json_part = response_text[first_brace:]
                    # Find matching closing brace
                    brace_count = 0
                    end_pos = -1
                    for i, char in enumerate(json_part):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i
                                break
                    if end_pos > 0:
                        json_str = json_part[:end_pos + 1]
                        try:
                            result = json.loads(json_str)
                            logger.info("Successfully extracted JSON from response with preamble")
                        except json.JSONDecodeError:
                            result = None  # Fall through to repair
                    else:
                        result = None
                
                # Check if response looks like explanation text rather than JSON
                # If it starts with bullet points, dashes, or explanation text, try to extract JSON from it
                if result is None and response_text.strip().startswith(('- ', '* ', '• ', 'If you', 'Example', 'Note:', 'IMPORTANT:', 'Here')):
                    logger.warning(f"Response appears to be explanation text, attempting to extract JSON from it")
                    # Try to find JSON object in the text
                    # Look for JSON object pattern
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    json_matches = re.finditer(json_pattern, response_text, re.DOTALL)
                    json_candidates = []
                    for match in json_matches:
                        try:
                            candidate = json.loads(match.group(0))
                            if isinstance(candidate, dict) and ('data' in candidate or 'evidence' in candidate):
                                json_candidates.append((len(match.group(0)), candidate))
                        except:
                            pass
                    
                    if json_candidates:
                        # Use the largest valid JSON object found
                        json_candidates.sort(reverse=True, key=lambda x: x[0])
                        result = json_candidates[0][1]
                        logger.info(f"Extracted JSON object from explanation text ({len(json_candidates)} candidates found)")
                    else:
                        # Fall through to repair_json
                        result = None
                
                if result is None:
                    # Show more context around the error
                    error_pos = getattr(e, 'pos', None)
                    if error_pos:
                        start = max(0, error_pos - 100)
                        end = min(len(response_text), error_pos + 100)
                        context = response_text[start:end]
                        logger.warning(f"Initial JSON parse failed at position {error_pos}: {e}")
                        logger.warning(f"Context around error: ...{context}...")
                    else:
                        logger.warning(f"Initial JSON parse failed: {e}. Response preview: {response_text[:300]}...")
                    logger.debug(f"Full response text ({len(response_text)} chars): {response_text}")
                    result = self._repair_json(response_text)

                if result is None:
                    logger.error(f"All JSON repair attempts failed. Response length: {len(response_text)} chars")
                    logger.error(f"Response text (first 1000 chars): {response_text[:1000]}")
                    raise LLMClientError(f"Failed to parse JSON response: {response_text[:500]}")

        # Post-process the result to fix common LLM output issues
        if isinstance(result, dict):
            # Fix: Convert "null" strings to actual null values
            if "data" in result and isinstance(result["data"], dict):
                for key, value in result["data"].items():
                    if value == "null" or value == "NULL":
                        result["data"][key] = None
                        logger.debug(f"Converted string 'null' to null for field '{key}'")
            
            # Fix: If evidence is nested inside data, move it out
            if "data" in result and isinstance(result["data"], dict):
                if "evidence" in result["data"] and "evidence" not in result:
                    result["evidence"] = result["data"].pop("evidence")
                    logger.info("Moved evidence from inside data to top level")
                    # If evidence was a string "null", convert to empty object
                    if result["evidence"] == "null" or result["evidence"] is None:
                        result["evidence"] = {}
            
            # Fix: Ensure evidence exists as an object
            if "evidence" not in result:
                result["evidence"] = {}
                logger.info("Added missing evidence object")
            elif result["evidence"] == "null" or result["evidence"] is None:
                result["evidence"] = {}
                logger.info("Converted null evidence to empty object")
        
        # Repair evidence structure if needed (LLM sometimes returns array instead of object)
        if isinstance(result, dict) and "evidence" in result:
            if isinstance(result["evidence"], list):
                logger.warning("LLM returned evidence as array instead of object, attempting repair")
                evidence_obj = {}
                data_fields = result.get("data", {})
                
                # Try to match evidence items to data fields by comparing quote text with data values
                evidence_array = result["evidence"]
                matched_evidence = set()
                
                for field_name, field_value in data_fields.items():
                    if field_value is None:
                        evidence_obj[field_name] = []
                        continue
                    
                    # Convert field value to string for comparison
                    field_value_str = str(field_value).strip()
                    evidence_obj[field_name] = []
                    
                    # Try to find evidence items that match this field value
                    for idx, item in enumerate(evidence_array):
                        if idx in matched_evidence:
                            continue
                        if isinstance(item, dict):
                            quote = item.get("quote", "").strip()
                            # Check if quote contains or matches the field value
                            if field_value_str.lower() in quote.lower() or quote.lower() in field_value_str.lower():
                                evidence_obj[field_name].append(item)
                                matched_evidence.add(idx)
                
                # Create empty evidence arrays for any remaining data fields
                for field_name in data_fields.keys():
                    if field_name not in evidence_obj:
                        evidence_obj[field_name] = []
                
                result["evidence"] = evidence_obj
                logger.info(f"Repaired evidence structure: {len(evidence_obj)} field keys, matched {len(matched_evidence)}/{len(evidence_array)} evidence items")

        if schema and not self._validate_schema(result, schema):
            raise LLMClientError(f"Response does not match expected schema: {result}")

        return result, response_text, curl_command, duration

    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Basic schema validation for LLM responses."""
        if not isinstance(data, dict):
            return False

        required_keys = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required keys
        for key in required_keys:
            if key not in data:
                return False

        # Check property types
        for key, value in data.items():
            if key in properties:
                prop_schema = properties[key]
                prop_type = prop_schema.get("type")
                if prop_type == "string" and not isinstance(value, str):
                    return False
                elif prop_type == "number" and not isinstance(value, (int, float)):
                    return False
                elif prop_type == "integer" and not isinstance(value, int):
                    return False
                elif prop_type == "boolean" and not isinstance(value, bool):
                    return False
                elif prop_type == "array" and not isinstance(value, list):
                    return False
                elif prop_type == "object" and not isinstance(value, dict):
                    return False

        return True

    async def check_health(self) -> bool:
        """Check if the LLM provider is reachable and model is available."""
        try:
            if self.provider == "vllm":
                # vLLM uses OpenAI-compatible /v1/models endpoint
                url = f"{self.base_url}/v1/models"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    data = response.json()
                    models = data.get("data", [])
                    model_ids = [model.get("id") for model in models]
                    
                    logger.info(f"vLLM available models: {model_ids}")
                    logger.info(f"Configured model: '{self.model}' - found: {self.model in model_ids}")
                    
                    return self.model in model_ids
            else:
                # Ollama uses /api/tags endpoint
                url = f"{self.base_url}/api/tags"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    data = response.json()
                    models = data.get("models", [])
                    model_names = [model.get("name") for model in models]

                    logger.info(f"Ollama available models: {model_names}")
                    logger.info(f"Configured model: '{self.model}' - found: {self.model in model_names}")

                    return self.model in model_names

        except Exception as e:
            logger.error(f"{self.provider} health check failed: {e}")
            return False