import json
import asyncio
from typing import Dict, Any, Optional, List
import httpx
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class LLMClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip('/')
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.max_retries = settings.ollama_max_retries

    def _generate_curl_command(self, url: str, payload: Dict[str, Any]) -> str:
        """Generate curl command equivalent of the HTTP request."""
        json_payload = json.dumps(payload, indent=2)
        return f"curl -X POST '{url}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{json_payload}'"

    async def _make_request(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> tuple[str, str]:
        """Make a request to Ollama API with retries. Returns (response_text, curl_command)."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 8192,  # Increased to prevent truncated JSON responses (especially for metadata extraction)
                "top_p": 0.9
            }
        }

        # Generate curl command
        curl_command = self._generate_curl_command(url, payload)

        # Log detailed request information
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        logger.info(f"LLM Request to model '{self.model}' - prompt length: {len(user_message)} chars")
        logger.info(f"LLM Request (curl equivalent):\n{curl_command}")

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    if "message" not in data or "content" not in data["message"]:
                        raise LLMClientError("Invalid response format from Ollama")

                    # Log response info
                    response_content = data["message"]["content"]
                    
                    # Check if response was truncated (Ollama may set done: false or truncate silently)
                    done = data.get("done", True)
                    if not done:
                        logger.warning(f"LLM response may be incomplete (done=false). Response length: {len(response_content)} chars")
                    
                    # Check for incomplete JSON in response
                    if response_content.count('{') > response_content.count('}') or response_content.count('[') > response_content.count(']'):
                        logger.warning(f"LLM response appears truncated (unbalanced braces/brackets). Response length: {len(response_content)} chars")
                    
                    logger.info(f"LLM Response received ({len(response_content)} chars): {response_content[:200]}{'...' if len(response_content) > 200 else ''}")

                    return response_content, curl_command

            except httpx.TimeoutException:
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise LLMClientError("LLM request timed out")
                await asyncio.sleep(1)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise LLMClientError(f"Model '{self.model}' not found on Ollama server")
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

    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair malformed JSON responses, including multiple JSON objects."""
        # Remove markdown code blocks
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
                
                # If we found an opening quote but no closing quote after it, we're mid-string
                if last_quote_pos >= 0:
                    # Check if there's content after the last quote (we're in a string)
                    remaining = text[last_quote_pos+1:].strip()
                    if remaining and not remaining.endswith(('}', ']', ',')):
                        # We're definitely in a string - close it
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
            logger.debug(f"JSON objects: {json_objects}")
            merged = self._merge_json_objects(json_objects)
            if merged:
                logger.info("JSON objects successfully merged")
                logger.debug(f"Merged result: {merged}")
            else:
                logger.warning("Failed to merge JSON objects, returning first object")
                return json_objects[0]  # Return first object as fallback
            return merged
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

        # Fallback: return the first object if merging fails
        logger.warning("Merging produced empty result, returning first object as fallback")
        return objects[0]

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
            response_text, curl_command = await self._make_request(messages)
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

    async def generate_json_with_raw(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], str, str]:
        """Generate JSON response from LLM, returning parsed JSON, raw response text, and curl command."""
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

        response_text, curl_command = await self._make_request(messages)

        # Try to parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
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

        return result, response_text, curl_command

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
        """Check if Ollama is reachable and model is available."""
        try:
            url = f"{self.base_url}/api/tags"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])
                model_names = [model.get("name") for model in models]

                return self.model in model_names

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False