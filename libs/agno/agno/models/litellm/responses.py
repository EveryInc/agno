import json
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type, Union

from pydantic import BaseModel
from typing_extensions import Literal

from agno.exceptions import ModelProviderError
from agno.media import File
from agno.models.base import MessageData, Model, _add_usage_metrics_to_assistant_message
from agno.models.message import Citations, Message, UrlCitation
from agno.models.response import ModelResponse
from agno.utils.log import log_debug, log_error, log_warning
from agno.utils.models.schema_utils import get_response_schema_for_provider
from agno.utils.openai import _format_file_for_message, audio_to_message, images_to_message

try:
    import litellm
    from litellm import validate_environment
except ImportError:
    raise ImportError("`litellm` not installed. Please install it via `pip install litellm`")


@dataclass
class LiteLLMResponses(Model):
    """
    A class for interacting with LiteLLM models using the Responses API.

    LiteLLM Responses API allows you to use a unified interface for various LLM providers
    with advanced features like reasoning models and session continuity.
    For more information, see: https://docs.litellm.ai/docs/response_api
    """

    id: str = "gpt-4o"
    name: str = "LiteLLMResponses"
    provider: str = "LiteLLM"
    supports_native_structured_outputs: bool = True

    # LiteLLM client parameters
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Dict[str, str]] = None
    client_params: Optional[Dict[str, Any]] = None

    # Responses API specific parameters
    include: Optional[List[str]] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Dict[str, Any]] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    store: Optional[bool] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    truncation: Optional[Literal["auto", "disabled"]] = None
    user: Optional[str] = None
    service_tier: Optional[Literal["auto", "default", "flex", "priority"]] = None
    request_params: Optional[Dict[str, Any]] = None

    # Parameters affecting built-in tools
    vector_store_name: str = "knowledge_base"

    client: Optional[Any] = None

    # The role to map the message role to.
    role_map: Dict[str, str] = field(
        default_factory=lambda: {
            "system": "developer",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
        }
    )

    def __post_init__(self):
        """Initialize the model after the dataclass initialization."""
        super().__post_init__()

        # Set up API key from environment variable if not already set
        if not self.api_key:
            self.api_key = getenv("LITELLM_API_KEY")
            if not self.api_key:
                # Check for other present valid keys, e.g. OPENAI_API_KEY if self.id is an OpenAI model
                env_validation = validate_environment(model=self.id, api_base=self.api_base)
                if not env_validation.get("keys_in_environment"):
                    log_warning(
                        "Missing required key. Please set the LITELLM_API_KEY or other valid environment variables."
                    )

        # Validate model-specific parameters
        self._validate_parameters()

    def _using_reasoning_model(self) -> bool:
        """Return True if the contextual used model is a known reasoning model."""
        return self.id.startswith("o3") or self.id.startswith("o4-mini") or self.id.startswith("gpt-5")

    def _validate_parameters(self) -> None:
        """Validate model parameters and fix common configuration issues."""
        from agno.utils.log import log_warning
        
        # GPT-5 specific validations
        if self.id.startswith("gpt-5"):
            if self.temperature is not None:
                log_warning("GPT-5 models don't support temperature parameter, ignoring")
                self.temperature = None
        
        # Claude specific validations
        if self.id.startswith("claude-"):
            # Check for thinking configuration
            thinking_enabled = (
                self.request_params is not None 
                and "thinking" in self.request_params
                and self.request_params["thinking"].get("type") == "enabled"
            )
            
            if thinking_enabled:
                # When thinking is enabled, temperature should be 1.0
                if self.temperature != 1.0:
                    log_warning("Claude thinking requires temperature=1.0, adjusting automatically")
                    self.temperature = 1.0
                
                # Don't use top_p with temperature for Claude
                if self.top_p is not None:
                    log_warning("Claude doesn't work well with both temperature and top_p, setting top_p=None")
                    self.top_p = None
        
        # General parameter conflict checks
        if self.temperature is not None and self.top_p is not None:
            if not self.id.startswith("gpt-"):
                log_warning("Using both temperature and top_p may cause unexpected behavior with some models")
        
        # Reasoning effort validation
        if self.reasoning_effort is not None:
            valid_efforts = ["low", "medium", "high"]
            if self.reasoning_effort not in valid_efforts:
                log_warning(f"Invalid reasoning_effort '{self.reasoning_effort}', using 'medium'")
                self.reasoning_effort = "medium"

    def get_client(self) -> Any:
        """
        Returns a LiteLLM client.

        Returns:
            Any: An instance of the LiteLLM client.
        """
        if self.client is not None:
            return self.client

        self.client = litellm
        return self.client

    def get_request_params(
        self,
        messages: Optional[List[Message]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        # Define base request parameters
        base_params: Dict[str, Any] = {
            "model": self.id,
            "include": self.include,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "metadata": self.metadata,
            "parallel_tool_calls": self.parallel_tool_calls,
            "store": self.store,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "truncation": self.truncation,
            "user": self.user,
            "service_tier": self.service_tier,
        }

        # Add LiteLLM specific params
        if self.api_key:
            base_params["api_key"] = self.api_key
        if self.api_base:
            base_params["api_base"] = self.api_base
        if self.timeout:
            base_params["timeout"] = self.timeout
        if self.max_retries:
            base_params["max_retries"] = self.max_retries
        if self.default_headers:
            base_params["default_headers"] = self.default_headers

        # Handle reasoning parameter - convert reasoning_effort to reasoning format
        if self.reasoning is not None:
            base_params["reasoning"] = self.reasoning
        elif self.reasoning_effort is not None:
            base_params["reasoning"] = {"effort": self.reasoning_effort}

        # Build text parameter
        text_params: Dict[str, Any] = {}

        # Add verbosity if specified
        if self.verbosity is not None:
            text_params["verbosity"] = self.verbosity

        # Set the response format
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                schema = get_response_schema_for_provider(response_format, "openai")
                text_params["format"] = {
                    "type": "json_schema",
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                }
            else:
                # JSON mode
                text_params["format"] = {"type": "json_object"}

        # Add text parameter if there are any text-level params
        if text_params:
            base_params["text"] = text_params

        # Filter out None values
        request_params: Dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}

        # Deep research models require web_search_preview tool or MCP tool
        if "deep-research" in self.id:
            if tools is None:
                tools = []

            # Check if web_search_preview tool is already present
            has_web_search = any(tool.get("type") == "web_search_preview" for tool in tools)

            # Add web_search_preview if not present - this enables the model to search
            # the web for current information and provide citations
            if not has_web_search:
                web_search_tool = {"type": "web_search_preview"}
                tools.insert(0, web_search_tool)
                log_debug(f"Added web_search_preview tool for deep research model: {self.id}")

        if tools:
            request_params["tools"] = self._format_tool_params(messages=messages or [], tools=tools)

        if tool_choice is not None:
            request_params["tool_choice"] = tool_choice

        # Handle reasoning tools for o3 and o4-mini models
        if self._using_reasoning_model() and messages is not None:
            request_params["store"] = True

            # Check if the last assistant message has a previous_response_id to continue from
            previous_response_id = None
            for msg in reversed(messages):
                if (
                    msg.role == "assistant"
                    and hasattr(msg, "provider_data")
                    and msg.provider_data
                    and "response_id" in msg.provider_data
                ):
                    previous_response_id = msg.provider_data["response_id"]
                    log_debug(f"Using previous_response_id: {previous_response_id}")
                    break

            if previous_response_id:
                request_params["previous_response_id"] = previous_response_id

        # Add additional request params if provided
        if self.request_params:
            request_params.update(self.request_params)

        if request_params:
            log_debug(f"Calling {self.provider} with request parameters: {request_params}", log_level=2)
        return request_params

    def _upload_file(self, file: File) -> Optional[str]:
        """Upload a file to the LiteLLM vector database (if supported)."""
        # Note: File upload may not be supported in all LiteLLM backends
        # This is a placeholder for potential future implementation
        log_warning("File upload not yet implemented for LiteLLM Responses API")
        return None

    def _create_vector_store(self, file_ids: List[str]) -> str:
        """Create a vector store for the files (if supported)."""
        # Note: Vector store creation may not be supported in all LiteLLM backends
        # This is a placeholder for potential future implementation
        log_warning("Vector store creation not yet implemented for LiteLLM Responses API")
        return "default_vector_store"

    def _format_tool_params(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Format the tool parameters for the LiteLLM Responses API."""
        formatted_tools = []
        if tools:
            for _tool in tools:
                if _tool.get("type") == "function":
                    _tool_dict = _tool.get("function", {})
                    _tool_dict["type"] = "function"
                    # Fix type arrays to single types (common issue with OpenAPI schemas)
                    for prop in _tool_dict.get("parameters", {}).get("properties", {}).values():
                        if isinstance(prop.get("type", ""), list):
                            prop["type"] = prop["type"][0]

                    formatted_tools.append(_tool_dict)
                else:
                    formatted_tools.append(_tool)

        # Note: File handling and vector stores may need different implementation for LiteLLM
        # For now, we'll handle files in the message formatting

        return formatted_tools

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Format messages for the LiteLLM Responses API.

        Args:
            messages (List[Message]): The messages to format.

        Returns:
            List[Dict[str, Any]]: The formatted messages.
        """
        formatted_messages: List[Dict[str, Any]] = []

        if self._using_reasoning_model():
            # Detect whether we're chaining via previous_response_id. If so, we should NOT
            # re-send prior function_call items; the Responses API already has the state and
            # expects only the corresponding function_call_output items.
            previous_response_id: Optional[str] = None
            for msg in reversed(messages):
                if (
                    msg.role == "assistant"
                    and hasattr(msg, "provider_data")
                    and msg.provider_data
                    and "response_id" in msg.provider_data
                ):
                    previous_response_id = msg.provider_data["response_id"]
                    break

        # Build a mapping from function_call id (fc_*) → call_id (call_*) from prior assistant tool_calls
        fc_id_to_call_id: Dict[str, str] = {}
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    fc_id = tc.get("id")
                    call_id = tc.get("call_id") or fc_id
                    if isinstance(fc_id, str) and isinstance(call_id, str):
                        fc_id_to_call_id[fc_id] = call_id

        for message in messages:
            if message.role in ["user", "system"]:
                message_dict: Dict[str, Any] = {
                    "role": self.role_map[message.role],
                    "content": message.content,
                }
                message_dict = {k: v for k, v in message_dict.items() if v is not None}

                # Handle multimodal content
                if message.images is not None and len(message.images) > 0:
                    if isinstance(message.content, str):
                        message_dict["content"] = [{"type": "input_text", "text": message.content}]
                        if message.images is not None:
                            message_dict["content"].extend(images_to_message(images=message.images))

                if message.audio is not None and len(message.audio) > 0:
                    log_warning("Audio input may not be supported by all LiteLLM providers.")

                if message.videos is not None and len(message.videos) > 0:
                    log_warning("Video input may not be supported by all LiteLLM providers.")

                # Handle files
                if message.files is not None and len(message.files) > 0:
                    if isinstance(message_dict["content"], str):
                        content_list = [{"type": "input_text", "text": message_dict["content"]}]
                    else:
                        content_list = message_dict["content"]
                    
                    for file in message.files:
                        file_part = _format_file_for_message(file)
                        if file_part:
                            content_list.append(file_part)
                    message_dict["content"] = content_list

                formatted_messages.append(message_dict)

            # Tool call result
            elif message.role == "tool":
                if message.tool_call_id and message.content is not None:
                    function_call_id = message.tool_call_id
                    # Normalize: if a fc_* id was provided, translate to its corresponding call_* id
                    if isinstance(function_call_id, str) and function_call_id in fc_id_to_call_id:
                        call_id_value = fc_id_to_call_id[function_call_id]
                    else:
                        call_id_value = function_call_id
                    formatted_messages.append(
                        {"type": "function_call_output", "call_id": call_id_value, "output": message.content}
                    )
            # Tool Calls
            elif message.tool_calls is not None and len(message.tool_calls) > 0:
                # Only skip re-sending prior function_call items when we have a previous_response_id
                # (reasoning models). For non-reasoning models, we must include the prior function_call
                # so the API can associate the subsequent function_call_output by call_id.
                if self._using_reasoning_model() and hasattr(self, '_has_previous_response_id') and self._has_previous_response_id:
                    continue

                for tool_call in message.tool_calls:
                    formatted_messages.append(
                        {
                            "type": "function_call",
                            "id": tool_call.get("id"),
                            "call_id": tool_call.get("call_id", tool_call.get("id")),
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                            "status": "completed",
                        }
                    )
            elif message.role == "assistant":
                # Handle null content by converting to empty string
                content = message.content if message.content is not None else ""
                formatted_messages.append({"role": self.role_map[message.role], "content": content})

        return formatted_messages

    def invoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Send a request to the LiteLLM Responses API.
        """
        try:
            request_params = self.get_request_params(
                messages=messages, response_format=response_format, tools=tools, tool_choice=tool_choice
            )

            # Use litellm.responses() method - this is the LiteLLM wrapper for responses API
            return self.get_client().responses(
                input=self._format_messages(messages),
                **request_params,
            )
        except Exception as exc:
            log_error(f"Error from LiteLLM Responses API: {exc}")
            raise ModelProviderError(message=str(exc), model_name=self.name, model_id=self.id) from exc

    async def ainvoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Sends an asynchronous request to the LiteLLM Responses API.
        """
        try:
            request_params = self.get_request_params(
                messages=messages, response_format=response_format, tools=tools, tool_choice=tool_choice
            )

            # Use litellm.aresponses() method for async responses
            return await self.get_client().aresponses(
                input=self._format_messages(messages),
                **request_params,
            )
        except Exception as exc:
            log_error(f"Error from LiteLLM Responses API: {exc}")
            raise ModelProviderError(message=str(exc), model_name=self.name, model_id=self.id) from exc

    def invoke_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Iterator[Any]:
        """
        Send a streaming request to the LiteLLM Responses API.
        """
        try:
            request_params = self.get_request_params(
                messages=messages, response_format=response_format, tools=tools, tool_choice=tool_choice
            )

            # Enable streaming
            request_params["stream"] = True

            yield from self.get_client().responses(
                input=self._format_messages(messages),
                **request_params,
            )
        except Exception as exc:
            log_error(f"Error from LiteLLM Responses API: {exc}")
            raise ModelProviderError(message=str(exc), model_name=self.name, model_id=self.id) from exc

    async def ainvoke_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[Any]:
        """
        Sends an asynchronous streaming request to the LiteLLM Responses API.
        """
        try:
            request_params = self.get_request_params(
                messages=messages, response_format=response_format, tools=tools, tool_choice=tool_choice
            )
            
            # Enable streaming
            request_params["stream"] = True

            async_stream = await self.get_client().aresponses(
                input=self._format_messages(messages),
                **request_params,
            )
            async for chunk in async_stream:
                yield chunk
        except Exception as exc:
            log_error(f"Error from LiteLLM Responses API: {exc}")
            raise ModelProviderError(message=str(exc), model_name=self.name, model_id=self.id) from exc

    def format_function_call_results(
        self, messages: List[Message], function_call_results: List[Message], tool_call_ids: List[str]
    ) -> None:
        """
        Handle the results of function calls.

        Args:
            messages (List[Message]): The list of conversation messages.
            function_call_results (List[Message]): The results of the function calls.
            tool_call_ids (List[str]): The tool call ids.
        """
        if len(function_call_results) > 0:
            for _fc_message_index, _fc_message in enumerate(function_call_results):
                if _fc_message_index < len(tool_call_ids):
                    _fc_message.tool_call_id = tool_call_ids[_fc_message_index]
                messages.append(_fc_message)

    def parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        """
        Parse the LiteLLM response into a ModelResponse.

        Args:
            response: Response from invoke() method (LiteLLM ResponsesAPIResponse)

        Returns:
            ModelResponse: Parsed response data
        """
        model_response = ModelResponse()

        # Check for error in response
        if hasattr(response, 'error') and response.error:
            raise ModelProviderError(
                message=str(response.error),
                model_name=self.name,
                model_id=self.id,
            )

        # Store the response ID for continuity (if available)
        if hasattr(response, 'id') and response.id:
            if model_response.provider_data is None:
                model_response.provider_data = {}
            model_response.provider_data["response_id"] = response.id

        # Add role
        model_response.role = "assistant"

        # Parse LiteLLM ResponsesAPIResponse format
        if hasattr(response, 'output') and response.output:
            for output in response.output:
                if hasattr(output, 'type'):
                    if output.type == "message":
                        # Handle message content - this is the main response
                        if hasattr(output, 'content') and output.content:
                            # Extract text from content array
                            text_parts = []
                            for content_item in output.content:
                                if hasattr(content_item, 'type') and content_item.type == "output_text":
                                    if hasattr(content_item, 'text'):
                                        text_parts.append(content_item.text)
                            
                            model_response.content = ''.join(text_parts)

                            # Handle citations/annotations if available
                            citations = Citations()
                            for content_item in output.content:
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    citations.raw = [annotation.model_dump() if hasattr(annotation, 'model_dump') else annotation for annotation in content_item.annotations]
                                    for annotation in content_item.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == "url_citation":
                                            if citations.urls is None:
                                                citations.urls = []
                                            citations.urls.append(UrlCitation(
                                                url=getattr(annotation, 'url', ''),
                                                title=getattr(annotation, 'title', '')
                                            ))
                            if citations.urls or citations.raw:
                                model_response.citations = citations

                    elif output.type == "reasoning":
                        # Handle reasoning content (both GPT-5 and Claude formats)
                        reasoning_parts = []
                        
                        # GPT-5 format: has summary field
                        if hasattr(output, 'summary') and output.summary:
                            for summary_item in output.summary:
                                if hasattr(summary_item, 'text'):
                                    reasoning_parts.append(summary_item.text)
                        
                        # Claude format: has content field like messages
                        elif hasattr(output, 'content') and output.content:
                            for content_item in output.content:
                                if hasattr(content_item, 'type') and content_item.type == "output_text":
                                    if hasattr(content_item, 'text'):
                                        reasoning_parts.append(content_item.text)
                        
                        if reasoning_parts:
                            model_response.reasoning_content = '\n'.join(reasoning_parts)

                    elif output.type == "function_call":
                        # Handle function calls
                        if model_response.tool_calls is None:
                            model_response.tool_calls = []
                        model_response.tool_calls.append(
                            {
                                "id": getattr(output, 'id', ''),
                                "call_id": getattr(output, 'call_id', getattr(output, 'id', '')),
                                "type": "function",
                                "function": {
                                    "name": getattr(output, 'name', ''),
                                    "arguments": getattr(output, 'arguments', ''),
                                },
                            }
                        )

                        model_response.extra = model_response.extra or {}
                        model_response.extra.setdefault("tool_call_ids", []).append(getattr(output, 'call_id', getattr(output, 'id', '')))

        elif hasattr(response, 'choices') and response.choices:
            # Fallback: Standard chat completion style response
            choice = response.choices[0]
            message = choice.message if hasattr(choice, 'message') else choice
            
            if hasattr(message, 'content') and message.content is not None:
                model_response.content = message.content

            if hasattr(message, 'tool_calls') and message.tool_calls:
                model_response.tool_calls = []
                for tool_call in message.tool_calls:
                    model_response.tool_calls.append(
                        {
                            "id": getattr(tool_call, 'id', ''),
                            "type": "function",
                            "function": {
                                "name": getattr(tool_call.function, 'name', '') if hasattr(tool_call, 'function') else '',
                                "arguments": getattr(tool_call.function, 'arguments', '') if hasattr(tool_call, 'function') else '',
                            },
                        }
                    )

        else:
            # Fallback: Simple string or direct response
            if isinstance(response, str):
                model_response.content = response
            elif hasattr(response, 'output_text'):
                model_response.content = response.output_text
            elif hasattr(response, 'content'):
                model_response.content = response.content

        # Add usage metrics if available
        if hasattr(response, 'usage') and response.usage is not None:
            model_response.response_usage = response.usage

        return model_response

    def _process_stream_response(
        self,
        stream_event: Any,
        assistant_message: Message,
        stream_data: MessageData,
        tool_use: Dict[str, Any],
    ) -> Tuple[Optional[ModelResponse], Dict[str, Any]]:
        """
        Common handler for processing stream responses from LiteLLM Responses API.

        Args:
            stream_event: The streamed response from LiteLLM
            assistant_message: The assistant message being built
            stream_data: Data accumulated during streaming
            tool_use: Current tool use data being built

        Returns:
            Tuple containing the ModelResponse to yield and updated tool_use dict
        """
        model_response = None

        # Handle different types of stream events
        # The exact event types may vary based on the LiteLLM provider
        if hasattr(stream_event, 'type'):
            if stream_event.type == "response.created":
                model_response = ModelResponse()
                # Store the response ID for continuity
                if hasattr(stream_event, 'response') and hasattr(stream_event.response, 'id') and stream_event.response.id:
                    if stream_data.response_provider_data is None:
                        stream_data.response_provider_data = {}
                    stream_data.response_provider_data["response_id"] = stream_event.response.id
                # Update metrics
                if not assistant_message.metrics.time_to_first_token:
                    assistant_message.metrics.set_time_to_first_token()

            elif stream_event.type in ["response.output_text.delta", "content_delta", "text_delta"]:
                model_response = ModelResponse()
                # Extract delta content
                delta_content = ""
                if hasattr(stream_event, 'delta'):
                    delta_content = stream_event.delta
                elif hasattr(stream_event, 'content'):
                    delta_content = stream_event.content
                elif hasattr(stream_event, 'text'):
                    delta_content = stream_event.text

                model_response.content = delta_content
                stream_data.response_content += delta_content

                if self.reasoning is not None:
                    model_response.reasoning_content = delta_content
                    stream_data.response_thinking += delta_content

            elif stream_event.type == "response.output_item.added":
                if hasattr(stream_event, 'item') and hasattr(stream_event.item, 'type') and stream_event.item.type == "function_call":
                    item = stream_event.item
                    tool_use = {
                        "id": getattr(item, "id", None),
                        "call_id": getattr(item, "call_id", None) or getattr(item, "id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", ""),
                        },
                    }

            elif stream_event.type in ["response.function_call_arguments.delta", "tool_call_delta"]:
                if hasattr(stream_event, 'delta') and tool_use:
                    tool_use["function"]["arguments"] += stream_event.delta

            elif stream_event.type == "response.output_item.done" and tool_use:
                model_response = ModelResponse()
                model_response.tool_calls = [tool_use]
                if assistant_message.tool_calls is None:
                    assistant_message.tool_calls = []
                assistant_message.tool_calls.append(tool_use)

                stream_data.extra = stream_data.extra or {}
                stream_data.extra.setdefault("tool_call_ids", []).append(tool_use["call_id"])
                tool_use = {}

            elif stream_event.type in ["response.completed", "stream_end"]:
                model_response = ModelResponse()
                # Add usage metrics if present
                if hasattr(stream_event, 'response') and hasattr(stream_event.response, 'usage') and stream_event.response.usage is not None:
                    model_response.response_usage = stream_event.response.usage
                elif hasattr(stream_event, 'usage') and stream_event.usage is not None:
                    model_response.response_usage = stream_event.usage

                if model_response.response_usage:
                    _add_usage_metrics_to_assistant_message(
                        assistant_message=assistant_message,
                        response_usage=model_response.response_usage,
                    )

        # Handle delta responses that might not have explicit types
        elif hasattr(stream_event, 'choices') and stream_event.choices:
            choice = stream_event.choices[0]
            if hasattr(choice, 'delta'):
                delta = choice.delta
                model_response = ModelResponse()
                
                if hasattr(delta, 'content') and delta.content:
                    model_response.content = delta.content
                    stream_data.response_content += delta.content

                # Handle thinking blocks from Claude streaming
                if hasattr(delta, 'thinking_blocks') and delta.thinking_blocks:
                    thinking_parts = []
                    for thinking_block in delta.thinking_blocks:
                        if hasattr(thinking_block, 'thinking') and thinking_block.thinking:
                            thinking_parts.append(thinking_block.thinking)
                    
                    if thinking_parts:
                        thinking_content = '\n'.join(thinking_parts)
                        model_response.reasoning_content = thinking_content
                        stream_data.response_thinking += thinking_content

                # Handle reasoning content from GPT-5 streaming
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    model_response.reasoning_content = delta.reasoning_content
                    stream_data.response_thinking += delta.reasoning_content

                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    # Handle streaming tool calls
                    for tc_delta in delta.tool_calls:
                        if hasattr(tc_delta, 'function'):
                            if hasattr(tc_delta.function, 'arguments') and tc_delta.function.arguments:
                                tool_use["function"]["arguments"] += tc_delta.function.arguments

        return model_response, tool_use

    def process_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Iterator[ModelResponse]:
        """Process the synchronous response stream."""
        tool_use: Dict[str, Any] = {}

        for stream_event in self.invoke_stream(
            messages=messages, tools=tools, response_format=response_format, tool_choice=tool_choice
        ):
            model_response, tool_use = self._process_stream_response(
                stream_event=stream_event,
                assistant_message=assistant_message,
                stream_data=stream_data,
                tool_use=tool_use,
            )

            if model_response is not None:
                yield model_response

    async def aprocess_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[ModelResponse]:
        """Process the asynchronous response stream."""
        tool_use: Dict[str, Any] = {}

        async for stream_event in self.ainvoke_stream(
            messages=messages, tools=tools, response_format=response_format, tool_choice=tool_choice
        ):
            model_response, tool_use = self._process_stream_response(
                stream_event=stream_event,
                assistant_message=assistant_message,
                stream_data=stream_data,
                tool_use=tool_use,
            )
            if model_response is not None:
                yield model_response

    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        """
        Parse the streaming provider response delta into ModelResponse objects.

        Args:
            response: Raw response chunk from the provider

        Returns:
            ModelResponse: Parsed response delta
        """
        model_response = ModelResponse()
        
        # This method is called for each streaming chunk
        # Implementation varies based on the specific stream format from LiteLLM
        if hasattr(response, 'content'):
            model_response.content = response.content
        elif hasattr(response, 'delta'):
            model_response.content = response.delta
        elif hasattr(response, 'text'):
            model_response.content = response.text

        return model_response