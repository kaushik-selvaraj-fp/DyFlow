"""Module for handling different model API clients."""

import os
from typing import Any, Dict

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

try:
    from openai import AsyncOpenAI, OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

# from vertexai.generative_models import GenerativeModel
# import google.generativeai as genai

from .config import ENV_VARS, MODEL_MAPPING
from .utils import retry_decorator
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Part,
    Content,
    GenerationConfig,
    ToolConfig,
)


# 1. Define the function
def get_gold_price(location: str, carret: str = "22K"):
    """Gets the current gold price for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA.
        unit: The carret unit, e.g. 22K, 18K.
    """
    # In a real scenario, this would call an API
    return {
        "gold_price": "16255",
        "currency": "INR",
        "location": location,
        "carret": carret,
        "unit": "1gm",
    }


def get_gst(location: str):
    """Gets the current GST for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA.
        unit: The carret unit, e.g. 22K, 18K.
    """
    # In a real scenario, this would call an API
    return {"GST": "18%", "location": location}


# 2. Map function names to actual functions
FUNCTION_MAP = {"get_gold_price": get_gold_price, "get_gst": get_gst}


def get_api_response(function_call):
    """
    Execute the appropriate function based on the function call.

    Args:
        function_call: The function call object with name and args

    Returns:
        dict: API response data from the executed function
    """
    function_name = function_call.name
    function_args = dict(function_call.args)

    # Get the actual function from the map
    if function_name in FUNCTION_MAP:
        function_to_call = FUNCTION_MAP[function_name]
        # Call the function with the provided arguments
        return function_to_call(**function_args)
    else:
        return {"error": f"Function {function_name} not found"}


def process_model_response(response, model, user_prompt, tools):
    """
    Process model response - either handle function calls or return text message.

    Args:
        response: The model's response object
        model: The model instance to use for follow-up calls
        user_prompt: The original user prompt text
        tools: List of tools/functions available to the model

    Returns:
        str: The final text response from the model
    """
    # Check if response contains function calls
    if response.candidates[0].function_calls:
        function_response_parts = []

        # Process each function call
        for function_call in response.candidates[0].function_calls:
            print(f"Function call: {function_call.name}")
            print(f"Arguments: {dict(function_call.args)}")

            # Get API response by executing the actual function
            api_response = get_api_response(function_call)
            print(f"API Response: {api_response}")

            # Add function response to parts
            function_response_parts.append(
                Part.from_function_response(
                    name=function_call.name, response={"content": api_response}
                )
            )

        # Create function response content
        function_response_contents = Content(role="user", parts=function_response_parts)

        # Submit everything back to the model
        final_response = model.generate_content(
            [
                Content(role="user", parts=[Part.from_text(user_prompt)]),
                response.candidates[0].content,  # Function call response
                function_response_contents,  # API output
            ],
            tools=tools,
        )

        return final_response.text
    else:
        # No function calls - just return the text message
        print("No function calls - returning direct response")
        return response.text


# get_gold_price_func = FunctionDeclaration.from_func(get_gold_price)
# get_gst_func = FunctionDeclaration.from_func(get_gst)

# 3. Define function declarations for the model
get_gold_price_func = FunctionDeclaration(
    name="get_gold_price",
    description="Gets the current gold price for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. Chennai, Tamil Nadu"
            },
            "carret": {
                "type": "string",
                "description": "The carret unit, e.g. 22K, 18K, 24K",
                "enum": ["22K", "18K", "24K"]
            }
        },
        "required": ["location"]
    }
)

get_gst_func = FunctionDeclaration(
    name="get_gst",
    description="Gets the current GST (Goods and Services Tax) for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. Chennai, Tamil Nadu"
            }
        },
        "required": ["location"]
    }
)

# # 2. Create the tool declaration
# gold_tool = FunctionDeclaration(
#     name="get_gold_price",
#     description="Get the current gold price in a given location",
#     parameters={
#         "type": "object",
#         "properties": {
#             "location": {
#                 "type": "string",
#                 "description": "The city and state, e.g. San Francisco, CA",
#             },
#             "carret": {"type": "string", "enum": ["22K", "18K", "14K"]},
#         },
#         "required": ["location"],
#     },
# )

# # 2. Create the tool declaration
# GST_tool = FunctionDeclaration(
#     name="get_gst",
#     description="Get the current GST for a given location",
#     parameters={
#         "type": "object",
#         "properties": {
#             "location": {
#                 "type": "string",
#                 "description": "The city and state, e.g. San Francisco, CA",
#             }
#         },
#         "required": ["location"],
#     },
# )
# 3. Create the tool
# gold_tool_object = Tool(function_declarations=[gold_tool])
# get_GST_tool_object = Tool(function_declarations=[GST_tool])
# ALL_TOOLS = [gold_tool_object, get_GST_tool_object]

ALL_TOOLS = [
    Tool(
        function_declarations=[
            get_gold_price_func,
            get_gst_func,
        ],
    )
]

tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        # allowed_function_names=["get_gold_price", "get_gst"],
    )
)


class ModelClients:
    """Class for managing different model API clients with lazy loading"""

    def __init__(self):
        """Initialize the client container without creating clients"""
        # Use private attributes that will be initialized on demand
        self._openai_client = None
        self._async_openai_client = None
        self._anthropic_client = None
        self._deepinfra_client = None
        self._async_deepinfra_client = None
        self._yi_client = None
        self._async_yi_client = None
        self._local_client = None
        self._google_client = None
        self._async_google_client = None

    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv(ENV_VARS["openai"])
        if not api_key:
            print(
                f"Warning: OpenAI API key not found in environment variable {ENV_VARS['openai']}"
            )
            return None
        if OpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return OpenAI(api_key=api_key)

    def _init_async_openai(self):
        """Initialize Async OpenAI client"""
        api_key = os.getenv(ENV_VARS["openai"])
        if not api_key:
            print(
                f"Warning: OpenAI API key not found in environment variable {ENV_VARS['openai']}"
            )
            return None
        if AsyncOpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return AsyncOpenAI(api_key=api_key)

    def _init_anthropic(self):
        """Initialize Anthropic client"""
        api_key = os.getenv(ENV_VARS["anthropic"])
        if not api_key:
            return None
        if anthropic is None:
            print("Warning: anthropic package not installed; returning None")
            return None
        return anthropic.Anthropic(api_key=api_key)

    def _init_anthropic_async(self):
        """Initialize Async Anthropic client"""
        api_key = os.getenv(ENV_VARS["anthropic"])
        if not api_key:
            return None
        if anthropic is None:
            print("Warning: anthropic package not installed; returning None")
            return None
        return anthropic.AsyncAnthropic(api_key=api_key)

    def _init_deepinfra(self):
        """Initialize DeepInfra client"""
        api_key = os.getenv(ENV_VARS["deepinfra"]["api_key"])
        base_url = os.getenv(ENV_VARS["deepinfra"]["base_url"])
        if not api_key or not base_url:
            return None
        if OpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return OpenAI(api_key=api_key, base_url=base_url)

    def _init_async_deepinfra(self):
        """Initialize Async DeepInfra client"""
        api_key = os.getenv(ENV_VARS["deepinfra"]["api_key"])
        base_url = os.getenv(ENV_VARS["deepinfra"]["base_url"])
        if not api_key or not base_url:
            return None
        if AsyncOpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _init_yi(self):
        """Initialize Yi client"""
        api_key = os.getenv(ENV_VARS["yi"]["api_key"])
        base_url = os.getenv(ENV_VARS["yi"]["base_url"])
        if not api_key or not base_url:
            return None
        if OpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return OpenAI(api_key=api_key, base_url=base_url)

    def _init_async_yi(self):
        """Initialize Async Yi client"""
        api_key = os.getenv(ENV_VARS["yi"]["api_key"])
        base_url = os.getenv(ENV_VARS["yi"]["base_url"])
        if not api_key or not base_url:
            return None
        if AsyncOpenAI is None:
            print("Warning: openai package not installed; returning None")
            return None
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _init_local(self):
        """Initialize local client"""
        if OpenAI is None:
            return None
        return OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1/")

    # Lazy loading properties
    @property
    def openai_client(self):
        """Lazy load OpenAI client"""
        if self._openai_client is None:
            self._openai_client = self._init_openai()
        return self._openai_client

    @property
    def async_openai_client(self):
        """Lazy load async OpenAI client"""
        if self._async_openai_client is None:
            self._async_openai_client = self._init_async_openai()
        return self._async_openai_client

    @property
    def google_client(self):
        """Lazy load Google client"""
        if self._google_client is None:
            # self._google_client = GenerativeModel("gemini-2.5-flash")
            # self._google_client = genai.GenerativeModel("gemini-2.5-flash")
            self._google_client = GenerativeModel("gemini-2.5-flash")
        return self._google_client

    @property
    def async_google_client(self):
        """Lazy load async Google client"""
        # if self._async_google_client is None:
        #     self._async_google_client = self._init_async_google()
        return self._async_google_client

    @property
    def anthropic_client(self):
        """Lazy load Anthropic client"""
        if self._anthropic_client is None:
            self._anthropic_client = self._init_anthropic()
        return self._anthropic_client

    @property
    def deepinfra_client(self):
        """Lazy load DeepInfra client"""
        if self._deepinfra_client is None:
            self._deepinfra_client = self._init_deepinfra()
        return self._deepinfra_client

    @property
    def async_deepinfra_client(self):
        """Lazy load async DeepInfra client"""
        if self._async_deepinfra_client is None:
            self._async_deepinfra_client = self._init_async_deepinfra()
        return self._async_deepinfra_client

    @property
    def yi_client(self):
        """Lazy load Yi client"""
        if self._yi_client is None:
            self._yi_client = self._init_yi()
        return self._yi_client

    @property
    def async_yi_client(self):
        """Lazy load async Yi client"""
        if self._async_yi_client is None:
            self._async_yi_client = self._init_async_yi()
        return self._async_yi_client

    @property
    def local_client(self):
        """Lazy load local client"""
        if self._local_client is None:
            self._local_client = self._init_local()
        return self._local_client

    def get_client(self, model_type: str):
        """Get the appropriate client for the model type (with lazy loading)"""
        # Use dictionary with property getters to ensure lazy loading
        clients = {
            "openai": self.openai_client,
            "anthropic": self.anthropic_client,
            "deepinfra": self.deepinfra_client,
            "yi": self.yi_client,
            "local": self.local_client,
            "google": self.google_client,
        }

        if model_type not in clients:
            raise ValueError(f"Unknown model type: {model_type}")

        client = clients[model_type]
        if client is None:
            env_var = ENV_VARS.get(model_type, f"{model_type}_api_key")
            raise ValueError(
                f"Client for {model_type} is not configured properly. Please check the {env_var} environment variable."
            )

        return client

    def get_async_client(self, model_type: str):
        """Get the appropriate async client for the model type (with lazy loading)"""
        # Use dictionary with property getters to ensure lazy loading
        clients = {
            "openai": self.async_openai_client,
            "anthropic": self.anthropic_client,  # Note: Anthropic might not have async API
            "deepinfra": self.async_deepinfra_client,
            "yi": self.async_yi_client,
            "local": self.local_client,  # Local client doesn't support async
        }

        if model_type not in clients:
            raise ValueError(f"Unknown model type: {model_type}")

        client = clients[model_type]
        if client is None:
            env_var = ENV_VARS.get(model_type, f"{model_type}_api_key")
            raise ValueError(
                f"Async client for {model_type} is not configured properly. Please check the {env_var} environment variable."
            )

        return client

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    def call_anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        max_tokens: int = 2048,
        msg: list = None,
    ) -> str:
        """Call Anthropic Claude models"""
        client = self.anthropic_client
        if msg is None:
            message = client.messages.create(
                model=MODEL_MAPPING[model],
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            )
        else:
            message = client.messages.create(
                model=MODEL_MAPPING[model],
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a helpful assistant.",
                messages=msg,
            )
        tokens = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return message.content[0].text, tokens

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    def call_openai_compatible(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        client_type: str = "openai",
        msg: list = None,
    ) -> str:
        """Call OpenAI-compatible API endpoints"""
        client = self.get_client(client_type)

        if msg is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = msg

        if model == "o3-mini":
            response = client.chat.completions.create(
                model=MODEL_MAPPING[model], messages=messages
            )

        else:
            response = client.chat.completions.create(
                model=MODEL_MAPPING[model], messages=messages, temperature=temperature
            )

        input_token = response.usage.prompt_tokens
        output_token = response.usage.completion_tokens

        tokens = {"input_tokens": input_token, "output_tokens": output_token}

        return response.choices[0].message.content, tokens

    @retry_decorator(max_retries=3, delay=2, backoff=2)
    def call_google_compatible(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        client_type: str = "google",
        msg: list = None,
    ) -> str:
        """Call OpenAI-compatible API endpoints"""
        # print(client_type)
        client = self.get_client(client_type)

        if msg is None:
            messages = prompt
        else:
            messages = msg

        generation_config = GenerationConfig(
            temperature=temperature, max_output_tokens=8192
        )

        response = client.generate_content(
            messages,
            generation_config=generation_config,
            tools=ALL_TOOLS,
        )

         # Process the response
        final_answer = process_model_response(
            response=response,
            model=client,
            user_prompt=messages,
            tools=ALL_TOOLS
        )

        # print(response)
        input_token = response.usage_metadata.prompt_token_count
        output_token = response.usage_metadata.total_token_count

        tokens = {"input_tokens": input_token, "output_tokens": output_token}

        return final_answer, tokens

    @retry_decorator(max_retries=3, delay=2, backoff=2)
    def call_google_compatible_old(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        client_type: str = "google",
        msg: list = None,
    ) -> str:
        """Call OpenAI-compatible API endpoints"""
        # print(client_type)
        client = self.get_client(client_type)

        if msg is None:
            messages = prompt
        else:
            messages = msg

        generation_config = GenerationConfig(
            temperature=temperature, max_output_tokens=8192
        )

        response = client.generate_content(
            messages,
            generation_config=generation_config,
        )

        # print(response)
        input_token = response.usage_metadata.prompt_token_count
        output_token = response.usage_metadata.total_token_count

        tokens = {"input_tokens": input_token, "output_tokens": output_token}
        print(response.text)
        return response.text, tokens

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    async def call_openai_compatible_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        client_type: str = "openai",
    ) -> str:
        """Call OpenAI-compatible API endpoints asynchronously"""
        client = self.get_async_client(client_type)

        messages = [{"role": "user", "content": prompt}]

        if model == "o3-mini":
            response = await client.chat.completions.create(
                model=MODEL_MAPPING[model], messages=messages
            )

        else:
            response = await client.chat.completions.create(
                model=MODEL_MAPPING[model], messages=messages, temperature=temperature
            )

        input_token = response.usage.prompt_tokens
        output_token = response.usage.completion_tokens

        tokens = {"input_tokens": input_token, "output_tokens": output_token}

        return response.choices[0].message.content, tokens

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    async def call_anthropic_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.001,
        max_tokens: int = 2048,
    ) -> str:
        """Call Anthropic Claude models asynchronously"""
        client = self.get_async_client("anthropic")
        # Note: Anthropic's Python client doesn't have async support yet
        # This is a synchronous call in an async function
        message = await client.messages.create(
            model=MODEL_MAPPING[model],
            max_tokens=max_tokens,
            temperature=temperature,
            system="",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        tokens = {
            "input_tokens": message.usage.prompt_tokens,
            "output_tokens": message.usage.completion_tokens,
        }
        return message.content[0].text, tokens

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    def call_structured(
        self,
        model: str,
        messages: list,
        temperature: float = 0.5,
        response_format: Dict = None,
    ) -> Any:
        """Call OpenAI with structured output format"""
        client = self.openai_client

        response = client.beta.chat.completions.parse(
            model=MODEL_MAPPING[model],
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )

        return response.choices[0].message.parsed

    @retry_decorator(max_retries=3, delay=1, backoff=2)
    def call_local(
        self, model: str, prompt: str, temperature: float = 0.001, msg: list = None
    ) -> str:
        """Call Local model server via OpenAI compatible API"""
        client = self.local_client

        if msg is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = msg
        models = client.models.list()
        model = models.data[0].id
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body={"top_k": 50},
        )

        # Local model might not provide token usage info
        try:
            input_token = response.usage.prompt_tokens
            output_token = response.usage.completion_tokens
        except (AttributeError, TypeError):
            # Provide estimated token counts if not available
            input_token = len(prompt) // 4  # Rough estimate
            output_token = (
                len(response.choices[0].message.content) // 4
            )  # Rough estimate

        tokens = {"input_tokens": input_token, "output_tokens": output_token}

        return response.choices[0].message.content, tokens
