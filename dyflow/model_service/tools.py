import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration

# 1. Define the function
def get_current_weather(location: str, unit: str = "celsius"):
    """Gets the current weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA.
        unit: The temperature unit, celsius or fahrenheit.
    """
    # In a real scenario, this would call an API
    return {"weather": "sunny", "temperature": 25, "unit": unit}

# 2. Create the tool declaration
weather_tool = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
)

# 3. Create the tool
weather_tool_object = Tool(function_declarations=[weather_tool])
