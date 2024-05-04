import inspect

from .common import execute_function_call, extract_func_args, vehicle
from .weather import get_weather, get_forecast
from .routing import find_route
from .vehicle import vehicle_status


def format_functions_for_prompt_raven(*functions):
    """Format functions for use in Prompt Raven.
    
    Args:
    *functions (function): One or more functions to format.
    """
    formatted_functions = []
    for func in functions:
        signature = f"{func.__name__}{inspect.signature(func)}"
        docstring = inspect.getdoc(func)
        formatted_functions.append(
            f"Function:\n<func_start>{signature}<func_end>\n<docstring_start>\n{docstring}\n<docstring_end>"
        )
    return "\n".join(formatted_functions)


SKILLS_PROMPT = format_functions_for_prompt_raven(get_weather, get_forecast, find_route)
