from datetime import datetime
import inspect

from .common import execute_function_call, extract_func_args, vehicle as vehicle_obj
from .weather import get_weather_current_location, get_weather, get_forecast
from .routing import find_route
from .poi import search_points_of_interests, search_along_route_w_coordinates
from .vehicle import vehicle_status
from .interpreter import code_interpreter



def date_time_info():
    """Get the current date and time."""
    time = getattr(vehicle_obj, "time")
    date = getattr(vehicle_obj, "date")
    datetime_obj = datetime.fromisoformat(f"{date}T{time}")
    human_readable_datetime = datetime_obj.strftime("%I:%M %p %A, %B %d, %Y")
    return f"It is {human_readable_datetime}."


def do_anything_else():
    """If the user wants to do anything else call this function. If the question doesn't match any of the functions use this one."""
    return True


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


SKILLS_PROMPT = format_functions_for_prompt_raven(get_weather, get_forecast, find_route, search_points_of_interests)
