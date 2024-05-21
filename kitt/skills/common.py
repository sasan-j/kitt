import re
from typing import Union, Optional


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

from .. import skills
from enum import Enum

class Settings(BaseSettings):
    WEATHER_API_KEY: str
    TOMTOM_API_KEY: str
    REPLICATE_API_KEY: Optional[str]

    model_config = SettingsConfigDict(env_file=".env")


class Speed(Enum):
        SLOW = "slow"
        FAST = "fast"


class VehicleStatus(BaseModel):
    location: str
    location_coordinates: tuple[float, float] # (latitude, longitude)
    date: str
    time: str
    destination: str
    speed: Speed = Speed.SLOW
    


def execute_function_call(text: str, dry_run=False) -> str:
    function_name_match = re.search(r"Call: (\w+)", text)
    function_name = function_name_match.group(1) if function_name_match else None
    arguments = eval(f"dict{text.split(function_name)[1].strip()}")
    function = getattr(skills, function_name) if function_name else None

    if dry_run:
        print(f"{function_name}(**{arguments})")
        return "Dry run successful"

    if function:
        out = function(**arguments)
    try:
        if function:
            out = function(**arguments)
    except Exception as e:
        out = str(e)
    return out


def extract_func_args(text: str) -> tuple[str, dict]:
    function_name_match = re.search(r"Call: (\w+)", text)
    function_name = function_name_match.group(1) if function_name_match else None
    if not function_name:
        raise ValueError("No function name found in text")
    arguments = eval(f"dict{text.split(function_name)[1].strip()}")
    return function_name, arguments


config = Settings() # type: ignore

vehicle = VehicleStatus(
    location="Rue Alphonse Weicker, Luxembourg",
    location_coordinates=(49.505, 6.28111),
    date="2025-05-06",
    time="08:00:00",
    destination="Rue Alphonse Weicker, Luxembourg"
)
