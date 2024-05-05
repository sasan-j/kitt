import re
from typing import Union


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

import skills

class Settings(BaseSettings):
    WEATHER_API_KEY: str
    TOMTOM_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


class VehicleStatus(BaseModel):
    location: str
    location_coordinates: tuple[float, float] # (latitude, longitude)
    date: str
    time: str
    destination: str


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
    location="Luxembourg Gare, Luxembourg",
    location_coordinates=(49.6002, 6.1296),
    date="2025-03-29",
    time="08:00:20",
    destination="Kirchberg Campus, Kirchberg"
)
