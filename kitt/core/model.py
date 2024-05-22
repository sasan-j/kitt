import ast
import json
import re
import uuid
from enum import Enum
from typing import List
import xml.etree.ElementTree as ET

from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools.base import StructuredTool
from ollama import Client
from pydantic import BaseModel
from loguru import logger

from kitt.skills import vehicle_status
from kitt.skills.common import config
from .validator import validate_function_call_schema


class FunctionCall(BaseModel):
    arguments: dict
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """

    name: str
    """The name of the function to call."""


class ResponseType(Enum):
    TOOL_CALL = "tool_call"
    TEXT = "text"


class AssistantResponse(BaseModel):
    tool_calls: List[FunctionCall]
    """The tool call to make to get the response."""

    response_type: ResponseType = (
        ResponseType.TOOL_CALL
    )  # The type of response to make to the user. Either 'tool_call' or 'text'.
    """The type of response to make to the user. Either 'tool_call' or 'text'."""

    response: str


schema_json = json.loads(FunctionCall.schema_json())
# schema_json = json.loads(AssistantResponse.schema_json())

HRMS_SYSTEM_PROMPT = """<|im_start|>system
You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:
<schema>
{schema}
<schema><|im_end|>"""


HRMS_SYSTEM_PROMPT = """<|im_start|>system
Role:
Your name is KITT. You are embodied in a Car. The user is a human who is a passenger in the car. You have autonomy to use the tools available to you to assist the user.
You are the AI assistant in the car. From the information in <car_status></car_status you know where you are, the destination, and the current date and time.
You are witty, helpful, and have a good sense of humor. You are a function calling AI agent with self-recursion.
You are provided with function signatures within <tools></tools> XML tags.
User preferences are provided in <user_preferences></user_preferences> XML tags. Use them if needed.

<car_status>
{car_status}
</car_status>

<user_preferences>
{user_preferences}
</user_preferences>

Objective:
You may use agentic frameworks for reasoning and planning to help with user query.
Please call one or two functions at a time, the function results to be provided to you immediately. Try to answer the user query, with as little back and forth as possible.
Don't make assumptions about what values to plug into function arguments.
Once you have called a function, results will be fed back to you within <tool_response></tool_response> XML tags.
Don't make assumptions about tool results if <tool_response> XML tags are not present since function hasn't been executed yet.
Analyze the data once you get the results and call another function.
At each iteration please continue adding the your analysis to previous summary.
Your final response should directly answer the user query. Don't tell what you are doing, just do it.


Tools:
Here are the available tools:
<tools> {tools} </tools>
Make sure that the json object above with code markdown block is parseable with json.loads() and the XML block with XML ElementTree.
When using tools, ensure to only use the tools provided and not make up any data and do not provide any explanation as to which tool you are using and why.

When asked for the weather or points of interest, use the appropriate tool with the current location of the car. Unless the user provides a location, then use that location.
Always assume user wants to travel by car.

Schema:
Use the following pydantic model json schema for each tool call you will make:
{schema}

Instructions:
At the very first turn you don't have <tool_results> so you shouldn't not make up the results.
Please keep a running summary with analysis of previous function results and summaries from previous iterations.
Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.
Calling multiple functions at once can overload the system and increase cost so call one function at a time please.
If you plan to continue with analysis, always call another function.
For each function call return a valid json object (using double quotes) with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"arguments": <args-dict>, "name": <function-name>}}
</tool_call>
If there are more than one function call, return multiple <tool_call></tool_call> XML tags, for example:
<tool_call>
{{"arguments": <args-dict>, "name": <function-name>}}
</tool_call>
<tool_call>
{{"arguments": <args-dict>, "name": <function-name>}}
</tool_call>
You have to open and close the XML tags for each function call.

<|im_end|>"""
AI_PREAMBLE = """
<|im_start|>assistant
"""
HRMS_TEMPLATE_USER = """
<|im_start|>user
{user_input}<|im_end|>"""
HRMS_TEMPLATE_ASSISTANT = """
<|im_start|>assistant
{assistant_response}<|im_end|>"""
HRMS_TEMPLATE_TOOL_RESULT = """
<|im_start|>tool
{result}
<|im_end|>"""


"""
Below are a few examples, but they are not exhaustive. You can call any tool as long as it is within the <tools></tools> XML tags. Also examples are simplified and don't include all the tags you will see in the conversation.
Example 1:
User: How is the weather?
Assistant:
<tool_call>
{{"arguments": {{"location": ""}}, "name": "get_weather"}}
</tool_call>

Example 2:
User: Is there a Spa nearby?
Assistant:
<tool_call>
{{"arguments": {{"search_query": "Spa"}}, "name": "search_points_of_interest"}}
</tool_call>


Example 3:
User: How long will it take to get to the destination?
Assistant:
<tool_call>
{{"arguments": {{"destination": ""}}, "name": "calculate_route"}}
</tool_call>
"""


def append_message(prompt, h):
    if h.type == "human":
        prompt += HRMS_TEMPLATE_USER.format(user_input=h.content)
    elif h.type == "ai":
        prompt += HRMS_TEMPLATE_ASSISTANT.format(assistant_response=h.content)
    elif h.type == "tool":
        prompt += HRMS_TEMPLATE_TOOL_RESULT.format(result=h.content)
    return prompt


def get_prompt(template, history, tools, schema, user_preferences, car_status=None):
    if not car_status:
        # car_status = vehicle.dict()
        car_status = vehicle_status()[0]

    # "vehicle_status": vehicle_status_fn()[0]
    kwargs = {
        "history": history,
        "schema": schema,
        "tools": tools,
        "car_status": car_status,
        "user_preferences": user_preferences,
    }

    prompt = template.format(**kwargs).replace("{{", "{").replace("}}", "}")

    if history:
        for h in history.messages:
            prompt = append_message(prompt, h)

    # if input:
    #     prompt += USER_QUERY_TEMPLATE.format(user_input=input)
    return prompt





def run_inference_ollama(prompt):
    data = {
        "prompt": prompt,
        # "streaming": False,
        # "model": "smangrul/llama-3-8b-instruct-function-calling",
        # "model": "elvee/hermes-2-pro-llama-3:8b-Q5_K_M",
        # "model": "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "model": "interstellarninja/hermes-2-pro-llama-3-8b",
        # "model": "dolphin-llama3:8b",
        # "model": "dolphin-llama3:70b",
        "raw": True,
        "options": {
            "temperature": 0.7,
            # "max_tokens": 1500,
            "num_predict": 1500,
            # "mirostat": 1,
            # "mirostat_tau": 2,
            "repeat_penalty": 1.2,
            "top_k": 25,
            "top_p": 0.5,
            "num_ctx": 8000,
            # "stop": ["<|im_end|>"]
            # "num_predict": 1500,
            # "max_tokens": 1500,
        },
    }

    client = Client(host="http://localhost:11434")
    # out = ollama.generate(**data)
    out = client.generate(**data)
    res = out.pop("response")
    # Report prompt and eval tokens
    logger.warning(
        f"Prompt tokens: {out.get('prompt_eval_count')}, Response tokens: {out.get('eval_count')}"
    )
    logger.debug(f"Response from Ollama: {res}\nOut:{out}")
    return res


def run_inference_step(
    depth, history, tools, schema_json, user_preferences, backend="ollama"
):
    # If we decide to call a function, we need to generate the prompt for the model
    # based on the history of the conversation so far.
    # not break the loop
    openai_tools = [convert_to_openai_tool(tool) for tool in tools]
    prompt = get_prompt(
        HRMS_SYSTEM_PROMPT,
        history,
        openai_tools,
        schema_json,
        user_preferences=user_preferences,
    )
    logger.debug(f"History is: {history.messages}")

    # if depth == 0:
    #     prompt += "\nThis is the first turn and you don't have <tool_results> to analyze yet."
    prompt += AI_PREAMBLE

    logger.info(f"Prompt is:\n{prompt}")

    if backend == "ollama":
        output = run_inference_ollama(prompt)
    else:
        output = run_inference_replicate(prompt)

    logger.debug(f"Response from model: {output}")
    return output


def run_inference_replicate(prompt):
    from replicate import Client

    replicate = Client(api_token=config.REPLICATE_API_KEY)

    input = {
        "prompt": prompt,
        "temperature": 0.5,
        "system_prompt": "",
        "max_new_tokens": 1024,
        "repeat_penalty": 1.1,
        "prompt_template": "{prompt}",
    }

    output = replicate.run(
        # "mikeei/dolphin-2.9-llama3-8b-gguf:0f79fb14c45ae2b92e1f07d872dceed3afafcacd903258df487d3bec9e393cb2",
        "sasan-j/hermes-2-pro-llama-3-8b:28b1dc16f47d9df68d9839418282315d5e78d9e2ab3fa6ff15728c76ae71a6d6",
        input=input,
    )
    out = "".join(output)

    logger.debug(f"Response from Ollama:\nOut:{out}")

    return out


def run_inference(prompt, backend="ollama"):
    prompt += AI_PREAMBLE

    logger.info(f"Prompt is:\n{prompt}")

    if backend == "ollama":
        output = run_inference_ollama(prompt)
    else:
        output = run_inference_replicate(prompt)

    logger.debug(f"Response from model: {output}")
    return output


def validate_and_extract_tool_calls(assistant_content):
    validation_result = False
    tool_calls = []
    error_message = None

    try:
        # wrap content in root element
        xml_root_element = f"<root>{assistant_content}</root>"
        root = ET.fromstring(xml_root_element)

        # extract JSON data
        for element in root.findall(".//tool_call"):
            json_data = None
            try:
                json_text = element.text.strip()

                try:
                    # Prioritize json.loads for better error handling
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        error_message = (
                            f"JSON parsing failed with both json.loads and ast.literal_eval:\n"
                            f"- JSON Decode Error: {json_err}\n"
                            f"- Fallback Syntax/Value Error: {eval_err}\n"
                            f"- Problematic JSON text: {json_text}"
                        )
                        logger.error(error_message)
                        continue
            except Exception as e:
                error_message = f"Cannot strip text: {e}"
                logger.error(error_message)

            if json_data is not None:
                tool_calls.append(json_data)
                validation_result = True

    except ET.ParseError as err:
        error_message = f"XML Parse Error: {err}"
        logger.error(f"XML Parse Error: {err}")

    # Return default values if no valid data is extracted
    return validation_result, tool_calls, error_message


def execute_function_call(tool_call, functions):
    function_name = tool_call.get("name")
    for tool in functions:
        if tool.name == function_name:
            function_to_call = tool
            break
    else:
        raise ValueError(f"Function {function_name} not found.")
    function_args = tool_call.get("arguments", {})

    logger.info(f"Invoking function call {function_name} ...")
    if isinstance(function_to_call, StructuredTool):
        function_response = function_to_call.invoke(input=function_args)
    else:
        function_response = function_to_call(*function_args.values())
    results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
    return results_dict


def process_completion_and_validate(completion):

    # I think I don't need this.
    # assistant_message = get_assistant_message(completion, eos_token="<|im_end|>")
    assistant_message = completion.strip()

    if assistant_message:
        validation, tool_calls, error_message = validate_and_extract_tool_calls(
            assistant_message
        )

        if validation:
            logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
            return tool_calls, assistant_message, error_message
        else:
            tool_calls = None
            return tool_calls, assistant_message, error_message
    else:
        logger.warning("Assistant message is None")
        raise ValueError("Assistant message is None")


UNRESOLVED_MSG = "I'm sorry, I'm not sure how to help you with that."


def get_assistant_message(completion, eos_token):
    """define and match pattern to find the assistant message"""
    completion = completion.strip()
    assistant_pattern = re.compile(
        r"<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$", re.DOTALL
    )
    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        assistant_content = assistant_match.group(1).strip()
        return assistant_content.replace(eos_token, "")
    else:
        assistant_content = None
        logger.info("No match found for the assistant pattern")
        return assistant_content


def generate_function_call(
    query, history, user_preferences, tools, functions, backend, max_depth=5
) -> str:
    """
    Largely taken from https://github.com/NousResearch/Hermes-Function-Calling
    """

    try:
        depth = 0
        # user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
        user_message = f"{query}"
        # chat = [{"role": "user", "content": user_message}]
        history.add_message(HumanMessage(content=user_message))

        # openai_tools = [convert_to_openai_function(tool) for tool in tools]
        prompt = get_prompt(
            HRMS_SYSTEM_PROMPT,
            history,
            tools,
            schema_json,
            user_preferences=user_preferences,
        )
        logger.debug(f"History is: {history.json()}")

        # if depth == 0:
        #     prompt += "\nThis is the first turn and you don't have <tool_results> to analyze yet."
        completion = run_inference(prompt, backend=backend)

        def recursive_loop(prompt, completion, depth) -> str:
            nonlocal max_depth
            tool_calls, assistant_message, error_message = (
                process_completion_and_validate(completion)
            )
            # prompt.append({"role": "assistant", "content": assistant_message})
            history.add_message(AIMessage(content=assistant_message))

            tool_message = (
                f"Agent iteration {depth} to assist with user query: {query}\n"
            )
            if tool_calls:
                logger.info(f"Assistant Message:\n{assistant_message}")
                for tool_call in tool_calls:
                    validation, message = validate_function_call_schema(
                        tool_call, tools
                    )
                    if validation:
                        try:
                            function_response = execute_function_call(
                                tool_call, functions=functions
                            )
                            tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                            logger.info(
                                f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}"
                            )
                        except Exception as e:
                            logger.warning(f"Could not execute function: {e}")
                            tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    else:
                        logger.error(message)
                        tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                # prompt.append({"role": "tool", "content": tool_message})
                history.add_message(
                    ToolMessage(content=tool_message, tool_call_id=uuid.uuid4().hex)
                )

                depth += 1
                if depth >= max_depth:
                    logger.warning(
                        f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                    )
                    return UNRESOLVED_MSG

                prompt = get_prompt(
                    HRMS_SYSTEM_PROMPT,
                    history,
                    tools,
                    schema_json,
                    user_preferences=user_preferences,
                )
                completion = run_inference(prompt, backend=backend)
                return recursive_loop(prompt, completion, depth)
            elif error_message:
                logger.info(f"Assistant Message:\n{assistant_message}")
                tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                prompt.append({"role": "tool", "content": tool_message})

                depth += 1
                if depth >= max_depth:
                    logger.warning(
                        f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                    )
                    return UNRESOLVED_MSG

                completion = run_inference(prompt, backend=backend)
                return recursive_loop(prompt, completion, depth)
            else:
                logger.info(f"Assistant Message:\n{assistant_message}")
                return assistant_message

        return recursive_loop(prompt, completion, depth)  # noqa

    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return UNRESOLVED_MSG
        # raise e
