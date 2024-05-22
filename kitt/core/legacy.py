import uuid
import json
import re
from loguru import logger

def use_tool(tool_call, tools):
    func_name = tool_call["name"]
    kwargs = tool_call["arguments"]
    for tool in tools:
        if tool.name == func_name:
            return tool.invoke(input=kwargs)
    raise ValueError(f"Tool {func_name} not found.")


def parse_tool_calls(text):
    logger.debug(f"Start parsing tool_calls: {text}")
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    if not text.startswith("<tool_call>"):
        if "<tool_call>" in text:
            raise ValueError("<text_and_tool_call>")

        if "<tool_response>" in text:
            raise ValueError("<tool_response>")
        return [], []

    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    errors = []
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in tool call: {e}")

    logger.debug(f"Tool calls: {tool_calls}, errors: {errors}")
    return tool_calls, errors

def process_response(user_query, res, history, tools, depth):
    """Returns True if the response contains tool calls, False otherwise."""
    logger.debug(f"Processing response: {res}")
    tool_results = f"Agent iteration {depth} to assist with user query: {user_query}\n"
    tool_call_id = uuid.uuid4().hex
    try:
        tool_calls, errors = parse_tool_calls(res)
    except ValueError as e:
        if "<text_and_tool_call>" in str(e):
            tool_results += "<tool_response>If you need to call a tool your response must be wrapped in <tool_call></tool_call>. Try again, you are great.</tool_response>"
            history.add_message(
                ToolMessage(content=tool_results, tool_call_id=tool_call_id)
            )
            return True, [], []
        if "<tool_response>" in str(e):
            tool_results += "<tool_response>Tool results are not allowed in the response.</tool_response>"
            history.add_message(
                ToolMessage(content=tool_results, tool_call_id=tool_call_id)
            )
            return True, [], []
    # TODO: Handle errors
    if not tool_calls:
        logger.debug("No tool calls found in response.")
        return False, tool_calls, errors
    # tool_results = ""

    for tool_call in tool_calls:
        # TODO: Extra Validation
        # Call the function
        try:
            result = use_tool(tool_call, tools)
            logger.debug(f"Tool call {tool_call} result: {result}")
            if isinstance(result, tuple):
                result = result[1]
            tool_results += f"<tool_response>\n{result}\n</tool_response>\n"
        except Exception as e:
            logger.error(f"Error calling tool: {e}")
    # Currently only to mimic OpneAI's behavior
    # But it could be used for tracking function calls

    tool_results = tool_results.strip()
    print(f"Tool results: {tool_results}")
    history.add_message(ToolMessage(content=tool_results, tool_call_id=tool_call_id))
    return True, tool_calls, errors


def process_query(
    user_query: str,
    history: ChatMessageHistory,
    user_preferences,
    tools,
    backend="ollama",
):
    # Add vehicle status to the history
    user_query_status = f"consider the vehicle status:\n{vehicle_status()[0]}\nwhen responding to the following query:\n{user_query}"
    history.add_message(HumanMessage(content=user_query_status))
    for depth in range(10):
        # out = run_inference_step(depth, history, tools, schema_json)
        out = run_inference_step(
            depth,
            history,
            tools,
            schema_json,
            user_preferences=user_preferences,
            backend=backend,
        )
        logger.info(f"Inference step result:\n{out}")
        history.add_message(AIMessage(content=out))
        to_continue, tool_calls, errors = process_response(
            user_query, out, history, tools, depth
        )
        if errors:
            history.add_message(AIMessage(content=f"Errors in tool calls: {errors}"))

        if not to_continue:
            print(f"This is the answer, no more iterations: {out}")
            return out
        # Otherwise, tools result is already added to history, we just need to continue the loop.
    # If we get here something went wrong.
    history.add_message(
        AIMessage(content="Sorry, I am not sure how to help you with that.")
    )
    return "Sorry, I am not sure how to help you with that."