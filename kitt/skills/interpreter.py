import inspect

# From https://github.com/NousResearch/Hermes-Function-Calling

def code_interpreter(code_markdown: str) -> dict | str:
    """
    Execute the provided Python code string on the terminal using exec.

    The string should contain valid, executable and pure Python code in markdown syntax.
    Code should also import any required Python packages.

    Args:
        code_markdown (str): The Python code with markdown syntax to be executed.
            For example: ```python\n<code-string>\n```

    Returns:
        dict | str: A dictionary containing variables declared and values returned by function calls,
            or an error message if an exception occurred.

    Note:
        Use this function with caution, as executing arbitrary code can pose security risks. Use it only for numerical calculations.
    """
    try:
        # Extracting code from Markdown code block
        code_lines = code_markdown.split('\n')[1:-1]
        code_without_markdown = '\n'.join(code_lines)

        # Create a new namespace for code execution
        exec_namespace = {}

        # Execute the code in the new namespace
        exec(code_without_markdown, exec_namespace)

        # Collect variables and function call results
        result_dict = {}
        for name, value in exec_namespace.items():
            if callable(value):
                try:
                    result_dict[name] = value()
                except TypeError:
                    # If the function requires arguments, attempt to call it with arguments from the namespace
                    arg_names = inspect.getfullargspec(value).args
                    args = {arg_name: exec_namespace.get(arg_name) for arg_name in arg_names}
                    result_dict[name] = value(**args)
            elif not name.startswith('_'):  # Exclude variables starting with '_'
                result_dict[name] = value

        return result_dict

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return error_message
