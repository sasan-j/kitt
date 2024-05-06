import gradio as gr
import requests

import skills
from skills.common import config, vehicle
from skills.routing import calculate_route
import ollama

### LLM Stuff ###
from langchain_community.llms import Ollama
from langchain.tools.base import StructuredTool

from skills import (
    get_weather,
    find_route,
    get_forecast,
    vehicle_status as vehicle_status_fn,
    search_points_of_interests,
    search_along_route_w_coordinates,
    do_anything_else,
    date_time_info
)
from skills import extract_func_args


global_context = {
    "vehicle": vehicle,
    "query": "How is the weather?",
    "route_points": [],
}


MODEL_FUNC = "nexusraven"
MODEL_GENERAL = "llama3:instruct"

RAVEN_PROMPT_FUNC = """You are a helpful AI assistant in a car (vehicle), that follows instructions extremely well. \
Answer questions concisely and do not mention what you base your reply on."

{raven_tools}

{history}

User Query: Question: {input}<human_end>
"""

def get_prompt(template, input, history, tools):
    # "vehicle_status": vehicle_status_fn()[0]
    kwargs = {"history": history, "input": input}
    prompt = "<human>:\n"
    for tool in tools:
        func_signature, func_docstring = tool.description.split(" - ", 1)
        prompt += f'Function:\n<func_start>def {func_signature}<func_end>\n<docstring_start>\n"""\n{func_docstring}\n"""\n<docstring_end>\n'
    kwargs["raven_tools"] = prompt

    if history:
        kwargs["history"] = f"Previous conversation history:{history}\n"

    return template.format(**kwargs).replace("{{", "{").replace("}}", "}")

def use_tool(func_name, kwargs, tools):
    for tool in tools:
        if tool.name == func_name:
            return tool.invoke(input=kwargs)
    return None

tools = [
    StructuredTool.from_function(get_weather),
    StructuredTool.from_function(find_route),
    # StructuredTool.from_function(vehicle_status),
    StructuredTool.from_function(search_points_of_interests),
    StructuredTool.from_function(search_along_route_w_coordinates),
    StructuredTool.from_function(date_time_info),
    StructuredTool.from_function(do_anything_else),
]
# llm = Ollama(model="nexusraven", stop=["\nReflection:", "\nThought:"], keep_alive=60*10)


# Generate options for hours (00-23)
hour_options = [f"{i:02d}:00" for i in range(24)]


def set_time(time_picker):
    vehicle.time = time_picker
    return vehicle.model_dump_json()


def get_vehicle_status(state):
    return state.value["vehicle"].model_dump_json()


def run_generic_model(query):
    print(f"Running the generic model with query: {query}")
    data = {
        "prompt": query,
        "model": MODEL_GENERAL,
        "options": {
            # "temperature": 0.1,
            # "stop":["\nReflection:", "\nThought:"]
        }
    }
    out = ollama.generate(**data)
    return out["response"]


def run_model(query):
    print("Query: ", query)
    global_context["query"] = query
    global_context["prompt"] = get_prompt(RAVEN_PROMPT_FUNC, query, "", tools)
    print("Prompt: ", global_context["prompt"])
    data = {
        "prompt": global_context["prompt"],
        # "streaming": False,
        "model": "nexusraven",
        # "model": "smangrul/llama-3-8b-instruct-function-calling",
        "raw": True,
        "options": {
            "temperature": 0.5,
            "stop":["\nReflection:", "\nThought:"]
        }
    }
    out = ollama.generate(**data)
    llm_response = out["response"]
    if "Call: " in llm_response:
        func_name, kwargs = extract_func_args(llm_response)
        print(f"Function: {func_name}, Args: {kwargs}")
        if func_name == "do_anything_else":
            return run_generic_model(query)
    
        return use_tool(func_name, kwargs, tools)
    return out["response"]


# to be able to use the microphone on chrome, you will have to go to chrome://flags/#unsafely-treat-insecure-origin-as-secure and enter http://10.186.115.21:7860/
# in "Insecure origins treated as secure", enable it and relaunch chrome

# example question:
# what's the weather like outside?
# What's the closest restaurant from here?


with gr.Blocks(theme=gr.themes.Default()) as demo:
    state = gr.State(
        value={
            # "context": initial_context,
            "query": "",
            "route_points": [],
        }
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            time_picker = gr.Dropdown(
                choices=hour_options,
                label="What time is it? (HH:MM)",
                value="08:00",
                interactive=True,
            )
            history = gr.Radio(
                ["Yes", "No"],
                label="Maintain the conversation history?",
                value="No",
                interactive=True,
            )
            voice_character = gr.Radio(
                choices=[
                    "Morgan Freeman",
                    "Eddie Murphy",
                    "David Attenborough",
                    "Rick Sanches",
                ],
                label="Choose a voice",
                value="Morgan Freeman",
                show_label=True,
                interactive=True,
            )
            emotion = gr.Radio(
                choices=["Cheerful", "Grumpy"],
                label="Choose an emotion",
                value="Cheerful",
                show_label=True,
            )
            origin = gr.Textbox(
                value="Luxembourg Gare, Luxembourg", label="Origin", interactive=True
            )
            destination = gr.Textbox(
                value="Kirchberg Campus, Luxembourg",
                label="Destination",
                interactive=True,
            )

        with gr.Column(scale=2, min_width=600):
            map_plot = gr.Plot()

            # map_if = gr.Interface(fn=plot_map, inputs=year_input, outputs=map_plot)

    with gr.Row():
        with gr.Column():
            recorder = gr.Audio(
                type="filepath", label="Input audio", elem_id="recorder"
            )
            input_text = gr.Textbox(
                value="How is the weather?", label="Input text", interactive=True
            )
            vehicle_status = gr.JSON(
                value=vehicle.model_dump_json(), label="Vehicle status"
            )
        with gr.Column():
            output_audio = gr.Audio(label="output audio")
            output_text = gr.TextArea(value="", label="Output text", interactive=False)
    # iface = gr.Interface(
    #     fn=transcript,
    #     inputs=[
    #         gr.Textbox(value=initial_context, visible=False),
    #         gr.Audio(type="filepath", label="input audio", elem_id="recorder"),
    #         voice_character,
    #         emotion,
    #         place,
    #         time_picker,
    #         history,
    #         gr.State(),  # This will keep track of the context state across interactions.
    #     ],
    #     outputs=[gr.Audio(label="output audio"), gr.Textbox(visible=False), gr.State()],
    #     head=shortcut_js,
    # )

    # Update plot based on the origin and destination
    # Sets the current location and destination
    origin.submit(
        fn=calculate_route,
        inputs=[origin, destination],
        outputs=[map_plot, vehicle_status],
    )
    destination.submit(
        fn=calculate_route,
        inputs=[origin, destination],
        outputs=[map_plot, vehicle_status],
    )

    # Update time based on the time picker
    time_picker.select(fn=set_time, inputs=[time_picker], outputs=[vehicle_status])

    # Run the model if the input text is changed
    input_text.submit(fn=run_model, inputs=[input_text], outputs=[output_text])

# close all interfaces open to make the port available
gr.close_all()
# Launch the interface.

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7860, ssl_verify=False)

# iface.launch(debug=True, share=False, server_name="0.0.0.0", server_port=7860, ssl_verify=False)
