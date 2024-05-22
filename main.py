import time
import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import pipeline
import typer

from kitt.skills.common import config, vehicle
from kitt.skills.routing import calculate_route
from kitt.core.tts import run_tts_replicate, run_tts_fast, run_melo_tts
import ollama

from langchain.tools.base import StructuredTool
from langchain.memory import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool
from loguru import logger


from kitt.skills import (
    get_weather,
    find_route,
    get_forecast,
    vehicle_status as vehicle_status_fn,
    set_vehicle_speed,
    search_points_of_interest,
    search_along_route_w_coordinates,
    set_vehicle_destination,
    do_anything_else,
    date_time_info,
    get_weather_current_location,
    code_interpreter,
)
from kitt.skills import extract_func_args
from kitt.core import voice_options, tts_gradio

# from kitt.core.model import process_query
from kitt.core.model import generate_function_call as process_query
from kitt.core import utils as kitt_utils


global_context = {
    "vehicle": vehicle,
    "query": "How is the weather?",
    "route_points": [],
}

speaker_embedding_cache = {}
history = ChatMessageHistory()

MODEL_FUNC = "nexusraven"
MODEL_GENERAL = "llama3:instruct"

RAVEN_PROMPT_FUNC = """You are a helpful AI assistant in a car (vehicle), that follows instructions extremely well. \
Answer questions concisely and do not mention what you base your reply on."

{raven_tools}

{history}

User Query: Question: {input}<human_end>
"""


HERMES_PROMPT_FUNC = """
<|im_start|>system
You are a helpful AI assistant in a car (vehicle), that follows instructions extremely well. \
Answer questions concisely and do not mention what you base your reply on.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

USER_PREFERENCES = "I love italian food\nI like doing sports"


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


# llm = Ollama(model="nexusraven", stop=["\nReflection:", "\nThought:"], keep_alive=60*10)


# Generate options for hours (00-23)
hour_options = [f"{i:02d}:00:00" for i in range(24)]


@tool
def search_along_route(query=""):
    """Search for points of interest along the route/way to the destination.

    Args:
        query (str, optional): The type of point of interest to search for. Defaults to "restaurant".

    """
    points = global_context["route_points"]
    # maybe reshape
    return search_along_route_w_coordinates(points, query)


def set_time(time_picker):
    vehicle.time = time_picker
    return vehicle.model_dump_json()


def get_vehicle_status(state):
    return state.value["vehicle"].model_dump_json()


tools = [
    # StructuredTool.from_function(get_weather),
    # StructuredTool.from_function(find_route),
    # StructuredTool.from_function(vehicle_status_fn),
    # StructuredTool.from_function(set_vehicle_speed),
    # StructuredTool.from_function(set_vehicle_destination),
    # StructuredTool.from_function(search_points_of_interest),
    # StructuredTool.from_function(search_along_route),
    # StructuredTool.from_function(date_time_info),
    # StructuredTool.from_function(get_weather_current_location),
    # StructuredTool.from_function(code_interpreter),
    # StructuredTool.from_function(do_anything_else),
]

functions = [
    set_vehicle_speed,
    set_vehicle_destination,
    get_weather,
    find_route,
    search_points_of_interest,
    search_along_route,
]
openai_tools = [convert_to_openai_tool(tool) for tool in functions]


def run_generic_model(query):
    print(f"Running the generic model with query: {query}")
    data = {
        "prompt": f"Answer the question below in a short and concise manner.\n{query}",
        "model": MODEL_GENERAL,
        "options": {
            # "temperature": 0.1,
            # "stop":["\nReflection:", "\nThought:"]
        },
    }
    out = ollama.generate(**data)
    return out["response"]


def clear_history():
    history.clear()


def run_nexusraven_model(query, voice_character, state):
    global_context["prompt"] = get_prompt(RAVEN_PROMPT_FUNC, query, "", tools)
    print("Prompt: ", global_context["prompt"])
    data = {
        "prompt": global_context["prompt"],
        # "streaming": False,
        "model": "nexusraven",
        # "model": "smangrul/llama-3-8b-instruct-function-calling",
        "raw": True,
        "options": {"temperature": 0.5, "stop": ["\nReflection:", "\nThought:"]},
    }
    out = ollama.generate(**data)
    llm_response = out["response"]
    if "Call: " in llm_response:
        print(f"llm_response: {llm_response}")
        llm_response = llm_response.replace("<bot_end>", " ")
        func_name, kwargs = extract_func_args(llm_response)
        print(f"Function: {func_name}, Args: {kwargs}")
        if func_name == "do_anything_else":
            output_text = run_generic_model(query)
        else:
            output_text = use_tool(func_name, kwargs, tools)
    else:
        output_text = out["response"]

    if type(output_text) == tuple:
        output_text = output_text[0]
    gr.Info(f"Output text: {output_text}\nGenerating voice output...")
    return (
        output_text,
        tts_gradio(output_text, voice_character, speaker_embedding_cache)[0],
    )


def run_llama3_model(query, voice_character, state):

    assert len(functions) > 0, "No functions to call"
    assert len(openai_tools) > 0, "No openai tools to call"

    output_text = process_query(
        query,
        history=history,
        user_preferences=state["user_preferences"],
        tools=openai_tools,
        functions=functions,
        backend=state["llm_backend"],
    )
    gr.Info(f"Output text: {output_text}\nGenerating voice output...")
    voice_out = None
    if state["tts_enabled"]:
        # voice_out = run_tts_replicate(output_text, voice_character)
        # voice_out = run_tts_fast(output_text)[0]
        voice_out = run_melo_tts(output_text, voice_character)    
        # voice_out = tts_gradio(output_text, voice_character, speaker_embedding_cache)[0]
    return (
        output_text,
        voice_out,
    )


def run_model(query, voice_character, state):
    model = state.get("model", "nexusraven")
    query = query.strip().replace("'", "")
    logger.info(
        f"Running model: {model} with query: {query}, voice_character: {voice_character} and llm_backend: {state['llm_backend']}, tts_enabled: {state['tts_enabled']}"
    )
    global_context["query"] = query
    if model == "nexusraven":
        text, voice = run_nexusraven_model(query, voice_character, state)
    elif model == "llama3":
        text, voice = run_llama3_model(query, voice_character, state)
    else:
        text, voice = "Error running model", None

    if not state["enable_history"]:
        history.clear()
    return text, voice, vehicle.model_dump_json()


def calculate_route_gradio(origin, destination):
    vehicle_status, points = calculate_route(origin, destination)
    plot = kitt_utils.plot_route(points, vehicle=vehicle.location_coordinates)
    global_context["route_points"] = points
    vehicle.location_coordinates = points[0]["latitude"], points[0]["longitude"]
    return plot, vehicle_status, 0


def update_vehicle_status(trip_progress, origin, destination):
    if not global_context["route_points"]:
        vehicle_status, points = calculate_route(origin, destination)
        global_context["route_points"] = points
    n_points = len(global_context["route_points"])
    index = min(int(trip_progress / 100 * n_points), n_points - 1)
    print(f"Trip progress: {trip_progress} len: {n_points}, index: {index}")
    new_coords = global_context["route_points"][index]
    new_coords = new_coords["latitude"], new_coords["longitude"]
    print(f"Trip progress: {trip_progress}, len: {n_points}, new_coords: {new_coords}")
    vehicle.location_coordinates = new_coords
    vehicle.location = ""
    plot = kitt_utils.plot_route(
        global_context["route_points"], vehicle=vehicle.location_coordinates
    )
    return vehicle.model_dump_json(), plot


device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)


def save_audio_as_wav(data, sample_rate, file_path):
    # make a tensor from the numpy array
    data = torch.tensor(data).reshape(1, -1)
    torchaudio.save(
        file_path, data, sample_rate=sample_rate, bits_per_sample=16, encoding="PCM_S"
    )


def save_and_transcribe_audio(audio):
    try:
        # capture the audio and save it to a file as wav or mp3
        # file_name = save("audioinput.wav")
        sr, y = audio
        # y = y.astype(np.float32)
        # y /= np.max(np.abs(y))

        # add timestamp to file name
        filename = f"recordings/audio{time.time()}.wav"
        save_audio_as_wav(y, sr, filename)

        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        text = transcriber({"sampling_rate": sr, "raw": y})["text"]
        gr.Info(f"Transcribed text is: {text}\nProcessing the input...")

    except Exception as e:
        print(f"Error: {e}")
        return "Error transcribing audio."
    return text


def save_and_transcribe_run_model(audio, voice_character, state):
    text = save_and_transcribe_audio(audio)
    out_text, out_voice, vehicle_status = run_model(text, voice_character, state)
    return text, out_text, out_voice, vehicle_status


def set_tts_enabled(tts_enabled, state):
    new_tts_enabled = tts_enabled == "Yes"
    logger.info(
        f"TTS enabled was {state['tts_enabled']} and changed to {new_tts_enabled}"
    )
    state["tts_enabled"] = new_tts_enabled
    return state


def set_llm_backend(llm_backend, state):
    new_llm_backend = "ollama" if llm_backend == "Ollama" else "replicate"
    logger.info(
        f"LLM backend was {state['llm_backend']} and changed to {new_llm_backend}"
    )
    state["llm_backend"] = new_llm_backend
    return state


def set_user_preferences(preferences, state):
    new_preferences = preferences
    logger.info(f"User preferences changed to: {new_preferences}")
    state["user_preferences"] = new_preferences
    return state


def set_enable_history(enable_history, state):
    new_enable_history = enable_history == "Yes"
    logger.info(
        f"Enable history was {state['enable_history']} and changed to {new_enable_history}"
    )
    state["enable_history"] = new_enable_history
    return state


# to be able to use the microphone on chrome, you will have to go to chrome://flags/#unsafely-treat-insecure-origin-as-secure and enter http://10.186.115.21:7860/
# in "Insecure origins treated as secure", enable it and relaunch chrome

# example question:
# what's the weather like outside?
# What's the closest restaurant from here?


ORIGIN = "Mondorf-les-Bains, Luxembourg"
DESTINATION = "Rue Alphonse Weicker, Luxembourg"
DEFAULT_LLM_BACKEND = "ollama"
ENABLE_HISTORY = True
ENABLE_TTS = True


def create_demo(tts_server: bool = False, model="llama3"):
    print(f"Running the demo with model: {model} and TTSServer: {tts_server}")
    with gr.Blocks(theme=gr.themes.Default()) as demo:
        state = gr.State(
            value={
                # "context": initial_context,
                "query": "",
                "route_points": [],
                "model": model,
                "tts_enabled": ENABLE_TTS,
                "llm_backend": DEFAULT_LLM_BACKEND,
                "user_preferences": USER_PREFERENCES,
                "enable_history": ENABLE_HISTORY,
            }
        )
        trip_points = gr.State(value=[])
        plot, vehicle_status, _ = calculate_route_gradio(ORIGIN, DESTINATION)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                time_picker = gr.Dropdown(
                    choices=hour_options,
                    label="What time is it? (HH:MM)",
                    value="08:00:00",
                    interactive=True,
                )
                voice_character = gr.Radio(
                    choices=voice_options,
                    label="Choose a voice",
                    value=voice_options[0],
                    show_label=True,
                )
                # voice_character = gr.Textbox(
                #     label="Choose a voice",
                #     value="freeman",
                #     show_label=True,
                # )
                origin = gr.Textbox(
                    value=ORIGIN,
                    label="Origin",
                    interactive=True,
                )
                destination = gr.Textbox(
                    value=DESTINATION,
                    label="Destination",
                    interactive=True,
                )
                preferences = gr.Textbox(
                    value=USER_PREFERENCES,
                    label="User preferences",
                    lines=3,
                    interactive=True,
                )

            with gr.Column(scale=2, min_width=600):
                map_plot = gr.Plot(value=plot, label="Map")
                trip_progress = gr.Slider(
                    0, 100, step=5, label="Trip progress", interactive=True
                )

                # map_if = gr.Interface(fn=plot_map, inputs=year_input, outputs=map_plot)

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    type="numpy",
                    sources=["microphone"],
                    label="Input audio",
                    elem_id="input_audio",
                )
                input_text = gr.Textbox(
                    value="How is the weather?", label="Input text", interactive=True
                )
                with gr.Accordion("Debug"):
                    input_audio_debug = gr.Audio(
                        type="numpy",
                        sources=["microphone"],
                        label="Input audio",
                        elem_id="input_audio",
                    )
                    input_text_debug = gr.Textbox(
                        value="How is the weather?",
                        label="Input text",
                        interactive=True,
                    )
                vehicle_status = gr.JSON(
                    value=vehicle.model_dump_json(), label="Vehicle status"
                )
                with gr.Accordion("Config"):
                    tts_enabled = gr.Radio(
                        ["Yes", "No"],
                        label="Enable TTS",
                        value="Yes" if ENABLE_TTS else "No",
                        interactive=True,
                    )
                    llm_backend = gr.Radio(
                        choices=["Ollama", "Replicate"],
                        label="LLM Backend",
                        value=DEFAULT_LLM_BACKEND.title(),
                        interactive=True,
                    )
                    enable_history = gr.Radio(
                        ["Yes", "No"],
                        label="Maintain the conversation history?",
                        value="Yes" if ENABLE_HISTORY else "No",
                        interactive=True,
                    )
                # Push button
                clear_history_btn = gr.Button(value="Clear History")
            with gr.Column():
                output_audio = gr.Audio(label="output audio", autoplay=True)
                output_text = gr.TextArea(
                    value="", label="Output text", interactive=False
                )

        # Update plot based on the origin and destination
        # Sets the current location and destination
        origin.submit(
            fn=calculate_route_gradio,
            inputs=[origin, destination],
            outputs=[map_plot, vehicle_status, trip_progress],
        )
        destination.submit(
            fn=calculate_route_gradio,
            inputs=[origin, destination],
            outputs=[map_plot, vehicle_status, trip_progress],
        )
        preferences.submit(
            fn=set_user_preferences, inputs=[preferences, state], outputs=[state]
        )

        # Update time based on the time picker
        time_picker.select(fn=set_time, inputs=[time_picker], outputs=[vehicle_status])

        # Run the model if the input text is changed
        input_text.submit(
            fn=run_model,
            inputs=[input_text, voice_character, state],
            outputs=[output_text, output_audio, vehicle_status],
        )
        input_text_debug.submit(
            fn=run_model,
            inputs=[input_text_debug, voice_character, state],
            outputs=[output_text, output_audio, vehicle_status],
        )

        # Set the vehicle status based on the trip progress
        trip_progress.release(
            fn=update_vehicle_status,
            inputs=[trip_progress, origin, destination],
            outputs=[vehicle_status, map_plot],
        )

        # Save and transcribe the audio
        input_audio.stop_recording(
            fn=save_and_transcribe_run_model,
            inputs=[input_audio, voice_character, state],
            outputs=[input_text, output_text, output_audio, vehicle_status],
        )
        input_audio_debug.stop_recording(
            fn=save_and_transcribe_audio,
            inputs=[input_audio_debug],
            outputs=[input_text_debug],
        )

        # Clear the history
        clear_history_btn.click(fn=clear_history, inputs=[], outputs=[])

        # Config
        tts_enabled.change(
            fn=set_tts_enabled, inputs=[tts_enabled, state], outputs=[state]
        )
        llm_backend.change(
            fn=set_llm_backend, inputs=[llm_backend, state], outputs=[state]
        )
        enable_history.change(
            fn=set_enable_history, inputs=[enable_history, state], outputs=[state]
        )

    return demo


# close all interfaces open to make the port available
gr.close_all()


demo = create_demo(False, "llama3")
demo.launch(
    debug=True,
    server_name="0.0.0.0",
    server_port=7860,
    ssl_verify=False,
    share=False,
)
app = typer.Typer()


@app.command()
def run(tts_server: bool = False):
    global demo
    demo = create_demo(tts_server)
    demo.launch(
        debug=True, server_name="0.0.0.0", server_port=7860, ssl_verify=True, share=True
    )


@app.command()
def dev(tts_server: bool = False, model: str = "llama3"):
    demo = create_demo(tts_server, model)
    demo.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        ssl_verify=False,
        share=False,
    )


if __name__ == "__main__":
    app()
