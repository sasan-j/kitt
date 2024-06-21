import gradio as gr
from langchain.memory import ChatMessageHistory
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from loguru import logger

from kitt.core import tts_gradio
from kitt.core import utils as kitt_utils
from kitt.core import voice_options
from kitt.core.model import generate_function_call as process_query
from kitt.core.stt import save_and_transcribe_audio
from kitt.core.tts import prep_for_tts, run_melo_tts, run_tts_replicate
from kitt.skills import (
    code_interpreter,
    date_time_info,
    do_anything_else,
    extract_func_args,
    find_route,
    get_forecast,
    get_weather,
    get_weather_current_location,
    search_along_route_w_coordinates,
    search_points_of_interest,
    set_vehicle_destination,
    set_vehicle_speed,
)
from kitt.skills.common import config, vehicle
from kitt.skills.routing import calculate_route, find_address

ORIGIN = "Luxembourg, Luxembourg"
DESTINATION = "Paris, France"
DEFAULT_LLM_BACKEND = "replicate"
ENABLE_HISTORY = True
ENABLE_TTS = True
TTS_BACKEND = "local"
USER_PREFERENCES = "User prefers italian food."

global_context = {
    "vehicle": vehicle,
    "query": "How is the weather?",
    "route_points": [],
    "origin": ORIGIN,
    "destination": DESTINATION,
    "enable_history": ENABLE_HISTORY,
    "tts_enabled": ENABLE_TTS,
    "tts_backend": TTS_BACKEND,
    "llm_backend": DEFAULT_LLM_BACKEND,
    "map_origin": ORIGIN,
    "map_destination": DESTINATION,
    "update_proxy": 0,
    "map": None,
}

speaker_embedding_cache = {}
history = ChatMessageHistory()


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
    return vehicle


functions = [
    # set_vehicle_speed,
    set_vehicle_destination,
    get_weather,
    find_route,
    search_points_of_interest,
    search_along_route,
]
openai_tools = [convert_to_openai_tool(tool) for tool in functions]


def clear_history():
    logger.info("Clearing the conversation history...")
    history.clear()


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
    output_text_tts = prep_for_tts(output_text)
    voice_out = None
    if global_context["tts_enabled"]:
        if "Fast" in voice_character:
            voice_out = run_melo_tts(output_text_tts, voice_character)
        elif global_context["tts_backend"] == "replicate":
            voice_out = run_tts_replicate(output_text_tts, voice_character)
        else:
            voice_out = tts_gradio(
                output_text_tts, voice_character, speaker_embedding_cache
            )[0]
    return (
        output_text,
        voice_out,
    )


def run_model(query, voice_character, state):
    model = state.get("model", "llama3")
    query = query.strip().replace("'", "")
    logger.info(
        f"Running model: {model} with query: {query}, voice_character: {voice_character} and llm_backend: {state['llm_backend']}, tts_enabled: {state['tts_enabled']}"
    )
    global_context["query"] = query
    text, voice = run_llama3_model(query, voice_character, state)

    if not state["enable_history"]:
        history.clear()
    global_context["update_proxy"] += 1

    return (
        text,
        voice,
        vehicle.model_dump(),
        state,
        dict(update_proxy=global_context["update_proxy"]),
    )


def calculate_route_gradio(origin, destination):
    _, points = calculate_route(origin, destination)
    plot = kitt_utils.plot_route(points, vehicle=vehicle.location_coordinates)
    global_context["map"] = plot
    global_context["route_points"] = points
    # state.value["route_points"] = points
    vehicle.location_coordinates = points[0]["latitude"], points[0]["longitude"]
    return plot, vehicle.model_dump(), 0


def update_vehicle_status(trip_progress, origin, destination, state):
    if not global_context["route_points"]:
        _, points = calculate_route(origin, destination)
        global_context["route_points"] = points
    global_context["destination"] = destination
    global_context["route_points"] = global_context["route_points"]
    n_points = len(global_context["route_points"])
    index = min(int(trip_progress / 100 * n_points), n_points - 1)
    logger.info(f"Trip progress: {trip_progress} len: {n_points}, index: {index}")
    new_coords = global_context["route_points"][index]
    new_coords = new_coords["latitude"], new_coords["longitude"]
    logger.info(
        f"Trip progress: {trip_progress}, len: {n_points}, new_coords: {new_coords}"
    )
    vehicle.location_coordinates = new_coords
    new_vehicle_location = find_address(new_coords[0], new_coords[1])
    vehicle.location = new_vehicle_location
    plot = kitt_utils.plot_route(
        global_context["route_points"], vehicle=vehicle.location_coordinates
    )
    return vehicle, plot, state


def save_and_transcribe_run_model(audio, voice_character, state):
    text = save_and_transcribe_audio(audio)
    out_text, out_voice, vehicle_status, state, update_proxy = run_model(
        text, voice_character, state
    )
    return None, text, out_text, out_voice, vehicle_status, state, update_proxy


def set_tts_enabled(tts_enabled, state):
    new_tts_enabled = tts_enabled == "Yes"
    logger.info(
        f"TTS enabled was {state['tts_enabled']} and changed to {new_tts_enabled}"
    )
    state["tts_enabled"] = new_tts_enabled
    global_context["tts_enabled"] = new_tts_enabled
    return state


def set_llm_backend(llm_backend, state):
    new_llm_backend = "ollama" if llm_backend == "Ollama" else "replicate"
    logger.info(
        f"LLM backend was {state['llm_backend']} and changed to {new_llm_backend}"
    )
    state["llm_backend"] = new_llm_backend
    global_context["llm_backend"] = new_llm_backend
    return state


def set_user_preferences(preferences, state):
    new_preferences = preferences
    logger.info(f"User preferences changed to: {new_preferences}")
    state["user_preferences"] = new_preferences
    global_context["user_preferences"] = new_preferences
    return state


def set_enable_history(enable_history, state):
    new_enable_history = enable_history == "Yes"
    logger.info(
        f"Enable history was {state['enable_history']} and changed to {new_enable_history}"
    )
    state["enable_history"] = new_enable_history
    global_context["enable_history"] = new_enable_history
    return state


def set_tts_backend(tts_backend, state):
    new_tts_backend = tts_backend.lower()
    logger.info(
        f"TTS backend was {state['tts_backend']} and changed to {new_tts_backend}"
    )
    state["tts_backend"] = new_tts_backend
    global_context["tts_backend"] = new_tts_backend
    return state


def conditional_update():
    if global_context["destination"] != vehicle.destination:
        global_context["destination"] = vehicle.destination

    if global_context["origin"] != vehicle.location:
        global_context["origin"] = vehicle.location

    if (
        global_context["map_origin"] != vehicle.location
        or global_context["map_destination"] != vehicle.destination
        or global_context["update_proxy"] == 0
    ):
        logger.info(f"Updating the map plot... in conditional_update")
        map_plot, _, _ = calculate_route_gradio(vehicle.location, vehicle.destination)
        global_context["map"] = map_plot
    return global_context["map"]


# to be able to use the microphone on chrome, you will have to go to chrome://flags/#unsafely-treat-insecure-origin-as-secure and enter http://10.186.115.21:7860/
# in "Insecure origins treated as secure", enable it and relaunch chrome

# example question:
# what's the weather like outside?
# What's the closest restaurant from here?


def create_demo(tts_server: bool = False, model="llama3"):
    print(f"Running the demo with model: {model} and TTSServer: {tts_server}")
    with gr.Blocks(theme=gr.themes.Default(), title="KITT") as demo:
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
                "tts_backend": TTS_BACKEND,
                "destination": DESTINATION,
            }
        )

        plot, _, _ = calculate_route_gradio(ORIGIN, DESTINATION)
        global_context["map"] = plot

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                vehicle_status = gr.JSON(
                    value=vehicle.model_dump(), label="Vehicle status"
                )
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

            # with gr.Column(scale=1, min_width=300):
            #     gr.Image("linkedin-1.png", label="Linkedin - Sasan Jafarnejad")
            #     gr.Image(
            #         "team-ubix.png",
            #         label="Research Team - UBIX - University of Luxembourg",
            #     )

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
                    update_proxy = gr.JSON(
                        value=dict(update_proxy=0),
                        label="Global context",
                    )
                with gr.Accordion("Config"):
                    tts_enabled = gr.Radio(
                        ["Yes", "No"],
                        label="Enable TTS",
                        value="Yes" if ENABLE_TTS else "No",
                        interactive=True,
                    )
                    tts_backend = gr.Radio(
                        ["Local", "Replicate"],
                        label="TTS Backend",
                        value=TTS_BACKEND.title(),
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
            outputs=[output_text, output_audio, vehicle_status, state, update_proxy],
        )
        input_text_debug.submit(
            fn=run_model,
            inputs=[input_text_debug, voice_character, state],
            outputs=[output_text, output_audio, vehicle_status, state, update_proxy],
        )

        # Set the vehicle status based on the trip progress
        trip_progress.release(
            fn=update_vehicle_status,
            inputs=[trip_progress, origin, destination, state],
            outputs=[vehicle_status, map_plot, state],
        )

        # Save and transcribe the audio
        input_audio.stop_recording(
            fn=save_and_transcribe_run_model,
            inputs=[input_audio, voice_character, state],
            outputs=[
                input_audio,
                input_text,
                output_text,
                output_audio,
                vehicle_status,
                state,
                update_proxy,
            ],
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
        tts_backend.change(
            fn=set_tts_backend, inputs=[tts_backend, state], outputs=[state]
        )
        llm_backend.change(
            fn=set_llm_backend, inputs=[llm_backend, state], outputs=[state]
        )
        enable_history.change(
            fn=set_enable_history, inputs=[enable_history, state], outputs=[state]
        )
        update_proxy.change(fn=conditional_update, inputs=[], outputs=[map_plot])

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
