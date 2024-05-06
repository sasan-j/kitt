import gradio as gr
import requests

import skills
from skills.common import config, vehicle
from skills.routing import calculate_route


# Generate options for hours (00-23)
hour_options = [f"{i:02d}:00" for i in range(24)]

def set_time(time_picker):
    vehicle.time = time_picker
    return vehicle.model_dump_json()

def get_vehicle_status(state):
    return state.value["vehicle"].model_dump_json()


# to be able to use the microphone on chrome, you will have to go to chrome://flags/#unsafely-treat-insecure-origin-as-secure and enter http://10.186.115.21:7860/
# in "Insecure origins treated as secure", enable it and relaunch chrome

# example question:
# what's the weather like outside?
# What's the closest restaurant from here?


model_answer = ""
general_context = ""
# Define the initial state with some initial context.
print(general_context)
initial_state = {"context": general_context}
initial_context = initial_state["context"]
# Create the Gradio interface.


with gr.Blocks(theme=gr.themes.Default()) as demo:
    state = gr.State(
        value={"context": initial_context, "query": "", "vehicle": vehicle, "route_points": []}
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            time_picker = gr.Dropdown(
                choices=hour_options, label="What time is it? (HH:MM)", value="08:00", interactive=True
            )
            history = gr.Radio(
                ["Yes", "No"], label="Maintain the conversation history?", value="No", interactive=True
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
            vehicle_status = gr.Textbox(
                value=get_vehicle_status(state), label="Vehicle status", interactive=False
            )
        with gr.Column():
            output_audio = gr.Audio(label="output audio")
            output_text = gr.TextArea(
                value="", label="Output text", interactive=False
            )
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
    origin.submit(fn=calculate_route, inputs=[origin, destination], outputs=[map_plot, vehicle_status])
    destination.submit(fn=calculate_route, inputs=[origin, destination], outputs=[map_plot, vehicle_status])

    # Update time based on the time picker
    time_picker.select(fn=set_time, inputs=[time_picker], outputs=[vehicle_status])

# close all interfaces open to make the port available
gr.close_all()
# Launch the interface.

if __name__ == "__main__":
    demo.launch(
        debug=True, server_name="0.0.0.0", server_port=7860, ssl_verify=False
    )

# iface.launch(debug=True, share=False, server_name="0.0.0.0", server_port=7860, ssl_verify=False)
