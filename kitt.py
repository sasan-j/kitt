import gradio as gr
import plotly.express as px
import requests

# INTERFACE WITH AUDIO TO AUDIO


def transcript(
    general_context, link_to_audio, voice, emotion, place, time, delete_history, state
):
    """this function manages speech-to-text to input Fnanswer function and text-to-speech with the Fnanswer output"""
    # load audio from a specific path
    audio_path = link_to_audio
    audio_array, sampling_rate = librosa.load(
        link_to_audio, sr=16000
    )  # "sr=16000" ensures that the sampling rate is as required

    # process the audio array
    input_features = processor(
        audio_array, sampling_rate, return_tensors="pt"
    ).input_features
    predicted_ids = modelw.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    quest_processing = FnAnswer(
        general_context, transcription, place, time, delete_history, state
    )
    state = quest_processing[2]
    print("langue " + quest_processing[3])

    tts.tts_to_file(
        text=str(quest_processing[0]),
        file_path="output.wav",
        speaker_wav=f"Audio_Files/{voice}.wav",
        language=quest_processing[3],
        emotion="angry",
    )

    audio_path = "output.wav"
    return audio_path, state["context"], state


# to be able to use the microphone on chrome, you will have to go to chrome://flags/#unsafely-treat-insecure-origin-as-secure and enter http://10.186.115.21:7860/
# in "Insecure origins treated as secure", enable it and relaunch chrome

# example question:
# what's the weather like outside?
# What's the closest restaurant from here?


import gradio as gr

shortcut_js = """
<script>
function shortcuts(e) {
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        break;
        default:
        if (e.key.toLowerCase() == "r" && e.ctrlKey) {
            console.log("recording")
            document.getElementById("recorder").start_recording();
        }
        if (e.key.toLowerCase() == "s" && e.ctrlKey) {
            console.log("stopping")
            document.getElementById("recorder").stop_recording();
        }
    }
}
document.addEventListener('keypress', shortcuts, false);
</script>
"""

# with gr.Blocks(head=shortcut_js) as demo:
#     action_button = gr.Button(value="Name", elem_id="recorder")
#     textbox = gr.Textbox()
#     action_button.click(lambda : "button pressed", None, textbox)

# demo.launch()


# Generate options for hours (00-23)
hour_options = [f"{i:02d}:00:00" for i in range(24)]

model_answer = ""
general_context = ""
# Define the initial state with some initial context.
print(general_context)
initial_state = {"context": general_context}
initial_context = initial_state["context"]
# Create the Gradio interface.


with gr.Blocks(theme=gr.themes.Default()) as demo:

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            time_picker = gr.Dropdown(
                choices=hour_options, label="What time is it?", value="08:00:00"
            )
            history = gr.Radio(
                ["Yes", "No"], label="Maintain the conversation history?", value="No"
            )
            voice_character = gr.Radio(
                choices=[
                    "Rick Sanches",
                    "Eddie Murphy",
                    "David Attenborough",
                    "Morgan Freeman",
                ],
                label="Choose a voice",
                value="Rick Sancher",
                show_label=True,
            )
            emotion = gr.Radio(
                choices=["Cheerful", "Grumpy"],
                label="Choose an emotion",
                value="Cheerful",
                show_label=True,
            )
            # place = gr.Radio(
            #     choices=[
            #         "Luxembourg Gare, Luxembourg",
            #         "Kirchberg Campus, Kirchberg",
            #         "Belval Campus, Belval",
            #         "Eiffel Tower, Paris",
            #         "Thionville, France",
            #     ],
            #     label="Choose a location for your car",
            #     value="Kirchberg Campus, Kirchberg",
            #     show_label=True,
            # )
            origin = gr.Textbox(
                value="Luxembourg Gare, Luxembourg", label="Origin", interactive=True
            )
            destination = gr.Textbox(
                value="Kirchberg Campus, Kirchberg",
                label="Destination",
                interactive=True,
            )
            recorder = gr.Audio(
                type="filepath", label="input audio", elem_id="recorder"
            )
        with gr.Column(scale=2, min_width=600):
            map_plot = gr.Plot()
            origin.submit(fn=calculate_route, outputs=map_plot)
            destination.submit(fn=calculate_route, outputs=map_plot)
            output_audio = gr.Audio(label="output audio")
            # map_if = gr.Interface(fn=plot_map, inputs=year_input, outputs=map_plot)

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

# close all interfaces open to make the port available
gr.close_all()
# Launch the interface.

demo.queue().launch(
    debug=True, server_name="0.0.0.0", server_port=7860, ssl_verify=False
)

# iface.launch(debug=True, share=False, server_name="0.0.0.0", server_port=7860, ssl_verify=False)
