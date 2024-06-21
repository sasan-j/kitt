import copy
import time

import gradio as gr
import numpy as np
import torch
import torchaudio
from loguru import logger
from transformers import pipeline

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


def transcribe_audio(audio):
    sample_rate, data = audio
    try:
        data = data.astype(np.float32)
        data /= np.max(np.abs(data))
        text = transcriber({"sampling_rate": sample_rate, "raw": data})["text"]
        gr.Info(f"Transcribed text is: {text}\nProcessing the input...")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise Exception("Error transcribing audio.")
    return text    


def save_and_transcribe_audio(audio, save=True):
    sample_rate, data = audio
    # add timestamp to file name
    filename = f"recordings/audio{time.time()}.wav"
    if save:
        save_audio_as_wav(data, sample_rate, filename)
    return transcribe_audio(audio)
