import os
import pathlib
import time
from collections import namedtuple
from typing import List

import numpy as np
import torch
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

tts_pipeline = None

Voice = namedtuple("voice", ["name", "neutral", "angry", "speed"])

file_full_path = pathlib.Path(os.path.realpath(__file__)).parent

voices = [
    Voice(
        "Fast",
        neutral="empty",
        angry=None,
        speed=1.0,
    ),
    Voice(
        "Attenborough",
        neutral=f"{file_full_path}/audio/attenborough/neutral.wav",
        angry=None,
        speed=1.2,
    ),
    Voice(
        "Rick",
        neutral=f"{file_full_path}/audio/rick/neutral.wav",
        angry=None,
        speed=1.2,
    ),
    Voice(
        "Freeman",
        neutral=f"{file_full_path}/audio/freeman/neutral.wav",
        angry=f"{file_full_path}/audio/freeman/angry.wav",
        speed=1.1,
    ),
    Voice(
        "Walken",
        neutral=f"{file_full_path}/audio/walken/neutral.wav",
        angry=None,
        speed=1.1,
    ),
    Voice(
        "Darth Wader",
        neutral=f"{file_full_path}/audio/darth/neutral.wav",
        angry=None,
        speed=1.15,
    ),
]


def load_tts_pipeline():
    # load model for text to speech
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps"
    tts_pipeline = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts_pipeline


def compute_speaker_embedding(voice_path: str, config, pipeline, cache):
    if voice_path not in cache:
        cache[voice_path] = pipeline.synthesizer.tts_model.get_conditioning_latents(
            audio_path=voice_path,
            gpt_cond_len=config.gpt_cond_len,
            gpt_cond_chunk_len=config.gpt_cond_chunk_len,
            max_ref_length=config.max_ref_len,
            sound_norm_refs=config.sound_norm_refs,
        )
    return cache[voice_path]


voice_options = []
for voice in voices:
    if voice.neutral:
        voice_options.append(f"{voice.name} - Neutral")
    if voice.angry:
        voice_options.append(f"{voice.name} - Angry")


def voice_from_text(voice):
    for v in voices:
        if voice == f"{v.name} - Neutral":
            return v.neutral
        if voice == f"{v.name} - Angry":
            return v.angry
    raise ValueError(f"Voice {voice} not found.")


def speed_from_text(voice):
    for v in voices:
        if voice == f"{v.name} - Neutral":
            return v.speed
        if voice == f"{v.name} - Angry":
            return v.speed


def tts_xtts(
    self,
    text: str = "",
    language_name: str = "",
    reference_wav=None,
    gpt_cond_latent=None,
    speaker_embedding=None,
    split_sentences: bool = True,
    **kwargs,
) -> List[int]:
    """🐸 TTS magic. Run all the models and generate speech.

    Args:
        text (str): input text.
        speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".
        language_name (str, optional): language id for multi-language models. Defaults to "".
        speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
        style_wav ([type], optional): style waveform for GST. Defaults to None.
        style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
        reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
        reference_speaker_name ([type], optional): speaker id of reference waveform. Defaults to None.
        split_sentences (bool, optional): split the input text into sentences. Defaults to True.
        **kwargs: additional arguments to pass to the TTS model.
    Returns:
        List[int]: [description]
    """
    start_time = time.time()
    use_gl = self.vocoder_model is None
    wavs = []

    if not text and not reference_wav:
        raise ValueError(
            "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
        )

    if text:
        sens = [text]
        if split_sentences:
            print(" > Text splitted to sentences.")
            sens = self.split_into_sentences(text)
        print(sens)

    if not reference_wav:  # not voice conversion
        for sen in sens:
            outputs = self.tts_model.inference(
                sen,
                language_name,
                gpt_cond_latent,
                speaker_embedding,
                # GPT inference
                temperature=0.75,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=50,
                top_p=0.85,
                do_sample=True,
                **kwargs,
            )
            waveform = outputs["wav"]
            if (
                torch.is_tensor(waveform)
                and waveform.device != torch.device("cpu")
                and not use_gl
            ):
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # # trim silence
            # if (
            #     "do_trim_silence" in self.tts_config.audio
            #     and self.tts_config.audio["do_trim_silence"]
            # ):
            #     waveform = trim_silence(waveform, self.tts_model.ap)

            wavs += list(waveform)
            wavs += [0] * 10000

    # compute stats
    process_time = time.time() - start_time
    audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
    print(f" > Processing time: {process_time}")
    print(f" > Real-time factor: {process_time / audio_time}")
    return wavs


def tts_gradio(text, voice, cache):
    global tts_pipeline
    if not tts_pipeline:
        tts_pipeline = load_tts_pipeline()

    voice_path = voice_from_text(voice)
    speed = speed_from_text(voice)
    (gpt_cond_latent, speaker_embedding) = compute_speaker_embedding(
        voice_path, tts_pipeline.synthesizer.tts_config, tts_pipeline, cache
    )
    out = tts_xtts(
        tts_pipeline.synthesizer,
        text,
        language_name="en",
        speaker=None,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        speed=speed,
        # file_path="out.wav",
    )
    return (22050, np.array(out)), dict(text=text, voice=voice)
