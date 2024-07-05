import copy
from collections import namedtuple

import torch
from loguru import logger
from replicate import Client

from kitt.skills.common import config

replicate = Client(api_token=config.REPLICATE_API_KEY)

Voice = namedtuple("voice", ["name", "neutral", "angry", "speed"])

voices_replicate = [
    Voice(
        "Fast",
        neutral="empty",
        angry=None,
        speed=1.0,
    ),
    Voice(
        "Attenborough",
        neutral="https://zebel.ams3.digitaloceanspaces.com/xtts/short/attenborough-neutral.wav",
        angry=None,
        speed=1.2,
    ),
    Voice(
        "Rick",
        neutral="https://zebel.ams3.digitaloceanspaces.com/xtts/short/rick-neutral.wav",
        angry=None,
        speed=1.2,
    ),
    Voice(
        "Freeman",
        neutral="https://zebel.ams3.digitaloceanspaces.com/xtts/short/freeman-neutral.wav",
        angry="https://zebel.ams3.digitaloceanspaces.com/xtts/short/freeman-angry.wav",
        speed=1.1,
    ),
    Voice(
        "Walken",
        neutral="https://zebel.ams3.digitaloceanspaces.com/xtts/short/walken-neutral.wav",
        angry=None,
        speed=1.1,
    ),
    Voice(
        "Darth Wader",
        neutral="https://zebel.ams3.digitaloceanspaces.com/xtts/short/darth-neutral.wav",
        angry=None,
        speed=1.15,
    ),
]


def prep_for_tts(text: str):
    text_tts = copy.deepcopy(text)
    text_tts = text_tts.replace("km/h", " kilometers per hour")
    text_tts = text_tts.replace("°C", " degree Celsius")
    text_tts = text_tts.replace("°F", " degree Fahrenheit")
    text_tts = text_tts.replace("km", " kilometers")
    return text_tts


def voice_from_text(voice, voices):
    for v in voices:
        if voice == f"{v.name} - Neutral":
            return v.neutral
        if voice == f"{v.name} - Angry":
            return v.angry
    raise ValueError(f"Voice {voice} not found.")


def speed_from_text(voice, voices):
    for v in voices:
        if voice == f"{v.name} - Neutral":
            return v.speed
        if voice == f"{v.name} - Angry":
            return v.speed


def run_tts_replicate(text: str, voice_character: str):
    voice = voice_from_text(voice_character, voices_replicate)

    input = {"text": text, "speaker": voice, "cleanup_voice": True}

    output = replicate.run(
        # "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71",
        "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
        input=input,
    )
    logger.info(f"sound output: {output}")
    return output


def load_melo_tts():
    from melo.api import TTS as MeloTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeloTTS(language="EN", device=device)
    return model


melo_tts = None


def run_melo_tts(text: str, voice: str):
    global melo_tts
    if melo_tts is None:
        try:
            melo_tts = load_melo_tts()
        except ImportError as e:
            logger.error(f"Error loading MeloTTS: {e}")
            melo_tts = None
            raise ValueError("MeloTTS not available.")

    speed = 1.0
    speaker_ids = melo_tts.hps.data.spk2id
    audio = melo_tts.tts_to_file(text, speaker_ids["EN-Default"], None, speed=speed)
    return melo_tts.hps.data.sampling_rate, audio
