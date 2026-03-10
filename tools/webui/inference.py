import html
import io
import re
from functools import partial
from typing import Any, Callable

import numpy as np
import soundfile as sf

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

MAX_CHARS_PER_CHUNK = 200


def split_text(text: str) -> list[str]:
    """Split long text into chunks by sentences, respecting a max character limit."""
    sentences = re.split(r'(?<=[.!?。！？\n])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) > MAX_CHARS_PER_CHUNK:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def audio_to_reference(audio_data: np.ndarray, sample_rate: int, text: str) -> list:
    """Convert generated audio numpy array to ServeReferenceAudio for reuse."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="wav")
    return [ServeReferenceAudio(audio=buf.getvalue(), text=text)]


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Splits long text into chunks. Uses the first chunk's output as
    reference audio for subsequent chunks to keep a consistent voice.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    chunks = split_text(text)
    all_segments = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        req = ServeTTSRequest(
            text=chunk,
            reference_id=reference_id if reference_id else None,
            references=references,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=int(seed) if seed else None,
            use_memory_cache=use_memory_cache,
        )

        for result in engine.inference(req):
            match result.code:
                case "final":
                    sample_rate, audio_data = result.audio
                    all_segments.append(audio_data)
                    # Use first chunk's audio as reference for remaining chunks
                    if i == 0 and not references and not reference_id:
                        references = audio_to_reference(audio_data, sample_rate, chunk)
                case "error":
                    return None, build_html_error_message(i18n(result.error))
                case _:
                    pass

    if not all_segments:
        return None, i18n("No audio generated")

    combined = np.concatenate(all_segments, axis=0)
    return (sample_rate, combined), None


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
