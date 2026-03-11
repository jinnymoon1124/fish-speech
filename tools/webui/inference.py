import html
import io
import re
from functools import partial
from typing import Any, Callable

import numpy as np
import soundfile as sf

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

MAX_CHARS_PER_CHUNK = 400
PAUSE_SENTENCE = 0.7   # pause after sentence-ending punctuation (.!?。！？)
PAUSE_COMMA = 0.35     # pause after commas (,，、)


def split_text(text: str) -> list[tuple[str, str]]:
    """Split text at punctuation marks, returning (chunk, separator_type) pairs.

    separator_type is 'sentence', 'comma', or 'none' (for the last chunk).
    """
    # Split at sentence-enders and commas, keeping the delimiter
    parts = re.split(r'([.!?。！？\n]+|[,，、]+)', text.strip())

    # Rebuild segments: pair each text part with its trailing punctuation type
    segments = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if i + 1 < len(parts):
            sep = parts[i + 1]
            chunk += sep  # keep punctuation in the text
            if re.match(r'[.!?。！？\n]+', sep):
                sep_type = "sentence"
            else:
                sep_type = "comma"
            i += 2
        else:
            sep_type = "none"
            i += 1
        if chunk.strip():
            segments.append((chunk.strip(), sep_type))

    if not segments:
        return [(text, "none")]

    # Merge small segments to stay within MAX_CHARS_PER_CHUNK
    merged = []
    current_text = ""
    current_sep = "none"
    for chunk, sep_type in segments:
        if current_text and len(current_text) + len(chunk) > MAX_CHARS_PER_CHUNK:
            merged.append((current_text.strip(), current_sep))
            current_text = chunk
            current_sep = sep_type
        else:
            current_text = current_text + " " + chunk if current_text else chunk
            current_sep = sep_type
    if current_text.strip():
        merged.append((current_text.strip(), current_sep))

    return merged if merged else [(text, "none")]


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
    all_segments = []  # list of (audio_data, sep_type) pairs
    sample_rate = None

    for i, (chunk, sep_type) in enumerate(chunks):
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
                    all_segments.append((audio_data, sep_type))
                    # Use first chunk's audio as reference for remaining chunks
                    if i == 0 and not references and not reference_id:
                        references = audio_to_reference(audio_data, sample_rate, chunk)
                case "error":
                    return None, build_html_error_message(i18n(result.error))
                case _:
                    pass

    if not all_segments:
        return None, i18n("No audio generated")

    if sample_rate and len(all_segments) > 1:
        parts = [all_segments[0][0]]
        for j in range(1, len(all_segments)):
            prev_sep = all_segments[j - 1][1]
            if prev_sep == "sentence":
                pause = PAUSE_SENTENCE
            elif prev_sep == "comma":
                pause = PAUSE_COMMA
            else:
                pause = PAUSE_COMMA
            parts.append(np.zeros(int(sample_rate * pause)))
            parts.append(all_segments[j][0])
        combined = np.concatenate(parts, axis=0)
    else:
        combined = np.concatenate([s[0] for s in all_segments], axis=0)

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
