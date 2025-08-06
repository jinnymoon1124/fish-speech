import os
import queue
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Callable, Dict, Any

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.content_sequence import (
    ContentSequence,
    TextPart,
    VQPart,
)
from fish_speech.tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: torch.Tensor = None,
) -> torch.Tensor:
    # print(x, torch.count_nonzero(vq_masks))
    x = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        # Convert logits to probs
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    return codebooks.T


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    i = 0  # 루프 외부에서도 사용하기 위해 초기화
    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 32
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        # 긴 텍스트 생성을 위해 조기 종료 조건을 완화
        # IM_END_TOKEN이 나와도 최소한의 토큰은 생성하도록 수정
        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            # 최소 32개 토큰은 생성하도록 보장 (약 1-2초 분량)
            if i >= 32:
                break
            else:
                # 조기 종료를 방지하기 위해 IM_END_TOKEN을 다른 토큰으로 대체
                logger.info(f"IM_END_TOKEN detected early at step {i}, continuing generation...")
                # 마지막 토큰을 semantic_begin_id로 대체하여 생성 계속
                cur_token[0, 0, -1] = model.tokenizer.semantic_begin_id

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=num_samples,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temperature = torch.tensor(
        sampling_kwargs["temperature"], device=device, dtype=torch.bfloat16
    )
    top_p = torch.tensor(sampling_kwargs["top_p"], device=device, dtype=torch.bfloat16)
    repetition_penalty = torch.tensor(
        sampling_kwargs["repetition_penalty"], device=device, dtype=torch.bfloat16
    )

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x
    return seq


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Initialize cache
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            # mode="max-autotune-no-cudagraphs",
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable[..., torch.Tensor],
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    # Type guard to ensure we have lists when use_prompt is True
    if use_prompt:
        assert isinstance(prompt_text, list) and isinstance(prompt_tokens, list), "Prompt text and tokens must be lists when using prompts"
        assert len(prompt_text) == len(prompt_tokens), "Prompt text and tokens must have the same length"

    if prompt_tokens is not None:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    # Log reference audio information
    if use_prompt and prompt_text is not None and prompt_tokens is not None:
        logger.info(f"🎵 Using reference audio prompts: {len(prompt_text)} references")
        for i, (text_item, tokens) in enumerate(zip(prompt_text, prompt_tokens)):
            logger.info(f"  Reference {i+1}: text='{text_item[:50]}...' tokens_shape={tokens.shape}")
    else:
        logger.info("🚫 No reference audio provided - using random voice")

    # Split text into chunks if iterative prompting is enabled
    if iterative_prompt and chunk_length > 0:
        # 강사 영상의 특성을 고려한 자연스러운 호흡과 말의 흐름을 위한 청킹
        # 움직임이 많고 핸드마이크 사용 등을 고려하여 더 자연스러운 구간 분할
        text_with_pauses = text
        
        # 다양한 언어의 문장 부호 처리 (한국어, 영어, 일본어 등)
        pause_replacements = [
            ('。', '。 '),   # 일본어 마침표
            ('！', '！ '),   # 일본어 느낌표  
            ('？', '？ '),   # 일본어 물음표
            ('.', '. '),     # 영어 마침표
            ('!', '! '),     # 영어 느낌표
            ('?', '? '),     # 영어 물음표
            (',', ', '),     # 쉼표 - 짧은 호흡
            (':', ': '),     # 콜론 - 설명 전 휴지
            (';', '; '),     # 세미콜론 - 문장 연결 휴지
            (' - ', ' - '),  # 대시 - 강조나 설명 휴지
            ('...', '... '), # 말줄임 - 자연스러운 여운
        ]
        
        for old, new in pause_replacements:
            text_with_pauses = text_with_pauses.replace(old, new)
        
        # 연속된 공백 정리
        import re
        text_with_pauses = re.sub(r'\s+', ' ', text_with_pauses)
        
        # 강사의 말하기 패턴을 고려한 문장 분할
        # 마침표, 느낌표, 물음표를 기준으로 주요 구분점 설정
        major_breaks = ['。 ', '！ ', '？ ', '. ', '! ', '? ']
        sentences = [text_with_pauses]
        
        for break_mark in major_breaks:
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split(break_mark)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:  # 마지막이 아닌 경우
                        new_sentences.append((part + break_mark).strip())
                    elif part.strip():  # 마지막이면서 내용이 있는 경우
                        new_sentences.append(part.strip())
            sentences = [s for s in new_sentences if s]
        
        text_chunks = []
        current_chunk = ""
        
        # 강사 영상 특성을 고려한 청크 크기 조정
        # 움직임이 많고 핸드마이크 등으로 인한 음성 변화를 고려하여 
        # 더 작은 청크로 분할하여 일관성 유지
        adaptive_chunk_length = min(chunk_length, 300)  # 최대 300자로 제한
        
        for sentence in sentences:
            # 문장이 너무 길면 의미 단위로 더 세분화
            if len(sentence) > adaptive_chunk_length:
                # 다양한 구분점을 활용한 세분화 (쉼표, 콜론, 세미콜론, 대시 등)
                sub_delimiters = [', ', ': ', '; ', ' - ', ' and ', ' but ', ' so ', ' because ']
                best_split = [sentence]  # 기본값
                
                # 가장 적절한 구분점 찾기
                for delimiter in sub_delimiters:
                    if delimiter in sentence:
                        potential_split = sentence.split(delimiter)
                        # 각 부분이 너무 작지 않고 적절한 크기인지 확인
                        if all(20 <= len(part.strip()) <= adaptive_chunk_length for part in potential_split if part.strip()):
                            best_split = [part.strip() + delimiter if i < len(potential_split)-1 else part.strip() 
                                        for i, part in enumerate(potential_split) if part.strip()]
                            break
                
                # 세분화된 문장들을 청크에 추가
                for sub_sentence in best_split:
                    if len(current_chunk) + len(sub_sentence) + 1 <= adaptive_chunk_length:
                        if current_chunk:
                            current_chunk += ' '
                        current_chunk += sub_sentence
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                        current_chunk = sub_sentence
            else:
                # 적절한 크기의 문장은 그대로 추가
                if len(current_chunk) + len(sentence) + 1 <= adaptive_chunk_length:
                    if current_chunk:
                        current_chunk += ' '
                    current_chunk += sentence
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        if current_chunk:
            text_chunks.append(current_chunk.strip())
        
        # 매우 짧은 텍스트는 단일 청크로 처리하되, 휴지 정보는 유지
        if len(text_chunks) <= 1 and len(text) < adaptive_chunk_length * 0.7:
            text_chunks = [text_with_pauses]
        
        logger.info(f"📊 강사 영상 최적화 청킹 완료: {len(sentences)}개 문장 -> {len(text_chunks)}개 청크 (최대 {adaptive_chunk_length}자)")
    else:
        text_chunks = [text]

    logger.info(f"📝 Processing text in {len(text_chunks)} chunks")
    for i, chunk in enumerate(text_chunks):
        logger.info(f"  Chunk {i+1}: '{chunk[:50]}...' (length: {len(chunk)})")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        all_codes = []
        original_reference_tokens = list(prompt_tokens) if prompt_tokens else None
        original_reference_texts = list(prompt_text) if prompt_text else None
        tokens_sec = None  # Initialize to track token generation speed
        
        logger.info(f"🎯 Starting sample {sample_idx + 1}/{num_samples}")
        
        # Process each chunk
        for chunk_idx, chunk_text in enumerate(text_chunks):
            base_content_sequence = ContentSequence(modality="interleave")
            max_length = model.config.max_seq_len
            
            logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{len(text_chunks)}")
            
            # Add reference prompts for the first chunk with MAXIMUM strength
            if chunk_idx == 0 and use_prompt and original_reference_texts is not None and original_reference_tokens is not None:
                logger.info(f"🔥 첫 번째 청크: MAXIMUM 원본 참조 오디오 강화")
                for i, (t, c) in enumerate(zip(original_reference_texts, original_reference_tokens)):
                    logger.info(f"  🎯 참조 {i+1} 추가: text='{t[:30]}...' tokens_shape={c.shape}")
                    
                    # 참조 오디오 토큰을 최대한 많이 사용 (300 토큰까지)
                    if c.shape[1] > 300:
                        truncated_tokens = c[:, :300]
                        logger.info(f"    참조 {i+1} 최대 활용: {c.shape} -> {truncated_tokens.shape}")
                    else:
                        truncated_tokens = c
                        logger.info(f"    참조 {i+1} 전체 사용: {c.shape}")
                    
                    # 첫 번째 청크에서도 참조를 3번 반복으로 극강 설정
                    for repeat in range(3):  # 3번 반복으로 극강 참조
                        base_content_sequence.append(
                            [
                                TextPart(text=t),
                                VQPart(codes=truncated_tokens),
                            ],
                            add_end=True,
                            speaker=0,
                        )
                        logger.info(f"    🔄 첫 청크 참조 {i+1} 극강화 반복 {repeat+1}/3")
            
            # For subsequent chunks, prioritize original reference for voice consistency
            elif chunk_idx > 0:
                logger.info(f"🔗 Handling reference for chunk {chunk_idx + 1} (강화된 목소리 일관성 모드)")
                
                # ALWAYS include original reference to maintain voice consistency - this is the priority
                if use_prompt and original_reference_tokens is not None and original_reference_texts is not None:
                    logger.info(f"🔥 MAXIMUM 목소리 일관성 모드: 원본 참조 극대화")
                    
                    # 목소리 드리프트 방지를 위해 참조 오디오를 극도로 강화
                    for i, (ref_text, ref_tokens) in enumerate(zip(original_reference_texts, original_reference_tokens)):
                        # 참조 오디오 토큰을 최대한 많이 사용 (300 토큰까지 확장)
                        if ref_tokens.shape[1] > 300:
                            truncated_tokens = ref_tokens[:, :300]
                            logger.info(f"  🎯 참조 {i+1} 사용 (최대 활용): {ref_tokens.shape} -> {truncated_tokens.shape}")
                        else:
                            truncated_tokens = ref_tokens
                            logger.info(f"  🎯 참조 {i+1} 전체 사용: {ref_tokens.shape}")
                        
                        # 목소리 일관성을 위해 참조를 3번 반복 삽입 (더 강력한 영향)
                        for repeat in range(3):  # 3번 반복으로 극강 참조
                            base_content_sequence.append(
                                [
                                    TextPart(text=ref_text),
                                    VQPart(codes=truncated_tokens),
                                ],
                                add_end=True,
                                speaker=0,
                            )
                            logger.info(f"    🔄 참조 {i+1} 강화 반복 {repeat+1}/3")
                    
                    # Store reference token count for logging  
                    ref_token_count = sum(min(tokens.shape[1], 300) for tokens in original_reference_tokens) * 3  # 3배 반영
                    logger.info(f"  📊 총 참조 토큰 수: {ref_token_count} (극대화됨)")
                else:
                    ref_token_count = 0
                
                # 목소리 드리프트 방지를 위해 이전 생성 오디오 사용을 극도로 제한
                # 원본 참조 오디오가 압도적으로 우선하도록 설정
                if len(all_codes) > 0 and chunk_idx <= 2:  # 처음 2-3개 청크에서만 제한적 사용
                    prev_codes = all_codes[-1].cpu()
                    logger.info(f"  이전 생성 코드 형태: {prev_codes.shape}")
                    
                    if len(prev_codes.shape) == 2 and prev_codes.shape[1] > 10:
                        # 목소리 드리프트 방지를 위해 전환 세그먼트를 극도로 축소
                        # 참조 오디오 대비 비율을 1:20 이하로 유지 (5% 이하)
                        max_transition_length = min(3, max(1, ref_token_count // 100))  # 참조의 1% 이하
                        
                        if max_transition_length >= 2:
                            start_idx = max(0, prev_codes.shape[1] - max_transition_length - 1)
                            end_idx = prev_codes.shape[1] - 1
                            
                            transition_codes = prev_codes[:, start_idx:end_idx]
                            
                            logger.info(f"  🔒 극소량 전환 세그먼트: {transition_codes.shape} (최대: {max_transition_length})")
                            logger.info(f"  🎯 전환/참조 비율: {transition_codes.shape[1]}/{ref_token_count} = {transition_codes.shape[1]/ref_token_count*100:.1f}% (극소)")
                            
                            # 매우 엄격한 조건으로만 사용
                            if transition_codes.shape[1] >= 1 and transition_codes.shape[1] <= 3:
                                base_content_sequence.append(
                                    [
                                        VQPart(codes=transition_codes),
                                    ],
                                    add_end=True,
                                    speaker=0,
                                )
                                logger.info(f"    ✅ 극소량 전환 적용 (목소리 드리프트 최소화)")
                            else:
                                logger.info(f"  ❌ 전환 세그먼트 거부: 목소리 일관성 우선")
                        else:
                            logger.info(f"  ❌ 전환 길이 부족: 원본 참조만 사용")
                    else:
                        logger.info(f"  ❌ 이전 코드 부족: 원본 참조만 사용")
                else:
                    if chunk_idx > 2:
                        logger.info(f"  🚫 청크 {chunk_idx+1}: 이전 오디오 사용 금지 (목소리 드리프트 방지)")
                    else:
                        logger.info(f"  ⚠️ 첫 청크: 이전 코드 없음")
                
                logger.info(f"  🎯 원본 참조 절대 우선 모드 활성화")
            
            base_content_sequence.append(
                [
                    TextPart(text=chunk_text),
                ],
                add_end=False,
                speaker=0,
            )

            encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )
            
            logger.info(f"  Encoded sequence length: {encoded.size(1)}")
            
            # 긴 텍스트 처리를 위해 컨텍스트 제한을 완화
            # 기존 2048 여유분을 1024로 줄여 더 많은 텍스트 처리 가능
            context_buffer = 1024
            if encoded.size(1) > max_length - context_buffer:
                logger.warning(f"  Chunk {chunk_idx} too long: {encoded.size(1)} > {max_length - context_buffer}, truncating")
                # 텍스트 부분을 우선적으로 보존하고 오디오 참조 부분을 축소
                if chunk_idx > 0:  # 첫 번째 청크가 아닌 경우
                    # 텍스트 토큰 길이 추정 (대략적으로)
                    text_token_estimate = len(chunk_text) // 2  # 한 글자당 약 0.5 토큰
                    min_required = text_token_estimate + context_buffer
                    
                    if encoded.size(1) > min_required:
                        # 오디오 참조 부분을 더 많이 축소
                        new_length = max(min_required, max_length - context_buffer // 2)
                        logger.info(f"    Preserving text context: truncating to {new_length} (estimated text tokens: {text_token_estimate})")
                    else:
                        new_length = max_length - context_buffer
                else:
                    new_length = max_length - context_buffer
                
                encoded = encoded[:, :new_length]
                audio_masks = audio_masks[:new_length]
                if audio_parts is not None:
                    audio_parts = audio_parts[:new_length]

            encoded = encoded.to(device=device)
            logger.info(f"🎬 Generating audio for chunk {chunk_idx + 1}: '{chunk_text[:50]}...'")

            prompt_length = encoded.size(1)

            # 목소리 일관성 최우선 - 보수적 파라미터로 안정성 확보
            # 랜덤성을 줄이고 참조 오디오의 영향력을 극대화
            
            # Temperature 조정: 목소리 드리프트 방지를 위해 더 보수적으로 설정
            try:
                base_temp = float(temperature.item())
            except (AttributeError, TypeError):
                base_temp = float(temperature)
            
            # 목소리 일관성을 위해 temperature를 낮춤 (랜덤성 감소)
            # 청크 간 변화를 최소화하여 동일한 목소리 유지
            if chunk_idx == 0:
                # 첫 청크: 참조 오디오와 최대한 유사하게
                chunk_temperature = torch.tensor(
                    max(0.6, min(0.8, base_temp - 0.1)), 
                    device=device, dtype=torch.float
                )
                temp_val = chunk_temperature.item() if hasattr(chunk_temperature, 'item') else float(chunk_temperature)
                logger.info(f"  🎯 첫 청크: 보수적 temperature = {temp_val:.3f} (참조 우선)")
            else:
                # 후속 청크: 일관성 유지를 위해 더욱 보수적
                chunk_temperature = torch.tensor(
                    max(0.5, min(0.7, base_temp - 0.2)), 
                    device=device, dtype=torch.float
                )
                temp_val = chunk_temperature.item() if hasattr(chunk_temperature, 'item') else float(chunk_temperature)
                logger.info(f"  🔒 청크 {chunk_idx+1}: 극보수적 temperature = {temp_val:.3f} (일관성 우선)")
            
            # Repetition penalty 조정: 목소리 특성 유지를 위해 적당히 설정
            try:
                base_rep_penalty = float(repetition_penalty.item())
            except (AttributeError, TypeError):
                base_rep_penalty = float(repetition_penalty)
            
            # 목소리 일관성을 위해 repetition penalty를 적절히 조정
            chunk_repetition_penalty = torch.tensor(
                max(1.02, min(1.08, base_rep_penalty)), 
                device=device, dtype=torch.float
            )
            
            logger.info(f"  🎛️ 목소리 일관성 우선 파라미터 적용 완료")
            chunk_temp_value = chunk_temperature.item() if hasattr(chunk_temperature, 'item') else float(chunk_temperature)
            chunk_rep_value = chunk_repetition_penalty.item() if hasattr(chunk_repetition_penalty, 'item') else float(chunk_repetition_penalty)
            temp_change = chunk_temp_value - base_temp
            rep_change = chunk_rep_value - base_rep_penalty
            logger.info(f"    Temperature: {base_temp:.3f} -> {chunk_temp_value:.3f} (변화: {temp_change:+.3f})")
            logger.info(f"    Rep_penalty: {base_rep_penalty:.3f} -> {chunk_rep_value:.3f} (변화: {rep_change:+.3f})")
            
            # 목소리 일관성을 위한 보수적 토큰 수 조정
            if max_new_tokens == 0:
                # 텍스트 특성에 따른 적응적 토큰 수 계산 (더 보수적)
                text_length = len(chunk_text)
                
                # 문장 부호 개수로 호흡점 파악
                pause_marks = chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?') + \
                             chunk_text.count(',') + chunk_text.count(':') + chunk_text.count(';')
                
                # 목소리 일관성을 위해 더 보수적인 토큰 비율 사용
                base_token_ratio = 2.0  # 2.2에서 2.0으로 축소하여 더 안정적
                
                # 호흡점 보너스도 축소하여 일관성 우선
                pause_bonus = min(pause_marks * 0.05, 0.3)  # 0.1에서 0.05로, 최대 0.5에서 0.3으로 축소
                adjusted_ratio = base_token_ratio + pause_bonus
                
                estimated_tokens = int(text_length * adjusted_ratio)
                
                # 청크 크기에 따른 더 보수적인 범위 설정 (목소리 드리프트 방지)
                if text_length < 50:  # 짧은 청크
                    min_tokens, max_tokens = 100, 300  # 축소
                elif text_length < 150:  # 중간 청크
                    min_tokens, max_tokens = 150, 600  # 축소
                else:  # 긴 청크
                    min_tokens, max_tokens = 200, 800  # 축소
                
                dynamic_max_tokens = max(min_tokens, min(estimated_tokens, max_tokens))
                
                logger.info(f"  🎯 목소리 일관성 우선 토큰 수 조정:")
                logger.info(f"    텍스트 길이: {text_length}, 호흡점: {pause_marks}개")
                logger.info(f"    보수적 토큰 비율: {base_token_ratio} + {pause_bonus:.2f} = {adjusted_ratio:.2f}")
                logger.info(f"    최종 토큰 수: {dynamic_max_tokens} (범위: {min_tokens}-{max_tokens}, 일관성 우선)")
            else:
                dynamic_max_tokens = max_new_tokens
            
            logger.info(f"  Generation parameters: temp={chunk_temperature:.3f}, rep_penalty={chunk_repetition_penalty:.3f}, max_tokens={dynamic_max_tokens}")

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=dynamic_max_tokens,  # 동적으로 계산된 토큰 수 사용
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=chunk_temperature,
                top_p=top_p,
                repetition_penalty=chunk_repetition_penalty,
            )

            if sample_idx == 0 and chunk_idx == 0 and compile:
                logger.info(f"⚡ Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t if t > 0 else 0 # Calculate tokens_sec here
            logger.info(
                f"✅ Chunk {chunk_idx + 1} completed: {tokens_generated} tokens in {t:.02f}s ({tokens_sec:.02f} tokens/sec)"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"🔥 GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end>, we remove last token
            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found"
            
            logger.info(f"  Generated codes shape: {codes.shape}")
            all_codes.append(codes)

            decoded = y[:, prompt_length:].clone()
            # But for global encoding, we should keep the <im_end> token
            global_encoded.append(decoded.cpu())
            
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=chunk_text)

        # Log total statistics
        total_tokens = sum(codes.size(1) for codes in all_codes)
        logger.info(f"🎉 Sample {sample_idx + 1} completed - Total tokens: {total_tokens}")
        if tokens_sec is not None:
            logger.info(f"📊 Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: Dict[str, Any]
    response_queue: queue.Queue[WrappedGenerateResponse]


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.8)
@click.option("--repetition-penalty", type=float, default=1.1)
@click.option("--temperature", type=float, default=0.8)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/openaudio-s1-mini",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_tokens: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text is not None and prompt_tokens is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
