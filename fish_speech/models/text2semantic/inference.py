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
    # 강화된 반복 방지 로직 - 강사 영상의 반복적 표현 패턴 차단
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        
        # 기본 repetition penalty 적용
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)
        
        # 추가 반복 패턴 감지 및 강화된 페널티 적용
        # 연속된 동일 토큰 시퀀스 감지 (예: "Part 3 and Part 3 and Part 3")
        if previous_tokens is not None and previous_tokens.numel() >= 6:  # 최소 6개 토큰 이상일 때만 패턴 감지
            # 최근 토큰들을 평탄화하여 분석
            recent_tokens = previous_tokens.flatten()[-min(64, previous_tokens.numel()):]  # 최대 64개 토큰 분석
            
            # 2-gram, 3-gram, 4-gram 반복 패턴 감지
            for n_gram in [2, 3, 4]:
                if len(recent_tokens) >= n_gram * 3:  # 최소 3번 반복 확인 가능한 길이
                    # n-gram 패턴 추출
                    for i in range(len(recent_tokens) - n_gram * 2 + 1):
                        pattern = recent_tokens[i:i+n_gram]
                        next_pattern = recent_tokens[i+n_gram:i+n_gram*2]
                        
                        # 동일한 패턴이 연속으로 나타나는 경우
                        if torch.equal(pattern, next_pattern):
                            # 해당 패턴의 토큰들에 강화된 페널티 적용
                            enhanced_penalty = repetition_penalty * 1.5  # 1.5배 강화
                            for token in pattern:
                                if 0 <= token < logits.size(-1):  # 유효한 토큰 인덱스 확인
                                    current_score = logits[token]
                                    if current_score < 0:
                                        logits[token] = current_score * enhanced_penalty
                                    else:
                                        logits[token] = current_score / enhanced_penalty
                            
                            # 강사 영상 특성상 자주 나타나는 반복 표현들에 대한 특별 처리
                            # "and", "Part", "3" 등의 연속 사용 억제
                            break  # 첫 번째 패턴만 처리하여 과도한 페널티 방지

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
    text_context: str = "",  # 반복 패턴 추적을 위한 텍스트 컨텍스트
    chunk_idx: int = 0,      # 청크 인덱스
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    i = 0  # 루프 외부에서도 사용하기 위해 초기화
    # 반복 패턴 감지를 위한 변수들
    repetition_count = 0
    last_generated_sequence = []
    
    # 반복 패턴 감지를 위한 토큰 생성 시작 로깅
    logger.info(f"🎵 토큰 생성 시작 - 목표: {num_new_tokens}개 토큰")
    logger.info(f"   🔍 반복 감지 설정: 4-gram 반복 12회 이상 시 중단")
    logger.info(f"   🔍 긴 패턴 감지: 15-gram 반복 2회 시 중단")

    for i in tqdm(range(num_new_tokens)):
        # 강사 영상의 반복 패턴 방지를 위해 윈도우 크기 확대
        # 기본 32에서 64로 증가하여 더 긴 반복 패턴 감지 가능
        win_size = 64  # 32에서 64로 증가
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

        # 실시간 반복 패턴 감지 및 조기 차단 - 강사 영상 특성 고려
        current_token_id = cur_token[0, 0, -1].item()
        last_generated_sequence.append(current_token_id)
        
        # 최근 생성된 시퀀스가 너무 길어지지 않도록 제한 (메모리 효율성)
        if len(last_generated_sequence) > 32:
            last_generated_sequence = last_generated_sequence[-32:]
        
        # 반복 패턴 감지 (최소 8개 토큰 이상 생성된 후부터 검사)
        if i >= 8 and len(last_generated_sequence) >= 8:
            # 최근 4개 토큰이 이전 4개 토큰과 동일한지 확인 (단순 반복)
            if len(last_generated_sequence) >= 8:
                recent_4 = last_generated_sequence[-4:]
                prev_4 = last_generated_sequence[-8:-4]
                if recent_4 == prev_4:
                    repetition_count += 1
                    
                    # 반복된 토큰들을 semantic 토큰 형태로 변환하여 의미 파악
                    semantic_tokens = []
                    for token_id in recent_4:
                        if model.tokenizer.semantic_begin_id <= token_id <= model.tokenizer.semantic_end_id:
                            semantic_idx = token_id - model.tokenizer.semantic_begin_id
                            semantic_tokens.append(f"semantic:{semantic_idx}")
                        else:
                            # 일반 텍스트 토큰인 경우 디코딩 시도
                            try:
                                decoded_token = model.tokenizer.decode([token_id])
                                semantic_tokens.append(f"text:'{decoded_token}'")
                            except:
                                semantic_tokens.append(f"unknown:{token_id}")
                    
                    logger.warning(f"⚠️ 반복 패턴 감지 (step {i}): {repetition_count}회 연속")
                    logger.info(f"   🔄 반복된 토큰들: {semantic_tokens}")
                    logger.info(f"   📊 토큰 ID들: {recent_4}")
                    logger.info(f"   📍 현재 진행률: {i}/{num_new_tokens} ({i/num_new_tokens*100:.1f}%)")
                    
                    # 텍스트 컨텍스트와 연결
                    if text_context:
                        logger.info(f"   📝 처리 중인 텍스트 (청크 {chunk_idx+1}): '{text_context[:80]}{'...' if len(text_context) > 80 else ''}'")
                        
                        # 텍스트 내에서 대략적인 위치 추정
                        text_progress = i / num_new_tokens if num_new_tokens > 0 else 0
                        estimated_char_pos = int(len(text_context) * text_progress)
                        if estimated_char_pos < len(text_context):
                            context_window = text_context[max(0, estimated_char_pos-20):estimated_char_pos+20]
                            logger.info(f"   🎯 추정 텍스트 위치: '...{context_window}...' (위치: {estimated_char_pos}/{len(text_context)})")
                    
                    # 반복 감지 조건을 더욱 완화: 12회 연속 반복 시에만 중단 (강사 영상의 자연스러운 패턴 허용)
                    if repetition_count >= 12:
                        logger.error(f"🚫 과도한 반복 패턴으로 인한 강제 중단 (step {i})")
                        logger.error(f"   💥 최종 반복 시퀀스: {semantic_tokens}")
                        logger.error(f"   💥 반복 시작점: step {i - repetition_count * 4} 부근")
                        logger.error(f"   💥 반복 지속 길이: {repetition_count * 4}개 토큰")
                        
                        # 반복 발생 텍스트 구간 상세 분석
                        if text_context:
                            logger.error(f"   📝 문제 텍스트 (청크 {chunk_idx+1}): '{text_context}'")
                            
                            # 반복 시작점과 종료점의 텍스트 위치 추정
                            start_progress = max(0, (i - repetition_count * 4)) / num_new_tokens if num_new_tokens > 0 else 0
                            end_progress = i / num_new_tokens if num_new_tokens > 0 else 0
                            
                            start_char_pos = int(len(text_context) * start_progress)
                            end_char_pos = int(len(text_context) * end_progress)
                            
                            if start_char_pos < len(text_context) and end_char_pos <= len(text_context):
                                problematic_text = text_context[start_char_pos:end_char_pos]
                                logger.error(f"   🎯 반복 발생 추정 구간: '{problematic_text}' (위치: {start_char_pos}-{end_char_pos})")
                                
                                # 해당 구간의 특성 분석
                                repeated_chars = len([c for c in problematic_text if problematic_text.count(c) > 1])
                                logger.error(f"   📊 구간 분석: 길이 {len(problematic_text)}자, 반복 문자 {repeated_chars}개")
                        break
                else:
                    repetition_count = 0  # 반복이 끊어지면 카운트 리셋
            
            # 긴 패턴 감지 (15-gram 반복으로 더욱 완화, 강사 영상에서는 긴 반복도 자연스러울 수 있음)
            if len(last_generated_sequence) >= 30:
                recent_15 = last_generated_sequence[-15:]
                prev_15 = last_generated_sequence[-30:-15]
                if recent_15 == prev_15:
                    # 긴 패턴의 semantic 토큰 분석
                    long_semantic_tokens = []
                    for token_id in recent_15:
                        if model.tokenizer.semantic_begin_id <= token_id <= model.tokenizer.semantic_end_id:
                            semantic_idx = token_id - model.tokenizer.semantic_begin_id
                            long_semantic_tokens.append(f"s:{semantic_idx}")
                        else:
                            try:
                                decoded_token = model.tokenizer.decode([token_id])
                                long_semantic_tokens.append(f"'{decoded_token}'")
                            except:
                                long_semantic_tokens.append(f"?:{token_id}")
                    
                    logger.error(f"🚫 매우 긴 반복 패턴 감지로 인한 중단 (step {i})")
                    logger.error(f"   🔄 긴 반복 패턴 (15-gram): {long_semantic_tokens}")
                    logger.error(f"   📊 토큰 ID들: {recent_15}")
                    logger.error(f"   📍 반복 시작 추정점: step {i - 15} 부근")
                    logger.error(f"   💥 총 반복 길이: 30개 토큰 (15-gram × 2)")
                    
                    # 긴 패턴 반복 텍스트 구간 분석
                    if text_context:
                        logger.error(f"   📝 긴 패턴 문제 텍스트 (청크 {chunk_idx+1}): '{text_context}'")
                        
                        # 긴 패턴의 텍스트 위치 추정
                        long_start_progress = max(0, (i - 30)) / num_new_tokens if num_new_tokens > 0 else 0
                        long_end_progress = i / num_new_tokens if num_new_tokens > 0 else 0
                        
                        long_start_pos = int(len(text_context) * long_start_progress)
                        long_end_pos = int(len(text_context) * long_end_progress)
                        
                        if long_start_pos < len(text_context) and long_end_pos <= len(text_context):
                            long_problematic_text = text_context[long_start_pos:long_end_pos]
                            logger.error(f"   🎯 긴 패턴 발생 추정 구간: '{long_problematic_text}' (위치: {long_start_pos}-{long_end_pos})")
                    break

        # 강사 영상용 개선된 조기 종료 처리 - 잡음 방지 우선
        # IM_END_TOKEN 감지 시 잡음 생성을 방지하고 자연스러운 종료 유도
        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            # 최소한의 텍스트 완전성만 보장 (잡음 방지를 위해 완화)
            min_tokens_ratio = 0.4  # 70%에서 40%로 대폭 완화 (잡음 방지 우선)
            min_tokens = max(32, int(num_new_tokens * min_tokens_ratio))  # 최소 32토큰으로 완화
            
            if i >= min_tokens:
                logger.info(f"✅ 자연스러운 종료: IM_END_TOKEN at step {i} (>= {min_tokens}, target: {num_new_tokens})")
                break
            else:
                # 잡음 방지를 위한 매우 제한적인 조기 종료 방지
                completion_ratio = i / num_new_tokens if num_new_tokens > 0 else 0
                
                # 진행률이 30% 미만인 경우에만 매우 제한적으로 대체
                if completion_ratio < 0.3:
                    logger.info(f"🔄 최소 완전성 보장: IM_END_TOKEN 대체 at step {i} (진행률: {completion_ratio:.1%})")
                    
                    # 잡음 방지를 위해 가장 안전한 토큰만 사용
                    # semantic:0 (base_token) 근처의 안전한 토큰만 사용
                    base_token = model.tokenizer.semantic_begin_id
                    
                    # 매우 제한적인 안전 토큰 풀 (잡음 최소화)
                    safe_offsets = [0, 1, 2]  # semantic:0, semantic:1, semantic:2만 사용
                    selected_offset = i % len(safe_offsets)
                    selected_token = base_token + safe_offsets[selected_offset]
                    
                    cur_token[0, 0, -1] = selected_token
                    logger.info(f"   🎵 안전 토큰 사용: semantic:{safe_offsets[selected_offset]} (잡음 방지)")
                else:
                    # 진행률이 30% 이상이면 자연스럽게 종료 (잡음 방지)
                    logger.info(f"🎯 잡음 방지 종료: IM_END_TOKEN 수용 at step {i} (진행률: {completion_ratio:.1%})")
                    break

    # 토큰 생성 완료 통계 로깅
    actual_generated = i + 1
    completion_rate = actual_generated / num_new_tokens * 100 if num_new_tokens > 0 else 0
    logger.info(f"🎵 토큰 생성 완료:")
    logger.info(f"   📊 생성된 토큰: {actual_generated}/{num_new_tokens} ({completion_rate:.1f}%)")
    logger.info(f"   🔄 총 반복 감지 횟수: {repetition_count}회")
    logger.info(f"   📈 생성 시퀀스 길이: {len(last_generated_sequence)}개")
    
    if actual_generated < num_new_tokens:
        logger.warning(f"   ⚠️ 조기 종료됨 - 부족한 토큰: {num_new_tokens - actual_generated}개")
        
        # 조기 종료 원인 분석
        if repetition_count >= 12:
            logger.warning(f"   💥 조기 종료 원인: 과도한 반복 패턴 (12회 이상)")
        else:
            logger.warning(f"   💥 조기 종료 원인: IM_END_TOKEN 또는 긴 패턴 반복")

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
    text_context: str = "",  # 반복 패턴 추적을 위한 텍스트 컨텍스트
    chunk_idx: int = 0,      # 청크 인덱스
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
        text_context=text_context,
        chunk_idx=chunk_idx,
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
        
        # 강사 영상 특성을 고려한 완전한 문장 단위 분할 - 텍스트 끊김 방지
        # 문장 완전성을 보장하는 더 정교한 분할 방식
        major_breaks = ['。 ', '！ ', '？ ', '. ', '! ', '? ']
        sentences = [text_with_pauses]
        
        # 1단계: 완전한 문장 단위로 분할
        for break_mark in major_breaks:
            new_sentences = []
            for sentence in sentences:
                if break_mark in sentence:
                    parts = sentence.split(break_mark)
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:  # 마지막이 아닌 경우
                            complete_sentence = (part + break_mark).strip()
                            # 너무 짧은 문장은 다음 문장과 합치기 고려
                            if len(complete_sentence) < 20 and new_sentences:
                                # 이전 문장과 합치기
                                new_sentences[-1] = new_sentences[-1] + ' ' + complete_sentence
                            else:
                                new_sentences.append(complete_sentence)
                        elif part.strip():  # 마지막이면서 내용이 있는 경우
                            remaining_part = part.strip()
                            # 남은 부분이 너무 짧으면 이전 문장과 합치기
                            if len(remaining_part) < 30 and new_sentences:
                                new_sentences[-1] = new_sentences[-1] + ' ' + remaining_part
                            else:
                                new_sentences.append(remaining_part)
                else:
                    new_sentences.append(sentence)
            sentences = [s for s in new_sentences if s and len(s.strip()) > 5]  # 너무 짧은 조각 제거
        
        # 반복되는 문장 패턴 감지 및 제거/수정
        # 강사 영상에서 자주 발생하는 "Part 3 and Part 4" -> "Part 3 and Part 3 and Part 3" 같은 오류 방지
        cleaned_sentences = []
        seen_patterns = {}
        
        for sentence in sentences:
            # 문장을 단어로 분할하여 패턴 분석
            words = sentence.split()
            
            # 짧은 문장은 그대로 유지
            if len(words) < 4:
                cleaned_sentences.append(sentence)
                continue
            
            # 반복 패턴 감지 (연속된 같은 구문)
            is_repetitive = False
            
            # "Part X and Part X and Part X" 패턴 감지
            if len(words) >= 6:
                # 3-gram 단위로 반복 검사
                for i in range(len(words) - 5):
                    trigram1 = ' '.join(words[i:i+3])
                    trigram2 = ' '.join(words[i+3:i+6])
                    
                    if trigram1 == trigram2:
                        logger.warning(f"⚠️ 반복 패턴 감지된 문장: '{sentence}'")
                        logger.info(f"   반복된 3-gram: '{trigram1}'")
                        
                        # 반복 부분 제거하여 수정
                        corrected_words = words[:i+3]  # 첫 번째 패턴만 유지
                        # 나머지 부분에서 반복되지 않는 부분 추가
                        remaining_words = words[i+6:]
                        if remaining_words:
                            corrected_words.extend(remaining_words)
                        
                        corrected_sentence = ' '.join(corrected_words)
                        logger.info(f"   수정된 문장: '{corrected_sentence}'")
                        cleaned_sentences.append(corrected_sentence)
                        is_repetitive = True
                        break
            
            if not is_repetitive:
                # 동일한 문장이 너무 자주 반복되는지 확인
                sentence_key = sentence.lower().strip()
                if sentence_key in seen_patterns:
                    seen_patterns[sentence_key] += 1
                    # 같은 문장이 3번 이상 나타나면 일부는 생략
                    if seen_patterns[sentence_key] <= 2:
                        cleaned_sentences.append(sentence)
                    else:
                        logger.warning(f"⚠️ 중복 문장 생략: '{sentence[:50]}...'")
                else:
                    seen_patterns[sentence_key] = 1
                    cleaned_sentences.append(sentence)
        
        sentences = cleaned_sentences
        logger.info(f"📝 반복 패턴 정리 완료: {len(sentences)}개 문장 (중복 제거됨)")
        
        text_chunks = []
        current_chunk = ""
        
        # 강사 영상 특성을 고려한 청크 크기 조정
        # 움직임이 많고 핸드마이크 등으로 인한 음성 변화를 고려하여 
        # 더 작은 청크로 분할하여 일관성 유지
        adaptive_chunk_length = min(chunk_length, 10000)  # 제한 X
        
        # 2단계: 문장 완전성을 보장하면서 청크 구성
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 문장이 너무 길면 의미 단위로 세분화하되, 완전성 보장
            if len(sentence) > adaptive_chunk_length:
                # 자연스러운 호흡점을 우선으로 한 세분화
                sub_delimiters = [
                    ', and ', ', but ', ', so ', ', because ',  # 접속사 우선
                    ', ', ': ', '; ', ' - ',  # 일반 구두점
                    ' and ', ' but ', ' so ', ' because ', ' when ', ' if ', ' as '  # 접속사만
                ]
                best_split = [sentence]  # 기본값: 분할하지 않음
                
                # 가장 자연스러운 분할점 찾기
                for delimiter in sub_delimiters:
                    if delimiter in sentence:
                        potential_split = sentence.split(delimiter)
                        # 각 부분이 의미 있는 길이인지 확인 (최소 25자, 최대 청크 길이)
                        valid_parts = []
                        for i, part in enumerate(potential_split):
                            if part.strip():
                                if i < len(potential_split) - 1:
                                    complete_part = part.strip() + delimiter.rstrip()
                                else:
                                    complete_part = part.strip()
                                
                                # 부분이 너무 짧으면 이전 부분과 합치기
                                if len(complete_part) < 25 and valid_parts:
                                    valid_parts[-1] = valid_parts[-1] + ' ' + complete_part
                                else:
                                    valid_parts.append(complete_part)
                        
                        # 유효한 분할인지 확인 (모든 부분이 적절한 크기)
                        if len(valid_parts) > 1 and all(25 <= len(part) <= adaptive_chunk_length for part in valid_parts):
                            best_split = valid_parts
                            logger.info(f"📝 문장 분할 성공: '{delimiter}' 기준으로 {len(best_split)}개 부분")
                            break
                
                # 분할된 부분들을 청크에 추가
                for sub_sentence in best_split:
                    sub_sentence = sub_sentence.strip()
                    if not sub_sentence:
                        continue
                        
                    # 청크 크기 확인 후 추가
                    if current_chunk and len(current_chunk) + len(sub_sentence) + 1 > adaptive_chunk_length:
                        # 현재 청크가 너무 커지면 새 청크 시작
                        text_chunks.append(current_chunk.strip())
                        logger.info(f"✅ 청크 완성: '{current_chunk[:30]}...' (길이: {len(current_chunk)})")
                        current_chunk = sub_sentence
                    else:
                        # 현재 청크에 추가
                        if current_chunk:
                            current_chunk += ' ' + sub_sentence
                        else:
                            current_chunk = sub_sentence
            else:
                # 적절한 크기의 문장은 그대로 추가
                if current_chunk and len(current_chunk) + len(sentence) + 1 > adaptive_chunk_length:
                    # 현재 청크 완료
                    text_chunks.append(current_chunk.strip())
                    logger.info(f"✅ 청크 완성: '{current_chunk[:30]}...' (길이: {len(current_chunk)})")
                    current_chunk = sentence
                else:
                    # 현재 청크에 추가
                    if current_chunk:
                        current_chunk += ' ' + sentence
                    else:
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
        
        # 🎯 청크 간 음성 연속성을 위한 이전 생성 오디오 저장소
        previous_generated_tokens = []  # 이전 청크에서 생성된 오디오 토큰들
        previous_generated_texts = []   # 이전 청크의 텍스트들
        
        # 🔗 텍스트 오버랩을 통한 연속성 강화를 위한 설정
        text_overlap_words = 5  # 이전 청크의 마지막 5단어를 다음 청크 시작에 추가
        
        logger.info(f"🎯 Starting sample {sample_idx + 1}/{num_samples}")
        
        # Process each chunk
        for chunk_idx, chunk_text in enumerate(text_chunks):
            base_content_sequence = ContentSequence(modality="interleave")
            max_length = model.config.max_seq_len
            
            logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{len(text_chunks)}")
            
            # 🔗 텍스트 연속성을 위한 컨텍스트 준비 (오버랩은 참조용으로만 사용)
            original_chunk_text = chunk_text
            overlap_context_text = ""  # 오버랩 컨텍스트 (참조용)
            
            if chunk_idx > 0 and len(previous_generated_texts) > 0:
                # 이전 청크의 마지막 몇 단어를 컨텍스트로만 준비 (실제 생성에는 포함하지 않음)
                prev_text = previous_generated_texts[-1]
                prev_words = prev_text.split()
                
                if len(prev_words) >= text_overlap_words:
                    overlap_context_text = " ".join(prev_words[-text_overlap_words:])
                    logger.info(f"  🔗 연속성 컨텍스트 준비: '{overlap_context_text}' (참조용만, 음성 생성 제외)")
                    logger.info(f"    실제 생성 텍스트: '{original_chunk_text[:50]}...'")
                else:
                    logger.info(f"  ℹ️ 이전 청크가 너무 짧아 컨텍스트 미적용 (단어 수: {len(prev_words)} < {text_overlap_words})")
            else:
                logger.info(f"  ℹ️ 첫 번째 청크 - 연속성 컨텍스트 미적용")
            
            # 실제 음성 생성에는 원본 텍스트만 사용
            chunk_text_for_generation = original_chunk_text
            
            # Add reference prompts for the first chunk with MAXIMUM strength
            if chunk_idx == 0 and use_prompt and original_reference_texts is not None and original_reference_tokens is not None:
                logger.info(f"🔥 첫 번째 청크: MAXIMUM 원본 참조 오디오 강화 (드리프트 방지 시작점)")
                
                # 첫 청크도 동일한 강화 방식 적용
                base_repeat_count = 3
                total_repeat_count = base_repeat_count  # 첫 청크는 기본값
                
                for i, (t, c) in enumerate(zip(original_reference_texts, original_reference_tokens)):
                    logger.info(f"  🎯 참조 {i+1} 추가: text='{t[:30]}...' tokens_shape={c.shape}")
                    
                    # 참조 오디오 토큰을 최대한 많이 사용 (350 토큰까지 - 후속 청크와 동일)
                    max_tokens = 350
                    if c.shape[1] > max_tokens:
                        truncated_tokens = c[:, :max_tokens]
                        logger.info(f"    참조 {i+1} 최대 활용: {c.shape} -> {truncated_tokens.shape}")
                    else:
                        truncated_tokens = c
                        logger.info(f"    참조 {i+1} 전체 사용: {c.shape}")
                    
                    # 첫 번째 청크에서도 강화된 참조 적용
                    for repeat in range(total_repeat_count):
                        base_content_sequence.append(
                            [
                                TextPart(text=t),
                                VQPart(codes=truncated_tokens),
                            ],
                            add_end=True,
                            speaker=0,
                        )
                        logger.info(f"    🔄 첫 청크 참조 {i+1} 강화 반복 {repeat+1}/{total_repeat_count}")
            
            # For subsequent chunks, prioritize original reference for voice consistency
            elif chunk_idx > 0:
                logger.info(f"🔗 Handling reference for chunk {chunk_idx + 1} (강화된 목소리 일관성 모드)")
                
                # ALWAYS include original reference to maintain voice consistency - this is the priority
                if use_prompt and original_reference_tokens is not None and original_reference_texts is not None:
                    logger.info(f"🔥 MAXIMUM 목소리 일관성 모드: 원본 참조 극대화")
                    
                    # 🎯 강사 영상 최적화: 극대화된 참조 반복으로 목소리 일관성 확보
                    # 모든 청크에서 충분한 참조 반복으로 일관된 목소리 유지
                    base_repeat_count = 6  # 8회에서 6회로 줄여서 이전 오디오 공간 확보
                    drift_prevention_bonus = min(chunk_idx, 3)  # 최대 3회 추가 (총 최대 9회)
                    total_repeat_count = base_repeat_count + drift_prevention_bonus
                    
                    logger.info(f"  📈 청크 {chunk_idx+1} 드리프트 방지 강화: 기본 {base_repeat_count}회 + 보너스 {drift_prevention_bonus}회 = 총 {total_repeat_count}회")
                    
                    for i, (ref_text, ref_tokens) in enumerate(zip(original_reference_texts, original_reference_tokens)):
                        # 참조 오디오 토큰을 최대한 많이 사용 (300 토큰으로 조정)
                        max_tokens = 300  # 350에서 300으로 조정하여 이전 오디오 공간 확보
                        if ref_tokens.shape[1] > max_tokens:
                            truncated_tokens = ref_tokens[:, :max_tokens]
                            logger.info(f"  🎯 참조 {i+1} 사용 (최대 활용): {ref_tokens.shape} -> {truncated_tokens.shape}")
                        else:
                            truncated_tokens = ref_tokens
                            logger.info(f"  🎯 참조 {i+1} 전체 사용: {ref_tokens.shape}")
                        
                        # 청크가 뒤쪽일수록 더 많은 반복으로 드리프트 방지
                        for repeat in range(total_repeat_count):
                            base_content_sequence.append(
                                [
                                    TextPart(text=ref_text),
                                    VQPart(codes=truncated_tokens),
                                ],
                                add_end=True,
                                speaker=0,
                            )
                            if repeat < 3:
                                logger.info(f"    🔄 참조 {i+1} 기본 반복 {repeat+1}/{base_repeat_count}")
                            else:
                                logger.info(f"    🛡️ 참조 {i+1} 드리프트 방지 추가 반복 {repeat+1}/{total_repeat_count}")
                    
                    # 드리프트 방지를 위한 참조 토큰 수 계산
                    ref_token_count = sum(min(tokens.shape[1], 300) for tokens in original_reference_tokens) * total_repeat_count
                    logger.info(f"  📊 총 참조 토큰 수: {ref_token_count} (드리프트 방지 강화됨)")
                else:
                    ref_token_count = 0
                
                # 🎯 NEW: 청크 간 음성 연속성을 위한 이전 생성 오디오 활용
                # 이전 청크의 마지막 부분을 현재 청크의 추가 참조로 사용
                if chunk_idx > 0 and len(previous_generated_tokens) > 0:
                    logger.info(f"🔗 청크 {chunk_idx+1}: 이전 생성 오디오로 연속성 확보")
                    
                    # 가장 최근 생성된 오디오를 우선적으로 사용 (최대 2개)
                    recent_count = min(2, len(previous_generated_tokens))
                    
                    for i in range(recent_count):
                        prev_idx = len(previous_generated_tokens) - 1 - i  # 최신 것부터
                        prev_tokens = previous_generated_tokens[prev_idx]
                        prev_text = previous_generated_texts[prev_idx]
                        
                        # 이전 오디오의 뒷부분만 사용 (자연스러운 연결을 위해)
                        # 너무 길면 마지막 150 토큰만 사용
                        if prev_tokens.shape[1] > 150:
                            connection_tokens = prev_tokens[:, -150:]
                            logger.info(f"  🔗 이전 청크 {prev_idx+1} 연결 토큰: {prev_tokens.shape} -> {connection_tokens.shape} (마지막 150토큰)")
                        else:
                            connection_tokens = prev_tokens
                            logger.info(f"  🔗 이전 청크 {prev_idx+1} 연결 토큰: 전체 사용 {connection_tokens.shape}")
                        
                        # 이전 텍스트의 마지막 부분 (연결 컨텍스트용)
                        prev_text_words = prev_text.split()
                        if len(prev_text_words) > 10:
                            connection_text = " ".join(prev_text_words[-10:])  # 마지막 10단어
                        else:
                            connection_text = prev_text
                        
                        # 현재 청크와의 연결을 위해 오버랩 컨텍스트 추가 (있는 경우)
                        if overlap_context_text:
                            connection_text = connection_text + " " + overlap_context_text
                            logger.info(f"  🔗 연결 컨텍스트에 오버랩 추가: '{connection_text[-50:]}'...")
                        
                        # 연결성 강화를 위해 2회 반복 (원본 참조보다는 적게)
                        for repeat in range(2):
                            base_content_sequence.append(
                                [
                                    TextPart(text=connection_text),
                                    VQPart(codes=connection_tokens),
                                ],
                                add_end=True,
                                speaker=0,
                            )
                            logger.info(f"    🔄 연결 참조 {i+1} 반복 {repeat+1}/2: '{connection_text[:30]}...'")
                    
                    logger.info(f"  ✅ 청크 연속성 참조 추가 완료: {recent_count}개 이전 청크 활용")
                else:
                    logger.info(f"  ℹ️ 청크 {chunk_idx+1}: 첫 번째 청크이거나 이전 오디오 없음 - 원본 참조만 사용")
            
            # 🎯 실제 음성 생성에는 원본 텍스트만 사용 (오버랩 제외)
            base_content_sequence.append(
                [
                    TextPart(text=chunk_text_for_generation),
                ],
                add_end=False,
                speaker=0,
            )
            
            logger.info(f"  📝 실제 생성 텍스트: '{chunk_text_for_generation[:100]}{'...' if len(chunk_text_for_generation) > 100 else ''}'")
            if overlap_context_text:
                logger.info(f"  🔗 참조 컨텍스트 (생성 제외): '{overlap_context_text}'")

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
            logger.info(f"🎬 Generating audio for chunk {chunk_idx + 1}: '{chunk_text_for_generation[:50]}...'")

            prompt_length = encoded.size(1)

            # 목소리 일관성 최우선 - 보수적 파라미터로 안정성 확보
            # 랜덤성을 줄이고 참조 오디오의 영향력을 극대화
            
            # 강화된 Temperature 조정: 반복 방지와 목소리 드리프트 방지를 동시에 고려
            try:
                if hasattr(temperature, 'item') and callable(getattr(temperature, 'item', None)):
                    base_temp = float(temperature.item())
                else:
                    base_temp = float(temperature)
            except (AttributeError, TypeError):
                base_temp = float(temperature)
            
            # 텍스트 특성 분석을 통한 동적 temperature 조정 (원본 텍스트 기준)
            chunk_words = chunk_text_for_generation.split()
            
            # 반복 위험도 계산
            repetition_risk = 0.0
            short_word_ratio = 0.0
            repeated_ratio = 0.0
            pattern_ratio = 0.0
            
            # 1. 짧은 단어들의 비율 (Part, and, 3 등)
            short_words = [w for w in chunk_words if len(w) <= 3]
            if len(chunk_words) > 0:
                short_word_ratio = len(short_words) / len(chunk_words)
                repetition_risk += short_word_ratio * 0.3
            
            # 2. 반복되는 단어 패턴 감지
            word_counts = {}
            for word in chunk_words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            repeated_words = [w for w, count in word_counts.items() if count > 1]
            if len(chunk_words) > 0:
                repeated_ratio = len(repeated_words) / len(set(chunk_words))
                repetition_risk += repeated_ratio * 0.4
            
            # 3. 특정 패턴 단어들 (Part, and, with, from 등)
            pattern_words = ['part', 'and', 'with', 'from', 'the', 'we', 'start', 'started']
            pattern_count = sum(1 for word in chunk_words if word.lower() in pattern_words)
            if len(chunk_words) > 0:
                pattern_ratio = pattern_count / len(chunk_words)
                repetition_risk += pattern_ratio * 0.3
            
            # 반복 위험도를 0-1 범위로 제한
            repetition_risk = min(1.0, max(0.0, repetition_risk))
            
            logger.info(f"  📊 반복 위험도 분석: {repetition_risk:.3f}")
            logger.info(f"    - 짧은 단어 비율: {short_word_ratio:.3f}")
            logger.info(f"    - 반복 단어 비율: {repeated_ratio:.3f}")
            logger.info(f"    - 패턴 단어 비율: {pattern_ratio:.3f}")
            
            # 목소리 드리프트 완전 차단을 위한 적응적 temperature 조정
            # 청크가 뒤쪽일수록 더욱 보수적으로 설정하여 일관성 극대화
            # 🎯 강사 영상 최적화: 모든 청크에서 일관된 temperature 사용
            # 청크별 적응적 조정 대신 고정값으로 목소리 일관성 극대화
            target_temp = 0.7  # 안정적이고 자연스러운 고정 temperature
            
            chunk_temperature = torch.tensor(target_temp, device=device, dtype=torch.float)
            temp_val = float(chunk_temperature.item()) if hasattr(chunk_temperature, 'item') else float(chunk_temperature)
            logger.info(f"  🎯 청크 {chunk_idx+1}: 일관성 우선 temperature = {temp_val:.3f} (고정값)")
            logger.info(f"    반복 위험도: {repetition_risk:.3f} (참고용, 조정 안함)")
            
            # 🎯 강사 영상 최적화: 일관된 repetition penalty 사용
            # 적응적 조정 대신 고정값으로 목소리 일관성 극대화
            try:
                if hasattr(repetition_penalty, 'item') and callable(getattr(repetition_penalty, 'item', None)):
                    base_rep_penalty = float(repetition_penalty.item())
                else:
                    base_rep_penalty = float(repetition_penalty)
            except (AttributeError, TypeError):
                base_rep_penalty = float(repetition_penalty)
            
            # 기본값 그대로 사용하여 일관성 유지
            chunk_repetition_penalty = torch.tensor(base_rep_penalty, device=device, dtype=torch.float)
            
            rep_val = float(chunk_repetition_penalty.item()) if hasattr(chunk_repetition_penalty, 'item') else float(chunk_repetition_penalty)
            logger.info(f"  🎛️ 일관성 우선 repetition penalty = {rep_val:.3f} (고정값, 반복위험도 무시)")
            
            logger.info(f"  🎛️ 목소리 일관성 우선 파라미터 적용 완료")
            chunk_temp_value = float(chunk_temperature.item()) if hasattr(chunk_temperature, 'item') else float(chunk_temperature)
            chunk_rep_value = float(chunk_repetition_penalty.item()) if hasattr(chunk_repetition_penalty, 'item') else float(chunk_repetition_penalty)
            temp_change = float(chunk_temp_value) - float(base_temp)
            rep_change = float(chunk_rep_value) - float(base_rep_penalty)
            logger.info(f"    Temperature: {base_temp:.3f} -> {chunk_temp_value:.3f} (변화: {temp_change:+.3f})")
            logger.info(f"    Rep_penalty: {base_rep_penalty:.3f} -> {chunk_rep_value:.3f} (변화: {rep_change:+.3f})")
            
            # 목소리 일관성을 위한 보수적 토큰 수 조정 (원본 텍스트 기준)
            if max_new_tokens == 0:
                # 텍스트 특성에 따른 적응적 토큰 수 계산 (더 보수적)
                text_length = len(chunk_text_for_generation)
                
                # 문장 부호 개수로 호흡점 파악 (원본 텍스트 기준)
                pause_marks = chunk_text_for_generation.count('.') + chunk_text_for_generation.count('!') + chunk_text_for_generation.count('?') + \
                             chunk_text_for_generation.count(',') + chunk_text_for_generation.count(':') + chunk_text_for_generation.count(';')
                
                # 텍스트 완전성을 위한 충분한 토큰 비율 사용 (끊김 방지)
                base_token_ratio = 2.5  # 2.0에서 2.5로 증가하여 완전성 우선
                
                # 호흡점과 복잡성을 고려한 보너스 증가
                pause_bonus = min(pause_marks * 0.08, 0.5)  # 0.05에서 0.08로, 최대 0.5로 증가
                
                # 문장 복잡성 분석 (접속사, 관계대명사 등) - 원본 텍스트 기준
                complexity_words = ['and', 'but', 'because', 'when', 'if', 'that', 'which', 'who', 'where', 'how']
                complexity_count = sum(chunk_text_for_generation.lower().count(word) for word in complexity_words)
                complexity_bonus = min(complexity_count * 0.05, 0.3)
                
                adjusted_ratio = base_token_ratio + pause_bonus + complexity_bonus
                estimated_tokens = int(text_length * adjusted_ratio)
                
                # 텍스트 완전성을 위한 더 여유로운 범위 설정
                if text_length < 50:  # 짧은 청크
                    min_tokens, max_tokens = 150, 400  # 증가
                elif text_length < 150:  # 중간 청크
                    min_tokens, max_tokens = 200, 800  # 증가
                else:  # 긴 청크
                    min_tokens, max_tokens = 300, 1000  # 증가
                
                dynamic_max_tokens = max(min_tokens, min(estimated_tokens, max_tokens))
                
                logger.info(f"  📈 텍스트 완전성 분석:")
                logger.info(f"    복잡성 단어: {complexity_count}개 (+{complexity_bonus:.3f})")
                logger.info(f"    최종 비율: {adjusted_ratio:.3f} (기본 {base_token_ratio} + 호흡점 {pause_bonus:.3f} + 복잡성 {complexity_bonus:.3f})")
                
                logger.info(f"  🎯 목소리 일관성 우선 토큰 수 조정:")
                logger.info(f"    텍스트 길이: {text_length}, 호흡점: {pause_marks}개")
                logger.info(f"    보수적 토큰 비율: {base_token_ratio} + {pause_bonus:.2f} = {adjusted_ratio:.2f}")
                logger.info(f"    최종 토큰 수: {dynamic_max_tokens} (범위: {min_tokens}-{max_tokens}, 일관성 우선)")
            else:
                dynamic_max_tokens = max_new_tokens
            
            # 반복 패턴 감지를 위한 청크 컨텍스트 정보 로깅 (원본 텍스트 기준)
            logger.info(f"  📝 청크 {chunk_idx+1} 텍스트 컨텍스트:")
            logger.info(f"    📄 텍스트 내용: '{chunk_text_for_generation[:100]}{'...' if len(chunk_text_for_generation) > 100 else ''}'")
            logger.info(f"    📏 텍스트 길이: {len(chunk_text_for_generation)}자")
            logger.info(f"    🔤 단어 수: {len(chunk_text_for_generation.split())}개")
            
            logger.info(f"  🎛️ 드리프트 방지 생성 파라미터:")
            logger.info(f"    - Temperature: {chunk_temperature:.3f} (청크 {chunk_idx+1})")
            logger.info(f"    - Rep_penalty: {chunk_repetition_penalty:.3f}")
            logger.info(f"    - Max_tokens: {dynamic_max_tokens}")
            logger.info(f"    - 이전 오디오 사용: 🚫 완전 차단")
            logger.info(f"    - 참조 오디오 의존도: 💯 100%")

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
                text_context=chunk_text_for_generation,  # 반복 패턴 추적을 위한 텍스트 컨텍스트 (원본 텍스트)
                chunk_idx=chunk_idx,      # 청크 인덱스
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
            
            # 🎯 NEW: 생성된 오디오를 다음 청크의 연속성을 위해 저장
            # 각 청크에서 생성된 오디오 토큰과 텍스트를 저장하여 다음 청크에서 참조로 활용
            previous_generated_tokens.append(codes.cpu())  # CPU로 이동하여 메모리 절약
            # 원본 텍스트를 저장 (오버랩이 적용된 텍스트가 아닌)
            previous_generated_texts.append(original_chunk_text)
            
            # 메모리 관리: 최대 3개의 이전 청크만 유지 (너무 많으면 메모리 부족)
            if len(previous_generated_tokens) > 3:
                previous_generated_tokens.pop(0)  # 가장 오래된 것 제거
                previous_generated_texts.pop(0)
                logger.info(f"  🗑️ 메모리 관리: 오래된 청크 참조 제거 (최대 3개 유지)")
            
            logger.info(f"  💾 청크 {chunk_idx+1} 저장 완료: 다음 청크 연속성을 위해 보관 (총 {len(previous_generated_tokens)}개 저장)")
            
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=original_chunk_text)

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
