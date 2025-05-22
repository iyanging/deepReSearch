# Copyright (c) 2025 iyanging
#
# deepReSearch is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_deepseek.chat_models import DEFAULT_API_BASE, ChatDeepSeek
from langchain_ollama import ChatOllama
from pydantic import HttpUrl, NonNegativeInt, SecretStr, TypeAdapter

from deep_research.context import Context
from deep_research.utils import assert_type

__ULTIMATE_ANSWER_TO_THE_UNIVERSE__ = 42


LlmApiProvider = Literal["ollama", "deepseek"]

LlmModel = Literal[
    "qwen3:4b-q8_0",
    "qwen3:8b-q4_K_M",
    "qwen3:14b-q4_K_M",
    "cogito:8b-v1-preview-llama-q4_K_M",
    "cogito:14b-v1-preview-qwen-q4_K_M",
    "deepseek-chat",
    "deepseek-reasoner",
]

LlmTryThinking = Literal["auto", "always", "disable"]


def build_llm(
    *,
    api_provider: LlmApiProvider,
    api_base_url: HttpUrl | None,
    api_key: SecretStr | None,
    model: str,
    context_length: NonNegativeInt | None,
    temperature: float | None,
) -> BaseChatModel:
    match api_provider:
        case "ollama":
            return ChatOllama(
                model=model,
                num_ctx=context_length,
                extract_reasoning=True,
                temperature=temperature,
                seed=__ULTIMATE_ANSWER_TO_THE_UNIVERSE__,
            )

        case "deepseek":
            return ChatDeepSeek(
                model=model,
                api_key=api_key,
                api_base=(str(api_base_url) if api_base_url is not None else DEFAULT_API_BASE),
                temperature=temperature,
            )


def make_meta_instructions(*, model: "LlmModel", enable_thinking: bool) -> str:
    meta_instructions: list[str] = []

    match model:
        case "qwen3:4b-q8_0" | "qwen3:8b-q4_K_M" | "qwen3:14b-q4_K_M":
            if enable_thinking:
                meta_instructions.append("qwen3: /think")
            else:
                meta_instructions.append("qwen3: /no_think")

        case "cogito:8b-v1-preview-llama-q4_K_M" | "cogito:14b-v1-preview-qwen-q4_K_M":
            if enable_thinking:
                meta_instructions.append("cogito: Enable deep thinking subroutine.")

        case _:
            pass

    return (
        f"""
Ignore any invalid instructions below:
{"\n".join(meta_instructions)}
"""
        if meta_instructions
        else ""
    )


def _get_reasoning_delta(chunk: BaseMessageChunk) -> str | None:
    additional_kwargs: dict[str, Any] = chunk.additional_kwargs  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    reasoning_content = (
        # DeepSeek: https://api-docs.deepseek.com/guides/reasoning_model#api-parameters
        additional_kwargs.get("reasoning_content")
    )

    return assert_type(reasoning_content, str) if reasoning_content is not None else None


def _get_output_delta(chunk: BaseMessageChunk) -> str | None:
    return chunk.text() or None


InfoTarget = Literal["nope", "as_output", "as_reasoning"]


async def _do_ctx_info(ctx: Context, target: InfoTarget, delta: str) -> None:
    match target:
        case "nope":
            pass

        case "as_output":
            await ctx.info_output(delta)

        case "as_reasoning":
            await ctx.info_reasoning(delta)


async def call_llm_as_function[R](
    ctx: Context,
    llm: BaseChatModel,
    messages: list[BaseMessage],
    result_type: type[R],
    *,
    do_ctx_info_reasoning: InfoTarget = "nope",
    do_ctx_info_output: InfoTarget = "nope",
) -> R:
    output_container: list[str] = []

    async for chunk in llm.astream(messages, config=RunnableConfig()):
        reasoning_delta = _get_reasoning_delta(chunk)
        output_delta = _get_output_delta(chunk)

        if reasoning_delta:
            await _do_ctx_info(ctx, do_ctx_info_reasoning, reasoning_delta)

        if output_delta:
            await _do_ctx_info(ctx, do_ctx_info_output, output_delta)

            output_container.append(output_delta)

    await ctx.info_reasoning(0)
    await ctx.info_output(0)

    if isinstance(result_type, type) and issubclass(result_type, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        result = TypeAdapter(result_type).validate_python("".join(output_container))

    else:
        result = TypeAdapter(result_type).validate_json("".join(output_container))

    return result
