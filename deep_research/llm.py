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

from collections.abc import AsyncIterator
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_deepseek.chat_models import DEFAULT_API_BASE, ChatDeepSeek
from langchain_ollama import ChatOllama
from langgraph.prebuilt import (
    create_react_agent,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import HttpUrl, NonNegativeInt, SecretStr

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


async def call_llm_as_function(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    result_collector: BaseTool,
    *,
    tools: list[BaseTool] | None = None,
) -> Any:
    cloned_result_collector = result_collector.model_copy(deep=True)
    cloned_result_collector.return_direct = True
    cloned_result_collector.handle_validation_error = True

    all_tools = [cloned_result_collector, *(tools or [])]

    return await create_react_agent(
        model=llm.bind_tools(  # pyright: ignore[reportUnknownMemberType]
            all_tools,
            tool_choice=cloned_result_collector.name,
        ),
        tools=all_tools,
    ).ainvoke({"messages": messages})


def call_llm_as_stream(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    *,
    tools: list[BaseTool] | None = None,
) -> AsyncIterator[BaseMessage]:
    if tools:
        return llm.bind_tools(tools).astream(messages)  # pyright: ignore[reportUnknownMemberType]

    return llm.astream(messages)
