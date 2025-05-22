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

import asyncio
from types import TracebackType
from typing import Literal, override
from uuid import uuid4

import streamlit as st
from pydantic import HttpUrl, NonNegativeFloat, NonNegativeInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from streamlit.delta_generator import (
    DeltaGenerator,
    _check_nested_element_violation,  # noqa: PLC2701 # pyright: ignore[reportPrivateUsage]
)

from deep_research.agents.analyst import Analyst
from deep_research.agents.planner import Planner
from deep_research.config import Config
from deep_research.context import Context, ContextCollaborator
from deep_research.llm import LlmApiProvider, LlmModel, LlmTryThinking
from deep_research.logger import setup_logging
from deep_research.retrievers.browser_google_search import BrowserGoogleSearch


class EnvVar(BaseSettings):
    model_config = SettingsConfigDict(
        BaseSettings.model_config,
        env_ignore_empty=True,
    )

    retriever_google_api_key: SecretStr
    retriever_google_search_engine_id: str

    planner_llm_api_base_url: HttpUrl | None = None
    planner_llm_api_provider: LlmApiProvider
    planner_llm_api_key: SecretStr
    planner_llm_model: LlmModel
    planner_llm_try_thinking: LlmTryThinking = "auto"
    planner_llm_temperature: NonNegativeFloat = 1.0

    analyst_llm_api_base_url: HttpUrl | None = None
    analyst_llm_api_provider: LlmApiProvider
    analyst_llm_api_key: SecretStr
    analyst_llm_model: LlmModel
    analyst_context_length: NonNegativeInt | None = None
    analyst_llm_try_thinking: LlmTryThinking = "auto"
    analyst_llm_temperature: NonNegativeFloat = 1.0


async def main(
    envs: EnvVar,
    original_question: str,
    collaborators: list[ContextCollaborator],
) -> str:
    config = Config(
        # planner
        planner_llm_api_provider=envs.planner_llm_api_provider,
        planner_llm_api_base_url=envs.planner_llm_api_base_url,
        planner_llm_api_key=envs.planner_llm_api_key,
        planner_llm_model=envs.planner_llm_model,
        planner_llm_try_thinking=envs.planner_llm_try_thinking,
        planner_llm_temperature=envs.planner_llm_temperature,
        planner_blocked_document_link_host_pattern={".csdn.net"},
        planner_max_fetch_count_of_original_question_documents=30,
        planner_max_valid_count_of_original_question_documents=10,
        planner_max_fetch_count_of_original_keywords_documents=30,
        planner_max_valid_count_of_original_keywords_documents=10,
        planner_max_fetch_count_of_translated_keywords_documents=30,
        planner_max_valid_count_of_translated_keywords_documents=10,
        planner_max_retries_of_searching_internet=3,
        planner_max_concurrency_of_checking_document_validity=4,
        planner_max_concurrency_of_calling_analyst=4,
        planner_min_character_count_of_document_content=99,
        # analyst
        analyst_llm_api_provider=envs.analyst_llm_api_provider,
        analyst_llm_api_base_url=envs.analyst_llm_api_base_url,
        analyst_llm_api_key=envs.analyst_llm_api_key,
        analyst_llm_model=envs.analyst_llm_model,
        analyst_llm_context_length=envs.analyst_context_length,
        analyst_llm_try_thinking=envs.analyst_llm_try_thinking,
        analyst_llm_temperature=envs.analyst_llm_temperature,
        analyst_max_retries_of_taking_document_note=3,
        analyst_max_retries_of_distilling_initial_note=3,
    )

    # retriever = GoogleSearch(
    #     google_api_key=envs.retriever_google_api_key,
    #     google_search_engine_id=envs.retriever_google_search_engine_id,
    # )
    # retriever = DuckDuckGoSearch("wt-wt")
    retriever = BrowserGoogleSearch()
    analyst = Analyst(
        config,
    )
    planner = Planner(
        config,
        analyst,
        retriever,
    )

    async with Context(
        original_question,
        run_id=str(uuid4()),
        collaborators=collaborators,
    ) as ctx:
        report = await planner.research(
            ctx,
            original_question,
        )

    return report


class StreamlitCollaborator(ContextCollaborator):
    canvas: DeltaGenerator
    reasoning_canvas: DeltaGenerator | None
    output_canvas: DeltaGenerator | None

    reasoning_buf: str
    output_buf: str

    INPUT_CURSOR = "▕"

    def __init__(self, canvas: DeltaGenerator) -> None:
        super().__init__()

        self.canvas = canvas
        self.reasoning_canvas = None
        self.output_canvas = None
        self.reasoning_buf = ""
        self.output_buf = ""

    def _get_delta_canvas(self, kind: Literal["reasoning", "output"]) -> DeltaGenerator:
        match kind:
            case "reasoning":
                if self.reasoning_canvas is None:
                    self.reasoning_canvas = self.canvas.markdown("")

                return self.reasoning_canvas

            case "output":
                if self.output_canvas is None:
                    self.output_canvas = self.canvas.markdown("")

                return self.output_canvas

    @override
    async def aenter(self, new_context: Context) -> "StreamlitCollaborator":
        sub_container = self.canvas.status(new_context.name, expanded=True)

        return StreamlitCollaborator(sub_container)

    @override
    async def aexit(
        self,
        context: Context,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = self.canvas.__exit__(None, None, None)

    @override
    async def info_reasoning(self, delta: str | Literal[0]) -> None:
        if delta:
            self.reasoning_buf += delta

        reasoning_canvas = self._get_delta_canvas("reasoning")

        if delta != 0:
            self.reasoning_canvas = reasoning_canvas.markdown(
                self.reasoning_buf + self.INPUT_CURSOR,
            )
        else:
            self.reasoning_canvas = reasoning_canvas.markdown(self.reasoning_buf)

    @override
    async def info_output(self, delta: str | Literal[0]) -> None:
        if delta:
            self.output_buf += delta

        output_canvas = self._get_delta_canvas("output")

        if delta != 0:
            self.output_canvas = output_canvas.markdown(
                self.output_buf + self.INPUT_CURSOR,
            )
        else:
            self.output_canvas = output_canvas.markdown(self.output_buf)

    @override
    async def info(self, message: str) -> None:
        _ = self.canvas.markdown(message)


def _mocked_check_nested_element_violation(*_: object, **__: object) -> None:
    # Silent all checks
    pass


# Monkey Patch
_check_nested_element_violation.__code__ = _mocked_check_nested_element_violation.__code__


def draw_page() -> tuple[str, DeltaGenerator] | None:
    st.set_page_config(
        page_title="deepReSearch",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    original_question = st.container().chat_input("What can I help you ? 🧐") or None

    if not original_question:
        return None

    else:
        canvas = st.empty().container(border=False)

        return (original_question, canvas)


def st_main() -> None:
    setup_logging()

    envs = EnvVar()  # pyright: ignore[reportCallIssue]

    original_question_and_canvas = draw_page()
    if not original_question_and_canvas:
        return

    original_question, canvas = original_question_and_canvas

    report = asyncio.run(
        main(
            envs,
            original_question,
            [StreamlitCollaborator(canvas)],
        )
    )

    _ = canvas.container().chat_message("Report").markdown(report)


st_main()
