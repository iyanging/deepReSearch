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

from pydantic import HttpUrl, NonNegativeFloat, NonNegativeInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.agents.analyst import Analyst
from deep_research.agents.planner import Planner
from deep_research.config import Config
from deep_research.llm import LlmApiProvider, LlmModel, LlmTryThinking
from deep_research.logger import setup_logging
from deep_research.retrievers.google_search import GoogleSearch


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
    analyst_llm_temperature: NonNegativeFloat = 0.3


async def main() -> None:
    setup_logging()

    envs = EnvVar()  # pyright: ignore[reportCallIssue]

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
        planner_max_fetch_count_of_keywords_documents=30,
        planner_max_valid_count_of_keywords_documents=10,
        planner_max_retries_of_searching_internet=3,
        planner_max_retries_of_taking_document_note=3,
        planner_max_retries_of_distilling_initial_note=3,
        planner_max_concurrency_of_checking_document_validity=4,
        planner_max_concurrency_of_calling_analyst=4,
        # analyst
        analyst_llm_api_provider=envs.analyst_llm_api_provider,
        analyst_llm_api_base_url=envs.analyst_llm_api_base_url,
        analyst_llm_api_key=envs.analyst_llm_api_key,
        analyst_llm_model=envs.analyst_llm_model,
        analyst_llm_context_length=envs.analyst_context_length,
        analyst_llm_try_thinking=envs.analyst_llm_try_thinking,
        analyst_llm_temperature=envs.analyst_llm_temperature,
    )

    retriever = GoogleSearch(
        google_api_key=envs.retriever_google_api_key,
        google_search_engine_id=envs.retriever_google_search_engine_id,
    )
    analyst = Analyst(
        config,
    )
    planner = Planner(
        config,
        analyst,
        retriever,
    )

    original_question = None
    while not original_question:
        original_question = input("User! > ")

    answer = await planner.solve(original_question)

    print("Agent >", answer)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
