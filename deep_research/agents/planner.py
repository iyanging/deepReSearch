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


import string

import backoff
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, HttpUrl, NonNegativeFloat, NonNegativeInt, PositiveInt, SecretStr

from deep_research.agents.analyst import Analyst
from deep_research.llm import LlmApiProvider, LlmModel, LlmTryThinking, build_llm
from deep_research.logger import root_logger
from deep_research.resources.document import Document, DocumentError
from deep_research.retrievers.base import RecoverableRetrieverError, Retriever
from deep_research.utils import count_without, parallel

logger = root_logger.getChild(__name__)


class PlannerConfig(BaseModel):
    planner_llm_api_provider: LlmApiProvider
    planner_llm_api_base_url: HttpUrl | None
    planner_llm_api_key: SecretStr | None
    planner_llm_model: LlmModel
    planner_llm_try_thinking: LlmTryThinking
    planner_llm_temperature: NonNegativeFloat
    planner_blocked_document_link_host_pattern: set[str]
    planner_max_fetch_count_of_original_question_documents: NonNegativeInt
    planner_max_valid_count_of_original_question_documents: NonNegativeInt
    planner_max_fetch_count_of_original_keywords_documents: NonNegativeInt
    planner_max_valid_count_of_original_keywords_documents: NonNegativeInt
    planner_max_fetch_count_of_translated_keywords_documents: NonNegativeInt
    planner_max_valid_count_of_translated_keywords_documents: NonNegativeInt
    planner_max_retries_of_searching_internet: NonNegativeInt
    planner_max_concurrency_of_checking_document_validity: PositiveInt
    planner_max_concurrency_of_calling_analyst: PositiveInt
    planner_min_character_count_of_document_content: PositiveInt


class Planner:
    cfg: PlannerConfig
    llm: BaseChatModel

    analyst: Analyst
    retriever: Retriever[str]

    def __init__(
        self,
        cfg: PlannerConfig,
        analyst: Analyst,
        retriever: Retriever[str],
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.llm = build_llm(
            api_provider=cfg.planner_llm_api_provider,
            api_base_url=cfg.planner_llm_api_base_url,
            api_key=cfg.planner_llm_api_key,
            model=cfg.planner_llm_model,
            context_length=None,
            temperature=cfg.planner_llm_temperature,
        )

        self.analyst = analyst
        self.retriever = retriever

    @staticmethod
    def _merge_document_set(a: set[Document], b: set[Document]) -> set[Document]:
        a_table = {hash(d): d for d in a}
        b_table = {hash(d): d for d in b}

        unique_set = (a - b) | (b - a)
        merge_set = {a_table[k].merge(b_table[k]) for k in (a_table.keys() & b_table.keys())}

        return unique_set | merge_set

    async def _check_if_document_valid(self, document: Document) -> tuple[Document, bool]:
        for blocked_host_pattern in self.cfg.planner_blocked_document_link_host_pattern:
            if blocked_host_pattern in (HttpUrl(document.link).host or ""):
                logger.info("Declare document invalid: link host is blocked")
                return (document, False)

        try:
            logger.info("Try fetch content")
            content = await document.fetch_content()

        except DocumentError as e:
            logger.warning("Cannot fetch content: %s", e)
            content = None

        if (
            not content
            or count_without(content, set(string.whitespace))
            <= self.cfg.planner_min_character_count_of_document_content
        ):
            logger.info("Declare document invalid: content is empty")
            return (document, False)

        return (document, True)

    async def search_internet_with_filter(
        self,
        query: str,
        *,
        max_fetch_count: int,
        max_valid_count: int,
    ) -> set[Document]:
        logger.info("Search with query >> %s", query)

        # Google Search API requires
        max_search_count = 10

        checked_document_to_validity: dict[Document, bool] = {}

        round_count = -1

        while True:
            fetch_count = len(checked_document_to_validity)
            valid_count = sum(1 for v in checked_document_to_validity.values() if v is True)

            if fetch_count >= max_fetch_count or valid_count >= max_valid_count:
                logger.info("Max fetch count or valid count reached")
                break

            round_count += 1
            logger.info("Start round %d", round_count)

            gap_count = min(max_fetch_count, max_valid_count) - valid_count
            if gap_count <= 0:
                break  # not possible

            logger.info("Gap count: %d", gap_count)

            search_count = min(max_search_count, gap_count)

            unchecked_documents = set(
                await backoff.on_exception(
                    backoff.constant,
                    RecoverableRetrieverError,
                    jitter=None,
                    max_tries=self.cfg.planner_max_retries_of_searching_internet,
                    interval=3,
                )(self.retriever.retrieve)(
                    query,
                    offset=len(checked_document_to_validity),  # checked document count
                    limit=search_count,
                )
            )
            logger.info("Fetched %d documents", len(unchecked_documents))

            if not unchecked_documents:
                logger.info("No more documents to fetch")
                break

            async for result in parallel(
                {
                    # pretend to be some kind of "rerank"
                    self._check_if_document_valid(unchecked_document)
                    for unchecked_document in unchecked_documents
                    if unchecked_document not in checked_document_to_validity
                },
                concurrency=self.cfg.planner_max_concurrency_of_checking_document_validity,
                ignore_exceptions=True,
            ):
                document, validity = result.result()

                checked_document_to_validity[document] = validity

                logger.info(
                    "Document checked validity, document: %s, validity: %s",
                    document,
                    validity,
                )

        valid_documents = {
            document for document, v in checked_document_to_validity.items() if v is True
        }

        logger.info("Selected %d valid documents", len(valid_documents))

        return valid_documents

    async def take_document_note(
        self,
        question: str,
        document: Document,
    ) -> tuple[Document, tuple[str, str] | None]:
        if document.may_be_useful is False:
            return (document, None)

        may_be_useful = await self.analyst.judge_document_useful(
            [question],
            document,
        )

        if not may_be_useful:
            return (document, None)

        note, expanded_question = await self.analyst.take_document_note(
            question,
            document,
        )

        return (document, (note, expanded_question))

    async def take_initial_note(self, original_question: str) -> str:
        logger.info("Extract keywords from original question")

        original_keywords, translated_keywords = await self.analyst.extract_keywords(
            original_question,
            "english",
        )

        logger.info("Original keywords: %s", original_keywords)
        logger.info("Translated keywords: %s", translated_keywords)

        logger.info("Search internet for original question and keywords")

        # TODO: design a DocumentFetcher to skip fetching those checked documents
        original_keywords_query = " ".join(original_keywords)
        translated_keywords_query = " ".join(translated_keywords)

        search_tasks = {
            self.search_internet_with_filter(
                original_question,
                max_fetch_count=self.cfg.planner_max_fetch_count_of_original_question_documents,
                max_valid_count=self.cfg.planner_max_valid_count_of_original_question_documents,
            ),
            self.search_internet_with_filter(
                original_keywords_query,
                max_fetch_count=self.cfg.planner_max_fetch_count_of_original_keywords_documents,
                max_valid_count=self.cfg.planner_max_valid_count_of_original_keywords_documents,
            ),
        }
        if translated_keywords_query:
            search_tasks.add(
                self.search_internet_with_filter(
                    translated_keywords_query,
                    max_fetch_count=self.cfg.planner_max_fetch_count_of_translated_keywords_documents,
                    max_valid_count=self.cfg.planner_max_valid_count_of_translated_keywords_documents,
                )
            )

        initial_documents: set[Document] = set()
        async for search_result in parallel(
            search_tasks,
            concurrency=len(search_tasks),
            ignore_exceptions=True,
        ):
            partial_keywords_documents = search_result.result()

            initial_documents = self._merge_document_set(
                initial_documents,
                partial_keywords_documents,
            )

        if not initial_documents:
            raise PlannerError("Has no useful documents")

        initial_sub_original_note: dict[Document, str] = {}
        initial_sub_expanded_note: dict[Document, str] = {}

        async for result in parallel(
            {
                self.take_document_note(original_question, document)
                for document in initial_documents
            },
            concurrency=self.cfg.planner_max_concurrency_of_calling_analyst,
            ignore_exceptions=True,
        ):
            document, note_and_question = result.result()

            if note_and_question is None:
                logger.info(
                    "Document judged useless, skip taking note, document: %s",
                    document,
                )
                continue

            original_note, expanded_question = note_and_question

            initial_sub_original_note[document] = original_note
            initial_sub_expanded_note[document] = expanded_question

        if not initial_sub_original_note:
            raise PlannerError("Has no useful documents notes")

        # TODO: handle expanded questions

        logger.info("Distill to initial note")

        initial_note = await self.analyst.distill_note(
            original_question,
            initial_sub_original_note.values(),
        )

        return initial_note

    async def research(self, original_question: str) -> str:
        initial_note = await self.take_initial_note(original_question)

        return initial_note


class PlannerError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
