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


import backoff
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, HttpUrl, NonNegativeFloat, NonNegativeInt, PositiveInt, SecretStr

from deep_research.agents.analyst import Analyst, AnalystError
from deep_research.llm import LlmApiProvider, LlmModel, LlmTryThinking, build_llm
from deep_research.logger import root_logger
from deep_research.resources.document import Document, DocumentError
from deep_research.retrievers.base import RecoverableRetrieverError, Retriever
from deep_research.utils import parallel

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
    planner_max_fetch_count_of_keywords_documents: NonNegativeInt
    planner_max_valid_count_of_keywords_documents: NonNegativeInt
    planner_max_retries_of_searching_internet: NonNegativeInt
    planner_max_retries_of_taking_document_note: NonNegativeInt
    planner_max_retries_of_distilling_initial_note: NonNegativeInt
    planner_max_concurrency_of_checking_document_validity: PositiveInt
    planner_max_concurrency_of_calling_analyst: PositiveInt


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

        # TODO: check validity of content
        try:
            logger.info("Try fetch content")
            content = await document.fetch_content()

        except DocumentError as e:
            logger.warning("Cannot fetch content: %s", e)
            content = None

        if not content:
            logger.info("Declare document invalid: content is empty")
            return (document, False)

        return (document, True)

    async def search_internet(
        self,
        query: str,
        *,
        max_fetch_count: NonNegativeInt,
        max_valid_count: NonNegativeInt,
    ) -> set[Document]:
        logger.info("Search with query >> %s", query)

        # Google Search API requires
        max_search_count = 10

        checked_document_to_validity: dict[Document, bool] = {}

        round_count = -1

        while True:
            round_count += 1
            logger.info("Start round %d", round_count)

            fetch_count = len(checked_document_to_validity)
            valid_count = sum(1 for v in checked_document_to_validity.values() if v is True)

            if fetch_count >= max_fetch_count or valid_count >= max_valid_count:
                logger.info("Max count reached")
                break

            gap_count = min(max_fetch_count, max_valid_count) - valid_count

            logger.info("Gap count: %d", gap_count)

            if gap_count <= 0:
                break

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
                    offset=len(checked_document_to_validity),
                    limit=search_count,
                )
            )
            logger.info("Fetched %d documents", len(unchecked_documents))

            async for result in parallel(
                {
                    self._check_if_document_valid(unchecked_document)
                    for unchecked_document in unchecked_documents
                    if unchecked_document not in checked_document_to_validity
                },
                concurrency=self.cfg.planner_max_concurrency_of_checking_document_validity,
            ):
                try:
                    document, validity = result.result()

                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "exception occurred during checking document validity, ignore it",
                        exc_info=e,
                    )
                    continue

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

    async def take_initial_note(self, original_question: str) -> str:
        logger.info("Search internet for original question")

        original_question_documents = await self.search_internet(
            original_question,
            max_fetch_count=self.cfg.planner_max_fetch_count_of_original_question_documents,
            max_valid_count=self.cfg.planner_max_valid_count_of_original_question_documents,
        )

        logger.info("Extract keywords from original question")

        keywords = await self.analyst.extract_keywords(original_question)

        logger.info("Keywords: %s", keywords)

        logger.info("Search internet for keywords")

        # TODO: design a DocumentFetcher to skip fetching those checked documents
        keywords_query = " ".join(keywords)
        keywords_documents = await self.search_internet(
            keywords_query,
            max_fetch_count=self.cfg.planner_max_fetch_count_of_keywords_documents,
            max_valid_count=self.cfg.planner_max_valid_count_of_keywords_documents,
        )

        logger.info("Merge question documents and keywords documents")

        initial_documents = self._merge_document_set(
            original_question_documents,
            keywords_documents,
        )

        del original_question_documents
        del keywords_documents

        async for result in parallel(
            {
                self.analyst.judge_document_useful(
                    [original_question],
                    document,
                )
                for document in initial_documents
            },
            concurrency=self.cfg.planner_max_concurrency_of_calling_analyst,
        ):
            try:
                document, is_useful = result.result()

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "exception occurred during judging document useful, ignore it",
                    exc_info=e,
                )
                continue

            logger.info(
                "Document judged useful, document: %s, useful: %s",
                document,
                is_useful,
            )

        initial_sub_note: dict[Document, str] = {}
        async for result in parallel(
            {
                backoff.on_exception(
                    backoff.constant,
                    AnalystError,
                    jitter=None,
                    max_tries=self.cfg.planner_max_retries_of_taking_document_note,
                )(self.analyst.take_document_note)(
                    original_question,
                    document,
                )
                for document in initial_documents
                if document.may_be_useful
            },
            concurrency=self.cfg.planner_max_concurrency_of_calling_analyst,
        ):
            try:
                document, document_note = result.result()

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "exception occurred during taking document note, ignore it",
                    exc_info=e,
                )
                continue

            initial_sub_note[document] = document_note

        if not initial_sub_note:
            raise PlannerError("Has no useful documents notes")

        logger.info("Distill to initial note")

        initial_note = await backoff.on_exception(
            backoff.constant,
            AnalystError,
            jitter=None,
            max_tries=self.cfg.planner_max_retries_of_distilling_initial_note,
        )(self.analyst.distill_note)(
            original_question,
            initial_sub_note.values(),
        )

        return initial_note

    async def solve(self, original_question: str) -> str:
        initial_note = await self.take_initial_note(original_question)

        return initial_note


class PlannerError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
