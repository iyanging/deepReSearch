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


from collections.abc import Collection
from typing import Annotated, Any, override

from annotated_types import Interval
from googleapiclient.discovery import (
    build,  # pyright: ignore[reportUnknownVariableType]
)
from langchain_core.runnables import run_in_executor
from pydantic import NonNegativeInt, PositiveInt, SecretStr

from deep_research.logger import root_logger
from deep_research.resources.document import Document
from deep_research.retrievers.base import Retriever, UnrecoverableRetrieverError
from deep_research.utils import assert_not_none

logger = root_logger.getChild(__name__)


GOOGLE_SEARCH_MAX_LIMIT = 10


class GoogleSearch(Retriever[str]):
    google_search_engine_id: str
    google_search: Any

    def __init__(
        self,
        google_api_key: SecretStr,
        google_search_engine_id: str,
    ) -> None:
        super().__init__()

        self.google_search_engine_id = google_search_engine_id
        self.google_search = build(
            "customsearch",
            "v1",
            developerKey=google_api_key.get_secret_value(),
            cache_discovery=False,
        ).cse()  # pyright: ignore[reportUnknownMemberType]

    @override
    async def retrieve(
        self,
        query: str,
        *,
        offset: NonNegativeInt,
        limit: PositiveInt,
    ) -> Collection[Document]:
        if limit > GOOGLE_SEARCH_MAX_LIMIT:
            raise UnrecoverableRetrieverError(
                "limit only supports [1, 10] in Google Custom Search API"
            )

        raw_result: dict[str, Any] = await run_in_executor(
            None,
            self._do_search,
            q=query,
            start=offset + 1,
            num=limit,
        )

        if "items" not in raw_result:
            return []

        return [
            Document(
                title=assert_not_none(
                    r.get("title", ""),
                    "Document title cannot be null",
                ),
                snippet=assert_not_none(
                    r.get("snippet", ""),
                    "Document snippet cannot be null",
                ),
                link=assert_not_none(
                    r.get("link"),
                    "Document link cannot be null",
                ),
                content=None,
            )
            for r in raw_result["items"]
        ]

    def _do_search(
        self,
        q: str,
        start: PositiveInt,
        num: Annotated[int, Interval(ge=1, le=10)],
    ) -> dict[str, Any]:
        return self.google_search.list(
            q=q,
            cx=self.google_search_engine_id,
            start=start,
            num=num,
        ).execute()
