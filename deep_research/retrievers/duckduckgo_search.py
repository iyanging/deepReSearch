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
from typing import override

from duckduckgo_search import DDGS
from duckduckgo_search import exceptions as ddgs_exceptions
from langchain_core.runnables import run_in_executor
from pydantic import NonNegativeInt, PositiveInt

from deep_research.logger import root_logger
from deep_research.resources.document import Document
from deep_research.retrievers.base import RecoverableRetrieverError, Retriever

logger = root_logger.getChild(__name__)


class DuckDuckGoSearch(Retriever[str]):
    ddgs: DDGS
    region: str

    def __init__(self, region: str) -> None:
        super().__init__()

        self.ddgs = DDGS()
        self.region = region

    @override
    async def retrieve(
        self,
        query: str,
        *,
        offset: NonNegativeInt,
        limit: PositiveInt,
    ) -> Collection[Document]:
        try:
            with self.ddgs:
                raw_result = await run_in_executor(
                    None,
                    self.ddgs.text,
                    keywords=query,
                    region=self.region,
                    backend="lite",
                    max_results=offset + limit,
                )

        except ddgs_exceptions.RatelimitException as e:
            raise RecoverableRetrieverError("DuckDuckGo rate limit exceeded") from e

        except ddgs_exceptions.TimeoutException as e:
            raise RecoverableRetrieverError("DuckDuckGo timeout") from e

        except ddgs_exceptions.DuckDuckGoSearchException as e:
            raise RecoverableRetrieverError("unknown DuckDuckGo error") from e

        return [
            Document(
                title=r["title"],
                snippet=r["body"],
                link=r["href"],
                content=None,
            )
            for r in raw_result
        ]
