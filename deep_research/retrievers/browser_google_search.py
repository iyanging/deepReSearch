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
from urllib.parse import quote_plus

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.models import CrawlResultContainer
from pydantic import NonNegativeInt, PositiveInt

from deep_research.resources.document import Document
from deep_research.retrievers.base import RecoverableRetrieverError, Retriever
from deep_research.utils import assert_type


class BrowserGoogleSearch(Retriever[str]):
    MAX_COUNT_PER_PAGE = 10

    @override
    async def retrieve(
        self,
        query: str,
        *,
        offset: NonNegativeInt,
        limit: PositiveInt,
    ) -> Collection[Document]:
        result: list[Document] = []

        extractor_title_and_link = JsonCssExtractionStrategy(
            {
                "baseSelector": 'div[data-snhf="0"]',
                "fields": [
                    {"name": "title", "selector": "h3", "type": "text"},
                    {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"},
                ],
            }
        )
        extractor_snippet = JsonCssExtractionStrategy(
            {
                "baseSelector": 'div[data-sncf="1"]',
                "fields": [
                    {"name": "snippet", "selector": "*", "type": "text"},
                ],
            }
        )

        async with AsyncWebCrawler(config=BrowserConfig(verbose=False)) as crawler:
            config = CrawlerRunConfig(
                verbose=False,
                only_text=True,
                target_elements=[
                    'div[data-async-context^="query:"]',  # whole result block
                    # 'div[data-snhf="0"]',  # title and link
                    # 'div[data-sncf="1"]',  # snippet
                ],
            )

            # "start" starts from 0
            for start in range(offset, offset + limit, self.MAX_COUNT_PER_PAGE):
                url = self._build_url(query, start)

                page = assert_type(
                    await crawler.arun(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        url,
                        config=config,
                    ),
                    CrawlResultContainer[CrawlResult],
                )

                if not page.success:
                    raise RecoverableRetrieverError("Cannot fetch web page: " + page.error_message)

                cleaned_html = assert_type(page.cleaned_html, str)

                title_and_link_list = extractor_title_and_link.extract(  # pyright: ignore[reportUnknownMemberType]
                    url,
                    cleaned_html,
                )
                snippet_list = extractor_snippet.extract(  # pyright: ignore[reportUnknownMemberType]
                    url,
                    cleaned_html,
                )

                for title_and_link, snippet in zip(
                    title_and_link_list,
                    snippet_list,
                    strict=False,
                ):
                    result.append(
                        Document(
                            title=title_and_link["title"],
                            link=title_and_link["link"],
                            snippet=snippet["snippet"],
                            content=None,
                        )
                    )

        return result

    @staticmethod
    def _build_url(query: str, start: int) -> str:
        # quote_plus() will encode " " to "+"
        # which is the same as google page
        encoded_query = quote_plus(query)
        return f"https://www.google.com/search?q={encoded_query}&start={start}"
