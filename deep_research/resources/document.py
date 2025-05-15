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

from typing import Self, override

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)
from crawl4ai.models import CrawlResultContainer
from pydantic import BaseModel, Field

from deep_research.logger import root_logger
from deep_research.utils import assert_type

logger = root_logger.getChild(__name__)


class Document(BaseModel):
    title: str = Field(
        description="The title declared by this document",
        repr=True,
    )
    snippet: str = Field(
        description="The concluded snippet of this document",
        repr=False,
    )
    link: str = Field(
        description="The URL of this document",
        repr=True,
    )
    content: str | None = Field(
        description="The content of this document",
        repr=False,
    )

    may_be_useful: bool | None = Field(
        None,
        init=False,
        repr=True,
    )

    @override
    def __hash__(self) -> int:
        return hash(self.link)

    @override
    def __str__(self) -> str:
        return repr(self)

    def render_markdown(
        self,
        *,
        level: int,
        with_link: bool,
        with_content: bool,
    ) -> str:
        result = [
            f"""
{"#" * level} Title
{self.title}
""",
            f"""
{"#" * level} Snippet
{self.snippet}
""",
        ]

        if with_content:
            result.append(f"""
{"#" * level} Content
```Markdown
{self.content}
```
""")

        if with_link:
            result.append(f"""
{"#" * level} Link
{self.link}
""")

        return "".join(result)

    async def fetch_content(self, *, refetch: bool = False) -> str:
        if self.content is not None and not refetch:
            return self.content

        async with AsyncWebCrawler(config=BrowserConfig(verbose=False)) as crawler:
            result = assert_type(
                await crawler.arun(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    str(self.link),
                    CrawlerRunConfig(
                        verbose=False,
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=PruningContentFilter(
                                # Lower → more content retained
                                # higher → more content pruned
                                threshold=0.45,
                                # "fixed" or "dynamic"
                                threshold_type="dynamic",
                                # Ignore nodes with <5 words
                                min_word_threshold=5,
                            ),
                        ),
                    ),
                ),
                CrawlResultContainer[CrawlResult],
            )

            if not result.success:
                raise DocumentError("Cannot fetch web page: " + result.error_message)

            markdown = assert_type(result.markdown.fit_markdown, str)

            self.content = markdown

        return self.content

    @staticmethod
    def merge_text(a: str, b: str) -> str:
        if a in b:
            return b
        if b in a:
            return a

        max_overlapped_len = min(len(a), len(b))
        for i in range(max_overlapped_len, 0, -1):
            if a[-i:] == b[:i]:
                return a + b[i:]
            if b[-i:] == a[:i]:
                return b + a[i:]

        return a + "..." + b

    def merge(self, same_link_document: "Document") -> Self:
        if same_link_document.link != self.link:
            raise DocumentError("Cannot merge from different source document")

        self.title = self.merge_text(self.title, same_link_document.title)
        self.snippet = self.merge_text(self.snippet, same_link_document.snippet)

        return self


class DocumentError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
