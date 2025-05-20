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
from typing import Annotated

import backoff
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel, HttpUrl, NonNegativeFloat, NonNegativeInt, SecretStr, TypeAdapter

from deep_research.llm import (
    LlmApiProvider,
    LlmModel,
    LlmTryThinking,
    build_llm,
    call_llm_as_function,
    call_llm_as_stream,
    make_meta_instructions,
)
from deep_research.logger import root_logger
from deep_research.resources.document import Document
from deep_research.utils import assert_type, retryable

logger = root_logger.getChild(__name__)


class AnalystConfig(BaseModel):
    analyst_llm_api_provider: LlmApiProvider
    analyst_llm_api_base_url: HttpUrl | None
    analyst_llm_api_key: SecretStr | None
    analyst_llm_model: LlmModel
    analyst_llm_context_length: NonNegativeInt | None
    analyst_llm_try_thinking: LlmTryThinking
    analyst_llm_temperature: NonNegativeFloat
    analyst_max_retries_of_taking_document_note: NonNegativeInt
    analyst_max_retries_of_distilling_initial_note: NonNegativeInt


class AnalystError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Analyst:
    cfg: AnalystConfig
    llm: BaseChatModel

    @staticmethod
    def __self_class(_: "Analyst") -> None: ...

    def __init__(self, cfg: AnalystConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.llm = build_llm(
            api_provider=cfg.analyst_llm_api_provider,
            api_base_url=cfg.analyst_llm_api_base_url,
            api_key=cfg.analyst_llm_api_key,
            model=cfg.analyst_llm_model,
            context_length=cfg.analyst_llm_context_length,
            temperature=cfg.analyst_llm_temperature,
        )

    def _make_system_prompt(self, *, default_thinking: bool) -> str:
        match self.cfg.analyst_llm_try_thinking:
            case "auto":
                enable_thinking = default_thinking
            case "always":
                enable_thinking = True
            case "disable":
                enable_thinking = False

        return f"""
# MetaInstruction
{
            make_meta_instructions(
                model=self.cfg.analyst_llm_model,
                enable_thinking=enable_thinking,
            )
        }

# Identity
You are a highly sophisticated and rigorous information analyst and linguist,
powered by "deepReSearch".
"""

    async def extract_keywords(
        self,
        paragraph: str,
        with_language: str | None,
    ) -> tuple[list[str], list[str]]:
        original_keywords_container: list[str] = []
        translated_keywords_container: list[str] = []

        @tool(return_direct=True)  # this is final call
        def _return_keywords(
            original_keywords: Annotated[
                list[str],
                "A list of extracted keywords, from original paragraph",
            ],
            translated_keywords: Annotated[
                list[str],
                """
                A list of extracted keywords, from translated paragraph;
                If you did not do translation, make the list empty.
                """,
            ],
        ) -> None:
            """Return extracted keywords."""

            original_keywords_container.clear()
            original_keywords_container.extend(original_keywords)

            translated_keywords_container.clear()
            translated_keywords_container.extend(translated_keywords)

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=False)
            + f"""
## Task

Extract keywords.

- Identify and extract ALL keywords from the provided paragraph, including:
    * Named entities (e.g., people, places, organizations)
    * Action verbs (e.g., "install", "compare")
    * Technical terms/phrases (maintain exact wording)
    * Numerical values/units
- Remove modifiers, e.g., particles, adjectives
- Exclude words that carry little substantive meaning, such as "how many", "whether"
- Preserve even misspelled or ambiguous keywords exactly as given.
- Keywords MUST be distinct and non-overlapping.
- Separate compound words and ensure that each separated word has substantial meaning.
- Result MUST be returned by using tool `{_return_keywords.name}`.

## ProhibitedActions
- NO synonym substitution
- NO term expansion/interpretation
- NO grammatical correction
- NO adding external knowledge
"""
        )
        user_prompt = HumanMessage(f"""
Isolated task:
- Extract keywords from original paragraph, using the original language
{
            f"- Extract keywords from translated paragraph, using language {with_language}"
            if with_language is not None
            else ""
        }

## Paragraph
{paragraph}
""")

        await call_llm_as_function(
            self.llm,
            [system_prompt, user_prompt],
            _return_keywords,
        )

        if not original_keywords_container:
            raise AnalystError("LLM does not provide the query fragments")

        # translated keywords may be empty

        return (
            original_keywords_container,
            translated_keywords_container,
        )

    async def judge_document_useful(
        self,
        questions: list[str],
        document: Document,
    ) -> bool:
        @tool(return_direct=True)  # this is final call
        def _return_document_useful(
            may_be_useful: Annotated[
                bool,
                "May this document be useful? True is useful, False is not useful.",
            ],
        ) -> None:
            """Mark the document as useful or not useful."""

            document.may_be_useful = may_be_useful

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=False)
            + f"""
## Task
Based on the provided document, judge if this document MAY be useful for answering the question.
If you do not have enough knowledge or information to make a decision,
use words match rather than semantic match.
Your judgement MUST be returned by using tool `{_return_document_useful.name}`.

## Question
{"\n".join(questions)}
"""
        )
        user_prompt = HumanMessage(f"""
Judge if this document MAY be useful for answering the question.

## Document
{document.render_markdown(level=3, with_link=False, with_content=False)}
""")

        await call_llm_as_function(
            self.llm,
            [
                system_prompt,
                user_prompt,
            ],
            _return_document_useful,
        )

        if document.may_be_useful is None:
            raise AnalystError("LLM does not provide the judgement of this document")

        return document.may_be_useful

    @retryable(
        backoff.constant,
        AnalystError,
        _=__self_class,
        max_tries=lambda x: x.cfg.analyst_max_retries_of_taking_document_note,
        interval=3,
    )
    async def take_document_note(
        self,
        question: str,
        document: "Document",
    ) -> tuple[str, str]:
        if not document.content:
            raise AnalystError("document content is empty")

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=True)
            + f"""
## Task
Based on the provided document,
take a note from content, which can help answer the specified questions.
And try to bring up a expanded new possible question.

### Note
The first note must fully reproduce all relevant information from the original text,
including specific details, data points, and examples, without summarization or abstraction.
The output should be a comprehensive and detailed paragraph
that preserves the original context and content.
Please ensure that the first note MUST include all original text information useful
for answering the question.

If content is useless for the first note, You MUST record `Useless content` directly.

### Expanded New Possible Question

Adopting an expanded perspective in documentation,
check whether the original question contains ambiguities or vague points,
then bring up a expanded new question to address those uncertainties.

If you are sure that the original question is accurate,
or content is useless for bring up a new question,
You MUST record `No doubt` directly.

## Output Example

Output a two-elements list,
the first element is the first note,
the second element is the new possible question.

Useful document, all notes are valid:
```JSON
["xxx", "xxx"]
```

Useful document, but only the note is valid:
```JSON
["xxx", "No doubt"]
```

Useful document, but only the new question is valid:
```JSON
["Useless content", "xxx"]
```

Useless document, none is valid:
```JSON
["Useless content", "No doubt"]
```

You MUST output json WITHOUT markdown tags like '```'.

Bad:
```JSON
["", ""]
```

Good: ["", ""]

## Question
{question}
"""
        )
        user_prompt = HumanMessage(f"""
Take a note from document, then try to bring up a expanded new question.

## Document
{document.render_markdown(level=3, with_link=False, with_content=True)}
""")

        logger.info("Invoke LLM")
        logging_buf_size = 32
        logging_buf: list[str] = []

        result_container: list[str] = []
        async for chunk in call_llm_as_stream(
            self.llm,
            [system_prompt, user_prompt],
        ):
            content = chunk.text()
            if content:
                result_container.append(content)

            if len(logging_buf) >= logging_buf_size:
                logger.info("chunks: %s", "".join(logging_buf))
                logging_buf = []

            logging_content = assert_type(
                content
                or chunk.additional_kwargs.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    "reasoning_content",
                    None,
                )
                or "",
                str,
            )
            if logging_content:
                logging_buf.append(logging_content)

        if logging_buf:
            logger.info("chunks: %s", "".join(logging_buf))
            logging_buf = []

        logger.info("Finished LLM invocation")

        if not result_container:
            raise AnalystError("LLM does not provide the note")

        result = "".join(result_container)
        original_note, expanded_question = TypeAdapter(tuple[str, str]).validate_json(result)

        return (original_note, expanded_question)

    @retryable(
        backoff.constant,
        AnalystError,
        _=__self_class,
        max_tries=lambda self: self.cfg.analyst_max_retries_of_distilling_initial_note,
        interval=3,
    )
    async def distill_note(
        self,
        question: str,
        documents_notes: Collection[str],
    ) -> str:
        if not documents_notes:
            raise AnalystError("documents notes is empty")

        note_container: list[str] = []

        @tool(return_direct=True)  # this is final call
        def _return_distilled_note(note: Annotated[str, ""]) -> None:
            """Return a distilled note of all documents notes."""

            note_container.clear()
            note_container.append(note)

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=True)
            + f"""
## Task
Based on the provided documents notes, distill/summarize a note.
The note should integrate all relevant information from the original text,
which can help answer the specified questions and form a coherent paragraph.
Please ensure that the note includes all original text information useful
for answering the question.
Your note MUST be returned by using tool `{_return_distilled_note.name}`.

## Question
{question}
"""
        )

        document_note_paragraphs = [
            f"""
## DocumentNote:{i}
{document_note}
"""
            for i, document_note in enumerate(documents_notes)
        ]
        user_prompt = HumanMessage(f"""
Distill a note from the following documents notes to help answer the specified questions.

{"\n".join(document_note_paragraphs)}
""")

        await call_llm_as_function(
            self.llm,
            [
                system_prompt,
                user_prompt,
            ],
            _return_distilled_note,
        )

        if not note_container:
            raise AnalystError("LLM does not provide the note")

        return note_container[0]


def analyst_type_hint(cls: type[Analyst]) -> None: ...
