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

import backoff
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, HttpUrl, NonNegativeFloat, NonNegativeInt, SecretStr

from deep_research.context import Context
from deep_research.llm import (
    LlmApiProvider,
    LlmModel,
    LlmTryThinking,
    build_llm,
    call_llm_as_function,
    make_meta_instructions,
)
from deep_research.logger import root_logger
from deep_research.resources.document import Document
from deep_research.utils import retryable

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
        ctx: Context,
        paragraph: str,
        with_language: str | None,
    ) -> tuple[list[str], list[str]]:
        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=False)
            + """
# Task

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

## ProhibitedActions

- NO synonym substitution
- NO term expansion/interpretation
- NO grammatical correction
- NO adding external knowledge
"""
        )
        user_prompt = HumanMessage(f"""
# Isolated task

- Extract keywords from original paragraph, using the original language.
{
            f"- Extract keywords from translated paragraph, using language {with_language}"
            if with_language is not None
            else ""
        }

# Output Example

Output a two-elements list,
the first element is the list of extracted keywords, from original paragraph,
the second element is the list of extracted keywords, from translated paragraph,
if you are asked to do translation.

Did translation:
```JSON
[["a", "b", "c"], ["d", "e"]]
```

No translation:
```JSON
[["a", "b", "c"], []]
```

You MUST output json WITHOUT markdown tags like '```'.

Bad:
```JSON
[["a", "b", "c"], ["d", "e"]]
```

Good: [["a", "b", "c"], ["d", "e"]]

# Paragraph
{paragraph}
""")

        original_keywords, translated_keywords = await call_llm_as_function(
            ctx,
            self.llm,
            [system_prompt, user_prompt],
            tuple[list[str], list[str]],
        )

        if not original_keywords:
            raise AnalystError("LLM does not provide the query fragments")

        # translated keywords may be empty

        return (
            original_keywords,
            translated_keywords,
        )

    async def judge_document_useful(
        self,
        ctx: Context,
        questions: list[str],
        document: Document,
    ) -> bool:
        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=False)
            + f"""
# Task

Based on the provided document, judge if this document MAY be useful for answering the question.
If you do not have enough knowledge or information to make a decision,
use words match rather than semantic match.

# Output example

document is useful:
```JSON
true
```

document is useless:
```JSON
false
```

You MUST output json WITHOUT markdown tags like '```'.

Bad:
```JSON
true
```

Good: true

# Question
{"\n".join(questions)}
"""
        )
        user_prompt = HumanMessage(f"""
Judge if this document MAY be useful for answering the question.

# Document
{document.render_markdown(level=3, with_link=False, with_content=False)}
""")

        may_be_useful = await call_llm_as_function(
            ctx,
            self.llm,
            [
                system_prompt,
                user_prompt,
            ],
            bool,
        )

        document.may_be_useful = may_be_useful

        return may_be_useful

    @retryable(
        backoff.constant,
        AnalystError,
        _=__self_class,
        max_tries=lambda x: x.cfg.analyst_max_retries_of_taking_document_note,
        interval=3,
    )
    async def take_document_note(
        self,
        ctx: Context,
        question: str,
        document: "Document",
    ) -> tuple[str | None, str | None]:
        if not document.content:
            raise AnalystError("document content is empty")

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=True)
            + f"""
# Task

- Based on the provided document, take a note from content,
    which can help answer the specified questions.
- Try to bring up a expanded new possible question.

## Note

- The first note must fully reproduce all relevant information from the original text,
    including specific details, data points, and examples, without summarization or abstraction.
- The output should be a comprehensive and detailed paragraph
    that preserves the original context and content.
- Please ensure that the first note MUST include all original text information useful
    for answering the question.
- Take note in the SAME LANGUAGE as the original question.
- Use multiple paragraphs to separate different ideas or points.
- No markdown formatting.
- If content is useless for the note, You MUST record `null` directly.

## Expanded New Possible Question

Adopting an expanded perspective in documentation,
check whether the original question contains ambiguities or vague points,
then bring up a expanded new question to address those uncertainties.

If you believe the original question is accurate and should not bring up a new question,
You MUST record `null` directly.

# Output Example

Output a two-elements list,
the first element is the note,
the second element is the expanded new possible question.

Useful document, the note and the expanded question are valid:
```JSON
["xxx", "xxx"]
```

Useful document, but only the note is valid:
```JSON
["xxx", null]
```

Useful document, but only the expanded question is valid:
```JSON
[null, "xxx"]
```

Useless document, none is valid:
```JSON
[null, null]
```

You MUST output json WITHOUT markdown tags like '```'.

Bad:
```JSON
["xxx", "xxx"]
```

Good: ["xxx", "xxx"]

# Question
{question}
"""
        )
        user_prompt = HumanMessage(f"""
Take a note from document, then try to bring up a expanded new question.

# Document
{document.render_markdown(level=2, with_link=False, with_content=True)}
""")

        original_note, expanded_question = await call_llm_as_function(
            ctx,
            self.llm,
            [system_prompt, user_prompt],
            tuple[str | None, str | None],
            do_ctx_info_reasoning="as_output",
        )

        await ctx.info(f"original note: {original_note}")
        await ctx.info(f"expanded question: {expanded_question}")

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
        ctx: Context,
        question: str,
        documents_notes: Collection[str],
    ) -> str:
        if not documents_notes:
            raise AnalystError("documents notes is empty")

        system_prompt = SystemMessage(
            self._make_system_prompt(default_thinking=True)
            + f"""
# Task

Based on the provided documents notes, distill/summarize a note.
The note should integrate all relevant information from the original documents notes,
which can help answer the specified questions and form a coherent paragraph.

The final note should write in the SAME LANGUAGE as the original question.

# Question
{question}
"""
        )

        document_note_paragraphs = [
            f"""
# DocumentNote:{i}
{document_note}
"""
            for i, document_note in enumerate(documents_notes)
        ]
        user_prompt = HumanMessage(f"""
Distill a note from the following documents notes to help answer the specified questions.

{"\n".join(document_note_paragraphs)}
""")

        note = await call_llm_as_function(
            ctx,
            self.llm,
            [
                system_prompt,
                user_prompt,
            ],
            str,
            do_ctx_info_reasoning="as_reasoning",
            do_ctx_info_output="as_output",
        )

        return note
