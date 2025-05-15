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
from typing import Protocol

from pydantic import NonNegativeInt, PositiveInt

from deep_research.resources.document import Document


class Retriever[QueryT](Protocol):
    async def retrieve(
        self,
        query: QueryT,
        *,
        offset: NonNegativeInt,
        limit: PositiveInt,
    ) -> Collection[Document]: ...


class RecoverableRetrieverError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnrecoverableRetrieverError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
