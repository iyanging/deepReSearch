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

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Final, Literal, NoReturn, overload, override

from deep_research.utils import assert_not_none


class ContextCollaborator(ABC):
    @abstractmethod
    async def aenter(self, new_context: "Context") -> "ContextCollaborator":
        raise NotImplementedError

    @abstractmethod
    async def aexit(
        self,
        context: "Context",
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    async def state(self, status: str) -> None:  # noqa: ARG002, PLR6301
        return

    async def info_reasoning(self, delta: str | Literal[0]) -> None:  # noqa: ARG002, PLR6301
        return

    async def info_output(self, delta: str | Literal[0]) -> None:  # noqa: ARG002, PLR6301
        return

    async def info(self, message: str) -> None:  # noqa: ARG002, PLR6301
        return


class Context(AbstractAsyncContextManager["Context"]):
    run_id: str
    name: Final[str]
    stack: "Final[list[Context]]"

    in_ctx_collaborators: list[ContextCollaborator]
    out_ctx_collaborators: list[ContextCollaborator]

    @overload
    def __init__(
        self,
        name: str,
        *,
        parent: None = None,
        run_id: str,
        collaborators: list[ContextCollaborator] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        parent: "Context",
        run_id: None = None,
        collaborators: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        parent: "Context",
        run_id: str | None = None,
        collaborators: list[ContextCollaborator],
    ) -> NoReturn: ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        parent: "Context",
        run_id: str,
        collaborators: list[ContextCollaborator] | None = None,
    ) -> NoReturn: ...

    def __init__(
        self,
        name: str,
        *,
        parent: "Context | None" = None,
        run_id: str | None = None,
        collaborators: list[ContextCollaborator] | None = None,
    ) -> None:
        super().__init__()

        self.name = name
        self.in_ctx_collaborators = []

        if parent is not None:
            self.run_id = parent.run_id
            self.stack = [parent, *parent.stack]
            self.out_ctx_collaborators = [*parent.in_ctx_collaborators]

        else:
            self.run_id = assert_not_none(run_id)
            self.stack = []
            # collaborators should not access Context.collaborators during the initialization,
            # otherwise it will raise AttributeError
            self.out_ctx_collaborators = [*collaborators] if collaborators is not None else []

    @override
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"stack={[ctx.name for ctx in self.stack]})"
        )

    def scope(self, name: str) -> "Context":
        return Context(name, parent=self)

    async def enter[R](
        self,
        name: str,
        func: Callable[["Context"], Awaitable[R]],
    ) -> R:
        async with self.scope(name) as sub:
            return await func(sub)

    @override
    async def __aenter__(self) -> "Context":
        self.in_ctx_collaborators = [await co.aenter(self) for co in self.out_ctx_collaborators]

        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        for co in self.in_ctx_collaborators:
            await co.aexit(
                self,
                exc_type,
                exc_value,
                traceback,
            )

        self.in_ctx_collaborators = []

        return False  # exception should go up.

    async def state(self, status: str) -> None:
        root = self.stack[-1] if self.stack else None
        if root is not None:
            return await root.state(status)

        else:
            for co in self.in_ctx_collaborators:
                await co.state(status)

    async def info_output(self, delta: str | Literal[0]) -> None:
        for co in self.in_ctx_collaborators:
            await co.info_output(delta)

    async def info_reasoning(self, delta: str | Literal[0]) -> None:
        for co in self.in_ctx_collaborators:
            await co.info_reasoning(delta)

    async def info(self, message: str) -> None:
        for co in self.in_ctx_collaborators:
            await co.info(message)
