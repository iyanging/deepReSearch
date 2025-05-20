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

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Hashable,
    Sequence,
    Set,
)
from functools import wraps
from typing import Any, Concatenate, get_origin

import backoff

from deep_research.logger import root_logger


def assert_type[T](obj: object, typ: type[T]) -> T:
    actual_type: type[T] = get_origin(typ) or typ

    if not isinstance(obj, actual_type):
        raise TypeError(f"Expected {typ}, got {type(obj)}")

    return obj


def assert_not_none[T](obj: T | None, msg: str | None = None) -> T:
    if obj is None:
        raise ValueError(msg or "Value cannot be None")

    return obj


def count_without[T: Hashable](raw: Collection[T], without: set[T]) -> int:
    c = 0
    for r in raw:
        if r not in without:
            c += 1

    return c


async def parallel[T](
    aws: Set[Awaitable[T]],
    *,
    concurrency: int,
    ignore_exceptions: bool,
) -> AsyncIterator[asyncio.Future[T]]:
    futures = {
        asyncio.ensure_future(
            _parallel_shield(
                aw,
                ignore_exceptions=ignore_exceptions,
            )
        )
        for aw in aws
    }
    nows: set[asyncio.Future[asyncio.Future[T] | None]] = set()

    while nows or futures:
        while len(nows) < concurrency and futures:
            future = futures.pop()
            nows.add(future)

        histories, nows = await asyncio.wait(nows, return_when=asyncio.FIRST_COMPLETED)

        for history in histories:
            fact = history.result()
            if fact is not None:
                yield fact


_parallel_logger = root_logger.getChild("parallel")


async def _parallel_shield[T](
    aw: Awaitable[T],
    *,
    ignore_exceptions: bool,
) -> asyncio.Future[T] | None:
    result: asyncio.Future[T] | None = asyncio.Future()

    try:
        result.set_result(await aw)
    except Exception as e:  # noqa: BLE001
        if ignore_exceptions:
            _parallel_logger.warning(
                "awaitable %s raises an exception, ignoring it due to config",
                aw,
                exc_info=e,
            )
            result = None

        else:
            result.set_exception(e)

    return result


def retryable[S, **P, R](
    wait_gen: Callable[..., Generator[float]],
    exception: type[Exception] | Sequence[type[Exception]],
    *,
    _: Callable[[S], None] | None = None,
    max_tries: int | Callable[[S], int],
    **wait_gen_kwargs: Any,
) -> Callable[[Callable[Concatenate[S, P], R]], Callable[Concatenate[S, P], R]]:
    def wrapper(func: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
        @wraps(func)
        def wrapped(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
            return backoff.on_exception(
                wait_gen,
                exception,
                jitter=None,
                max_tries=(max_tries if not callable(max_tries) else max_tries(self)),
                **wait_gen_kwargs,
            )(func)(self, *args, **kwargs)

        return wrapped

    return wrapper
