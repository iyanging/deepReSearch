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
from collections.abc import AsyncIterator, Awaitable
from typing import get_origin


def assert_type[T](obj: object, typ: type[T]) -> T:
    actual_type: type[T] = get_origin(typ) or typ

    if not isinstance(obj, actual_type):
        raise TypeError(f"Expected {typ}, got {type(obj)}")

    return obj


def assert_not_none[T](obj: T | None, msg: str | None = None) -> T:
    if obj is None:
        raise ValueError(msg or "Value cannot be None")

    return obj


async def parallel[T](
    aws: set[Awaitable[T]],
    *,
    concurrency: int,
) -> AsyncIterator[asyncio.Future[T]]:
    futures = {asyncio.ensure_future(_parallel_shield(aw)) for aw in aws}
    nows: set[asyncio.Future[asyncio.Future[T]]] = set()

    while nows or futures:
        while len(nows) < concurrency and futures:
            future = futures.pop()
            nows.add(future)

        histories, nows = await asyncio.wait(nows, return_when=asyncio.FIRST_COMPLETED)

        for history in histories:
            yield history.result()


async def _parallel_shield[T](aw: Awaitable[T]) -> asyncio.Future[T]:
    result: asyncio.Future[T] = asyncio.Future()

    try:
        result.set_result(await aw)
    except Exception as e:  # noqa: BLE001
        result.set_exception(e)

    return result
