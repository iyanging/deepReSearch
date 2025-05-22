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

import logging
from datetime import datetime
from typing import override

import colorlog

__all__ = ["root_logger", "setup_logging"]

root_logger = logging.getLogger("deep_research")


class StdColoredFormatter(colorlog.ColoredFormatter):
    @override
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        dt = datetime.fromtimestamp(record.created).astimezone()

        return dt.strftime(datefmt) if datefmt else dt.isoformat()


def setup_logging() -> None:
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(
        StdColoredFormatter(
            fmt=(
                "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(process)-5d %(taskName)-8s "
                "[%(funcName)8s] [%(filename)16s:%(lineno)-4d] "
                "%(log_color)s%(message)s%(reset)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S.%f%z",
            log_colors={
                logging.getLevelName(logging.DEBUG): "cyan",
                logging.getLevelName(logging.INFO): "green",
                logging.getLevelName(logging.WARNING): "yellow",
                logging.getLevelName(logging.ERROR): "red",
                logging.getLevelName(logging.CRITICAL): "bold_red",
            },
        )
    )

    for existed_handler in root_logger.handlers:
        if not isinstance(existed_handler, StdColoredFormatter):
            root_logger.removeHandler(existed_handler)

    root_logger.addHandler(handler)
