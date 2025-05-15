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

import colorlog

__all__ = ["root_logger", "setup_logging"]

root_logger = logging.getLogger("deep_research")


def setup_logging() -> None:
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt=(
                "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(process)-5d "
                "[%(funcName)8s] [%(filename)16s:%(lineno)-4d] "
                "%(log_color)s%(message)s%(reset)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            log_colors={
                logging.getLevelName(logging.DEBUG): "cyan",
                logging.getLevelName(logging.INFO): "green",
                logging.getLevelName(logging.WARNING): "yellow",
                logging.getLevelName(logging.ERROR): "red",
                logging.getLevelName(logging.CRITICAL): "bold_red",
            },
        )
    )

    root_logger.addHandler(handler)
