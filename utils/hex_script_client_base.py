#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-12-02
################################################################

import threading

from hex_robo_utils import (
    HexRate,
    HexTeleopUtilKeyboard,
)

from .hex_client_base import HexClientBase


class HexScriptClientBase(HexClientBase):

    def __init__(self):
        super().__init__()
        self._record_max = None

    def init_record(self, config: dict = {}):
        super().init_record(config)
        self._record_max = config["record_max"]

    def init_teleop(self, config: dict = {}):
        self._teleop_util = HexTeleopUtilKeyboard()
        self._teleop_thread = threading.Thread(
            target=self.teleop_thread,
            daemon=True,
        )

    def teleop_thread(self):
        print("### Script instructions: ###")
        print("Press 'q' to exit.")

        rate = HexRate(200.0)
        while self.is_working():
            rate.sleep()

            value = self._teleop_util.pop_value()
            if value is None:
                continue

            if value["key"] == "q":
                print("Got 'q'. Exit by keyboard.")
                self._save_record()
                print("Exit process finished.")
                self.exit()
