#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-12-02
################################################################

import os, threading
from abc import abstractmethod
from enum import auto, Enum

import numpy as np

from hex_robo_utils import (
    HexRate,
    hex_ts_to_ns,
    interp_joint,
    mit_cmd,
    HexFricUtil,
    HexTeleopBase,
    HexDataWriterBase,
    HexRerunWriter,
    HexHdf5Writer,
)


class HexClientBase:

    class WorkState(Enum):
        INIT = auto()
        RUNNING = auto()
        FINISH = auto()
        EXIT = auto()

    def __init__(self):
        self._work_event: threading.Event = threading.Event()
        self._curr_state = self.WorkState.INIT
        self._last_state = None
        self._finish_event = threading.Event()

        self._teleop_thread: threading.Thread = None
        self._teleop_util: HexTeleopBase = None

        self._camera_thread: threading.Thread = None

        self._record_flag: bool = False
        self._record_name: str = None
        self._record_idx: int = None
        self._record_util: HexDataWriterBase = None

        self._dofs: dict = None
        self._dyn_utils: dict = None
        self._fric_util: None | HexFricUtil = None
        self._clients: dict = None
        self._ctrl_cfg: dict = None

    @abstractmethod
    def init_client(self, config: dict = {}):
        raise NotImplementedError(
            "**init_client** should be implemented by the child class")

    @abstractmethod
    def init_utils(self, config: dict = {}):
        raise NotImplementedError(
            "**init_utils** should be implemented by the child class")

    def init_record(self, config: dict = {}):
        self._record_flag = False
        self._record_name = config["record_name"]
        self._record_idx = config["record_start"] - 1
        os.makedirs(self._record_name, exist_ok=True)
        if config["visual"]:
            print("Using rerun writer for record.")
            self._record_util = HexRerunWriter()
        else:
            print("Using hdf5 writer for record.")
            self._record_util = HexHdf5Writer()

    @abstractmethod
    def init_camera(self, config: dict = {}):
        raise NotImplementedError(
            "**init_camera** should be implemented by the child class")

    @abstractmethod
    def init_teleop(self, config: dict = {}):
        raise NotImplementedError(
            "**init_teleop** should be implemented by the child class")

    def is_working(self) -> bool:
        return self._work_event.is_set()

    def start(self):
        if self._teleop_util is not None:
            self._teleop_util.start()
        self._work_event.set()
        if self._teleop_thread is not None:
            self._teleop_thread.start()
        if self._camera_thread is not None:
            self._camera_thread.start()

    def exit(self):
        if self._curr_state == self.WorkState.EXIT:
            self._work_event.clear()
        else:
            self._finish_event.set()
        self._record_util.stop_record()

    def close(self):
        self.exit()
        if self._teleop_thread is not None:
            self._teleop_thread.join()
        if self._camera_thread is not None:
            self._camera_thread.join()
        if self._teleop_util is not None:
            self._teleop_util.close()

    @abstractmethod
    def teleop_thread(self):
        raise NotImplementedError(
            "**teleop_thread** should be implemented by the child class")

    @abstractmethod
    def camera_thread(self):
        raise NotImplementedError(
            "**camera_thread** should be implemented by the child class")

    def work_loop(self):
        rate = HexRate(2e3)

        while self.is_working():
            rate.sleep()

            if self._curr_state == self.WorkState.INIT:
                self._init_loop()
            elif self._curr_state == self.WorkState.RUNNING:
                self._running_loop()
            elif self._curr_state == self.WorkState.FINISH:
                self._finish_loop()
            elif self._curr_state == self.WorkState.EXIT:
                self.exit()

    @abstractmethod
    def _init_loop(self):
        raise NotImplementedError(
            "**_init_loop** should be implemented by the child class")

    @abstractmethod
    def _running_loop(self):
        raise NotImplementedError(
            "**_running_loop** should be implemented by the child class")

    @abstractmethod
    def _finish_loop(self):
        raise NotImplementedError(
            "**_finish_loop** should be implemented by the child class")

    def _start_record(self):
        if self._record_flag:
            print("Record is already started")
            return

        if self._curr_state != self.WorkState.RUNNING:
            print("You can only start record in running state")
            return

        self._record_idx += 1
        if self._record_idx < 0:
            raise ValueError(f"record_idx is less than 0: {self._record_idx}")

        self._record_flag = True
        self._record_util.start_record(self._record_name,
                                       f"{self._record_idx:04d}")

    def _save_record(self):
        if not self._record_flag:
            print("Record is not started")
            return

        self._record_flag = False
        self._record_util.stop_record()

    def _state_ctrl_func(
        self,
        tar_state: np.ndarray | None,
        role_name: str,
        robot_name: str | None = None,
        err_threshold: float = 0.03,
        arrive_threshold: float = 0.06,
    ) -> tuple[bool, bool]:
        has_state, arrived = False, False

        states_hdr, states = self._get_states(role_name, robot_name)
        if states_hdr is not None:
            has_state = True

            # set cmds
            cmds, arrived = self._state_prepare_cmds(
                states,
                tar_state,
                role_name,
                err_threshold,
                arrive_threshold,
            )
            self._set_cmds(cmds, role_name, robot_name)
        return has_state, arrived

    def _state_prepare_cmds(
        self,
        cur_state,
        tar_state,
        role_name,
        err_threshold: float = 0.03,
        arrive_threshold: float = 0.06,
    ):
        zeros_np = np.zeros(self._dofs["robot_sum"])
        cur_q = cur_state[:, 0]
        cur_dq = cur_state[:, 1]
        tau_comp = self._tau_comp(cur_q, cur_dq, role_name)

        # calc cmds
        tar_pos = zeros_np
        tar_vel = zeros_np
        tar_kp = zeros_np
        tar_kd = zeros_np
        arrived = False
        if tar_state is not None:
            tar_pos = tar_state[:, 0].copy()
            tar_pos, _, arrived = self._mid_pos(
                cur_q,
                tar_pos,
                err_threshold=err_threshold,
                arrive_threshold=arrive_threshold,
            )
            tar_vel = tar_state[:, 1].copy()
            tar_kp = self._ctrl_cfg["kp"]
            tar_kd = self._ctrl_cfg["kd"]
        cmds = mit_cmd(
            pos=tar_pos,
            vel=tar_vel,
            tau=tau_comp,
            kp=tar_kp,
            kd=tar_kd,
        )
        return cmds, arrived

    def _get_states(self, role_name, robot_name=None):
        states_hdr, states = None, None
        hdr_buffer, states_buffer = self._clients[role_name].get_states(
        ) if robot_name is None else self._clients[role_name].get_states(
            robot_name)
        while hdr_buffer is not None:
            states_buffer = np.concat([
                states_buffer,
                np.zeros((self._dofs["robot_sum"] -
                          self._dofs[role_name]["robot_sum"], 3)),
            ])
            prefix = f"{role_name}_state" if robot_name is None else f"{role_name}_{robot_name}_state"
            self._record_util.append_data({
                "ts_ns":
                hex_ts_to_ns(hdr_buffer["ts"]),
                f"{prefix}/pos":
                states_buffer[:, 0],
                f"{prefix}/vel":
                states_buffer[:, 1],
            })
            states_hdr, states = hdr_buffer, states_buffer
            hdr_buffer, states_buffer = self._clients[role_name].get_states(
            ) if robot_name is None else self._clients[role_name].get_states(
                robot_name)
        return states_hdr, states

    def _tau_comp(self, cur_q, cur_dq, role_name):
        tau_comp = np.zeros(self._dofs["robot_sum"])
        tau_comp[:self.
                 _dofs["robot_arm"]] = self._dyn_utils[role_name].compensation(
                     cur_q[:self._dofs["robot_arm"]],
                     cur_dq[:self._dofs["robot_arm"]])
        if self._fric_util is not None:
            tau_comp += self._fric_util(cur_dq)
        return tau_comp

    def _mid_pos(
        self,
        cur_q,
        tar_pos,
        err_threshold: float = 0.1,
        arrive_threshold: float = 0.2,
    ):
        mid_pos = tar_pos.copy()
        mid_pos[:self._dofs["robot_arm"]], interped, arrived = interp_joint(
            cur_q[:self._dofs["robot_arm"]],
            mid_pos[:self._dofs["robot_arm"]],
            err_limit=err_threshold,
            arrive_limit=arrive_threshold,
        )
        return mid_pos, interped, arrived

    def _set_cmds(self, cmds, role_name, robot_name=None):
        cmds = cmds[:self._dofs[role_name]["robot_sum"], ...]
        if robot_name is None:
            _ = self._clients[role_name].set_cmds(cmds)
        else:
            _ = self._clients[role_name].set_cmds(cmds, robot_name)
