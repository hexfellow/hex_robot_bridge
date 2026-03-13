#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-03-10
################################################################

import os, sys, argparse, json, traceback

BASE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.append(BASE_DIR)
from utils import HexScriptClientBase

import threading
import numpy as np
from hex_robo_yoco import HexYocoE3Desktop
from hex_robo_utils import (
    HexRate,
    hex_ts_to_ns,
    wait_client,
    dof_parser,
    HexDynUtil as DynUtil,
)


class OpenpiBridgeClient(HexScriptClientBase):

    def __init__(self, config: dict):
        super().__init__()
        self.__client_dict, self.__record_dict, self.__util_dict = self.__parse_cfg(
            config)

        self.init_client(self.__client_dict)
        self.init_utils(self.__util_dict)
        self.init_record(self.__record_dict)
        self.init_camera(self.__record_dict)
        self.init_teleop()

        self.start()

    def __parse_cfg(self, config: dict):
        client_dict, record_dict, util_dict = {}, {}, {}
        try:
            # client
            client_dict["yoco"] = config["yoco"]
            client_dict["net"] = config["net"]
            # record
            record_dict["record_name"] = config["record_cfg"]["data_name"]
            record_dict["record_start"], record_dict["record_max"] = 0, 0
            record_dict["visual"] = config["record_cfg"]["visual"]
            # util
            util_dict["stable_state"] = np.array([
                config["stable_pos"],
                np.zeros(len(config["stable_pos"])),
            ]).T
            util_dict["model_path"] = config["model_path"]
            util_dict["mit_cfg"] = {
                "kp": np.array(config["mit_cfg"]["kp"]),
                "kd": np.array(config["mit_cfg"]["kd"]),
            }
        except KeyError as ke:
            missing_key = ke.args[0]
            raise ValueError(f"cfg is not valid, missing key: {missing_key}")
        return client_dict, record_dict, util_dict

    def init_client(self, config: dict = {}):
        self._clients = {
            "robot":
            HexYocoE3Desktop(
                yoco_config=config["yoco"],
                net_config=config["net"],
            )
        }
        for client in self._clients.values():
            wait_client(client)
        self._dofs = {"robot": dof_parser(self._clients["robot"].get_dofs())}
        self._dofs["robot_arm"] = self._dofs["robot"]["robot_arm"]
        self._dofs["robot_sum"] = self._dofs["robot"]["robot_sum"]

        self._ctrl_cfg = {
            "kp": self.__util_dict["mit_cfg"]["kp"][:self._dofs["robot_sum"]],
            "kd": self.__util_dict["mit_cfg"]["kd"][:self._dofs["robot_sum"]],
        }

    def init_utils(self, config: dict = {}):
        self._dyn_utils = {"robot": DynUtil(model_path=config["model_path"])}

    def init_camera(self, config: dict = {}):
        self.__cam_state = self._clients["robot"].get_cam_state()
        has_camera = False
        for cam_name in ["head", "left", "right"]:
            has_camera |= self.__cam_state["use_rgb"][
                cam_name] or self.__cam_state["use_depth"][cam_name]
        if has_camera:
            intri = self._clients["robot"].get_intri()
            intri_serial = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in intri.items()
            }
            json.dump(
                intri_serial,
                open(f"{config['record_name']}/intri.json", "w"),
            )
            self._camera_thread = threading.Thread(
                target=self.camera_thread,
                daemon=True,
            )

    def camera_thread(self):
        rate = HexRate(60)
        while self.is_working():
            rate.sleep()
            for cam_name in ["head", "left", "right"]:
                if self.__cam_state["use_rgb"][cam_name]:
                    rgb_hdr, rgb = self._clients["robot"].get_rgb(cam_name)
                    if rgb_hdr is not None:
                        self._record_util.append_data({
                            "ts_ns":
                            hex_ts_to_ns(rgb_hdr["ts"]),
                            f"{cam_name}/rgb":
                            rgb,
                        })
                if self.__cam_state["use_depth"][cam_name]:
                    depth_hdr, depth = self._clients["robot"].get_depth(
                        cam_name)
                    if depth_hdr is not None:
                        self._record_util.append_data({
                            "ts_ns":
                            hex_ts_to_ns(depth_hdr["ts"]),
                            f"{cam_name}/depth":
                            depth,
                        })

    def _init_loop(self):
        if self._last_state != self.WorkState.INIT:
            print("Start init.")

        self._last_state = self._curr_state
        if self._finish_event.is_set():
            self._curr_state = self.WorkState.FINISH
            return

        for robot_name in ["left", "right"]:
            _, _ = self._state_ctrl_func(
                self.__util_dict["stable_state"],
                "robot",
                robot_name,
            )

        self._curr_state = self.WorkState.RUNNING

    def _running_loop(self):
        if self._last_state != self.WorkState.RUNNING:
            print("Start test.")

        self._last_state = self._curr_state
        if self._finish_event.is_set():
            self._curr_state = self.WorkState.FINISH
            return

        for robot_name in ["left", "right"]:
            _, _ = self._state_ctrl_func(
                self.__util_dict["stable_state"],
                "robot",
                robot_name,
            )

    def _finish_loop(self):
        if self._last_state != self.WorkState.FINISH:
            print("Finish loop")
            self.__arrived = {
                robot_name: False
                for robot_name in ["left", "right"]
            }

        self._last_state = self._curr_state

        for robot_name in ["left", "right"]:
            has_state, arrived = self._state_ctrl_func(
                self.__util_dict["stable_state"],
                "robot",
                robot_name,
            )
            if has_state:
                self.__arrived[robot_name] = arrived

        if all(self.__arrived.values()):
            self._curr_state = self.WorkState.EXIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.cfg)

    client = None
    try:
        client = OpenpiBridgeClient(cfg)
        client.work_loop()
    except Exception:
        traceback.print_exc()
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    main()
