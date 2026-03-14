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
from utils import HexOpenpiClient

import cv2
import threading
import numpy as np
from collections import deque
from hex_robo_yoco import HexYocoE3Desktop
from hex_robo_utils import (
    HexRate,
    hex_ts_to_ns,
    time_interp,
    time_nearest,
    wait_client,
    dof_parser,
    HexDynUtil as DynUtil,
    ns_now,
)

PREPARE_FINISH_CNT = 1_000
PREPARE_INTERVAL = 1_000


class OpenpiBridgeClient(HexScriptClientBase):

    def __init__(self, config: dict):
        super().__init__()
        self.__client_dict, self.__record_dict, self.__util_dict, self.__policy_dict = self.__parse_cfg(
            config)

        self.init_client(self.__client_dict)
        self.init_utils(self.__util_dict)
        self.init_record(self.__record_dict)
        self.init_camera(self.__record_dict)
        self.init_teleop()
        self.init_policy(self.__policy_dict)

        self.start()

    def __parse_cfg(self, config: dict):
        client_dict, record_dict, util_dict, policy_dict = {}, {}, {}, {}
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
            # policy
            policy_dict["host"] = config["policy_cfg"]["host"]
            policy_dict["port"] = config["policy_cfg"]["port"]
            policy_dict["prompt"] = config["policy_cfg"]["prompt"]
            policy_dict["image_shape"] = config["policy_cfg"]["image_shape"]
            policy_dict["predict_ns"] = config["policy_cfg"]["predict_ns"]
        except KeyError as ke:
            missing_key = ke.args[0]
            raise ValueError(f"cfg is not valid, missing key: {missing_key}")
        return client_dict, record_dict, util_dict, policy_dict

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

    def init_policy(self, config: dict = {}):
        self._policy_client = HexOpenpiClient(
            host=config["host"],
            port=config["port"],
        )
        self._policy_client.start()
        self.__cam_deque = {}
        self.__cam_ts_deque = {}
        for cam_name in ["left", "right"]:
            if self.__cam_state["use_rgb"][cam_name]:
                self.__cam_deque[f"{cam_name}/rgb"] = deque(maxlen=30)
                self.__cam_ts_deque[f"{cam_name}/rgb"] = deque(maxlen=30)

        self.__state_deque = {}
        self.__state_ts_deque = {}
        for key in ["left", "right"]:
            self.__state_deque[key] = deque(maxlen=500)
            self.__state_ts_deque[key] = deque(maxlen=500)

        self.__act_result = None
        self.__act_ts = None
        self.__act_base_ts = np.arange(50, dtype=np.int64) * int(1e9 / 30.0)

    def camera_thread(self):
        rate = HexRate(60)
        send_interval = 3
        send_cnt = 0
        while self.is_working():
            rate.sleep()
            for cam_name in ["head", "left", "right"]:
                if self.__cam_state["use_rgb"][cam_name]:
                    rgb_hdr, rgb = self._clients["robot"].get_rgb(cam_name)
                    if rgb_hdr is not None:
                        if cam_name == "head":
                            send_cnt = (send_cnt + 1) % send_interval
                            if send_cnt % send_interval == 0:
                                self.__send_obs(rgb,
                                                hex_ts_to_ns(rgb_hdr["ts"]))
                        else:
                            self.__cam_deque[f"{cam_name}/rgb"].append(rgb)
                            self.__cam_ts_deque[f"{cam_name}/rgb"].append(
                                hex_ts_to_ns(rgb_hdr["ts"]))
                        self._record_util.append_data({
                            "ts_ns":
                            hex_ts_to_ns(rgb_hdr["ts"]),
                            f"{cam_name}/rgb":
                            rgb,
                        })

    def _init_loop(self):
        if self._last_state != self.WorkState.INIT:
            print("Start init.")
            self.__prepare_cnt = 0
            self.__arrived = {
                robot_name: False
                for robot_name in ["left", "right"]
            }

        self._last_state = self._curr_state
        if self._finish_event.is_set():
            self._curr_state = self.WorkState.FINISH
            return

        for robot_name in ["left", "right"]:
            has_state, arrived = self._state_ctrl_func(
                self.__util_dict["stable_state"], "robot", robot_name)
            if has_state:
                self.__arrived[robot_name] = arrived

        if all(self.__arrived.values()):
            if self.__prepare_cnt == 0:
                print("Prepare to start running")
            self.__prepare_cnt += 1
            if self.__prepare_cnt % PREPARE_INTERVAL == 0:
                print(f"Prepare: {self.__prepare_cnt}/{PREPARE_FINISH_CNT}")
            if self.__prepare_cnt >= PREPARE_FINISH_CNT:
                self._curr_state = self.WorkState.RUNNING

    def _running_loop(self):
        if self._last_state != self.WorkState.RUNNING:
            print("Start test.")

        self._last_state = self._curr_state
        if self._finish_event.is_set():
            self._curr_state = self.WorkState.FINISH
            return

        cur_tar = self.__get_cur_cmd(ns_now())
        for robot_name in ["left", "right"]:
            _, _ = self._state_ctrl_func(
                cur_tar[robot_name],
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
                self.__util_dict["stable_state"], "robot", robot_name)
            if has_state:
                self.__arrived[robot_name] = arrived

        if all(self.__arrived.values()):
            self._curr_state = self.WorkState.EXIT

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
            self.__state_deque[robot_name].append(states_buffer[:, 0].copy())
            self.__state_ts_deque[robot_name].append(
                hex_ts_to_ns(hdr_buffer["ts"]))
            states_hdr, states = hdr_buffer, states_buffer
            hdr_buffer, states_buffer = self._clients[role_name].get_states(
            ) if robot_name is None else self._clients[role_name].get_states(
                robot_name)
        return states_hdr, states

    def __send_obs(self, rgb: np.ndarray, ts_ns: int):
        obs = {
            "state": np.zeros((14, )),
            "images": {
                "head":
                self.__encode_jpeg(
                    cv2.resize(rgb, self.__policy_dict["image_shape"])),
                "left":
                None,
                "right":
                None,
            },
            "prompt": self.__policy_dict["prompt"],
        }

        # get cam data
        for cam_name in ["left", "right"]:
            if self.__cam_state["use_rgb"][cam_name]:
                ts_arr = np.array(list(self.__cam_ts_deque[f"{cam_name}/rgb"]))
                img_arr = np.array(list(self.__cam_deque[f"{cam_name}/rgb"]))
                img = cv2.resize(time_nearest(ts_ns, ts_arr, img_arr),
                                 self.__policy_dict["image_shape"])
                obs["images"][cam_name] = self.__encode_jpeg(img)

        # get state data
        for key in ["left", "right"]:
            ts_arr = np.array(list(self.__state_ts_deque[key]))
            state_arr = np.array(list(self.__state_deque[key]))
            if key == "left":
                obs["state"][:self._dofs["robot_sum"]] = time_interp(
                    ts_ns, ts_arr, state_arr)
            elif key == "right":
                obs["state"][self._dofs["robot_sum"]:] = time_interp(
                    ts_ns, ts_arr, state_arr)

        self._policy_client.send_obs(obs, ts_ns)

    def __encode_jpeg(self, img: np.ndarray):
        return cv2.imencode(".jpg", img)[1].tobytes()

    def __get_cur_cmd(self, cur_ts):
        act_result, act_ts = self._policy_client.get_act(is_pop=True)
        if (act_ts is not None) and (act_result is not None):
            # update act deque
            self.__act_ts = act_ts + self.__act_base_ts
            self.__act_result = act_result["actions"].copy()

        if (self.__act_result is None) or (self.__act_ts is None):
            return {"left": None, "right": None}

        tar_ts = cur_ts + self.__policy_dict["predict_ns"]
        cur_pos = time_interp(tar_ts, self.__act_ts, self.__act_result)[0]

        cur_tar = {
            "left":
            np.array([
                cur_pos[:self._dofs["robot_sum"]].copy(),
                np.zeros(self._dofs["robot_sum"]),
            ]).T,
            "right":
            np.array([
                cur_pos[self._dofs["robot_sum"]:].copy(),
                np.zeros(self._dofs["robot_sum"]),
            ]).T
        }

        return cur_tar


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
