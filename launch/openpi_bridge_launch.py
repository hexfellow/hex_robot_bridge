#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-25
################################################################

import os
from hex_zmq_servers import HexLaunch, HexNodeConfig
from hex_zmq_servers import HEXARM_URDF_PATH_DICT
from hex_robo_yoco import HEX_YOCO_DRIVER_PATH_DICT

# YOCO config
CONTROL_HZ = 500
HEXARM_CFG = {"arm_type": "archer_y6", "gripper_type": "gp80"}
MIT_CFG = {
    "kp": [400.0, 400.0, 500.0, 200.0, 100.0, 100.0, 10.0],
    "kd": [5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 0.5],
}
HEXARM_DEVICE_IP = "172.18.22.245"
HEXARM_DEVICE_LEFT_PORT = 8439
HEXARM_DEVICE_RIGHT_PORT = 9439
YOCO = {
    "use_sim": True,
    "control_hz": CONTROL_HZ,
    "cam_type": ["rgb", "rgb", "rgb"],
    "srv_port": {
        "mujoco_port": 12345,
        "left_robot_port": 12346,
        "right_robot_port": 12347,
        "head_camera_port": 12348,
        "left_camera_port": 12349,
        "right_camera_port": 12350,
    },
    "params": {
        "mujoco": {
            "headless": True,
        },
        "robot": {
            "mit_kp": MIT_CFG["kp"],
            "mit_kd": MIT_CFG["kd"],
            "arm_type": HEXARM_CFG["arm_type"],
        },
        "rgb": {
            "resolution": [640, 480],
            "crop": [0, 640, 0, 480],
            "exposure": 70,
            "temperature": 0,
        },
        "realsense": {
            "resolution": [640, 480],
        },
        "berxel": {
            "exposure": 10000,
            "gain": 100,
        },
    },
    "device": {
        "left_robot": {
            "device_ip": HEXARM_DEVICE_IP,
            "device_port": HEXARM_DEVICE_LEFT_PORT,
        },
        "right_robot": {
            "device_ip": HEXARM_DEVICE_IP,
            "device_port": HEXARM_DEVICE_RIGHT_PORT,
        },
        "head_camera": {},
        "left_camera": {},
        "right_camera": {},
    },
}

# launch params
LAUNCH_PATH_DICT = {
    "driver": (HEX_YOCO_DRIVER_PATH_DICT["e3_desktop"], YOCO),
}
LAUNCH_PARAMS_DICT = {"driver": {}}

# node params
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = f"{SCRIPT_DIR}/.."
NODE_PARAMS_DICT = {
    # cli
    "openpi_bridge_cli": {
        "name": "openpi_bridge_cli",
        "node_path": f"{BASE_DIR}/client/openpi_bridge_cli.py",
        "cfg_path": f"{BASE_DIR}/config/openpi_bridge_cfg.json",
        "cfg": {
            "record_cfg": {
                "data_name": f"{BASE_DIR}/datasets/openpi_bridge",
                "visual": CONTROL_HZ < 1000,
            },
            "model_path":
            HEXARM_URDF_PATH_DICT[
                f'{HEXARM_CFG["arm_type"]}_{HEXARM_CFG["gripper_type"]}'],
            "mit_cfg": {
                "kp": YOCO["params"]["robot"]["mit_kp"],
                "kd": YOCO["params"]["robot"]["mit_kd"],
            },
            "yoco":
            YOCO,
            "net": {
                "mujoco_net": {
                    "port": YOCO["srv_port"]["mujoco_port"]
                },
                "left_robot_net": {
                    "port": YOCO["srv_port"]["left_robot_port"]
                },
                "right_robot_net": {
                    "port": YOCO["srv_port"]["right_robot_port"]
                },
                "head_camera_net": {
                    "port": YOCO["srv_port"]["head_camera_port"]
                },
                "left_camera_net": {
                    "port": YOCO["srv_port"]["left_camera_port"]
                },
                "right_camera_net": {
                    "port": YOCO["srv_port"]["right_camera_port"]
                },
            },
        },
    },
}


def get_node_cfgs(node_params_dict: dict = NODE_PARAMS_DICT,
                  launch_args: dict | None = None):
    launch_node_cfg = HexNodeConfig.get_launch_params_cfgs(
        launch_params_dict=LAUNCH_PARAMS_DICT,
        launch_default_params_dict=LAUNCH_PARAMS_DICT,
        launch_path_dict=LAUNCH_PATH_DICT,
    )
    node_default_params_config = HexNodeConfig(node_params_dict)
    node_default_params_config.add_cfgs(launch_node_cfg)
    node_default_params_dict = node_default_params_config.get_cfgs(
        use_list=False)
    return HexNodeConfig.parse_node_params_dict(
        node_params_dict,
        node_default_params_dict,
    )


def main():
    node_cfgs = get_node_cfgs()
    launch = HexLaunch(node_cfgs)
    launch.run()


if __name__ == '__main__':
    main()
