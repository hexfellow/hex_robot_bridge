#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-03-13
################################################################

import argparse
import io
import logging
import threading
import time
from collections import deque
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client
from PIL import Image
from hex_robo_utils import HexRate, ns_now


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape
        }
    if isinstance(obj, np.generic):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str
        }
    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        dtype = obj[b"dtype"]
        if isinstance(dtype, bytes):
            dtype = dtype.decode()
        shape = obj[b"shape"]
        if isinstance(shape, list):
            shape = tuple(shape)
        return np.frombuffer(obj[b"data"],
                             dtype=np.dtype(dtype)).reshape(shape).copy()
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        dtype = obj[b"dtype"]
        if isinstance(dtype, bytes):
            dtype = dtype.decode()
        return np.dtype(dtype).type(obj[b"data"])
    return obj


def pack_obs(obs: dict) -> bytes:
    return msgpack.packb(obs, default=_pack_array)


def unpack_response(raw: bytes) -> dict:
    return msgpack.unpackb(raw,
                           object_hook=_unpack_array,
                           strict_map_key=False)


def _encode_jpeg(arr: np.ndarray) -> bytes:
    """Encode (H, W, C) uint8 array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def hex_obs() -> dict:
    head_raw = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    left_raw = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    right_raw = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    head_jpg = _encode_jpeg(head_raw)
    left_jpg = _encode_jpeg(left_raw)
    right_jpg = _encode_jpeg(right_raw)
    return {
        "state": np.ones((14, )),
        "images": {
            "head": head_jpg,
            "left": left_jpg,
            "right": right_jpg,
        },
        "prompt": "do something",
    }


class HexOpenpiClient:

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int | None = 8000,
                 api_key: str | None = None):
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            uri += f":{port}"
        self._uri = uri
        self._api_key = api_key
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._metadata: dict | None = None
        self._obs_deque: deque[tuple[dict, int | None]] = deque()
        self._act_deque: deque[tuple[dict, int | None]] = deque()
        self._worker_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> dict:
        """连接 server 并启动工作线程。返回 server 的 metadata。"""
        if self._ws is not None:
            return self._metadata or {}
        logging.info("Connecting to %s ...", self._uri)
        headers = ({
            "Authorization": f"Api-Key {self._api_key}"
        } if self._api_key else None)
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )
        self._metadata = unpack_response(self._ws.recv())
        logging.info("Server metadata: %s", self._metadata)
        self._stop.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop,
                                               daemon=True)
        self._worker_thread.start()
        return self._metadata

    def _worker_loop(self) -> None:
        rate = HexRate(2e3)
        while not self._stop.is_set() and self._ws is not None:
            rate.sleep()
            if not self._obs_deque:
                continue

            obs_pair = self._obs_deque.popleft()
            if obs_pair is None:
                continue

            obs, ts_ns = obs_pair
            act = self.__infer(obs)
            self._act_deque.append((act, ts_ns))

    def __infer(self, obs: dict) -> dict:
        if self._ws is None:
            self.connect()
        self._ws.send(pack_obs(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        return unpack_response(response)

    def send_obs(self, obs: dict, ts_ns: int | None = None) -> None:
        """将 (obs, ts_ns) 打包放入 obs deque，非阻塞。"""
        if self._ws is None:
            raise RuntimeError("Not connected. Call start() first.")
        self._obs_deque.append((obs, ts_ns))

    def get_act(self, is_pop: bool = False) -> tuple[dict | None, int | None]:
        """
        is_pop=True: 从 act deque 弹出并返回 (act, ts_ns)
        is_pop=False: 返回最新一对（act deque 末尾），不弹出
        无结果时返回 (None, None)。
        """
        if not self._act_deque:
            return (None, None)
        if is_pop:
            return self._act_deque.popleft()
        return self._act_deque[-1]

    def wait_act(self, is_pop: bool = False) -> tuple[dict | None, int | None]:
        """等待 act deque 有结果，返回最新一对（act deque 末尾）。无则 (None, None)。"""
        while True:
            act, ts = self.get_act(is_pop=is_pop)
            if act is not None and ts is not None:
                return act, ts
            time.sleep(0.001)

    def close(self) -> None:
        """关闭连接并停止工作线程。"""
        self._stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._metadata = None
        self._obs_deque.clear()
        self._act_deque.clear()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="OpenPI WebSocket policy client")
    p.add_argument("--host", default="172.18.15.102")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--timing-file", default=None, help="Save timings to CSV")
    args = p.parse_args()

    client = HexOpenpiClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    try:
        client.start()
        t0 = ns_now()
        for _ in range(2):
            client.send_obs(hex_obs(), ts_ns=ns_now() - t0)
            _, ts = client.wait_act(is_pop=True)

        times = []
        for step in range(args.num_steps):
            client.send_obs(hex_obs(), ts_ns=ns_now() - t0)
            _, ts = client.wait_act(is_pop=True)
            times.append((ns_now() - ts - t0) / 1e6)
            print(f"Step {step + 1}: {times[-1]:.1f} ms")

        times = np.array(times)
        print("\n--- Client inference (ms) ---")
        print(
            f"  mean: {times.mean():.1f}, std: {times.std():.1f}, p50: {np.percentile(times, 50):.1f}, p99: {np.percentile(times, 99):.1f}"
        )
        if args.timing_file:
            np.savetxt(args.timing_file, times, fmt="%.2f", header="ms")
            logging.info("Wrote %s", args.timing_file)
    finally:
        client.close()


if __name__ == "__main__":
    main()
