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
import queue
import threading
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client
from PIL import Image


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
    """
    整合的 OpenPI WebSocket 客户端：
    - start(): 连接 server 并启动接收线程
    - send_obs(obs, ts_ns): 发送 obs 与时间戳（非阻塞）
    - get_act(): 返回最新的 (act, ts_ns)，无结果时返回 (None, None)
    """

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
        self._pending_ts_ns: queue.Queue[int | None] = queue.Queue()
        self._latest_act: dict | None = None
        self._latest_ts_ns: int | None = None
        self._lock = threading.Lock()
        self._recv_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> dict:
        """连接 server 并启动接收线程。返回 server 的 metadata。"""
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
        self._recv_thread = threading.Thread(target=self._recv_loop,
                                             daemon=True)
        self._recv_thread.start()
        return self._metadata

    def _recv_loop(self) -> None:
        while not self._stop.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv()
            except Exception as e:
                if not self._stop.is_set():
                    logging.warning("Recv thread error: %s", e)
                break
            try:
                ts_ns = self._pending_ts_ns.get_nowait()
            except queue.Empty:
                ts_ns = None
            if isinstance(raw, str):
                logging.warning("Server error: %s", raw)
                continue
            try:
                act = unpack_response(raw)
            except Exception as e:
                logging.warning("Unpack response error: %s", e)
                continue
            with self._lock:
                self._latest_act = act
                self._latest_ts_ns = ts_ns

    def send_obs(self, obs: dict, ts_ns: int | None = None) -> None:
        """发送 obs 与时间戳（非阻塞）。接收线程会将结果与 ts_ns 放入队列。"""
        if self._ws is None:
            raise RuntimeError("Not connected. Call start() first.")
        self._pending_ts_ns.put(ts_ns)
        self._ws.send(pack_obs(obs))

    def get_act(self) -> tuple[dict | None, int | None]:
        """返回最新的 (act, ts_ns)，没有则返回 (None, None)。"""
        with self._lock:
            act = self._latest_act
            ts_ns = self._latest_ts_ns
        return (act, ts_ns)

    def close(self) -> None:
        """关闭连接并停止接收线程。"""
        self._stop.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._metadata = None
        with self._lock:
            self._latest_act = None
            self._latest_ts_ns = None


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="OpenPI WebSocket policy client")
    p.add_argument("--host", default="127.0.0.1")
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
        for _ in range(2):
            client.send_obs(hex_obs(), ts_ns=None)
            time.sleep(0.05)
            act, ts = client.get_act()
            logging.info("Warmup: act=%s ts=%s", type(act), ts)
        times = []
        for step in range(args.num_steps):
            t0 = time.perf_counter()
            client.send_obs(hex_obs(), ts_ns=step)
            time.sleep(0.001)
            act, ts = client.get_act()
            while act is None and ts is None:
                time.sleep(0.001)
                act, ts = client.get_act()
            times.append((time.perf_counter() - t0) * 1000)
            if (step + 1) % 5 == 0 or step == 0:
                logging.info("Step %d: %.1f ms", step + 1, times[-1])
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
