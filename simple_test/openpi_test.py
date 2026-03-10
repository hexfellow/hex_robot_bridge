#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-03-10
################################################################

import argparse
import io
import logging
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


class StandalonePolicyClient:

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int | None = 8000,
                 api_key: str | None = None):
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            uri += f":{port}"
        self._uri = uri
        self._api_key = api_key
        self._ws = None
        self._metadata = None

    def connect(self) -> dict:
        logging.info("Connecting to %s ...", self._uri)
        headers = {
            "Authorization": f"Api-Key {self._api_key}"
        } if self._api_key else None
        self._ws = websockets.sync.client.connect(self._uri,
                                                  compression=None,
                                                  max_size=None,
                                                  additional_headers=headers)
        self._metadata = unpack_response(self._ws.recv())
        logging.info("Server metadata: %s", self._metadata)
        return self._metadata

    def infer(self, obs: dict) -> dict:
        if self._ws is None:
            self.connect()
        self._ws.send(pack_obs(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        return unpack_response(response)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Standalone HEX WebSocket policy client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--timing-file", default=None, help="Save timings to CSV")
    args = p.parse_args()

    client = StandalonePolicyClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    try:
        client.connect()
        for _ in range(2):
            client.infer(hex_obs())
        times = []
        for step in range(args.num_steps):
            t0 = time.perf_counter()
            client.infer(hex_obs())
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
