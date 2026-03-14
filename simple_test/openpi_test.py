#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-03-10
################################################################

import argparse, io, os, sys
import numpy as np
from PIL import Image
from hex_robo_utils import ns_now

BASE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.append(BASE_DIR)
from utils import HexOpenpiClient


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


def main() -> None:
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
            act, ts = client.wait_act(is_pop=True)
            print(act["actions"].shape)
            print(ts)
            print("-" * 100)

        times = []
        for step in range(args.num_steps):
            client.send_obs(hex_obs(), ts_ns=ns_now() - t0)
            act, ts = client.wait_act(is_pop=True)
            times.append((ns_now() - ts - t0) / 1e6)
            print(f"Step {step + 1}: {times[-1]:.1f} ms")

        times = np.array(times)
        print("\n--- Client inference (ms) ---")
        print(
            f"  mean: {times.mean():.1f}, std: {times.std():.1f}, p50: {np.percentile(times, 50):.1f}, p99: {np.percentile(times, 99):.1f}"
        )
        if args.timing_file:
            np.savetxt(args.timing_file, times, fmt="%.2f", header="ms")
            print(f"Wrote {args.timing_file}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
