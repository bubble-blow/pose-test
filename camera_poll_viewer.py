#!/usr/bin/env python3
"""Poll a fixed URL for JPEG frames and display them continuously."""

from __future__ import annotations

import time

import cv2
import numpy as np
import requests

# 修改为你的固定后端地址
JPEG_URL = "http://127.0.0.1:8000/frame.jpg"
REQUEST_TIMEOUT = 5  # 秒
RETRY_DELAY = 0.2  # 出错时重试间隔（秒）
WINDOW_NAME = "Camera Stream (press q to quit)"


def fetch_jpeg(url: str, timeout: float) -> np.ndarray | None:
    """Fetch one JPEG image and decode it into an OpenCV BGR frame."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    data = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame


def main() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        try:
            frame = fetch_jpeg(JPEG_URL, REQUEST_TIMEOUT)
            if frame is None:
                print("收到的数据不是有效 JPEG，准备重试…")
                time.sleep(RETRY_DELAY)
                continue

            height, width = frame.shape[:2]
            cv2.resizeWindow(WINDOW_NAME, width, height)
            cv2.imshow(WINDOW_NAME, frame)

        except requests.RequestException as exc:
            print(f"HTTP 请求失败: {exc}")
            time.sleep(RETRY_DELAY)
        except Exception as exc:  # 解码或显示相关未知错误
            print(f"处理图像失败: {exc}")
            time.sleep(RETRY_DELAY)

        # 每次接收完成后都会回到循环顶部，再次发送 HTTP 请求
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q 或 ESC 退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
