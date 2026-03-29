#!/usr/bin/env python3
"""Poll a fixed URL for JPEG frames, detect hands, and display results."""

from __future__ import annotations

import time
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
import requests

# 修改为你的固定后端地址
JPEG_URL = "http://127.0.0.1:8000/frame.jpg"
REQUEST_TIMEOUT = 5  # 秒
RETRY_DELAY = 0.2  # 出错时重试间隔（秒）
WINDOW_NAME = "Camera Stream (press q to quit)"

# MediaPipe Hands 配置
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 手势触发配置
PHOTO_HINT_TEXT = "Snap!"
PHOTO_HINT_DURATION = 0.5  # 秒

GestureName = Literal["rock", "scissors", "other", "none"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def fetch_jpeg(url: str, timeout: float) -> np.ndarray | None:
    """Fetch one JPEG image and decode it into an OpenCV BGR frame."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    data = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame


def is_finger_extended(
    landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark],
    tip_idx: int,
    pip_idx: int,
) -> bool:
    """Heuristic for non-thumb fingers in image coordinates (y smaller means higher)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def classify_gesture(
    landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark],
) -> GestureName:
    """Classify simple hand gestures: rock / scissors / other."""
    index_up = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_up = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_up = is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_up = is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)

    # 简化处理：不强依赖拇指状态，重点识别“石头->剪刀”
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "rock"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "scissors"
    return "other"


def process_hand_and_draw(
    frame: np.ndarray,
    hands: mp_hands.Hands,
) -> tuple[np.ndarray, GestureName]:
    """Run MediaPipe Hands, draw skeleton, and return first hand gesture class."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_gesture: GestureName = "none"
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            if idx == 0:
                current_gesture = classify_gesture(hand_landmarks.landmark)

    return frame, current_gesture


def draw_photo_hint(frame: np.ndarray, text: str) -> None:
    """Draw a centered photo hint on the frame."""
    h, w = frame.shape[:2]
    org = (int(w * 0.35), int(h * 0.12))
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_gesture: GestureName = "none"
    photo_hint_until = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            try:
                frame = fetch_jpeg(JPEG_URL, REQUEST_TIMEOUT)
                if frame is None:
                    print("收到的数据不是有效 JPEG，准备重试…")
                    time.sleep(RETRY_DELAY)
                    continue

                frame, current_gesture = process_hand_and_draw(frame, hands)

                # 相邻两帧：石头 -> 剪刀，触发拍照提示 0.5 秒
                now = time.monotonic()
                if last_gesture == "rock" and current_gesture == "scissors":
                    photo_hint_until = now + PHOTO_HINT_DURATION

                if now < photo_hint_until:
                    draw_photo_hint(frame, PHOTO_HINT_TEXT)

                last_gesture = current_gesture

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
