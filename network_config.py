"""Network configuration for camera polling."""

# 修改为你的固定后端地址
JPEG_URL = "http://127.0.0.1:8000/frame.jpg"
REQUEST_TIMEOUT = 5  # 秒
RETRY_DELAY = 0.2  # 出错时重试间隔（秒）
