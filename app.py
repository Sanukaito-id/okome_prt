import cv2
import numpy as np
import streamlit as st
import av
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh


# ==========================================
# ページ設定
# ==========================================

st.set_page_config(
    page_title="リアルタイムシャー芯カウンター",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ リアルタイムシャー芯カウンター")
st.write("カメラ映像から白い背景上のシャー芯を検出し、本数を推定します。")


# ==========================================
# サイドバー：設定
# ==========================================

st.sidebar.header("⚙️ システム設定")

st.sidebar.subheader("📷 カメラ設定")

st.sidebar.write("ブラウザからカメラを選択・許可してください。")

st.sidebar.markdown("---")

st.sidebar.subheader("🎨 色・明るさ設定")

threshold = st.sidebar.slider(
    "黒色判定のしきい値",
    min_value=0,
    max_value=255,
    value=100,
    help="数値を大きくすると、少し明るい灰色の芯も検出しやすくなります。"
)

st.sidebar.markdown("---")

st.sidebar.subheader("📐 検出精度設定")

min_area = st.sidebar.slider(
    "最小面積（ノイズ除去）",
    min_value=1,
    max_value=500,
    value=20
)

min_length = st.sidebar.slider(
    "最小の長さ（ピクセル）",
    min_value=1,
    max_value=200,
    value=10
)


# ==========================================
# シャー芯検出クラス
# ==========================================

class LeadProcessor(VideoProcessorBase):

    def __init__(self, threshold, min_area, min_length):

        self.threshold = threshold
        self.min_area = min_area
        self.min_length = min_length

        self.count = 0
        self.lock = threading.Lock()

    def recv(self, frame):

        # カメラ映像を取得
        img = frame.to_ndarray(format="bgr24")

        # グレースケール変換
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # ノイズを軽減
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # ==================================
        # 二値化処理
        # ==================================

        # 黒い部分を白、背景を黒に変換
        _, mask = cv2.threshold(
            gray,
            self.threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # ==================================
        # 輪郭・連結成分の検出
        # ==================================

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        result = img.copy()

        count = 0

        # 検出したシャー芯を囲む
        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area = stats[i, cv2.CC_STAT_AREA]

            # 面積と長さによるノイズ除去
            if area < self.min_area:
                continue

            if max(w, h) < self.min_length:
                continue

            count += 1

            # 検出部分を緑色の四角で囲む
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # 検出番号を表示
            cv2.putText(
                result,
                str(count),
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ==================================
        # 計測値を保存
        # ==================================

        with self.lock:
            self.count = count

        # ==================================
        # 表示用画像の作成
        # ==================================

        mask_color = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        # 画像にタイトルを追加
        cv2.putText(
            img, "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        cv2.putText(
            mask_color, "Binary Mask",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        cv2.putText(
            result, "Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        # 3画面を横に並べる
        combined = cv2.hconcat([
            img,
            mask_color,
            result
        ])

        # Streamlitに映像を返す
        return av.VideoFrame.from_ndarray(
            combined,
            format="bgr24"
        )


# ==========================================
# メイン画面：カメラ
# ==========================================

st.markdown("---")

st.subheader("📸 リアルタイムカメラ")

ctx = webrtc_streamer(

    key="lead-counter",

    video_processor_factory=lambda: LeadProcessor(
        threshold,
        min_area,
        min_length
    ),

    media_stream_constraints={
        "video": {
            "facingMode": {
                "ideal": "environment"
            }
        },
        "audio": False
    },

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    async_processing=True
)


# ==========================================
# 計測結果
# ==========================================

st.markdown("---")

st.subheader("📊 現在の計測結果")

metric_placeholder = st.empty()

# 0.5秒ごとに画面を更新
st_autorefresh(
    interval=500,
    key="count_refresh"
)

if ctx.video_processor:

    processor = ctx.video_processor

    with processor.lock:
        current_count = processor.count

    metric_placeholder.metric(
        label="検出したシャー芯の本数",
        value=f"{current_count} 本"
    )

else:

    st.info("カメラを起動すると計測を開始します。")


# ==========================================
# 使用上の注意
# ==========================================

st.markdown("---")

st.subheader("📌 使用上の注意")

st.write("""
- 背景には白い紙や白い板を使用してください。
- シャー芯同士が重ならないように配置してください。
- カメラはできるだけ真上から撮影してください。
- 周囲の照明を一定にすると検出精度が安定します。
- しきい値や最小面積を調整して検出精度を改善してください。
""")
