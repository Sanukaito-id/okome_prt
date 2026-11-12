import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO 

# ページの設定
st.set_page_config(page_title="虫検出カメラ", page_icon="🪲")
st.title("🪲 虫検出 Webカメラアプリ")
st.write("下のカメラで虫を撮影すると、AIが自動で検出します。")

# 1. モデルの読み込み（初回のみロードしてキャッシュ）
@st.cache_resource
def load_model():
    # 自分の虫検出モデルがある場合は 'best.pt' に変更してください
    return YOLO('yolov8n.pt')

model = load_model()

# 2. Webブラウザのカメラ起動UI
camera_image = st.camera_input("カメラで撮影")

# 3. 写真が撮影されたら検出処理を実行
if camera_image is not None:
    # 撮影された画像をPIL形式で開く
    image = Image.open(camera_image)
    
    with st.spinner("AIが虫を検出中..."):
        # PIL画像をNumPy配列（OpenCV形式）に変換
        img_array = np.array(image)
        
        # YOLOで検出（確信度50%以上）
        results = model(img_array, conf=0.5)
        
        # 検出結果の枠線を描画した画像を取得
        res_plotted = results[0].plot()
        
        # 色空間をBGRからRGBに変換
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 結果の表示
        st.subheader("📸 検出結果")
        st.image(res_rgb, caption="検出画像", use_container_width=True)
        
        # 検出されたオブジェクト（虫）の数を表示
        num_detected = len(results[0].boxes)
        if num_detected > 0:
            st.success(f"{num_detected} 匹の対象を検出しました！")
        else:
            st.info("対象は検出されませんでした。")