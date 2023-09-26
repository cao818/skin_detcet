import opencv-python   #cv2==4.8.0
import numpy as np#numpy ==1.24.3
import streamlit as st#
import time

st.title("实时皮肤检测应用程序")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("无法打开摄像头。")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("无法获取摄像头帧。")
            break

        ycbcr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        lower_skin = np.array([0, 133, 77], dtype="uint8")
        upper_skin = np.array([255, 230, 255], dtype="uint8")
        skin_mask = cv2.inRange(ycbcr_frame, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        skin_detected = cv2.bitwise_and(frame, frame, mask=skin_mask)

        st.image(skin_detected, channels="BGR", caption="皮肤检测结果", use_column_width=True)

        stop_button_id = str(time.time())

        if st.button("停止按钮" + stop_button_id):
            break

        # 控制每秒只显示2张图像
        time.sleep(1.0)

cap.release()
cv2.destroyAllWindows()
