import cv2
import time
from ultralytics import YOLO
from config.saha_config import (
    KAMERA1_KONUM, KAMERA2_KONUM,
    KAMERA1_RTSP, KAMERA2_RTSP,
    SAHA_GENISLIK, SAHA_UZUNLUK
)
from utils.geometry import mesafe_3d

# YOLOv8x modeli (en yüksek doğruluk)
model = YOLO("yolov8x.pt")

# RTSP kameraları bağla
caps = [
    cv2.VideoCapture(KAMERA1_RTSP),
    cv2.VideoCapture(KAMERA2_RTSP)
]

KAMERA_KONUM = [KAMERA1_KONUM, KAMERA2_KONUM]
aktif_index = 0

# Saniyede 5 tespit yapılacak → her 200ms
DETECTION_INTERVAL = 0.2  # saniye
last_detection_time = 0

def top_konumu_3b(box, frame_shape):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    x_gercek = cx / frame_shape[1] * SAHA_GENISLIK
    y_gercek = cy / frame_shape[0] * SAHA_UZUNLUK
    z_gercek = 0
    return (x_gercek, y_gercek, z_gercek)

while True:
    frameler = []
    top_konumlari = []

    # Tüm RTSP akışlarından kare al
    for cap in caps:
        ret, frame = cap.read()
        frameler.append(frame if ret else None)

    now = time.time()
    if now - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = now

        # YOLO ile top tespiti
        for i, frame in enumerate(frameler):
            top_konum = None
            if frame is not None:
                results = model.predict(frame, classes=[32], conf=0.25, verbose=False)
                for box in results[0].boxes:
                    if int(box.cls[0]) == 32:
                        top_konum = top_konumu_3b(box, frame.shape)
                        break
            top_konumlari.append(top_konum)

        # En yakın kamerayı seç
        min_mesafe = float("inf")
        en_yakin_index = aktif_index
        for i, konum in enumerate(top_konumlari):
            if konum:
                uzaklik = mesafe_3d(konum, KAMERA_KONUM[i])
                if uzaklik < min_mesafe:
                    min_mesafe = uzaklik
                    en_yakin_index = i
        aktif_index = en_yakin_index

    # Aktif kamerayı göster
    aktif_frame = frameler[aktif_index]
    if aktif_frame is not None:
        ekran = aktif_frame.copy()
        cv2.putText(ekran, f"Aktif Kamera: {aktif_index + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Aktif Kamera", ekran)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kapat
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
