import cv2
import numpy as np
import joblib
import pickle as pk
from skimage.feature import hog
from sklearn.decomposition import PCA

# Tải mô hình SVM và PCA đã huấn luyện
clf = joblib.load('C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\c2_svm_classifier.pkl')
pca = pk.load(open("C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\c1_PCA.pkl", 'rb'))
face_cascade = cv2.CascadeClassifier("C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\haarcascade_frontalface_default.xml")

# Mở webcam (dùng CAP_DSHOW để tránh lỗi MSMF)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Kiểm tra camera có mở được không
if not cam.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra lại kết nối hoặc driver.")
    exit()

print("[INFO] Đang khởi động hệ thống nhận diện khuôn mặt... Nhấn 'q' để thoát.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Không lấy được khung hình.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.equalizeHist(face_roi)  # Cân bằng sáng
        resized = cv2.resize(face_roi, (128, 128))

        # Trích xuất đặc trưng HOG
        hog_features = hog(resized,
                           orientations=12,
                           pixels_per_cell=(4, 4),
                           cells_per_block=(3, 3),
                           block_norm='L2-Hys',
                           visualize=False)

        # Áp dụng PCA
        hog_features_pca = pca.transform([hog_features])

        # Dự đoán
        probabilities = clf.predict_proba(hog_features_pca)[0]
        confidence = np.max(probabilities)
        label = clf.predict(hog_features_pca)[0]

        # Ngưỡng phân biệt người lạ
        if confidence < 0.75:
            label_text = "Unknown"
            color = (0, 0, 255)
        else:
            if label == 0:
                label_text = "J97"
            elif label == 1:
                label_text = "Trinh_Quoc_Viet"
            elif label == 2:
                label_text = "Trinh_Hoai_Duc"
            elif label == 3:
                label_text = "Le_Trung_Thanh"
            elif label == 4:
                label_text = "Dao_Ha_Thai_Son"
            else:
                label_text = "Unknown"
            color = (0, 255, 0) if label_text != "Unknown" else (0, 0, 255)

        # Hiển thị kết quả
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label_text} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"[INFO] {label_text} | Confidence: {confidence:.2f}")

    # Hiển thị khung hình
    cv2.imshow("Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cam.release()
cv2.destroyAllWindows()
