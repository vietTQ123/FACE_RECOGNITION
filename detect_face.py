import cv2

from skimage.feature import hog
from sklearn.decomposition import PCA

import pickle as pk
import joblib
import numpy as np

clf = joblib.load('C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\c2_svm_classifier.pkl')
pca = pk.load(open("C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\c1_PCA.pkl",'rb'))
face_cascade = cv2.CascadeClassifier("C:\\Users\\Admin\\Documents\\EMBEDDED_LAB\\FACE_RECOGNITION\\BTL_THCS\\code\\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
while True:
    cnt_Viet = 0
    cnt_unknown = 0
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)        
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  
        resized = cv2.resize(face_roi, (128, 128))
        
        # Match HOG parameters with training code
        hog_features = hog(resized, 
                          orientations=12,  # Changed from 9 to 12
                          pixels_per_cell=(4, 4),
                          cells_per_block=(3, 3),
                          block_norm='L2-Hys',
                          visualize=False)
        
        hog_features_pca = pca.transform([hog_features])
        probabilities = clf.predict_proba(hog_features_pca)
        confidence = np.max(probabilities)
        label = clf.predict(hog_features_pca)[0]
        if confidence < 0.9:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cnt_unknown += 1
            print("Unknown\n")
        elif (label == 0):
            cv2.putText(frame, f'Trinh_Quoc_Viet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print("Viet\n")
            cnt_Viet += 1
            #pass
        else:
            # cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # print("Person\n")
            pass
        print(f"Confidence: {confidence:.2f}")
        print(f"Label: {label}")
        print(f"Count Viet: {cnt_Viet}")
        print(f"Count Unknown: {cnt_unknown}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition with Saved Model', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()