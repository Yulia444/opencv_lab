import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, old_frame = cap.read()
old_point, new_point = None, ()
mask = np.zeros_like(old_frame)
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        new_point = (x + w // 2, y + h // 2)
        if old_point is None:
            old_point = new_point
        print(old_point, new_point)
        mask = cv2.line(mask, new_point, old_point, (255, 0, 0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    img = cv2.add(frame, mask)
    old_point = new_point
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()