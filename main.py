import cv2
import numpy as np

img = cv2.imread(r'C:\Users\nzanosiev\Desktop\Nikita\py\12.18\OpenCV\vid7\images\face4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier(r'C:\Users\nzanosiev\Desktop\Nikita\py\12.18\OpenCV\vid7\faces.xml')

results = faces.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('result', img)
cv2.waitKey()
