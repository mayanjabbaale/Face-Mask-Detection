import matplotlib.pyplot as plt
import cv2
from pathlib import Path

image_path = state_dict = Path('./Datasets/face-mask-data/images/maksssksksss600.png')
face_model = cv2.CascadeClassifier(
    r'C:\Users\p\OneDrive\Desktop\CoVision\cv-env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
)
img = cv2.imread(image_path)

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

#plotting
for (x,y,w,h) in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),1)
plt.figure(figsize=(12,12))
plt.imshow(out_img)
plt.show()