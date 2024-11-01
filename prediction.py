import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/weights/runs/segment/train/weights/best.pt")

image_path = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/Chip_Segmentation.v1i.yolov8/valid/images/28_jpg.rf.9213dca6f734cd426e252e0d4a4dbf6b.jpg"
results = model.predict(source = image_path, conf=0.25, save=True) 

annotated_image = cv2.imread(f'/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/weights/runs/segment/predict/{image_path.split("/")[-1]}')
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.title("Result of the prediction")
plt.axis('off')
plt.show()