import time
from ultralytics import YOLO
import cv2

model = YOLO("runs/runs/detect/YOLO weld detection project/weld_yolo_training/weights/best.pt")        # or the full runs/... path

image = cv2.imread("datasets/train/images/90ba92f8-fd03ae25-cracked-sun-trike-frame_jpg.rf.010bbb65a6f77c04dded749e5608979a.jpg")

start = time.time()
results = model(image)
end = time.time()

latency = end - start
print("Inference Time:", latency, "seconds")
