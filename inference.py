from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/runs/detect/YOLO weld detection project/weld_yolo_training/weights/best.pt")

# Load an image
image = cv2.imread("datasets/train/images/0e538f39-1d30436c-c23_jpg.rf.9af91763a613bf193d03c39400da2ec6.jpg")

# Run inference
results = model(image)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Label text
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

# Show output
cv2.imshow("Weld Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
