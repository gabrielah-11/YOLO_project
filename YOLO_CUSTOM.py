from ultralytics import YOLO
# Load a COCO-pretrained YOLO11n model
model = YOLO(r"yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=r".vscode\FRC-v2025-4\data.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model(r"C:\Users\Gabriela\Downloads\FRC v2025.v1-yolo-test.coco\test1.jpeg", save=True, show=True)