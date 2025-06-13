from ultralytics import YOLO
# Load a COCO-pretrained YOLO11n model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("yolo bus in city.jpg", save=True, show=True)


