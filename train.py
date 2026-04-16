from ultralytics import YOLO

model = YOLO("yolo12s.pt")
model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    workers=4
)