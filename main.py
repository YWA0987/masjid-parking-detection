from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source="",
    conf=0.3,
    imgsz=1300,
    show=True,
    save=True,
    classes=[2, 7],      
)
