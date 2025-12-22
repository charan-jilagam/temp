from ultralytics import YOLO
model = YOLO("data/activation_best.pt")
print(model.names)