from ultralytics import YOLO

# Load a model
model = YOLO.load(this, "yolo-sort-demo.pt")

# Run inference on an image
result = model.predict(source="", save=True)
print(result)
