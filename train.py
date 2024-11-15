from ultralytics import YOLO
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load a model
    model = YOLO("yolo11n-cls")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="caltech101", epochs=50, imgsz=416, batch=-1, save_period=5, verbose=True,
                          lr0=0.01, resume=True)
    # results = model.train(resume=True, batch=0.4)

    # Validate the model
    metric = model.val(data="caltech101", imgsz=416)  # evaluate model performance on the validation set
    print(metric)

    # Save the model
    model.save("yolo-sort-demo_50_epoch.pt")
