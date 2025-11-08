from ultralytics import YOLO

model = YOLO('yolo11m-seg.pt')  # load a pretrained model
results = model.train(
    # --- Your Original Parameters ---
    data='peduncle_mega.yaml',
    epochs=400,
    imgsz=640,
    val=True,
    batch=12,
    single_cls=True,

    # 1. Early Stopping
    patience=50,          # Stop training if no improvement is seen for 50 epochs.

    # 2. Optimizer and Regularization
    lr0=0.005,            # Set a slightly lower initial learning rate (default is 0.01).
    weight_decay=0.0005,  # Add weight decay for regularization.

    # 3. Data Augmentation
    hsv_h=0.025,          # Hue augmentation (color shift)
    hsv_s=0.80,           # Saturation augmentation (color intensity)
    hsv_v=0.50,           # Value augmentation (brightness)
)