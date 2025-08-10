# Fall Safety Vision

A YOLOv8-based computer vision system for detecting and preventing fall hazards.  
Includes GPU-accelerated training and visualization of performance metrics.  
Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), PyTorch, and OpenCV.

---

## 🚀 Features
- **GPU Acceleration** – Automatic CUDA detection for faster training.
- **Custom Dataset Support** – Train on your own images and labels.
- **Transfer Learning** – Starts from a pre-trained YOLOv8 model for better accuracy with fewer epochs.
- **Visualized Results** – Automatically displays training metrics after completion.
- **Configurable Parameters** – Easily adjust epochs, batch size, and image size.

---

## 📂 Project Structure
```
fall-safety-vision/
│
├── train_yolo.py             # Main training script
├── requirements.txt          # Python dependencies
├── yolo_dataset/             # Dataset folder (images + labels + data.yaml)
│   ├── train/
│   ├── val/
│   └── data.yaml
└── README.md                 # This file
```

---

## 🛠 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/<Abins2004>/fall-safety-vision.git
   cd fall-safety-vision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install YOLOv8**
   ```bash
   pip install ultralytics
   ```

4. **Verify GPU (Optional)**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 📦 Dataset Setup

1. Create a folder named `yolo_dataset` in your Google Drive (or local machine).
2. Inside it, create:
   ```
   train/  # Training images and labels
   val/    # Validation images and labels
   data.yaml
   ```
3. Example `data.yaml`:
   ```yaml
   train: /content/drive/MyDrive/yolo_dataset/train
   val: /content/drive/MyDrive/yolo_dataset/val
   nc: 1
   names: ['fall']
   ```

---

## ▶️ Usage

Run the training script:
```bash
python train_yolo.py
```

The script will:
- Load the dataset configuration from `data.yaml`
- Train a YOLOv8 small model (`yolov8s.pt`)
- Save results in `/content/drive/MyDrive/runs/detect/fall_safety_yolov8_run`
- Display training graphs

---

## 📊 Example Results

After training, a `results.png` file is generated showing:
- Training & validation loss
- Precision, recall, and mAP curves
- Learning rate schedule

Example:
![Training Metrics](runs/detect/fall_safety_yolov8_run/results.png)

---

## ⚙️ Configuration

Edit `train_yolo.py` to adjust:
- **epochs** – Number of training cycles (default: 50)
- **imgsz** – Image resolution (default: 640)
- **batch** – Batch size (default: 8)
- **device** – `'cpu'` or GPU index (default: auto-detect)

---

## 📜 License
This project is released under the [MIT License](LICENSE).

---

## 🤝 Contributing
Pull requests are welcome!  
If you find a bug or want to add features, open an issue or submit a PR.

---

## ⭐ Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
