PROJECT_DETECT_OBJECT — Real-time Object Recognition System (YOLOv11 + SAM2.1 + Classifier)

![Detect Object Preview](https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)

🚀 Overview

    Hệ thống Realtime Object Detection & Segmentation kết hợp nhiều mô hình AI mạnh mẽ:
        - YOLOv11 (Pretrained) — Dò tìm vật thể nhanh và chính xác.
        - SAM2.1 (Segment Anything 2) — Phân vùng chính xác (segmentation) từng vật thể được YOLO phát hiện.
        - ImageSearcher (Embedding-based Similarity Search) — Khi xác suất thấp, hệ thống tìm vật thể tương tự trong thư viện annotated/.
        - Object Tracking + Label Stabilization — Theo dõi vật thể qua khung hình để tránh nhấp nháy nhãn.
        - Tất cả được xử lý real-time từ webcam, với giao diện hiển thị mask, bounding box, và tên vật thể ngay trên màn hình.

🏗️ System Architecture

1️⃣ Input Layer — Webcam Frame Capture
    - Luồng video lấy trực tiếp từ webcam (qua cv2.VideoCapture).
    - Mỗi frame được đưa vào hàng đợi (frame_queue) cho xử lý nền (thread).

2️⃣ YOLOv11 Detector
    - Model YOLOv11 pretrained (ultralytics.YOLO) xử lý detection nhanh chóng.
    - Xuất ra danh sách các bounding box [x1, y1, x2, y2].

3️⃣ SAM2.1 Segmenter
    - Dựa trên YOLO bounding boxes → SAM2.1 tạo segmentation mask chính xác cho từng vật thể.
    - Trọng số tùy chỉnh nạp từ: data/final_pth_to_webcam/sam2_inference_weights_latest.pth
    - File cấu hình: configs/sam2.1/sam2.1_hiera_b+.yaml
4️⃣ Custom Classifier (ResNet18 Fine-tuned)
    - Model ResNet18 được huấn luyện riêng trên dataset 102 lớp.
    - Checkpoint: /media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/final_pth_to_webcam/sam2_inference_weights_latest.pth
    - Khi phát hiện vật thể, phần ảnh được crop theo mask → phân loại qua classifier.

5️⃣ Image Searcher (Backup Matching)
    - Nếu độ tin cậy của classifier < 0.85, hệ thống tìm ảnh tương tự nhất trong thư viện data/annotated/ bằng cosine similarity giữa feature embedding.
6️⃣ Object Tracker

    - Theo dõi các bounding box qua khung hình (IOU-based tracking).

    - Làm mượt tọa độ và nhãn vật thể qua bbox_smooth_alpha.

    - Giúp nhãn không nhấp nháy khi camera di chuyển.

7️⃣ Display Layer

    - Hiển thị bounding box, mask (màu khác nhau) và label trực tiếp trên video.

    - FPS được tính theo thời gian thực.

    - Có thể dùng cv2.imshow hoặc fallback matplotlib nếu OpenCV không mở được cửa sổ.

⚙️ Environment Setup

1️⃣ Create Environment
    cd PROJECT_DETECT_OBJECT
    python3 -m venv .venv
    source .venv/bin/activate

2️⃣ Install Dependencies
    pip install -r requirements.txt

3️⃣ Checkpoint Preparation

| Model                           | Path                                                           | Description                    |
| ------------------------------- | -------------------------------------------------------------- | ------------------------------ |
| **SAM2.1**                | `data/final_pth_to_webcam/sam2_inference_weights_latest.pth` | Custom finetuned SAM weights   |
| **Classifier (ResNet18)** | `output/experiments/checkpoints/static_finetune_epoch12.pth` | Finetuned classification model |
| **YOLOv11 Pretrained**    | `checkpoints/yolov11n.pt`                                    | Pretrained detection model     |
| **Config**                | `configs/sam2.1/sam2.1_hiera_b+.yaml`                        | SAM2 architecture config       |

▶️ Run Real-time Detection
    python scripts/inference_webcam.py

🧩 Options

    - Press q to quit webcam window.

    - Modify cam_id if multiple cameras:
        inferencer.run(cam_id=1)
    - Adjust max_draw (number of displayed masks):
        inferencer = WebcamInferencer(..., max_draw=5)

💡 How the Pipeline Works Internally
    1. Capture Frame
        Reads image from webcam in a loop.

    2. Queue Handling
        Frame sent to inference_worker thread.

    3. YOLOv11 Inference
        Detects rough object bounding boxes.

    4. SAM2 Prediction
        Refines detection → pixel-level masks.

    5. Classifier + Image Searcher
        Assigns label using deep classification and similarity matching.

    5. Tracking
        Matches objects across frames using IoU.

    5. Display
        Draw masks, boxes, and names on live webcam feed.

🧠 Performance Notes

    - Uses multi-threading to separate webcam capture and AI inference.

    - Supports both CPU and GPU automatically (cuda or cpu).

    - Can handle ~10–15 FPS on RTX 3060 or similar GPU.

🧾 Logs & Debugging

| Level             | Prefix                             | Description |
| ----------------- | ---------------------------------- | ----------- |
| `[INFO]`        | General system info                |             |
| `[WARN]`        | Missing files / fallback defaults  |             |
| `[SUCCESS]`     | Successful model or label loading  |             |
| `[FATAL ERROR]` | Critical load or inference failure |             |

🧩 Extensions
🔹 Replace YOLOv11 model checkpoint with custom trained weights.

🔹 Fine-tune SAM2.1 with custom masks dataset.

🔹 Add new annotated images for stronger Image Searcher performance.

🔹 Integrate SORT/ByteTrack for more stable multi-object tracking.

🎯 Summary

| Component     | Framework          | Purpose             |
| ------------- | ------------------ | ------------------- |
| YOLOv11       | Ultralytics        | Object Detection    |
| SAM2.1        | Meta FAIR          | Mask Segmentation   |
| ImageSearcher | Custom             | Similarity Matching |
| Tracker       | Custom (IOU-based) | Temporal Stability  |

🖼️ Output Example
    When webcam runs successfully, you'll see:
        - Colored mask overlay per object
        - Bounding box with label name and confidence
        - Live FPS counter in terminal

🧑‍💻 Tác giả

👤 Nguyen Sanh Tien

📍 Đại học Giao thông vận tải

📧 Email: nguyensanhtien2411@gmail.com
