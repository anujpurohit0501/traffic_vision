
🎯 Moving Object Detection with YOLOv7
A real-time moving object detection system that combines a pretrained YOLOv7 object detector with classical motion analysis (background subtraction + multi-frame centroid tracking) to answer two questions at once for every object in a video:

What is it? (person, car, dog, ... — any of the 80 COCO classes)
Is it moving? (vs. a static object that just happens to be in frame)
It ships with a polished, ready-to-demo Streamlit frontend supporting video upload, image upload, and a live webcam feed.

✨ Features
🔍 Pretrained YOLOv7 object detection — no training required, works out of the box
🏃 Motion classification per object — green boxes = moving, gray boxes = static
🧵 Object tracking — persistent IDs + motion trails across frames via a lightweight centroid tracker
🎚️ Fully configurable — confidence/IoU thresholds, motion sensitivity, class filters, inference resolution
📹 Three input modes — upload a video, upload a single image, or use a live webcam
📊 Live stats & charts — moving/static object counts, per-frame chart, processing FPS
⬇️ Downloadable output — export the fully annotated video
🎨 Clean, presentation-ready UI — custom styling, metric cards, tabs
🧠 How It Works
                ┌──────────────┐
   video/frame  │   YOLOv7     │  boxes + labels + confidence
  ────────────► │  (detector)  │ ─────────────┐
                └──────────────┘               │
                                                ▼
                ┌──────────────┐        ┌───────────────┐        ┌──────────────┐
   video/frame  │ MOG2 Motion  │  mask  │ Motion Scoring │  →     │   Centroid   │
  ────────────► │  Subtractor  │ ─────► │ (mask overlap  │        │   Tracker    │
                └──────────────┘        │  + displacement)│ ◄────► (IDs + trails)│
                                         └───────┬────────┘        └──────────────┘
                                                 ▼
                                     MOVING (green) / STATIC (gray)
                                          annotated output
Detection – Every frame runs through YOLOv7 to get bounding boxes, class labels, and confidence scores for all detected objects.
Motion masking – An MOG2 background subtractor continuously models the static background and flags pixels that deviate from it (the foreground/motion mask).
Motion scoring – For each YOLO box we compute:
Overlap ratio — the fraction of the box's area that falls inside the motion mask.
Centroid displacement — how far the object's tracked center has moved over the last several frames (via the lightweight centroid tracker). An object is flagged MOVING if either signal (configurable to require both) passes its threshold — this keeps the system robust against noisy backgrounds and against slow-moving objects that a mask alone might miss.
Tracking – A simple greedy centroid + IoU tracker assigns stable IDs across frames, enabling motion trails and per-object history.
This two-stage design (semantic detection + independent motion signal) is what makes this a moving object detector rather than a plain object detector — it will correctly leave a parked car unhighlighted while tagging a pedestrian walking past it.

📂 Project Structure
moving-object-detection/
├── app.py            # Streamlit frontend (video/image/webcam modes, UI, charts)
├── detector.py        # YOLOv7Detector wrapper + MotionEstimator (MOG2 background subtraction)
├── tracker.py          # CentroidTracker — assigns IDs, tracks centroid history/displacement
├── utils.py             # COCO class names, colors, drawing helpers, FPS meter
├── requirements.txt      # Python dependencies
└── README.md              # This file
🚀 Getting Started
1. Clone and set up an environment
git clone <your-repo-url> moving-object-detection
cd moving-object-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
GPU users: install a CUDA-matching build of torch/torchvision from https://pytorch.org/get-started/locally/ before running pip install -r requirements.txt, then select cuda as the device in the sidebar.

2. Download pretrained YOLOv7 weights
The app expects a YOLOv7 .pt weights file in the project root (default: yolov7.pt). Download the official weights from the YOLOv7 authors' release page:

# Standard model (~74 MB) — good accuracy/speed balance
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# OR the tiny model (~12 MB) — much faster, slightly less accurate, better for CPU/live demo
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
(No wget? Just open the URL in a browser and save the file, or use curl -L -o yolov7.pt <url>.)

The first time the app runs, torch.hub will also clone the WongKinYiu/yolov7 repository (needs internet access once; it's cached afterwards under ~/.cache/torch/hub). For fully offline / air-gapped deployment, clone that repo locally yourself and pass its path via the local_repo_dir argument in detector.py.

3. Run the app
streamlit run app.py
Open the URL Streamlit prints (typically http://localhost:8501).

🎛️ Using the App
Tab	What it does
📹 Upload Video	Upload an .mp4/.mov/.avi/.mkv file, click Run detection, watch a live preview + progress bar, then preview/download the fully annotated output video. Also plots a moving-vs-static objects-per-frame chart.
🖼️ Upload Image	Upload a single image for one-shot object detection (no motion classification — a single frame has no history).
🔴 Live Webcam	Real-time detection straight from your browser's webcam via streamlit-webrtc.
ℹ️ About	In-app explanation of the pipeline (handy during a live demo/Q&A).
Sidebar controls:

Model: weights path, device (cpu/cuda), inference resolution
Detection: confidence threshold, NMS IoU threshold
Motion sensitivity: minimum mask-overlap ratio, minimum centroid displacement (px), and whether the two signals combine with OR (more sensitive) or AND (more strict)
Filters: restrict to specific COCO classes, toggle static-object display, motion trails, and the on-frame stats HUD
⚙️ Configuration Tips
CPU-only machine? Use yolov7-tiny.pt and drop the inference size to 416 or 320 for smoother live-webcam performance.
Too many false "moving" flags (e.g. from camera shake or lighting flicker)? Raise the mask-overlap and displacement thresholds, or switch motion logic to AND.
Missing slow-moving objects? Lower the displacement threshold, or use OR logic so a strong mask-overlap alone is enough.
Static camera vs. moving camera: MOG2 background subtraction assumes a mostly static camera. For a panning/handheld camera, rely more heavily on the tracking/ displacement signal (raise the mask-overlap threshold high, or switch to AND).
🛠️ Tech Stack
YOLOv7 — pretrained object detector (PyTorch)
OpenCV — video I/O, MOG2 background subtraction, drawing
Streamlit — frontend/UI
streamlit-webrtc — live webcam streaming in-browser
Pandas — stats/chart data handling
📈 Possible Extensions
Swap the centroid tracker for ByteTrack / DeepSORT for more robust ID persistence through occlusion
Add zone-based alerts (e.g. notify when a moving object enters a defined region)
Export per-object trajectories as CSV for downstream analytics
Add direction/speed estimation using calibrated pixel-to-real-world scale
Swap YOLOv7 for a fine-tuned custom model for domain-specific classes
📄 License & Credits
YOLOv7: WongKinYiu/yolov7 (GPL-3.0) — cite their paper "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors" if used in academic work.
This project's own code (app/detector/tracker/utils) is provided as-is for educational and portfolio use.
