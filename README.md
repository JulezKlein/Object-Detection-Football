# SoccerTrack
Object Detection and Tracking with SoccerTrack dataset

## Streamlit inference UI

Run a web interface to upload an image or video, select a model from a dropdown, and stream detection results.

### 1) Install dependencies

```bash
pip install streamlit ultralytics opencv-python omegaconf onnxruntime pillow
```

### 2) Start the app

```bash
streamlit run streamlit_app.py
```

### 3) Use the interface

- Select a model from the dropdown (auto-discovered from `models/`)
- Choose input source:
	- Upload an image/video, or
	- Select an existing video from `data/`
- Click **Run inference**
- For videos, frames are streamed in the page and you can download the processed result
