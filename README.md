# Football Player Detection
Object Detection with Football Datasets

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

## Datasets

This repo supports two main dataset sources for training:

1. **Roboflow football dataset** (source referenced in text file)
2. **Handheld MOTAF-style dataset** (raw source referenced in text file, then converted)

### Source references (where to download)

- Roboflow source and metadata:
	- `data/football_player_detection/readme_roboflow_dataset.txt`
	- `data/football_player_detection/readme_roboflow.txt`
- Handheld raw dataset source:
	- `data/handheld_dataset/readme_handheld_dataset.txt`

### Football dataset (Roboflow)

If you re-download this dataset from the URL in the `readme_roboflow_dataset.txt`, keep the YOLO-formatted folder structure and place it under:

```text
data/football_player_detection/
```

Make sure `data.yaml` exists at:

```text
data/football_player_detection/data.yaml
```

### Handheld dataset -> combined YOLO dataset

1. Download/extract the raw handheld data from the URL listed in:

```text
data/handheld_dataset/readme_handheld_dataset.txt
```

2. Place extracted sequence folders under:

```text
data/handheld_dataset/data/
```

3. Build the combined YOLO dataset:

```bash
python data/handheld_dataset/preprocess_handheld.py
```

This creates:

```text
data/combined_dataset/train/{images,labels}
data/combined_dataset/valid/{images,labels}
```

## Training

### Local training script

Main local trainer:

```text
train/train_ultralytics_local.py
```

Default config:

```text
configs/train_config.yaml
```

Run training:

```bash
python train/train_ultralytics_local.py --config configs/train_config.yaml
```

### Choose dataset in config

Edit `configs/train_config.yaml`:

- `dataset.type: football` to train on football/combined dataset
- `paths.football.dataset_dir` to either:
	- `./data/football_player_detection`, or
	- `./data/combined_dataset`

### Colab / notebook training scripts

Notebook versions are available in:

- `train/train_ultralytics_colab.ipynb`
- `train/train_rtdetr_colab.ipynb`

Open those notebooks in Colab or Jupyter and adapt paths to your mounted dataset location.

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

- Select a model from the dropdown (auto-discovered from `models/`; two models are provided for demo use only)
- Choose input source:
	- Upload an image/video (two example images from the roboflow test dataset are provided under `data/`), or
	- Select an existing video from `data/`
- Click **Run inference**
- For videos, frames are streamed in the page and you can download the processed result
- Example result:

![Streamlit example](img/streamlit_example.png)
