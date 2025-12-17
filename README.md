# End-to-End Image Classification API

It trains a convolutional neural network (CNN) on the CIFAR-10 dataset and exposes the trained model through a **REST API** using FastAPI. The API accepts an image upload and returns a **numeric class prediction**.

This repository is intentionally structured as a **real backend service**, not a notebook-only ML project.

---

## 🚀 What This Project Does (High Level)

1. **Trains a CNN image classifier** on CIFAR-10 (10 classes)
2. **Saves the trained model** as a reusable artifact (`model.pth`)
3. **Serves the model through a FastAPI backend**
4. **Packages everything in Docker** for reproducible deployment

At runtime, the API:
- Accepts an image upload (`multipart/form-data`)
- Resizes the image to match training input size (32×32)
- Runs inference using the trained model
- Returns a numeric class ID

---

## 🧠 Model Output

The API currently returns a **single integer**:

```json
{
  "prediction": 7
}
```

This number corresponds to the predicted class index from the CIFAR-10 dataset:

| ID | Class |
|----|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

> Note: The model **always returns a class**, even for images outside the CIFAR-10 domain. Confidence scores and class names will be added in later iterations.

---

## 📁 Project Structure

```
.
├── data/                 # Dataset storage (CIFAR-10)
├── src/
│   ├── api/              # FastAPI inference service
│   ├── data/             # Dataset loading utilities
│   ├── models/           # CNN model definitions
│   ├── training/         # Training scripts
│   └── __init__.py
├── model.pth             # Trained model artifact
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠 Requirements

### Local (non-Docker)
- Python 3.10+
- pip

### Docker (recommended)
- Docker Desktop or Docker Engine

---

## 📦 Installation (Local)

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

From the project root:

```bash
python -m src.training.train
```

This will:
- Download CIFAR-10
- Train the CNN
- Save the model to `model.pth`

> The API **requires** `model.pth` to exist before starting.

---

## 🌐 Running the API (Local)

```bash
uvicorn src.api.main:app --reload
```

Open the interactive docs:

```
http://localhost:8000/docs
```

Use **POST /predict** to upload an image.

---

## 🐳 Running with Docker

### Build the image

```bash
docker build -t cs230_project .
```

### Run the container

```bash
docker run -p 8000:8000 cs230_project
```

Open:

```
http://localhost:8000/docs
```

---

## 🔌 API Usage

### Endpoint

```
POST /predict
```

### Request
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

### Example (curl)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.png"
```

### Response

```json
{
  "prediction": 3
}
```
