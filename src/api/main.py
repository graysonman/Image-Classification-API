from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.simple_cnn import SimpleCNN
from pathlib import Path

MODEL_PATH = Path("artifacts/model.pth")

app = FastAPI()

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
def load_model():
    global model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

# Transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Image Classification Demo</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h1>Image Classification Demo</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required />
                <br><br>
                <button type="submit">Upload & Predict</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    predicted_idx = predicted_idx.item()
    confidence = confidence.item() * 100
    label = CIFAR10_CLASSES[predicted_idx]

    html = f"""
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h2>Prediction Result</h2>
            <p><strong>Class Index:</strong> {predicted_idx}</p>
            <p><strong>Label:</strong> {label}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            <br>
            <a href="/">Try another image</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html)
