import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.models.simple_cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from pathlib import Path

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

def train_model(epochs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(train_loader)

    for epoch in range(epochs):
        model.train()
        loss = 0.0
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pth")
    print("Model saved to artifacts/model.pth")

if __name__ == "__main__":
    train_model(epochs=5, lr=0.001)
