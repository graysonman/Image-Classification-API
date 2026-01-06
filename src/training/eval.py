import torch
from src.models.simple_cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.training.adversarial import fgsm_attack

def evaluate_model(model_path, epsilon=0.03):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            clean_correct += (predicted == labels).sum().item()

        adv_images = fgsm_attack(model, images, labels, epsilon)
        outputs_adv = model(adv_images)
        _, predicted_adv = torch.max(outputs_adv, 1)
        adv_correct += (predicted_adv == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total

    print(f"Clean accuracy: {clean_acc:.2f}%")
    print(f"Adversarial accuracy (ε={epsilon}): {adv_acc:.2f}%")

    return clean_acc, adv_acc

if __name__ == "__main__":
    evaluate_model(model_path="artifacts/model.pth")