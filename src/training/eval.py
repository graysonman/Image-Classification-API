import torch
from src.models.simple_cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.training.adversarial import fgsm_attack

def ascii_bar(value, max_value=100, width=30):
    filled = int((value / max_value) * width)
    return "█" * filled + "-" * (width - filled)

def evaluate_model(model_path, epsilons=None):
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.02, 0.03, 0.05]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {}

    for epsilon in epsilons:
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if epsilon > 0:
                images = fgsm_attack(model, images, labels, epsilon)

            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        results[epsilon] = acc

    print("Adversarial Robustness Evaluation")
    print("-" * 50)

    for epsilon, acc in results.items():
        bar = ascii_bar(acc)
        print(f"ε={epsilon:>4} | {acc:6.2f}% | {bar}")

    return results

if __name__ == "__main__":
    evaluate_model(model_path="artifacts/model.pth")