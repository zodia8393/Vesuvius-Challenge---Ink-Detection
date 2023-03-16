import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def calculate_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total

def validate_model(model, x_val, y_val, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = F.cross_entropy(output, y)

            accuracy = calculate_accuracy(output, y)

            running_loss += loss.item() * x.size(0)
            running_accuracy += accuracy * x.size(0)

    avg_loss = running_loss / len(val_dataloader.dataset)
    avg_accuracy = running_accuracy / len(val_dataloader.dataset)

    return avg_loss, avg_accuracy

if __name__ == '__main__':
    model = ...  # Load your trained model here
    _, (x_val, y_val), _ = ...  # Load your preprocessed and split dataset here

    val_loss, val_accuracy = validate_model(model, x_val, y_val)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
