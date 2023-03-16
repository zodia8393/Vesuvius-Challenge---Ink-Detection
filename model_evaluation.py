import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def calculate_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total

def evaluate_model(model, x_test, y_test, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = F.cross_entropy(output, y)

            accuracy = calculate_accuracy(output, y)

            running_loss += loss.item() * x.size(0)
            running_accuracy += accuracy * x.size(0)

    avg_loss = running_loss / len(test_dataloader.dataset)
    avg_accuracy = running_accuracy / len(test_dataloader.dataset)

    return avg_loss, avg_accuracy

if __name__ == '__main__':
    model = ...  # Load your trained model here
    _, _, (x_test, y_test) = ...  # Load your preprocessed and split dataset here

    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
