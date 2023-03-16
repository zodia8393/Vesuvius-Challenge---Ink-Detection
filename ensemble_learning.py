import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def ensemble_prediction(models, x):
    predictions = []

    for model in models:
        with torch.no_grad():
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted)

    ensemble_preds = torch.stack(predictions)
    ensemble_preds, _ = torch.mode(ensemble_preds, dim=0)

    return ensemble_preds

def evaluate_ensemble(models, x_test, y_test, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for model in models:
        model.eval()

    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            ensemble_preds = ensemble_prediction(models, x)
            correct = (ensemble_preds == y).sum().item()
            total = y.size(0)

            running_correct += correct
            running_total += total

    accuracy = running_correct / running_total

    return accuracy

if __name__ == '__main__':
    trained_models = [...]  # Load your trained models here
    _, _, (x_test, y_test) = ...  # Load your preprocessed and split dataset here

    ensemble_accuracy = evaluate_ensemble(trained_models, x_test, y_test)
    print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')
