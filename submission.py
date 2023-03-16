import numpy as np
import pandas as pd
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

def create_submission(models, x_submission, submission_template, output_file='submission.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    submission_dataset = TensorDataset(torch.from_numpy(x_submission))
    submission_dataloader = DataLoader(submission_dataset, batch_size=32, shuffle=False)

    for model in models:
        model.eval()

    predictions = []

    with torch.no_grad():
        for x in submission_dataloader:
            x = x[0].to(device)

            ensemble_preds = ensemble_prediction(models, x)
            predictions.extend(ensemble_preds.tolist())

    submission = submission_template.copy()
    submission['Predicted'] = predictions
    submission.to_csv(output_file, index=False)

    print(f'Submission file saved to {output_file}')

if __name__ == '__main__':
    trained_models = [...]  # Load your trained models here
    x_submission = ...  # Load your preprocessed submission dataset here
    submission_template = pd.read_csv('sample_submission.csv')  # Load your submission template file here

    create_submission(trained_models, x_submission, submission_template)
