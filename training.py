import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = F.cross_entropy(output, y)

            running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train_model(model, x_train, y_train, x_val, y_val, num_epochs=50, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, device)

        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    return model

if __name__ == '__main__':
    model = ...  # Load your modified pre-trained model here
    (x_train, y_train), (x_val, y_val), _ = ...  # Load your preprocessed and split dataset here

    model = train_model(model, x_train, y_train, x_val, y_val)
    print('Model training complete')
