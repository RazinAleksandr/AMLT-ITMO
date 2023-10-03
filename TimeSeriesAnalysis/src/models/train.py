from tqdm import tqdm
import torch


def run_train_loop(epochs, model, train_loader, val_loader, optimizer, criterion, device):
    train_losses = []
    val_losses = []
    total_iterations = len(train_loader) + len(val_loader)

    for epoch in range(1, epochs+1):    
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{epochs}")

        # Training
        train_loss = train(model, train_loader, optimizer, criterion, device, pbar)
        train_losses.append(train_loss)

        # Validation
        val_loss = validate(model, val_loader, criterion, device, pbar)
        val_losses.append(val_loss)

        pbar.close()
        print(f"Epoch {epoch}/{epochs} => Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(' ')

    return train_losses, val_losses

def train(model, train_loader, optimizer, criterion, device, pbar):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.update(1)

    return total_loss / len(train_loader)

def validate(model, validation_loader, criterion, device, pbar):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_val_loss += loss.item()

            pbar.set_description(f"Validation Loss: {loss.item():.4f}")
            pbar.update(1)

    return total_val_loss / len(validation_loader)

