import torch
import torch.nn as nn
import torch.optim as optim
from model import TrafficSignCNN
from data_loader import get_data_loaders
import config

def train_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN().to(device)
    train_loader, val_loader = get_data_loaders(data_dir)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    # Point this to your downloaded dataset folder
    train_model("./data/gtsrb")