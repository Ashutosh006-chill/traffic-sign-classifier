import torch
import cv2
from torchvision import transforms
from model import TrafficSignCNN
import config
import sys

def predict(image_path):
    try:
        # Graceful error handling for missing files
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error: {e}")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    model = TrafficSignCNN()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence, predicted_class = torch.max(output, 1)
        
    print(f"Predicted Class ID: {predicted_class.item()} (Confidence raw score: {confidence.item():.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
    else:
        predict(sys.argv[1])