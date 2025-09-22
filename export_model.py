import torch
import torch.nn as nn
import numpy as np

class CNN_DuelingDQN(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(CNN_DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2), nn.ReLU()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 4, 4)
            conv_out_size = self.conv(dummy_input).flatten().shape[0]
        self.value_stream = nn.Sequential(nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

def main():
    PYTORCH_MODEL_PATH = "2048_best_new.pth" 
    ONNX_MODEL_PATH = "frontend/model.onnx" 
    
    device = torch.device("cpu")
    model = CNN_DuelingDQN(in_channels=1, action_dim=4).to(device)
    
 
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['q_net'])
    model.eval()
    print("PyTorch model loaded successfully.")

    dummy_input = torch.randn(1, 1, 4, 4, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        input_names=['input'],
        output_names=['output'], 
        opset_version=11
    )
    print(f"Model successfully converted and saved to {ONNX_MODEL_PATH}")

if __name__ == '__main__':
    main()
