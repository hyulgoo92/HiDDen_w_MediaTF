import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


class MediatfNoise(nn.Module):
    def __init__(self, device, model_path='./noise_layers/latest_net_G.pth'):
        super(MediatfNoise, self).__init__()

        self.device = device

        # JIT 모델 로드
        self.model = torch.jit.load(model_path, map_location=self.device).to(self.device)
        self.model.eval()

    def forward(self, encoded_and_cover):
        encoded_image, _ = encoded_and_cover
        with torch.no_grad():
            noised_image = self.model(encoded_image)

        return [noised_image, encoded_and_cover[1]]