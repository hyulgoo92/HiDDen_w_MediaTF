import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        
        self.decoder = Decoder(config)
    
    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)

        ## check
        #img = noised_image[0].detach().cpu().clamp(0, 1)
        #if img.shape[0] == 1:
        #    img = img.squeeze(0)  # (H, W)
        #else:
        #    img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        #img.numpy()

        #noised_img_np = img

        #img = encoded_image[0].detach().cpu().clamp(0, 1)
        #if img.shape[0] == 1:
        #    img = img.squeeze(0)  # (H, W)
        #else:
        #    img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        #img.numpy()

            # 이미지 전처리
        #encoded_img_np = img

        # 두 이미지 나란히 출력
        #fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        #axes[0].imshow(encoded_img_np, cmap='gray' if encoded_img_np.ndim == 2 else None)
        #axes[0].set_title('Encoded Image')
        #axes[0].axis('off')

        #axes[1].imshow(noised_img_np, cmap='gray' if noised_img_np.ndim == 2 else None)
        #axes[1].set_title('Noised Image')
        #axes[1].axis('off')

        #plt.tight_layout()
        #plt.show()

        return encoded_image, noised_image, decoded_message
    
