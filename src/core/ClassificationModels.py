import os
import sys
sys.path.append(os.getcwd())  # NOQA

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import src.lightning_models as LM
import pytorch_lightning as pl
from PIL import Image


class Transform:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)['image']


class MotorBikeModels(pl.LightningModule):
    def __init__(self, model: str, num_classes: int = 3, **kwargs):
        super().__init__()

        if model == 'resnet50':
            self.model = LM.ResNet50(num_classes=num_classes)
        elif model == 'vit':
            self.model = LM.VisionTransformerBase(num_classes=num_classes)
        elif model == 'vit_tiny':
            self.model = LM.VisionTransformerTiny(num_classes=num_classes)
        elif model == 'swinv2_base':
            self.model = LM.SwinV2Base(num_classes=num_classes)
        elif model == 'mobilenetv3_large':
            self.model = LM.MobileNetV3Large(num_classes=num_classes)
        elif model == 'resnet18':
            self.model = LM.ResNet18(num_classes=num_classes)

        if kwargs.get('weight'):
            self.load_weight(kwargs.get('weight'))

    def load_weight(self, weight_path: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(weight_path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'], strict=False)

    def forward(self, x):
        return self.model(x)

    def infer(self, image_path: str) -> int:
        img_np = np.array(Image.open(image_path).convert('RGB'))
        img = Transform()(img_np)

        self.eval()
        with torch.no_grad():
            pred = self(img.unsqueeze(0))

        return torch.argmax(pred, dim=1).item()


if __name__ == '__main__':
    model = MotorBikeModels(
        model='resnet18',
        weight='src/configs/weights/classify/resnet18.ckpt'
    )

    img_path = 'demo/2742_motorcycle.jpg'
    print(model.infer(img_path))
