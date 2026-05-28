from torchvision import models
import torch 

MODEL_PATH = {
    'resnet18': (models.resnet18, 'resnet18_weights.pth'),
    'resnet101': (models.resnet101, 'resnet101-63fe2227.pth'),
    'squeezenet': (models.squeezenet1_1, 'squeezenet1_1-b8a52dc0.pth'),
    'convnext': (models.convnext_tiny,'monvnext_tiny-983f1562.pth'), 
    'googlenet': (models.googlenet, 'googlenet-1378be20.pth'), 
}

PRETRAINED = False
from torchsummary import summary

for model_name in MODEL_PATH.keys():
    class_ = MODEL_PATH[model_name][0]
    weights_path = MODEL_PATH[model_name][1]
    model = class_(pretrained=PRETRAINED)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    print(model_name)
    # print(missing_keys, unexpected_keys)
    print(model)
    summary(model, (3, 224, 224), device='cpu')
    print()
