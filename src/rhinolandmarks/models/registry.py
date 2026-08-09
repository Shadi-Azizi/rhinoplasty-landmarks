from .unet import UNet
from .unet_pretrained import UNetResNet34
from .hrnet_timm import HRNetW18

MODEL_REGISTRY = {
    "unet": UNet,
    "unet_resnet34": UNetResNet34,
    "hrnet_w18": HRNetW18,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)