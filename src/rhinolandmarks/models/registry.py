from .unet import UNet
from .hrnet import HRNetLite

MODEL_REGISTRY = {
    "unet": UNet,
    "hrnet": HRNetLite,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)