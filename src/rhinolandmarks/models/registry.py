from .unet import UNet

MODEL_REGISTRY = {
    "unet": UNet,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)