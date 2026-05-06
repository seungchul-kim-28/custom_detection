from src.models.detr import DETR

MODEL_REGISTRY = {
    "DETR": DETR,
}


def build_model(config):
    model_name = config.model.name

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown Model : {model_name}')
    
    return MODEL_REGISTRY[model_name](config)