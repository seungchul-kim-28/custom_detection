from src.loss_fn.detr_criterion import DETRCriterion

CRITERION_REGISTRY={
    'DETRCriterion': DETRCriterion
}


def build_criterion(config):
    crit_name = config.criterion.name

    if crit_name not in CRITERION_REGISTRY:
        raise ValueError(f"Unknown Criterion : {crit_name}")
    
    return CRITERION_REGISTRY[crit_name](config)
