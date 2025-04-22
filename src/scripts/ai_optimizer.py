import bitsandbytes as bnb


def get_optimizer_grouped_parameters_with_llrd(model):
    """layerwise learning rate decay implementation"""
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": 2e-5,
            "weight_decay": 1e-3,
        },
    ]

    # initialize lrs for backbone layers
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()
    lr = 2e-5

    for layer in layers:
        lr *= 0.9

        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-3,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def get_optimizer(model):
    """optimizer for model training"""

    optimizer_grouped_parameters = get_optimizer_grouped_parameters_with_llrd(model)

    optimizer_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "lr": 2e-5}

    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
