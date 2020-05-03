from torchvision import models


def resnet(num_layers: int, num_classes: int, pretrain: bool):
    """
    Args:
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain: Use pretrained ResNet weights.
    """

    if num_classes < 3:
        num_classes = 1

    assert num_layers in (
        18, 34, 50, 101, 152
    ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
    model_constructor = getattr(models, f"resnet{num_layers}")
    model = model_constructor(num_classes=num_classes)

    if pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)
    return model
