import torch
import torchvision
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, num_layers: int, num_classes: int, pretrain: bool, spatial_sensitive: bool = False,
                 n_spatial_features: int = 64):
        """
        Args:
            num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
            num_classes: Number of classes in the dataset.
            pretrain: Use pretrained ResNet weights.
        """
        super().__init__()
        self._num_layers = num_layers
        self._pretrain = pretrain
        self._spatial_sensitive = spatial_sensitive and n_spatial_features > 0

        if num_classes < 3:
            self._num_classes = 1
        else:
            self._num_classes = num_classes

        self._resnet = self._make_resnet()
        fc_in_features = self._resnet.fc.in_features
        self._resnet.fc = torch.nn.Identity()
        if self._spatial_sensitive:
            self._spatial_transformer = torch.nn.Linear(2, n_spatial_features)
            fc_in_features += n_spatial_features
        self._fc = torch.nn.Linear(in_features=fc_in_features, out_features=self._num_classes)

    def forward(self, *inputs):
        patches = inputs[0]
        appearance_features = self._resnet(patches)
        if self._spatial_sensitive:
            x_coords = inputs[1].view(-1, 1).float()
            y_coords = inputs[2].view(-1, 1).float()
            coords = torch.cat((x_coords, y_coords), dim=1)
            spatial_features = F.relu(self._spatial_transformer(coords))
            all_features = torch.cat((appearance_features, spatial_features), dim=1)
        else:
            all_features = appearance_features
        return self._fc(all_features)

    def _make_resnet(self):
        assert self._num_layers in (
            18, 34, 50, 101, 152
        ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {self._num_layers}"
        model_constructor = getattr(torchvision.models, f"resnet{self._num_layers}")
        resnet_model = model_constructor()

        if self._pretrain:
            pretrained = model_constructor(pretrained=True).state_dict()
            if self._num_classes != pretrained["fc.weight"].size(0):
                del pretrained["fc.weight"], pretrained["fc.bias"]
            resnet_model.load_state_dict(state_dict=pretrained, strict=False)
        return resnet_model
