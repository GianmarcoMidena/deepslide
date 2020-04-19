import logging

import torch

from code.utils import get_classes


class Configurer:
    def __init__(self, args):
        self._args = args

    def with_device(self):
        # Device to use for PyTorch code.
        self._args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self

    def with_classes(self):
        self._args.classes = get_classes(folder=self._args.all_wsi)
        return self

    def with_num_classes(self):
        self._args.num_classes = len(self._args.classes)
        return self

    def build(self):
        self._log_configuration()
        return self._args

    def _log_configuration(self):
        configuration = f"CONFIGURATION:\n"
        for key, value in self._args.__dict__.items():
            configuration += f"{key}: {value}\n"
        logging.info(configuration)
