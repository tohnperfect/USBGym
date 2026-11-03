"""nnU-Net v2 trainer variants for the v2.1 pluggability study (EXPLORATORY). Reached only through
nnUNet_extTrainer=<this directory> (scripts/plug/train_nnunet_plug.py sets it); nothing in nnunetv2 is
edited. Each class changes the schedule length only; network, loss, augmentation, sampling, optimizer
and the poly learning-rate schedule are nnU-Net's defaults.

  nnUNetTrainer_USBGsmoke    1 epoch of 5 training and 2 validation iterations (smoke tests only)
  nnUNetTrainer_USBGtiming   1 epoch of 40 training and 5 validation iterations (cost measurement only)
  nnUNetTrainer_USBGbudget   as many 250-iteration epochs as needed to see the v2 U-Net's sample budget
                             (4000 iterations x batch 8 = 32,000 frames) at nnU-Net's planned batch size,
                             rounded up; the final-iterate checkpoint is used, like the v2 U-Net
The 20- and 100-epoch schedules use nnunetv2's own nnUNetTrainer_20epochs / nnUNetTrainer_100epochs.
"""
import math

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

V2_SAMPLE_BUDGET = 4000 * 8


class nnUNetTrainer_USBGsmoke(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2


class nnUNetTrainer_USBGbudget(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        bs = int(self.configuration_manager.batch_size)
        self.num_epochs = max(1, math.ceil(V2_SAMPLE_BUDGET / (self.num_iterations_per_epoch * bs)))


class nnUNetTrainer_USBGtiming(nnUNetTrainer):
    """Cost measurement only: 1 epoch of 40 training and 5 validation iterations."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 40
        self.num_val_iterations_per_epoch = 5
