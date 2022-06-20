# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

import json_tricks as json
import numpy as np

from dataset.JointsDataset import JointsDataset



logger = logging.getLogger(__name__)


class TICaMDataset(JointsDataset):
    
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        pass
    
    
    
    
    def evaluate(self):
        pass