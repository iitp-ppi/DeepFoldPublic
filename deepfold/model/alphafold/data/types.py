from typing import Dict, MutableMapping

import numpy as np
import torch

FeatureDict = MutableMapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
