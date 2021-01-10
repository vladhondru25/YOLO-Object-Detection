import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


ANCHORS = [(12,16), (19,36), (40,28), (36,75), (76,55), (72,146), (142,110), (192,243), (459,401)]