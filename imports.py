apt -yq install swig >/dev/null
pip -q install "gymnasium[box2d]" "torch" "torchvision" "opencv-python-headless" "imageio" "imageio-ffmpeg" "tqdm" "einops""numpy"

import os, math, random, time, imageio, numpy as np
import gymnasium as gym
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from tqdm.auto import trange
import cv2