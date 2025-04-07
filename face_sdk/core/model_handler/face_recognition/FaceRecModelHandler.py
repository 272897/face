"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import logging.config
import os

from face_sdk.utils.BuzException import FaseChannelError, FalseImageSizeError, InputError

# Tworzenie katalogu 'logs', jeśli nie istnieje
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/sdk.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('api')

import numpy as np
import torch

from face_sdk.core.model_handler.BaseModelHandler import BaseModelHandler
from face_sdk.utils.BuzException import *

class FaceRecModelHandler(BaseModelHandler):
    """Implementation of face recognition model handler.

    This version flattens the output (C,H,W) to a single 1D vector (C*H*W,).
    """

    def __init__(self, model, device, cfg):
        """
        Init FaceRecModelHandler settings.
        """
        super().__init__(model, device, cfg)
        self.mean = self.cfg['mean']
        self.std = self.cfg['std']
        self.input_height = self.cfg['input_height']
        self.input_width = self.cfg['input_width']

    def inference_on_image(self, image):
        """
        Get the inference from the face recognition model.
        We flatten the feature map to a single 1D vector.

        Returns:
            A numpy array of shape (C*H*W,) after flatten.
        """
        try:
            image = self._preprocess(image)
        except Exception as e:
            raise e

        # (B,3,112,112)
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            output = self.model(image)

            # Jeżeli model zwraca krotkę (embedding, coś_innego) → weź pierwszy element
            if isinstance(output, tuple):
                output = output[0]

            # Sprawdzamy wymiar:
            if output.dim() == 4:
                # => (B, C, H, W), spłaszczamy do (B, C*H*W)
                output = output.view(output.size(0), -1)
            elif output.dim() == 3:
                # => (C, H, W), spłaszczamy do (C*H*W)
                output = output.view(-1)

            feature = output.cpu().numpy()
            feature = np.squeeze(feature)  # np. (B, X) → (X,) jeżeli B=1
        return feature

    def _preprocess(self, image):
        """
        Preprocess the input image (BGR to CHW, normalize).
        Expects image of shape (112,112,3).
        """
        if not isinstance(image, np.ndarray):
            logger.error('The input should be an ndarray read by cv2!')
            raise InputError()

        height, width, channels = image.shape
        if height != self.input_height or width != self.input_width:
            raise FalseImageSizeError()

        # Ewentualna korekta wymiarów
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.ndim == 4:
            image = image[:, :, :3]
        if image.ndim > 4:
            raise FaseChannelError(image.ndim)

        # Normalizacja: (pix - mean)/std, transpozycja do (3,112,112)
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image