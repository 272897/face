"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com 
"""
import logging.config
import os



# Tworzenie katalogu 'logs', jeśli nie istnieje
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Ustawienie podstawowej konfiguracji logowania
logging.basicConfig(
    level=logging.DEBUG,  # Możesz zmienić na INFO, ERROR, itd.
    format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Wyświetlanie logów w konsoli
        logging.FileHandler("logs/sdk.log", encoding="utf-8")  # Zapisywanie do pliku
    ]
)

logger = logging.getLogger('api')

import torch
from face_sdk.core.model_loader.BaseModelLoader import BaseModelLoader

class FaceDetModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        # Ścieżka do katalogu modelu
        model_root_dir = os.path.join("face_sdk", "models")
        meta_file_path = os.path.join(model_root_dir, meta_file)

        logger.info(
            'Start to analyze the face detection model, model path: %s, model category: %s, model name: %s' %
            (model_root_dir, model_category, model_name)
        )

        # Przekazujemy poprawioną ścieżkę do klasy nadrzędnej
        super().__init__(model_root_dir, model_category, model_name, meta_file)

        self.cfg['min_sizes'] = self.meta_conf['min_sizes']
        self.cfg['steps'] = self.meta_conf['steps']
        self.cfg['variance'] = self.meta_conf['variance']
        self.cfg['in_channel'] = self.meta_conf['in_channel']
        self.cfg['out_channel'] = self.meta_conf['out_channel']
        self.cfg['confidence_threshold'] = self.meta_conf['confidence_threshold']

    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'], weights_only=False)
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face detection model!')
            return model, self.cfg
    def selik(self):
        return self.cfg
