"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
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

class FaceRecModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face recognition model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['mean'] = self.meta_conf['mean']
        self.cfg['std'] = self.meta_conf['std']
        
    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'], map_location=torch.device('cpu'), weights_only=False)
            if hasattr(model, 'module'):
                model = model.module  # Extract the actual model from DataParallel wrapper
            model = model.to('cpu')  # Ensure the model is on CPU
            model.eval()  # Set the model to evaluation mode
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                        % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face recognition model!')
            return model, self.cfg