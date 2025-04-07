"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import os
import sys
import logging.config
import json
from abc import ABCMeta, abstractmethod

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

sys.path.append(os.path.join('face_sdk', 'models', 'network_def'))  # Poprawiona ścieżka

class BaseModelLoader(metaclass=ABCMeta):
    """Base class for all model loaders.
    All model loaders must inherit this base class, 
    and each new model must implement the "load_model" method.
    """
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        # Poprawiona ścieżka do katalogu modelu
        model_root_dir = os.path.abspath(os.path.join("face_sdk", "models", model_category, model_name))
        meta_file_path = os.path.join(model_root_dir, meta_file)
        self.cfg = {}

        try:
            with open(meta_file_path, 'r') as meta_file:
                self.meta_conf = json.load(meta_file)
        except IOError as e:
            logger.error('The configuration file meta.json was not found or failed to parse the file! Path: %s', meta_file_path)
            raise e
        except Exception as e:
            logger.info('The configuration file format is wrong!')
            raise e
        else:
            logger.info('Successfully parsed the model configuration file meta.json!')

        # Common configs for all models
        self.cfg['model_path'] = model_root_dir
        self.cfg['model_category'] = model_category
        self.cfg['model_name'] = model_name
        self.cfg['model_type'] = self.meta_conf.get('model_type', 'unknown')
        self.cfg['model_info'] = self.meta_conf.get('model_info', {})
        self.cfg['model_file_path'] = os.path.join("face_sdk", "models", "face_detection", "face_detection_1.0", "face_detection_retina.pkl")
        self.cfg['release_date'] = self.meta_conf.get('release_date', 'unknown')
        self.cfg['input_height'] = self.meta_conf.get('input_height', 0)
        self.cfg['input_width'] = self.meta_conf.get('input_width', 0)

    @abstractmethod
    def load_model(self):
        """Should be overridden by all subclasses.
        Different models may have different configuration information,
        such as mean, so each model implements its own loader.
        """
        pass
