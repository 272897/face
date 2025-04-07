"""
@author: JiXuan Xu, Jun Wang
@date: 20201024
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
log_dir = "logs"
import os
os.makedirs(log_dir, exist_ok=True)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "face_sdk"))
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

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('face_sdk/config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    # common setting for all models, need not modify.
    model_path = 'models'

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]    
    logger.info('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face recognition model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')
    image_path = 'face_sdk/api_usage/test_images/test1.jpg'
    
    from PIL import Image

    try:
        img = Image.open(image_path)
        logger.info("PIL wczytał obraz poprawnie!")
    except Exception as e:
        logger.error(f"PIL nie może otworzyć obrazu: {e}")
    # read image and get face features.
    
    if os.path.exists(image_path):
        logger.info(f"Plik istnieje: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                logger.info(f"Plik {image_path} otwarty poprawnie.")
        except Exception as e:
            logger.error(f"Błąd przy otwieraniu pliku: {e}")
    else:
        logger.error(f"Plik NIE istnieje: {image_path}")
    image = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)
    
    if image is None:
        logger.error("Błąd: Nie można wczytać obrazu! Sprawdź ścieżkę do pliku.")
        sys.exit(-1)
    face_cropper = FaceRecImageCropper()
    print("1️⃣ Obraz wczytany poprawnie. Przekazuję do faceDetModelHandler.inference_on_image()")
    
    try:
        dets = faceDetModelHandler.inference_on_image(image)
        print("2️⃣ Wynik detekcji twarzy:", dets)
        face_nums = dets.shape[0]
        for det in dets:
            x1, y1, x2, y2, conf = det  # Współrzędne twarzy
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Jeśli jest więcej niż jedna twarz, zrób to dla każdej
        for i in range(dets.shape[0]):
            # Dla każdej wykrytej twarzy - znajdź landmarki
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
            print("land",i,landmarks)
            # Rysowanie landmarków na obrazie
            for (x, y) in landmarks.astype(np.int32):
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Czerwone kropki na landmarkach

        # Wyświetl obraz
        cv2.imshow("Landmarki", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if face_nums != 2:
            logger.info('Input image should contain two faces to compute similarity!')
        
        feature_list = []
        for i in range(face_nums):
            print(f"3️⃣ Rozpoczynam przetwarzanie twarzy {i+1}")
            print(f"📏 Rozmiar obrazu: {image.shape}")
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
            print(f"4️⃣ Landmarki dla twarzy {i+1}:", landmarks)
            
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
            print(f"5️⃣ Przycięty obraz twarzy {i+1} gotowy.")
            
            feature = faceRecModelHandler.inference_on_image(cropped_image)
            print(f"6️⃣ Wektor cech dla twarzy {i+1}:", feature)

            feature_list.append(feature)

        score = np.dot(feature_list[0], feature_list[1])
        logger.info('The similarity score of two faces: %f' % score)

    except Exception as e:
        logger.error('❌ Pipeline failed!')
        logger.error(e)
        sys.exit(-1)

    else:
        logger.info('Success!')
