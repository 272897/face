import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import sys
import logging
import yaml
import cv2
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "face_sdk"))

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
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

# Load model config
with open('face_sdk/config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

scene = 'non-mask'
model_path = 'models'

logger.info('Loading face detection model...')
try:
    faceDetModelLoader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
    det_model, det_cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(det_model, 'cpu', det_cfg)
    logger.info('Face detection model loaded successfully.')
except Exception as e:
    logger.error('Failed to load face detection model.')
    logger.error(e)
    sys.exit(1)

logger.info('Loading face recognition model...')
try:
    faceRecModelLoader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
    rec_model, rec_cfg = faceRecModelLoader.load_model()
    faceRecModelHandler = FaceRecModelHandler(rec_model, 'cpu', rec_cfg)  # <-- we flatten inside FaceRecModelHandler
    logger.info('Face recognition model loaded successfully.')
except Exception as e:
    logger.error('Failed to load face recognition model.')
    logger.error(e)
    sys.exit(1)


def normalize_embedding(emb):
    """L2-normalizacja wektora embeddingu."""
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection Capture")
        self.root.geometry("640x520")
        self.root.resizable(False, False)

        self.cap = None
        self.frame = None
        self.img_counter = 0

        if not os.path.isdir("tmp"):
            os.mkdir("tmp")

        self.canvas = tk.Label(root)
        self.canvas.place(x=0, y=0, width=640, height=480)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.place(x=0, y=480, width=640, height=40)

        self.cam_selector = ttk.Combobox(self.btn_frame, state="readonly")
        self.cam_selector.pack(side=tk.LEFT, padx=5)
        self.cam_selector.bind("<<ComboboxSelected>>", self.change_camera)

        tk.Button(self.btn_frame, text="Plik wideo", command=self.use_video_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zapisz zdjęcie", command=self.save_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zamknij", command=self.quit_app).pack(side=tk.LEFT, padx=5)

        self.known_embeddings = self.load_tmp_embeddings()

        # Bufor detekcji i stabilności
        self.last_boxes = []
        self.buffer_size = 20
        self.stable_threshold = 10
        self.ready_for_recognition = True
        self.last_recognition_time = 0
        self.recognition_cooldown = 5
        self.last_info_time = 0
        self.info_interval = 3

        self.scan_cameras()
        self.update()

    def scan_cameras(self):
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ret, _ = cap.read()
            if cap.isOpened() and ret:
                available.append(f"Kamera {i}")
            cap.release()
        self.cam_selector["values"] = available
        if available:
            self.cam_selector.current(0)
            index = int(available[0].split(" ")[1])
            self.set_camera(index)

    def set_camera(self, index):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > 0 and height > 0:
            self.root.geometry(f"{width}x{height + 40}")
            self.canvas.place(x=0, y=0, width=width, height=height)
            self.canvas.config(width=width, height=height)
            self.btn_frame.place(x=0, y=height, width=width, height=40)

    def change_camera(self, event):
        selected = self.cam_selector.get()
        if selected.startswith("Kamera"):
            index = int(selected.split(" ")[1])
            self.set_camera(index)

    def use_video_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                self.root.geometry(f"{width}x{height + 40}")
                self.canvas.place(x=0, y=0, width=width, height=height)
                self.canvas.config(width=width, height=height)
                self.btn_frame.place(x=0, y=height, width=width, height=40)

    def save_image(self):
        if self.frame is not None:
            username = simpledialog.askstring("Nazwa użytkownika", "Podaj nazwę użytkownika:")
            if not username:
                print("Nie podano nazwy użytkownika — anulowano zapis.")
                return

            dets = faceDetModelHandler.inference_on_image(self.frame)
            if dets is None or len(dets) == 0:
                print("Nie wykryto twarzy.")
                return

            best_face = max(dets, key=lambda x: x[-1])
            x1, y1, x2, y2 = map(int, best_face[:4])
            h, w = self.frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = self.frame[y1:y2, x1:x2]

            if face.shape[0] < 30 or face.shape[1] < 30:
                print("Twarz za mała, nie zapisano.")
                return

            filename = os.path.join("tmp", f"{username}.png")
            cv2.imwrite(filename, face)
            print(f"✅ Zapisano twarz użytkownika {username} jako {filename}")
            self.known_embeddings = self.load_tmp_embeddings()

    def get_embeddings_from_image(self, image):
        dets = faceDetModelHandler.inference_on_image(image)
        if dets is None or len(dets) == 0:
            return None

        best_face = max(dets, key=lambda x: x[-1])
        x1, y1, x2, y2 = map(int, best_face[:4])
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face = image[y1:y2, x1:x2]

        if face is None or face.size == 0:
            logger.warning("⚠️ Wycięta twarz pusta – pomijam.")
            return None

        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            emb = faceRecModelHandler.inference_on_image(face)
        except Exception as e:
            logger.error("Błąd przy generowaniu embeddingu:", exc_info=True)
            return None

        # Normalizacja L2
        emb = normalize_embedding(emb)
        logger.debug(f"get_embeddings_from_image => shape: {emb.shape}, first5: {emb[:5]}")
        return emb

    def load_tmp_embeddings(self):
        embeddings = {}
        for file in os.listdir("tmp"):
            if file.endswith(".png"):
                path = os.path.join("tmp", file)
                img = cv2.imread(path)
                emb = self.get_embeddings_from_image(img)
                if emb is not None:
                    name = os.path.splitext(file)[0]
                    embeddings[name] = emb
                    logger.info(f"Załadowano embedding {name}: shape={emb.shape}, first5={emb[:5]}")
        return embeddings

    def recognize_face(self, frame, threshold=0.75):
        current_emb = self.get_embeddings_from_image(frame)
        if current_emb is None:
            return None, None

        best_match, best_score = None, 0
        for name, emb in self.known_embeddings.items():
            score = cosine_similarity(
                current_emb.reshape(1, -1),
                emb.reshape(1, -1)
            )[0][0]
            logger.debug(f"Porównanie z {name}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score

    def is_stable_face(self, new_box):
        if new_box is None:
            self.last_boxes.clear()
            return False
        if len(self.last_boxes) < self.buffer_size:
            self.last_boxes.append(new_box)
            return False
        self.last_boxes.pop(0)
        self.last_boxes.append(new_box)
        x_diff = max(abs(b[0] - new_box[0]) for b in self.last_boxes)
        y_diff = max(abs(b[1] - new_box[1]) for b in self.last_boxes)
        w_diff = max(abs((b[2] - b[0]) - (new_box[2] - new_box[0])) for b in self.last_boxes)
        h_diff = max(abs((b[3] - b[1]) - (new_box[3] - new_box[1])) for b in self.last_boxes)
        return all(diff < self.stable_threshold for diff in [x_diff, y_diff, w_diff, h_diff])

    def quit_app(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def update(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                org_frame = frame.copy()
                try:
                    dets = faceDetModelHandler.inference_on_image(frame)
                    for box in dets:
                        _box = list(map(int, box))
                        cv2.rectangle(frame, (_box[0], _box[1]), (_box[2], _box[3]), (0, 0, 255), 2)
                        cv2.putText(frame, "{:0.2f}%".format(box[-1] * 100), (_box[0] + 10, _box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    if len(dets) > 0:
                        box = list(map(int, dets[0][:4]))
                        now = time.time()

                        if self.is_stable_face(box):
                            if self.ready_for_recognition or (
                                now - self.last_recognition_time
                            ) > self.recognition_cooldown:
                                self.ready_for_recognition = False
                                self.last_recognition_time = now

                                name, score = self.recognize_face(org_frame)
                                if name:
                                    print(f"✅ Rozpoznano: {name} ({score * 100:.2f}%)")
                                elif score:
                                    print(f"❌ Nie rozpoznano. Najlepszy wynik: {score * 100:.2f}%")
                            elif (now - self.last_info_time) > self.info_interval:
                                print("⏳ Odczekaj chwilę przed kolejną próbą...")
                                self.last_info_time = now
                        else:
                            self.ready_for_recognition = True
                            if (now - self.last_info_time) > self.info_interval:
                                print("🕒 Czekam na stabilną twarz...")
                                self.last_info_time = now

                except Exception as e:
                    logger.error('Detection failed')
                    logger.error(e)

                self.frame = org_frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.configure(image=imgtk)

        self.root.after(10, self.update)


if __name__ == '__main__':
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()