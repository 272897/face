import cv2

# Sprawdzenie dostępnych urządzeń kamery
def list_video_devices():
    # Przeszukuje urządzenia od 0 do 10 (możesz zwiększyć zakres, jeśli masz więcej kamer)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera {i} dostępna")
            cap.release()

list_video_devices()