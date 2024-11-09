import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

FACE_CLASSIFIER = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
CLASSIFIER = load_model('model.h5')


class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")

        self.video_capture = cv2.VideoCapture(0)

        self.video_label = ttk.Label(root)
        self.video_label.pack(pady=10)

        self.emotion_label = ttk.Label(root, text="", font=("Helvetica", 24, "bold"), foreground="white", background="black")
        self.emotion_label.pack(pady=10)

        self.quit_button = ttk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

        self.update()

    def quit_app(self):
        self.video_capture.release()
        self.root.destroy()

    def detect_emotion(self, roi_gray):
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = CLASSIFIER.predict(roi)[0]
            label = EMOTION_LABELS[prediction.argmax()]
            return label
        else:
            return "No Faces Detected"

    def update(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CLASSIFIER.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            label = self.detect_emotion(roi_gray)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            self.emotion_label.config(text=f"Detected Emotion: {label}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        self.video_label.config(image=img)
        self.video_label.image = img

        self.root.after(10, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
    cv2.destroyAllWindows()
