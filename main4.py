import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tkinter as tk
from PIL import Image, ImageTk
import dlib
from time import sleep

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")

        self.video_capture = cv2.VideoCapture(0)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.emotion_frame = tk.Frame(root)
        self.emotion_frame.pack(pady=20)

        self.emotion_label = tk.Label(self.emotion_frame, text="", font=("Helvetica", 16))
        self.emotion_label.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

        self.update()

    def quit_app(self):
        self.video_capture.release()
        self.root.destroy()

    def update(self):
        _, frame = self.video_capture.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.emotion_label.config(text=f"Detected Emotion: {label}")
            else:
                self.emotion_label.config(text="No Faces Detected")

            # Detect face landmarks
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = shape_predictor(gray, dlib_rect)

            # Access detailed facial features through landmarks.points()
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), -1)

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
