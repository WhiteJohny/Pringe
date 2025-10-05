import cv2
from transformers import pipeline
from PIL import Image
import logging
import time


logging.getLogger("transformers").setLevel(logging.ERROR)


class VideoEmotionAnalyzer:
    def __init__(self):
        """Инициализация модели для распознавания эмоций"""
        self.model_name = "trpakov/vit-face-expression"

        self.emotion_mapping = {
            'angry': {'text': 'Angry', 'color': (0, 0, 255)},
            'disgust': {'text': 'Disgust', 'color': (0, 128, 128)},
            'fear': {'text': 'Fear', 'color': (128, 0, 128)},
            'happy': {'text': 'Happy', 'color': (0, 255, 0)},
            'sad': {'text': 'Sad', 'color': (255, 128, 0)},
            'surprise': {'text': 'Surprise', 'color': (255, 255, 0)},
            'neutral': {'text': 'Neutral', 'color': (128, 128, 128)}
        }

        self.emotion_cache = {}
        self.last_processing_time = 0
        self.processing_interval = 2

        self.face_cascade = None
        self.emotion_classifier = None
        self._load_models()

    def _load_models(self):
        """Загрузка моделей для детекции лиц и классификации эмоций"""
        try:
            print("🔄 Loading models...")

            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            if self.face_cascade.empty():
                print("❌ Failed to load OpenCV face detector")
                return False

            print("📦 Loading emotion model...")
            self.emotion_classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=-1
            )

            print("✅ Models loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def detect_faces(self, frame):
        """Обнаружение лиц в кадре"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(64, 64),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def analyze_emotion(self, face_roi):
        """Анализ эмоции на обнаруженном лице"""
        try:
            if face_roi.shape[0] < 64 or face_roi.shape[1] < 64:
                return None, 0

            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            results = self.emotion_classifier(pil_image)

            if results:
                top_emotion = results[0]
                emotion_label = top_emotion['label'].lower()
                confidence = top_emotion['score']

                print(f"🔍 Detected: '{emotion_label}' ({confidence:.2f})")
                return emotion_label, confidence

            return None, 0

        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return None, 0

    def get_emotion_display(self, emotion_label):
        """Получение отображаемой информации об эмоции"""
        if emotion_label in self.emotion_mapping:
            return self.emotion_mapping[emotion_label]

        return {'text': 'Unknown', 'color': (255, 255, 255)}

    def process_webcam(self):
        """Анализ эмоций в реальном времени с веб-камеры"""
        print("🎥 Starting webcam...")
        print("💡 Press 'q' to exit")
        print("💡 Emotions update every 2 seconds for stability")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        frame_count = 0
        self.last_processing_time = time.time()
        self.emotion_cache.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame from webcam")
                break

            frame_count += 1

            faces = self.detect_faces(frame)

            current_time = time.time()
            should_process_emotions = (current_time - self.last_processing_time) >= self.processing_interval

            if should_process_emotions and len(faces) > 0:
                print(f"🔄 Processing emotions for {len(faces)} faces...")
                self.emotion_cache.clear()
                self.last_processing_time = current_time

                for i, (x, y, w, h) in enumerate(faces):
                    padding = 15
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    face_roi = frame[y1:y2, x1:x2]

                    if face_roi.size == 0:
                        continue

                    emotion_label, confidence = self.analyze_emotion(face_roi)

                    if emotion_label and confidence > 0.6:
                        emotion_info = self.get_emotion_display(emotion_label)
                        self.emotion_cache[i] = {
                            'emotion': emotion_info['text'],
                            'confidence': confidence,
                            'color': emotion_info['color']
                        }

            for i, (x, y, w, h) in enumerate(faces):
                if i in self.emotion_cache:
                    emotion_data = self.emotion_cache[i]
                    emotion_text = f"{emotion_data['emotion']} ({emotion_data['confidence']:.2f})"
                    color = emotion_data['color']
                else:
                    emotion_text = "Analyzing..."
                    color = (255, 255, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)

                cv2.putText(frame, emotion_text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            info_text = f"Faces: {len(faces)} | Frame: {frame_count} | Q to quit"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            time_until_update = max(0, self.processing_interval - (current_time - self.last_processing_time))
            update_text = f"Next emotion update in: {time_until_update:.1f}s"
            cv2.putText(frame, update_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Webcam Emotion Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Главная функция"""
    print("""
╔══════════════════════════════════════════════╗
║           VIDEO EMOTION ANALYSIS             ║
║           Real-time Emotion Detection        ║
╚══════════════════════════════════════════════╝
    """)

    analyzer = VideoEmotionAnalyzer()

    while True:
        print("\n🎯 SELECT MODE:")
        print("1. Webcam analysis (real-time)")
        print("2. Exit")

        choice = input("\nEnter choice (1-2) > ").strip()

        if choice == '1':
            analyzer.process_webcam()
        elif choice == '2':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
