import sys
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class SentimentModel:
    def __init__(self):
        """Инициализация многоязычной модели"""
        self.model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = None
        self.model = None
        self.config = None
        self._load_model()

    def _load_model(self):
        """Загрузка модели с обработкой ошибок"""
        try:
            print("🔄 Loading multilingual sentiment analysis model...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try installing required packages: pip install -r requirements.txt")
            sys.exit(1)

    def preprocess(self, text):
        """Предобработка текста (как в вашем примере)"""
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def analyze_text(self, text):
        """Анализ тональности текста с выводом всех эмоций"""
        if not text.strip():
            return {"error": "Empty text", "success": False}

        try:
            processed_text = self.preprocess(text)
            encoded_input = self.tokenizer(processed_text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            ranking = np.argsort(scores)[::-1]

            emotions = []
            for i in range(scores.shape[0]):
                label = self.config.id2label[ranking[i]]
                score = scores[ranking[i]]
                emotions.append({
                    "label": label,
                    "score": round(float(score), 4)
                })

            return {"emotions": emotions, "success": True}

        except Exception as e:
            return {"error": str(e), "success": False}
