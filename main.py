import sys
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class SentimentCLI:
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
            print("🌍 This model supports multiple languages including Russian")
            print(f"📋 Available for analysis. Type your text or 'quit' to exit.\n")

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
            return {"error": "Empty text"}

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
                    "score": round(float(score), 4),
                    "percentage": round(float(score) * 100, 2)
                })

            return {
                "emotions": emotions,
                "top_emotion": emotions[0],
                "success": True
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def print_banner(self):
        """Печать баннера при запуске"""
        banner = """
╔══════════════════════════════════════════════╗
║         MULTILINGUAL SENTIMENT ANALYSIS      ║
║     Powered by XLM-RoBERTa Multilingual      ║
╚══════════════════════════════════════════════╝
        """
        print(banner)

    def print_help(self):
        """Печать справки"""
        help_text = """
📖 **HOW TO USE:**
- Enter any text in English or Russian to analyze
- Commands:
  • 'quit', 'exit', 'q' - exit program
  • 'help' - show this help
  • 'clear' - clear screen

🎯 **SENTIMENT LABELS:**
😊 Positive - Positive emotion
😐 Neutral  - Neutral/no strong emotion  
😠 Negative - Negative emotion

💡 **EXAMPLES (English):**
  "I love this product!"
  "This is terrible"
  "The weather is okay today"

💡 **EXAMPLES (Russian):**
  "Это просто прекрасно!"
  "Ужасное качество"
  "Нормально, ничего особенного"
        """
        print(help_text)

    def run(self):
        """Запуск основного цикла CLI"""
        self.print_banner()
        self.print_help()

        while True:
            try:
                user_input = input("\n📝 Enter text to analyze > ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                elif user_input.lower() == 'clear':
                    print("\n" * 50)
                    continue
                elif not user_input:
                    print("❌ Please enter some text")
                    continue

                print("⏳ Analyzing...")
                result = self.analyze_text(user_input)

                if result.get("success"):
                    print(f"\n🎯 ANALYSIS RESULTS:")
                    print(f"📝 Text: '{user_input}'")
                    print(
                        f"🏆 Top emotion: {self.get_emoji(result['top_emotion']['label'])} "
                        f"{result['top_emotion']['label']} "
                        f"({result['top_emotion']['percentage']}%)"
                    )

                    print(f"\n📊 DETAILED BREAKDOWN:")
                    print("-" * 40)
                    for i, emotion in enumerate(result['emotions'], 1):
                        emoji = self.get_emoji(emotion['label'])
                        print(
                            f"{i:2d}. {emoji} {emotion['label']:<10} "
                            f"{emotion['percentage']:>6}%  "
                            f"(score: {emotion['score']:.4f})"
                        )

                else:
                    print(f"❌ Analysis error: {result.get('error', 'Unknown error')}")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

    def get_emoji(self, label):
        """Возвращает эмодзи для метки эмоции"""
        emoji_map = {
            "positive": "😊",
            "negative": "😠",
            "neutral": "😐"
        }
        return emoji_map.get(label.lower(), "❓")


def main():
    """Точка входа в программу"""
    try:
        cli = SentimentCLI()
        cli.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
