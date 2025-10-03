import sys
import logging
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class LargeLanguageModelCLI:
    def __init__(self):
        """Инициализация LLM"""
        self.model_name = "google/gemma-2-2b-it"
        self.tokenizer = None
        self.model = None
        self.config = None
        self._load_model()

    def _load_model(self):
        """Загрузка модели с обработкой ошибок"""
        try:
            print("🔄 Loading LLM...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            print("✅ Model loaded successfully!")
            print("🌍 This model supports multiple languages including Russian")
            print(f"📋 Type your message or 'quit' to exit.\n")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try installing required packages: pip install -r requirements.txt")
            sys.exit(1)

    def get_reply(self, text: str):
        """Обработка сообщения и вывод ответа"""
        if not text.strip():
            return {"error": "Empty text"}

        try:
            messages = [
                {"role": "user", "content": text}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=256
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "reply": generated_text,
                "success": True
            }

        except Exception as e:
            print(e)
            return {"error": e, "success": False}

    @staticmethod
    def print_help():
        """Печать справки"""
        print(
            "📖 **HOW TO USE:**\n"
            "- Enter any text in English or Russian to get an answer from the model\n"
            "- Commands:\n"
            "  • 'quit', 'exit', 'q' - exit program\n"
            "  • 'help' - show this help\n"
            "  • 'clear' - clear screen"
        )

    def run(self):
        """Запуск основного цикла CLI"""
        self.print_help()

        while True:
            try:
                user_input = input("\n📝 Enter your message > ").strip()

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

                print("⏳ Thinking...")
                result = self.get_reply(user_input)

                if result.get("success"):
                    print(result.get("reply"))
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


def main():
    """Точка входа в программу"""
    try:
        cli = LargeLanguageModelCLI()
        cli.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
