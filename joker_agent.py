import requests
import random


class JokerAgent:
    def __init__(self, model="qwen3:4b"):
        self.ollama_url = "http://localhost:11434"
        self.model = model
        self.joke_apis = {
            "jokeapi": "https://v2.jokeapi.dev/joke/Any?type=single&safe-mode",
            "official_joke": "https://official-joke-api.appspot.com/random_joke"
        }

        self.system_prompt = """
        Ты - веселый ассистент с доступом к базе шуток. Твои задачи:

        1. Если пользователь просит шутку, рассказывать анекдот или шутку
        2. Определять тему шутки по контексту (программирование, животные, наука и т.д.)
        3. В остальных случаях отвечать как обычный полезный ассистент

        Формат ответа со шуткой:
        "🎭 [Шутка]"

        Если шутка не сгенерирована тобой, а получена из API, можешь добавить комментарий.
        """

    def get_joke_from_api(self):
        """Получает шутку из случайного API"""
        try:
            api_choice = random.choice(list(self.joke_apis.keys()))
            api_url = self.joke_apis[api_choice]

            if api_choice == "jokeapi":
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("joke", "Не удалось получить шутку")

            elif api_choice == "official_joke":
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    return f"{data['setup']} - {data['punchline']}"

            return "Не удалось получить шутку из API"

        except Exception as e:
            return f"Ошибка при получении шутки: {e}"

    def detect_joke_request(self, message):
        """Определяет, просит ли пользователь шутку"""
        joke_keywords = [
            'шутк', 'анекдот', 'шути', 'рассмеши', 'юмор',
            'смешн', 'прикол', 'joke', 'funny', 'laugh',
            'развесели', 'улыбни', 'поюмори'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in joke_keywords)


    def send_to_ollama(self, message):
        """Отправляет сообщение в OLLAMA"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ],
            "stream": False
        }

        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Ошибка при обращении к OLLAMA: {e}"

    def process_query(self, user_query):
        """Основной метод обработки сообщений"""

        if self.detect_joke_request(user_query):
            api_joke = self.get_joke_from_api()

            response = f"🎭 Вот шутка для тебя:\n\n{api_joke}"

            llm_comment = self.send_to_ollama(
                f"Пользователь попросил шутку. Я уже получил шутку из API: '{api_joke}'. "
                f"Добавь краткий веселый комментарий к этой шутке (1-2 предложения)."
            )

            return f"{response}\n\n💬 {llm_comment}"

        else:
            return self.send_to_ollama(user_query)


def main():
    agent = JokerAgent()

    print("🤖 Веселый ИИ-агент с шутками активирован!")
    print("Просто напиши что-нибудь, а если хочешь шутку - попроси!")
    print("Для выхода введите 'выход' или 'quit'")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['выход', 'quit', 'exit']:
                print("👋 До свидания! Возвращайтесь за новыми шутками!")
                break

            if not user_input:
                continue

            print("\n🤖 Агент: ", end="")
            response = agent.process_query(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
