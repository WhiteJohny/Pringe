import os
import time
import logging

from datetime import datetime
from langfuse import Langfuse
from dotenv import load_dotenv

from calendar_agent import CalendarAgent
from joker_agent import JokerAgent

load_dotenv()

logging.basicConfig(
    filename='calendar_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DoubleAgent(CalendarAgent, JokerAgent):
    def __init__(self, model="deepseek-r1:8b"):
        super().__init__(model)
        self.model = model
        self.conversation_history = []
        self.max_history_length = 5
        self.system_prompt = self._build_system_prompt()
        self.joke_apis = {
            "jokeapi": "https://v2.jokeapi.dev/joke/Any?type=single&safe-mode",
            "official_joke": "https://official-joke-api.appspot.com/random_joke"
        }

        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            base_url=os.getenv("LANGFUSE_BASE_URL")
        )

    
    def _build_system_prompt(self):
        """Генерирует системный промпт с актуальными датой/временем."""
        now = datetime.now()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")

        system_prompt = f"""
            Роль
            Ты — интеллектуальный ассистент с доступом к API календаря. Твоя задача — обрабатывать запросы пользователя, чётко разделяя:
            - операции с календарём (просмотр, создание, редактирование событий);
            - обычное общение.


            Текущее время и дата (используй для обработки запросов):
            - Дата: {current_date}
            - Время: {current_time}
            - День недели: {day_of_week}
            - Полное время: {current_datetime}

            Основные функции
            1. Показывать предстоящие события (с учётом текущей даты и времени).
            2. Создавать новые события в календаре (с валидацией времени).
            3. Искать события в календаре по названию.
            4. Удалять события по ID.
            4. Поддерживать естественный диалог по несвязанным с календарём темам.
            5. Показывать предстоящие события и брать оттуда информацию


            Правила обработки запросов
            1. Распознавание календарных интентов
            - Триггеры: «встреча», «событие», «напомни», «запланируй», «когда», «расписание», «свободное время» и аналогичные.
            - Для относительных временных указателей («завтра», «через неделю») конвертируй в абсолютные значения, используя текущую дату {current_date}.
            - При неоднозначности задавай уточняющий вопрос: «Вы хотите добавить событие в календарь?»

            2. Работа с датой и временем
            - Всегда используй актуальные значения выше для расчётов.
            - Если время не указано, уточни его у пользователя.
            - Указывай часовой пояс, если это критично.

            3. Формат ответа
            - Для календарных операций: возвращай JSON‑структуру (только при явном/неявном запросе к календарю).
            - Для обычного общения: отвечай текстом в естественном стиле.

            4. Обработка недостающих данных
            - Нет времени → «На какое время запланировать событие?»
            - Нет названия → «Как назовём это событие?»

            5. Если достаточно информации, не нужно переспрашивать о выполнении дейсвтия, если это не название, время начала и дата. Просто выполни его.

            6. Не нужно заставлять пользователя вводить какие-то команды, общайся с ним, как при живом общении.


            Формат JSON для календарных операций get_events|create_event|unknown"
            {{
                "action": "get_events|create_event|unknown",
                "parameters": {{
                    "max_results": 10,
                    "summary": "Название события (обязательно)",
                    "start_time": "YYYY-MM-DDTHH:MM:SS",
                    "end_time": "YYYY-MM-DDTHH:MM:SS",
                    "description": "Описание события (опционально)"
                }}
            }}
            Формат JSON для календарных операций search_events_by_title
            {{
                "action": "search_events_by_title",
                "parameters": {{
                    "title": "Название события (обязательно)",
                    "max_results": 10
                }}
            }}
            Формат JSON для календарных операций delete_event
            {{
                "action": "delete_event",
                "parameters": {{
                    "event_id": "id"
                }}
            }}

            Также ты и веселый ассистент с доступом к базе шуток. Твои задачи:

            1. Если пользователь просит шутку, рассказывать анекдот или шутку
            2. Определять тему шутки по контексту (программирование, животные, наука и т.д.)
            3. В остальных случаях отвечать как обычный полезный ассистент

            Формат ответа со шуткой:
            "🎭 [Шутка]"

            Если шутка не сгенерирована тобой, а получена из API, можешь добавить комментарий.
        """
        return system_prompt

    
    def process_query(self, user_query):
        """Обрабатывает запрос пользователя, вызывает LLM и выполняет действие."""
        messages = self.get_contextual_messages(user_query)
        
        try:
            start_time = time.time()
            with self.langfuse.start_as_current_observation(as_type='span', name="process_query") as trace:
                logging.info(
                    "process_query_start"
                )

                trace.update(
                    metadata={
                        "user_query": user_query,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Метрика: сложность пользовательского запроса
                query_complexity = self._calculate_query_complexity(user_query)
                trace.score(
                    name="query_complexity",
                    value=query_complexity,
                    data_type="NUMERIC",
                    comment=f"Сложность запроса: {query_complexity}"
                )
                
                if self.detect_joke_request(user_query):
                    joke_start_time = time.time()

                    logging.info(
                        f"joke_request_detected - {user_query}"
                    )

                    with trace.start_as_current_observation(as_type='generation', name="joke-processing") as observation:
                        logging.info(
                            "joke_api_request"
                        )
                        
                        # Получаем шутку из API
                        api_joke = self.get_joke_from_api()

                        logging.info(
                            f"joke_api_response - success: {bool(api_joke)}"
                        )
                        
                        # Метрика: успешность получения шутки
                        joke_success = 1 if api_joke else 0
                        observation.score(
                            name="joke_api_success",
                            value=joke_success,
                            data_type="NUMERIC",
                            comment=f"Успешно получена шутка: {bool(api_joke)}"
                        )
                        
                        query = (
                            f"Пользователь попросил шутку. Я уже получил шутку из API: '{api_joke}'. "
                            "Добавь краткий веселый комментарий к этой шутке (1-2 предложения)."
                        )
                        messages = self.get_contextual_messages(query)

                        logging.info(
                            f"llm_request - model: {self.model} - query_type: joke_comment"
                        )

                        llm_response = self._call_ollama(messages)
                        llm_comment = llm_response['message']['content']

                        logging.info(
                            f"llm_response - response_length: {len(llm_comment)} - response_time: {llm_response.get('response_time', 0)}"
                        )
                        
                        # Метрика: время ответа модели
                        if 'response_time' in llm_response:
                            observation.score(
                                name="llm_response_time",
                                value=llm_response['response_time'],
                                data_type="NUMERIC",
                                comment=f"Время ответа LLM: {llm_response['response_time']:.2f} сек"
                            )
                        
                        # Метрика: длина ответа модели
                        observation.score(
                            name="llm_response_length",
                            value=len(llm_comment),
                            data_type="NUMERIC",
                            comment=f"Длина ответа LLM: {len(llm_comment)} символов"
                        )
                        
                        observation.update(
                            input={"joke": api_joke},
                            output={"comment": llm_comment}
                        )
                        
                        result = f"🎭 Вот шутка для тебя:\n\n{api_joke}\n\n💬 {llm_comment}"
                        self._update_conversation_history(user_query, result)

                        logging.info(
                            f"joke_processing_complete total_time: {time.time() - joke_start_time}"
                        )

                        return result
                
                logging.info(
                    "general_query_processing - query_type: general"
                )
                
                with trace.start_as_current_observation(as_type='generation', name="llm-response") as observation:
                    logging.info(
                        f"llm_request - model: {self.model} - query_type: general"
                    )

                    response = self._call_ollama(messages)
                    result = response['message']['content']

                    logging.info(
                        f"llm_response - response_length: {len(result)} - response_time: {response.get('response_time', 0)}"
                    )
                    
                    # Метрика: длина ответа модели
                    observation.score(
                        name="llm_response_length",
                        value=len(result),
                        data_type="NUMERIC",
                        comment=f"Длина ответа LLM: {len(result)} символов"
                    )
                    
                    # Метрика: время ответа модели
                    if 'response_time' in response:
                        observation.score(
                            name="llm_response_time",
                            value=response['response_time'],
                            data_type="NUMERIC",
                            comment=f"Время ответа LLM: {response['response_time']:.2f} сек"
                        )
                    
                    observation.update(
                        input={"query": user_query},
                        output={"response": result}
                    )
                    
                    self._update_conversation_history(user_query, result)
                    
                    action_data = self._extract_json_from_response(result)
                    if not action_data:
                        logging.info(
                            f"query_completed_no_action - response_type: text_only"
                        )

                        # Метрика: общее качество обработки
                        trace.score_trace(
                            name="overall_quality",
                            value=1.0,
                            data_type="NUMERIC",
                            comment="Успешная обработка без действий"
                        )
                        return result
                    
                    # Определяем тип действия
                    action_type = action_data.get('action', 'unknown')

                    logging.info(
                        f"action_{action_type}_start - action_type: {action_type}"
                    )
                    
                    with trace.start_as_current_observation(as_type='span', name=f"execute-{action_type}") as action_span:
                        action_start_time = time.time()
                        action_result = self._execute_action(action_data)
                        action_time = time.time() - action_start_time
                        
                        # Метрика: время выполнения операции
                        action_span.score(
                            name="execution_time",
                            value=action_time,
                            data_type="NUMERIC",
                            comment=f"Время выполнения {action_type}: {action_time:.2f} сек"
                        )
                        
                        # Метрика: успешность операции
                        success_value = 1 if action_result != "Неизвестное действие" else 0
                        action_span.score(
                            name="operation_success",
                            value=success_value,
                            data_type="NUMERIC",
                            comment=f"Успешность {action_type}: {success_value}"
                        )
                        
                        action_span.update(
                            input={"action_data": action_data},
                            output={"result": action_result}
                        )
                        
                        # Метрика: общее качество обработки
                        total_time = time.time() - start_time
                        trace.score_trace(
                            name="overall_quality",
                            value=self._calculate_overall_quality(action_result, total_time),
                            data_type="NUMERIC",
                            comment="Общее качество обработки запроса"
                        )

                        logging.info(
                            f"action_{action_type}_complete - success: {success_value} - execution_time: {action_time}"
                        )
                        
                        return action_result

        except Exception as e:
            logging.error(
                f"processing_error - error_type: {type(e).__name__} - error_message: {str(e)}",
            )

            # Метрика: ошибка обработки
            self.langfuse.score_current_trace(
                name="processing_error",
                value=1,
                data_type="NUMERIC",
                comment=f"Ошибка обработки: {str(e)[:100]}"
            )
            print(e)
            return f"❌ Ошибка при обработке запроса: {str(e)}"


def main(agent):
    """Интерактивный чат с агентом."""
    try:
        if agent.langfuse.auth_check():
            print("✅ Langfuse успешно подключён!")
        else:
            print("❌ Ошибка аутентификации")
            return
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return

    print("🤖 Веселый & Календарный агент запущен! (для выхода введите 'выход' или 'quit')")
    print("Просто напиши что-нибудь, а если хочешь шутку - попроси!")
    print("Или")
    print("Доступные команды:")
    print("- 'покажи мои события'")
    print("- 'создай событие ...'")
    print("- 'какие встречи на неделе?'")
    print("- 'Поиск события название - события'")
    print("- 'удали событие id - события'")
    print()

    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            if user_input.lower() in ['выход', 'quit', 'exit']:
                print("До свидания!")
                break
            if not user_input:
                continue

            print("🤖 Агент: ", end="")
            response = agent.process_query(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        
        except Exception as e:
            print(f"❌ Произошла ошибка: {str(e)}")


# Запуск агента
if __name__ == "__main__":
    agent = DoubleAgent()
    main(agent)
