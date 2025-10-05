import os

from dotenv import load_dotenv

from evaluate import load_model, evaluate_mood
from mood_features import get_audio_sentiments, convert_to_wav

SENTIMENTS = ['funny', 'happy', 'sad', 'scary', 'tender', 'trance']


def get_model():
    print("Загрузка модели...")
    load_dotenv()
    model = load_model(os.getenv("MODEL_NAME"))
    print("Модель загружена")
    return model


def get_model_msg(sentiments):
    sentiments = sentiments.split(" ")
    return f'{", ".join([SENTIMENTS[i] for i in range(0, len(SENTIMENTS)) if sentiments[i] == "1"])}'


def main(model):
    while True:
        audio_path = input("Введите путь до аудио: ")
        # audio_path = r"C:\Users\Егор\PycharmProjects\pythonProject1\Zapreshhjonnye_barabanshhiki_-_Ubili_negra_64353318 (1).mp3"
        if not os.path.exists(audio_path) and os.path.isfile(audio_path):
            print("Неправильный путь")
            continue

        print("Конвертация в формат .wav...")
        convert_to_wav(audio_path, "audio.wav")
        print("Конвертация завершена")

        print("Извлечение признаков...")
        features_vector = get_audio_sentiments("audio.wav")
        if features_vector is None:
            print("Не удалось извлечь признаки.")
            continue
        else:
            print("Признаки извлечены")

        print("Определение сантимента...")
        print(get_model_msg(evaluate_mood(model, features_vector)))


if __name__ == "__main__":
    model = get_model()
    main(model)
