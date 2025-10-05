import subprocess

import librosa
import numpy as np


def convert_to_wav(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        '-y',
        output_file
    ]
    subprocess.run(command, check=True)


def get_audio_sentiments(file_path: str, duration: float = None, sr: int = 22050,) -> list | None:
    """
    Извлекает признаки для классификации музыкального настроения из аудиофайла.

    Параметры:
        file_path (str): Путь к аудиофайлу

    Возвращает:
        np.ndarray: Нормализованный вектор признаков (1D)
    """
    try:
        # 1. Загрузка аудио --------------------------------------------------------
        audio, _ = librosa.load(file_path, duration=duration, sr=sr)
        # Использование 30 секунд обеспечивает одинаковую длину для всех треков

        # 2. Извлечение признаков ---------------------------------------------------

        # Темп (BPM) $ ------------------------------------------------------------
        # Темп (beats per minute) - критичен для определения энергии трека
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # Явное преобразование в скаляр (librosa иногда возвращает массив)
        tempo = np.array([tempo]).item() if isinstance(tempo, np.ndarray) else tempo

        # Гармонические/ударные компоненты --------------------------------------
        # Разделение на гармоническую (мелодия) и ударную (ритм) части $
        harmonic, percussive = librosa.effects.hpss(audio)
        # Энергия гармонической составляющей (средний квадрат амплитуды)
        harmonic_energy = np.mean(harmonic ** 2)
        # Энергия ударной составляющей (например, барабаны)
        percussive_energy = np.mean(percussive ** 2)

        # MFCC (Mel-Frequency Cepstral Coefficients) $ ----------------------------
        # 20 коэффициентов MFCC (расширено для лучшего описания тембра)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Средние значения по времени для каждого коэффициента (форма: (20,))
        mfccs_mean = mfccs.mean(axis=1).flatten()
        # Стандартное отклонение по времени (показывает изменчивость тембра)
        mfccs_std = mfccs.std(axis=1).flatten()

        # Chroma CENS $ ----------------------------------------------------------
        # Chroma Energy Normalized (CENS) - устойчив к изменениям громкости
        chroma = librosa.feature.chroma_cens(y=audio, sr=sr)
        # Средние значения по 12 хроматическим нотам (форма: (12,))
        chroma_mean = chroma.mean(axis=1).flatten()
        # Максимальное значение chroma (уверенность в доминирующей тональности)
        key_score = np.array([chroma.max()]).flatten()

        # Динамика RMS $+-(Root Mean Square Energy) -------------------------------
        # RMS - мера громкости сигнала
        rmse = librosa.feature.rms(y=audio)
        # Средняя громкость
        # rmse_mean = rmse.mean() не берем
        # Стандартное отклонение громкости (показывает динамические изменения)
        rmse_std = rmse.std()

        # Спектральные характеристики ------------------------------------------

        # Спектральный центроид $---------------------------------------------------
        # "Центр тяжести" спектра (показатель яркости звука)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = spectral_centroid.mean()  # Среднее значение
        spectral_centroid_std = spectral_centroid.std()  # Изменчивость яркости

        # Спектральная ширина $-----------------------------------------------------
        # Ширина распределения энергии вокруг центроида
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = spectral_bandwidth.mean()  # Средняя ширина
        spectral_bandwidth_std = spectral_bandwidth.std()  # Изменчивость ширины

        # Спектральный контраст $---------------------------------------------------
        # Разница между пиками и впадинами в частотных полосах
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = spectral_contrast.mean(axis=1).flatten()  # Средний контраст
        spectral_contrast_std = spectral_contrast.std(axis=1).flatten()  # Изменчивость контраста

        # Формирование вектора признаков ----------------------------------------
        feature_vectors = [
            # Темп (1 признак)
            np.array([tempo]).flatten(),  # [tempo]

            # Энергии гармоник/ударных (2 признака)
            np.array([harmonic_energy, percussive_energy]),  # [harm_energy, perc_energy]

            # Уверенность в тональности (1 признак)
            key_score.flatten(),  # [key_score]

            # Динамика RMS (2 признака)
            np.array([rmse_std]),  # [rmse_std]

            # MFCC (20 + 20 = 40 признаков)
            mfccs_mean,  # 20 mean
            mfccs_std,  # 20 std

            # Chroma (12 признаков)
            chroma_mean,  # 12 chroma bins

            # Спектральные характеристики (2+2+7+7=18 признаков)
            np.array([spectral_centroid_mean, spectral_centroid_std]),  # centroid (mean, std)
            np.array([spectral_bandwidth_mean, spectral_bandwidth_std]),  # bandwidth (mean, std)
            spectral_contrast_mean,  # 7 contrast means
            spectral_contrast_std  # 7 contrast stds
        ]

        # Гарантия одномерности всех массивов ----------------------------------
        # Преобразуем все массивы в строго 1D с помощью ravel()
        feature_vectors = [arr.ravel() for arr in feature_vectors]

        # Проверка размерностей (отладочный код) -------------------------------
        for i, vec in enumerate(feature_vectors):
            if vec.ndim != 1:
                raise ValueError(f"Ошибка размерности: Признак #{i} имеет {vec.ndim} измерений. Форма: {vec.shape}")

        # Объединение всех признаков в один вектор -----------------------------
        feature_vector = np.concatenate(feature_vectors)
    except Exception as e:
        return None

    return feature_vector.flatten().tolist()  # Возвращаем 1D вектор