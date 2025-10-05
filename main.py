import os
import cv2

from ultralytics import YOLO

# Параметры модели
# results = model.predict(
#     source='image.jpg',
#     conf=0.25,           # Confidence threshold
#     iou=0.7,             # NMS IoU threshold
#     imgsz=640,           # Размер изображения
#     device='cuda',       # 'cpu' или 'cuda' или 0,1,2,3
#     half=False,          # Использование половинной точности
#     max_det=300,         # Максимальное количество детекций
#     classes=[0, 2],      # Фильтр по классам (0: person, 2: car и т.д.)
#     augment=False,       # Аугментация при инференсе
#     agnostic_nms=False,  # Class-agnostic NMS
#     retina_masks=False,  # Использование retina masks для сегментации
#     show=False,          # Показывать результат
#     save=False,          # Сохранять результат
#     verbose=True         # Вывод информации
# )

# Параметры обучения
# model.train(
#     data='coco128.yaml',
#     epochs=100,
#     patience=50,          # Ранняя остановка
#     batch=16,
#     imgsz=640,
#     save=True,
#     save_period=-1,
#     cache=False,
#     device=0,
#     workers=8,
#     project='runs/detect',
#     name='exp',
#     exist_ok=False,
#     pretrained=True,
#     optimizer='SGD',      # SGD, Adam, AdamW, RMSprop
#     lr0=0.01,             # Начальный LR
#     lrf=0.01,             # Финальный LR
#     momentum=0.937,
#     weight_decay=0.0005,
#     warmup_epochs=3.0,
#     warmup_momentum=0.8,
#     box=7.5,              # Weight of box loss
#     cls=0.5,              # Weight of cls loss
#     dfl=1.5,              # Weight of dfl loss
#     pose=12.0,            # Weight of pose loss (pose)
#     kobj=1.0,             # Weight of keypoint obj loss (pose)
#     label_smoothing=0.0,
#     nbs=64,               # Nominal batch size
#     overlap_mask=True,
#     mask_ratio=4,
#     dropout=0.0,
#     val=True,             # Validate during training
# )

# Параметры валидации
# model.val(
#     data='coco128.yaml',
#     batch=16,
#     imgsz=640,
#     conf=0.001,
#     iou=0.6,
#     device='cuda',
#     split='val',          # 'val', 'test', or 'train'
#     save_json=False,      # Save results to JSON
#     save_hybrid=False,
#     half=True,
#     dnn=False,
#     plots=True           # Save plots
# )

# Параметры экспорта
# model.export(
#     format='onnx',        # Форматы: torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu,
#                                       tfjs, paddle
#     imgsz=[640, 640],
#     batch=1,              # Batch size
#     device='cpu',
#     half=False,           # FP16 quantization
#     int8=False,           # INT8 quantization
#     dynamic=False,        # Dynamic axes
#     simplify=False,       # ONNX simplify
#     opset=12,            # ONNX opset version
#     workspace=4,
#     nms=False,
# )


def load_model():
    print("Загрузка модели...")
    while True:
        try:
            size = input('Выберите размер модели: nano/small/medium/large/xlarge\n')
            if size.lower() in ("nano", "small", "medium", "large", "xlarge"):
                task = input("Выберите задачу (detection/segmentation): ").lower()
                if task == "segmentation":
                    model = YOLO(f'yolo11{size[0]}-seg.pt')
                else:
                    model = YOLO(f'yolo11{size[0]}.pt')
                print("Модель загружена")
            else:
                print("Такой модели не существует :(")
                continue
            print("Модель распознает следующие образы:")
            print(*model.names.values(), sep="\n")
            return model
        except Exception as e:
            print(f"Ошибка при загрузке модели :(\n{e}")
            return None


def process_image(model):
    while True:
        path = input("q - Назад\nВведите путь до изображения: ")

        if path.lower() == "q":
            break

        if not os.path.exists(path) and os.path.isfile(path):
            print("Файл не найден!")
            continue

        try:
            results = model(path)
            annotated_frame = results[0].plot()

            cv2.imshow('YOLO11', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Возникла ошибка - {e}")


def process_video(model):
    flag = True
    while flag:
        path = input("q - Назад\nВведите путь до изображения: ")

        if path.lower() == "q":
            break

        if not os.path.exists(path) and os.path.isfile(path):
            print("Файл не найден!")
            continue

        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model(frame)
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"Возникла ошибка - {e}")
                continue

            cv2.imshow('YOLO11', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                flag = False
                break

        cap.release()
        cv2.destroyAllWindows()


def process_camera(model):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame)
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"Возникла ошибка - {e}")
            continue

        cv2.imshow('YOLO11 Live', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = load_model()
    if model:
        while True:
            mode = input('Выберите режим:\n1 - Image, 2 - Video, 3 - webcamera\n')
            if mode.lower() == "q":
                print("До новых встреч")
                break
            elif mode == "1":
                process_image(model)
            elif mode == "2":
                process_video(model)
            elif mode == "3":
                process_camera(model)
            else:
                print("Такой режим не существует")


if __name__ == "__main__":
    main()
