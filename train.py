from ultralytics import RTDETR
import warnings
import multiprocessing

warnings.filterwarnings('ignore')


def main():
    model = RTDETR(r"D:/LQ/ultralytics-rtdetrmain/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet50-WTConv-HiLo.yaml")
    model.train(
        data=r'D:\LQ\ultralytics-rtdetrmain\ultralytics-main\DATA6000next\data.yaml',
        epochs=200,
        imgsz=640,
        workers=2,
        batch=4,
        device=0,
        optimizer='AdamW',
        amp=False,
        cache='disk',
        patience=20,
        save_period=10,
        verbose=False,
        lr0=0.0001,
        lrf=0.01
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

