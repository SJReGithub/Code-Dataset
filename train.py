import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-final.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='F:/YOLO/yolov8-main/dataset/fireandsmoke/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
