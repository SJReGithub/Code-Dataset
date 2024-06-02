import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/RCSEMA-ASFF/weights/best.pt')
    model.val(data='F:/YOLO/yolov8-main/dataset/fireandsmoke/data.yaml',
              split='val',
              imgsz=640,
              batch=32,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )