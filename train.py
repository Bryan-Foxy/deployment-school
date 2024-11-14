from ultralytics import YOLO

"""
Perfom prediction with YOLO8
"""

def load_model(yolo_path = "yolov8n-seg.pt"):
    model = YOLO(yolo_path)
    return model

if __name__ == "__main__":
    model = load_model()
    # Training
    model.train(
    data = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/Chip_Segmentation.v1i.yolov8/data.yaml",  
    epochs = 25,          
    imgsz = 640,           
    batch = 16,           
    device = 'mps'
)
