import cv2
import torch 
import supervision as sv
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DEVICE_CPU = "cpu"

print(f"Using device: {DEVICE}")
print(f"Using {DEVICE_CPU} because MPS is not compatible")
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/weights/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint = 
                                     CHECKPOINT_PATH).to(device = DEVICE_CPU)

def read_img(img_path):
    """ Read Images"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_automaticaly_mask(sam, img_path):
    """ Generate mask automaticaly using SAM """
    img = read_img(img_path)
    img2mask = SamAutomaticMaskGenerator(sam)
    sam_result = img2mask.generate(img)
    return sam_result

def imshow_mask(img_path, result_sam):
    """ Imshow Image """
    img = read_img(img_path)
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result = result_sam)
    annotated_img = mask_annotator.annotate(scene = img.copy(), detections = detections)
    sv.plot_images_grid(
        images = [img, annotated_img],
        grid_size = [1, 2],
        titles = ["source image", "segmented image"]
    )

if __name__ == "__main__":
    img_path = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/Void detection on X-ray/train/25_jpg.rf.893f4286e0c8a3cef2efb7612f248147.jpg"
    result = generate_automaticaly_mask(sam, img_path)
    imshow_mask(img_path, result)

