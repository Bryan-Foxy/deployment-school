import json

def convert_polygon_to_yolo_bbox(points, img_width, img_height):
    # Convertir les points en coordonnées de boîte englobante (x_min, y_min, x_max, y_max)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculer le centre, la largeur et la hauteur de la boîte englobante normalisés
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    
    return x_center, y_center, bbox_width, bbox_height

json_path = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/Void detection on X-ray/annotation_train/02_JPG.rf.d6063f8ca200e543da7becc1bf260ed5.json"
image_width = 640  
image_height = 480  

# Lire le fichier JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Créer un fichier de sortie YOLO
yolo_output_path = json_path.replace('.json', '.txt')
with open(yolo_output_path, 'w') as yolo_file:
    for shape in data['shapes']:
        if shape['label'] == 'object':  
            bbox = convert_polygon_to_yolo_bbox(shape['points'], image_width, image_height)
            yolo_line = f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
            yolo_file.write(yolo_line)

print(f"Annotations YOLO enregistrées dans {yolo_output_path}")