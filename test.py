import gradio as gr
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt

# Charger le modèle SAM et ses poids
sam_checkpoint = "/Users/armandbryan/Documents/aivancity/PGE5/Deployment AI/Project/weights/sam_vit_h_4b8939.pth"  # Remplacez par le chemin de votre modèle
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Fonction pour la segmentation basée sur un clic et réattribution de classes
def segment_and_reassign_class(image, points, class_label):
    if not points:
        return "Veuillez cliquer sur une région de l'image."
    
    x, y = points[0]
    image_np = np.array(image)
    predictor.set_image(image_np)

    # Point de clic de l'utilisateur
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # Label pour indiquer un point positif

    # Prédire le masque
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    # Annoter et afficher l'image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    for mask in masks:
        ax.imshow(mask, alpha=0.5, cmap="jet")
        # Annoter la région avec la classe sélectionnée
        ax.text(x, y, f"Classe: {class_label}", color="white", fontsize=12)
    ax.axis("off")
    plt.title(f"Segmenté - Classe attribuée : {class_label}")
    plt.show()
    return fig

# Interface Gradio
interface = gr.Interface(
    fn=segment_and_reassign_class,
    inputs=[
        gr.Image(type="pil", tool="point", label="Télécharger une image et cliquez pour sélectionner une région"),
        gr.Dropdown(choices=["Classe 1", "Classe 2", "Classe 3", "Classe 4"], label="Sélectionner une classe")
    ],
    outputs="plot",
    title="Annotation Interactive et Attribution de Classe avec SAM"
)

# Lancer l'interface
interface.launch()