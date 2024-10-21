import torch
import argparse
import numpy as np

from PIL import Image
import matplotlib.pylab as plt
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Resize, Normalize



from src import Range, load_model, tensorboard_runner



def load_model_weights(model, args):
    """Charge les poids du modèle en fonction du dispositif disponible."""
    if torch.cuda.is_available() and args.device == "cuda":
        model.load_state_dict(torch.load(
            'result/DFL_241021_223155/DFL.pt'))
        model.to(args.device)
    else:
        model.load_state_dict(torch.load(
            'result/DFL_241021_223155/DFL.pt', map_location=torch.device('cpu')))

def get_transform(args):
    """Retourne les transformations d'images."""
    return Compose([
        Resize((args.resize, args.resize)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def load_and_transform_images(img_paths, transform, device):
    """Charge et transforme une liste d'images."""
    with torch.no_grad():
        return torch.stack([transform(Image.open(img_path)).to(device) for img_path in img_paths])

def make_predictions(model, validation_batch):
    """Fait des prédictions sur un lot d'images."""
    model.eval()  # Met le modèle en mode évaluation
    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    return np.argmax(pred_probs, axis=1)

def display_results(img_list, predicted_classes, class_names):
    """Affiche les images avec leurs classes prédites, 4 par ligne."""
    num_images = len(img_list)
    num_cols = 5
    # num_rows = (num_images + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axs = axs.flatten()  # Aplatir la grille pour itérer facilement

    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title(f"{class_names[predicted_classes[i]]}")
        ax.imshow(img)

    # Désactiver les axes pour les sous-graphiques vides
    for i in range(num_images, num_rows * num_cols):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()



def main(args):
    model, args = load_model(args)
    tensorboard_runner(args)

    # Liste des chemins des images de validation
    validation_img_paths = [
        "./dataset/validation/0/img_37.jpg",
    ]

    # Liste des noms des classes correspondants
    class_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ]

    # Transformer les images
    transform = get_transform(args)

    # Charger et transformer les images
    img_list = [Image.open(img_path) for img_path in validation_img_paths]
    validation_batch = load_and_transform_images(validation_img_paths, transform, args.device)

    # Faire des prédictions
    predicted_classes = make_predictions(model, validation_batch)

    # Afficher les résultats
    display_results(img_list, predicted_classes, class_names)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Arguments de l'interface en ligne de commande
    parser.add_argument('--resize', type=int, default=28)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--imnorm', action='store_true')
    # parser.add_argument('--randrot', type=int, default=None)
    # parser.add_argument('--randhf', type=float, choices=[Range(0., 1.)], default=None)
    # parser.add_argument('--randvf', type=float, choices=[Range(0., 1.)], default=None)
    # parser.add_argument('--randjit', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--hidden_size', type=int, default=64)

    parser.add_argument('--model_name', type=str, choices=[
        'TwoNN', 'TwoCNN', 'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
        'ResNet10', 'ResNet18', 'ResNet34',
    ], default='TwoNN')

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--log_path', help='Chemin des logs',
    #                     type=str, default='./log')
    # parser.add_argument('--tb_port', help='TensorBoard',
    #                     type=int, default=6006)

    args = parser.parse_args()
    main(args)
