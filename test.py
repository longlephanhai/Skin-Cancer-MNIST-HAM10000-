from torchvision.transforms import Compose, RandomAffine, Resize, ToTensor, Normalize, ColorJitter
from dataset import SkinDiseaseDataset
from torch.utils.data import DataLoader
import numpy as np
# from model import SkinDiseaseModel
from transfer_learning import SkinDiseaseModelResNet
from argparse import ArgumentParser
from torchsummary import summary
import torch
import torch.nn as nn
import cv2

def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image-path", "-p", type=str, default=None)
    parser.add_argument("--image-size", "-i", type=int,
                        default=224, help="Image size")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    categories = ["Acne", "Actinic_Keratosis", "Benign_Tumors", "Bullous", "Candidiasis", "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus", "Moles",
                           "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer", "Sun_Sunlight_Damage",  "Tinea", "Unknown_Normal", "Vascular_Tumors", "Vasculitis", "Vitiligo",  "Warts"]

    model = SkinDiseaseModelResNet(num_classes=22).to(device)
    summary(model, (3, 224, 224))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint found!")
        exit(0)
        
    model.eval()
    
    
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.0
    image = image[None, :, :, :]   # 1 x 3 x 224 x 224
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)

    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    print("The test image is about {} with confident score of {}".format(
        predicted_class, probs[0, max_idx]))
    cv2.imshow("{}:{:.2f}%".format(predicted_class,
               probs[0, max_idx]*100), ori_image)
    cv2.waitKey(0)