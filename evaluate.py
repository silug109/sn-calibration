import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
import time
import segmentation_models_pytorch as smp


from src.soccerpitch import SoccerPitch
from src.dataloader import SoccerNetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=29):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def onehot_to_rgb(onehot, colormap):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def main():
    dataset_path = "soccernet_data/calibration"
    train_split = "train"

    n_classes = 29
    height = 384
    width = 640

    soccernet = SoccerNetDataset(dataset_path, split=train_split, width=width, height=height, mean="resources/mean.npy",
                                 std="resources/std.npy")


    model = torch.load("net.pt", map_location=device)

    model.eval()
    model.to(device)



    for i in range(5):



        idx = np.random.randint(0, len(soccernet), (1))[0]

        print(i, idx)

        img, mask = soccernet[idx]

        img_t = torch.from_numpy(img).to(device)[None,...]

        output = model(img_t)

        out_mask = output.permute(0,2,3,1).argmax(dim=-1)[0]

        vis = np.concatenate((out_mask.cpu().numpy(), mask), axis=1)

        cv2.imwrite(f"out_{i}.jpg", vis)


if __name__ == "__main__":
    main()