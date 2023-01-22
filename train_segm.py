

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

import matplotlib.pyplot as plt


from src.soccerpitch import SoccerPitch
from src.dataloader import SoccerNetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(
#             inputs, targets, reduction='none', ignore_index=self.ignore_index)
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
#         if self.size_average:
#             return focal_loss.mean()
#         else:
#             return focal_loss.sum()


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


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = [];
    val_acc = []
    train_iou = [];
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1;
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data

            image = image_tiles.to(device);
            mask = mask_tiles.to(device).long();
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device);
                    mask = mask_tiles.to(device).long();
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}

    labels_value = {"train_loss": {"xlabel": "epochs", "ylabel": None},
                    "val_loss": {"xlabel": "epochs", "ylabel": None},
                    "train_miou": {"xlabel": "epochs", "ylabel": None},
                    "val_miou": {"xlabel": "epochs", "ylabel": None},
                    "train_acc": {"xlabel": "epochs", "ylabel": None},
                    "val_acc": {"xlabel": "epochs", "ylabel": None},
                    "lrs": {"xlabel": "iterations", "ylabel": None},
                    }


    result_outpath = "results"
    if not(os.path.exists(result_outpath)):
        os.mkdir(result_outpath)

    for k, v in history.items():
        try:
            fig = plt.figure()
            plt.title(k)
            plt.plot(v)

            labels_key = labels_value.get(k, {"xlabel": None, "ylabel": None})

            xlabel_key = labels_key.get("xlabel", None)
            if xlabel_key:
                plt.xlabel(xlabel_key)

            ylabel_key = labels_key.get("ylabel", None)
            if ylabel_key:
                plt.xlabel(ylabel_key)

            plt.savefig(os.path.join(result_outpath, k))
            plt.close(fig)
        except Exception as e:
            print(f"Error with {k} saving. Exception: {e}")



    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history

def main():


    dataset_path = "soccernet_data/calibration"
    train_split = "train"
    test_split = "test"

    n_classes = 29
    # height = 384
    # width = 640
    height = 544
    width = 960


    soccernet = SoccerNetDataset(dataset_path, split=train_split, width=width, height=height, mean="resources/mean.npy", std="resources/std.npy")
    soccernet_test = SoccerNetDataset(dataset_path, split=test_split, width=width, height=height, mean="resources/mean.npy", std="resources/std.npy")

    n_epochs = 15
    batch_size = 3

    train_dataloader = DataLoader(soccernet, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(soccernet_test, batch_size=batch_size, shuffle=True)

    # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # model.classifier[4] = torch.nn.Conv2d(256, n_classes, 1)
    # model.aux_classifier[4] = torch.nn.Conv2d(256, n_classes, 1)

    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])



    lr = 1e-5
    weight_decay = 0.99

    max_lr = 1e-3
    weight_decay = 1e-4

    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
    #     {'params': model.classifier.parameters(), 'lr': lr},
    # ], lr=lr, momentum=0.9, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=n_epochs,
                                                steps_per_epoch=len(train_dataloader))

    # criterion = FocalLoss()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # os.mkdir("checkpoints")

    cur_itrs = 0

    # def save_ckpt(path):
    #     """ save current model
    #     """
    #     torch.save({
    #         "cur_itrs": cur_itrs,
    #         "model_state": model.module.state_dict(),
    #         "optimizer_state": optimizer.state_dict(),
    #     }, path)
    #     print("Model saved as %s" % path)

    history = fit(n_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, sched)

    torch.save(model, 'net.pt')


if __name__ == "__main__":

    main()