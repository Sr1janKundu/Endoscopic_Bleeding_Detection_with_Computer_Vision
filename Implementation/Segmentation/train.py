import torch
#from torchvision.transforms import v2
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import UNET
from utils import (
    #load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    #save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
#LOAD_MODEL = False
data_path = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TrainData\\"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)
        #print(f"\nInput image batch shape: {data.size()}, Input target batch shape: {targets.size()}")         # sanity check
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print(f"Model preiction tensor shape: {predictions.size()}")              # sanity check
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    '''
    From pytorch hub, with pretrained weights ; [won't work because of RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same]
    '''
    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)      
    '''
    Self defined, no pretrained weights
    '''
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)         
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(data_path)

    #if LOAD_MODEL:
    #    load_checkpoint(torch.load("my_checkpoint.pth"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'| Epoch {epoch}:')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
        
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()