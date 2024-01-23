import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torcheval.metrics import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from tqdm.auto import tqdm
from dataset import BleedLoader
from model import resnet_X

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
#RANDOM_SEED = 42
DATASET_PATH = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TrainData"
#MODEL_SAVE_PATH = "C:\\Users\\sr1ja\\Desktop\New folder (2)\\New folder\\saved_models\\model_resnet18.pth"
MODEL_SAVE_PATH = r"C:\Users\sr1ja\Desktop\New folder (2)\New folder\saved_models\model_resnet18.pth"
LEARNING_RATE = 0.01
CLASSES = 1
EPOCH = 5


# Define augmentations and transformations

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.RandomRotation(degrees=(0, 180)),
])

data_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Data loaders
train_dl, val_dl, test_dl = BleedLoader(path=DATASET_PATH,
                                        batch_size = BATCH_SIZE,
                                        transforms=data_transforms,
                                        )


# Model
## Self Defined
resnet_18 = resnet_X(layers=[2, 2, 2, 2],       #for resnet18
                     classes=CLASSES).to(DEVICE)

## From torchvision with pre-trained ImageNet weights
model_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model_resnet.to(DEVICE)


# Train
metric = BinaryF1Score(device=DEVICE)
prec = BinaryPrecision(device=DEVICE)
recall = BinaryRecall(device=DEVICE)
acc = BinaryAccuracy(device=DEVICE)

def train(epochs, model):
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(epochs):
        #tinme.sleep(1)
        model.train()
        running_loss = 0
        for _, (inputs, labels) in enumerate(tqdm(train_dl)):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                # preds = torch.round(outputs.sigmoid()).squeeze()
                loss = loss_func(outputs, labels.float().unsqueeze(1))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
        print(f"| Epoch {epoch+1}/{epochs} running loss: {running_loss}")
        model.eval()
        with torch.inference_mode():
            valid_loss = 0
            for inputs, labels in val_dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                preds = outputs.sigmoid().squeeze()
                loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels.float().unsqueeze(1))
                valid_loss += loss.item()
                metric.update(preds, labels)
                prec.update(preds, labels)
                recall.update(preds, labels)
                acc.update(preds, labels)
            print(f"| Total validation loss for epoch {epoch+1}: {valid_loss:.3f}, accuracy: {acc.compute():.3f}, precision: {prec.compute():.3f}, recall: {recall.compute():.3f}, F1Score: {metric.compute():.3f}")
    print('Saving model...')
    torch.save(model_resnet.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved at {MODEL_SAVE_PATH}')


#def main():
#    # Defining the transformations 
#    data_transforms = v2.Compose([
#    v2.ToImage(),
#    v2.ToDtype(torch.float32, scale=True),
#    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#    ])
#
#    # Defining the model
#    model_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
#    num_ftrs = model_resnet.fc.in_features
#    model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
#    model_resnet.to(DEVICE)
#    
#    # Data loaders
#    train_dl, val_dl, _ = BleedLoader(path=DATASET_PATH, 
#                                      batch_size = BATCH_SIZE, 
#                                      transforms=data_transforms)
#    
#    
#    pass


if __name__ == "__main__":
    train(epochs=EPOCH, model=resnet_18)