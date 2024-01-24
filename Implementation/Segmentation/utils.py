import torch
import torchvision
#from torchvision.transforms import v2
from dataset import BleedDataset


def get_loaders(path):
    batch_size = 32
    random_seed = 42
    torch.manual_seed(random_seed)
    
    dataset = BleedDataset(root_dir=path, transform=False)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)
    
    return train_dl, valid_dl


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    IoU = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            intersection = (preds * y).sum()
            union = (preds + y).sum() - intersection
            IoU += (intersection + 1e-8) / (union + 1e-8)

    print(f"\nPredicted {num_correct} pixels correct out of {num_pixels} pixels, with accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU: {IoU / len(loader)}")
    model.train()


#def save_predictions_as_imgs(loader, model, folder, device="cuda"):
#    model.eval()
#    for idx, (x, y) in enumerate(loader):
#        x = x.to(device=device)
#        with torch.no_grad():
#            preds = torch.sigmoid(model(x))
#            preds = (preds > 0.5).float()
#        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
#        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
#
#    model.train()