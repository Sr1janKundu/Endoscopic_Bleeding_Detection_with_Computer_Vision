import torch
from torchvision import datasets
from torch.utils.data import  DataLoader, random_split



def BleedLoader(path, batch_size, split_ratio = [0.7, 0.1], random_seed = 42, transforms = None, shuffle = False):
    dataset = datasets.ImageFolder(root=path, transform=transforms)
    #print(dataset.classes)     # Sanity check
    train_len = int(split_ratio[0]*len(dataset))
    val_len = int(split_ratio[1]*len(dataset))
    test_len = len(dataset) - (train_len + val_len)
    
    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle = shuffle)

    return train_dl, val_dl, test_dl