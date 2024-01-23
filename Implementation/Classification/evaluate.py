import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torcheval.metrics import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from dataset import BleedLoader
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
metric = BinaryF1Score(device=DEVICE)
prec = BinaryPrecision(device=DEVICE)
recall = BinaryRecall(device=DEVICE)
acc = BinaryAccuracy(device=DEVICE)

DATASET_PATH = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TrainData\\"
TEST_DATASET_1_PATH = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TestData\\Test Dataset 1\\"
TEST_DATASET_2_PATH = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TestData\\Test Dataset 2\\"
MODEL_PATH = "C:\\Users\\sr1ja\\Desktop\New folder (2)\\New folder\\saved_models\\model_resnet18.pth"

model_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model_resnet.load_state_dict(torch.load(MODEL_PATH))
model_resnet.to(DEVICE)


data_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_, _, test_dl = BleedLoader(path=DATASET_PATH,
                                        batch_size = BATCH_SIZE,
                                        transforms=data_transforms,
                                        )

def evaluate(model, test_data, eval_metric):
  model.eval()
  with torch.inference_mode():
    valid_loss = 0
    total_acc = 0
    len_preds = 0
    for inputs, labels in test_data:
      inputs = inputs.cuda()
      labels = labels.cuda()
      outputs = model(inputs)
      preds = torch.round(outputs.sigmoid()).squeeze(1)
      eval_metric.update(preds, labels)
      prec.update(preds, labels)
      recall.update(preds, labels)
      loss = F.binary_cross_entropy_with_logits(outputs, labels.float().unsqueeze(1))
      valid_loss += loss.item()
      total_acc += torch.tensor(torch.sum(preds == labels).item())
      len_preds += len(preds)
    print(f"[*] total binary cross entropy loss: {valid_loss} accuracy: {total_acc/len_preds} F1Score: {metric.compute()} BinaryRecall: {recall.compute()} Precision: {prec.compute()}")
    eval_metric.reset()
    prec.reset()
    recall.reset()


metric.reset()
prec.reset()
recall.reset()
acc.reset()
evaluate(model_resnet, test_dl, metric)


d = {0.:0,
     1.:0}
transform = data_transforms
for i in os.listdir(TEST_DATASET_1_PATH):
  img = Image.open(os.path.join(TEST_DATASET_1_PATH, i))
  img = transform(img)
  img = img.to(DEVICE)
  img = torch.unsqueeze(img, dim=0)
  with torch.inference_mode():
    output = model_resnet(img).squeeze()
    pred = torch.round(output.sigmoid())
    print(i, pred, output.sigmoid())
    d[pred.item()] += 1


print(d[0.], d[1.])



d = {0.:0,
     1.:0}
transform = data_transforms
for i in os.listdir(TEST_DATASET_2_PATH):
  img = Image.open(os.path.join(TEST_DATASET_2_PATH, i))
  img = transform(img)
  img = img.to(DEVICE)
  img = torch.unsqueeze(img, dim=0)
  with torch.inference_mode():
    output = model_resnet(img).squeeze()
    pred = torch.round(output.sigmoid())
    print(i, pred, output.sigmoid())
    d[pred.item()] += 1


print(d[0.], d[1.])