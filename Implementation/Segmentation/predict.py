import os
import torch
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from models import UNET
from utils import load_checkpoint

TEST_IMAGES_FOLDERS = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\TestData\\"
OUTPUT_MASKS_FOLDERS = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\saved_images\\"
MODEL_PATH = "D:\\Projects\\Endoscopic_Bleeding_Detection_with_Computer_Vision\\Implementation\\Segmentation\\my_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(MODEL_PATH), MODEL)
MODEL.eval()

def predict(input_path, output_path, model = MODEL):
    test_img_list = os.listdir(input_path)
    for img_name in test_img_list:
        img_path = os.path.join(input_path, img_name)
        img_np = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor.to(device=DEVICE))

        predicted_mask = (output > 0.5).float().cpu()

        mask_array = predicted_mask.squeeze().numpy()
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
        mask_image.save(os.path.join(output_path, f'{img_name.split(".")[0]}_mask.png'))


def main():
    input_folders = os.listdir(TEST_IMAGES_FOLDERS)
    output_folders = os.listdir(OUTPUT_MASKS_FOLDERS)
    input_folder_path = [os.path.join(TEST_IMAGES_FOLDERS, folder) for folder in input_folders]
    output_folder_paths = [os.path.join(OUTPUT_MASKS_FOLDERS, folder) for folder in output_folders]
    predict(input_folder_path[0], output_folder_paths[0])
    predict(input_folder_path[1], output_folder_paths[1])
    print("| Prediction and saving complete.")


if __name__ == '__main__':
    main()