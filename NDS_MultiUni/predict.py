
import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import torch
#from sklearn.utils import shuffle
from model import CompNet
from crf import apply_crf

def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Hyperparameters """
    size = (512, 512)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    images_paths = glob(os.path.join("/project/def-sponsor00/endocv2021-test-noCopyAllowed-v2/*/*.jpg"))
    # images_paths = glob(os.path.join("../test/*/*.jpg"))
    #images_paths= glob(os.path.join('../../endocv2021-test-noCopyAllowed-v1/*/*.jpg'))
    
    
    time_taken = []
    for img_path in tqdm(images_paths, total=len(images_paths)):
        """ Directory to save predicted mask"""
        dir_name = img_path.split("/")[-2]
        save_dir = f"segmentation/{dir_name}_pred/"
        create_dir(save_dir)

        """ Extracting the image name and extension. """
        name = img_path.split("/")[-1].split(".")[0]
        extn = img_path.split("/")[-1].split(".")[1]

        """ Reading image """
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ori_img = image
        ori_h, ori_w, _ = image.shape

        image = cv2.resize(image, size)
        resize_img = image

        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Predicting Mask """
        with torch.no_grad():
            """ FPS """
            start_time = time.time()
            pred = torch.sigmoid(model(image))
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred = pred[0].cpu().numpy()
            pred = np.squeeze(pred, axis=0)
            pred = pred > 0.5
            pred = pred.astype(np.int32)
            pred = apply_crf(resize_img, pred)
            pred = pred * 255
            pred = np.array(pred, dtype=np.uint8)


        pred = cv2.resize(pred, (ori_w, ori_h))
        save_path = f"{save_dir}/{name}_mask.{extn}"
        cv2.imwrite(save_path, pred)

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print(f"Mean Time Taken: {mean_time_taken} - Mean FPS: {mean_fps}")






    ##
