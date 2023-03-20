import os
import torch.nn.functional as F
import warnings
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Resize
import torch
import random
from model import VisualSimilarityModel
from data_loader import PhysDataset
from utils import report
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2333)
parser.add_argument("--data_dir", type=str, default="../data/PAN-phys/")
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
SEED = args.seed
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 
torch.autograd.set_detect_anomaly(True)


def cal_loss_cam_cos(cam, gold_gaze):
    result = F.cosine_similarity(cam.reshape(1, 49), gold_gaze.reshape(1, 49), dim=-1)
    return result


"""
    To test whether DPA can generalize to unseen real world images.
"""


test_dataset = PhysDataset(data_dir=args.data_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)
model = VisualSimilarityModel(device=device)
model.load_state_dict(torch.load("./checkpoint_best.pth"))
model.eval()
test_pred_list, test_gold_list, test_sc_list = [], [], []
tqdm_test_loader = tqdm(test_loader)
for (index, gaze_score, original_img, gaze_image, gaze_score_distribution) in tqdm_test_loader:
    original_img = original_img.to(device)
    gaze_score = gaze_score.to(device)
    gaze_image = gaze_image.to(device)
    gaze_score_distribution = gaze_score_distribution.to(device)
    y, cam, _ = model(original_img, gaze_score_distribution, device)
    for index in range(args.bs):
        s_c = cal_loss_cam_cos(Resize((7, 7))(cam[index].unsqueeze(0)), Resize((7, 7))(gaze_image[index].unsqueeze(0)))
        test_sc_list.append(float(s_c.detach().cpu().numpy()))
    test_pred_list += y.data.cpu().numpy().tolist()
    test_gold_list += gaze_score.data.cpu().numpy().tolist()

spearman, pearson, kendall, rmse = report(test_pred_list, test_gold_list)
print(spearman, pearson, kendall, rmse, np.mean(test_sc_list))
