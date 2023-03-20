import argparse
import os
import random
import traceback
import warnings
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import SelfDataset
from utils import report
from model import VisualSimilarityModel
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--lamb1", type=float, default=8)
parser.add_argument("--lamb2", type=float, default=3)
parser.add_argument("--data_dir", type=str, default="../data/PAN/")
parser.add_argument("--train", default=False, action='store_true')
parser.add_argument("--eval", default=False, action='store_true')
parser.add_argument("--test", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=2333)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

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
log_dir = ""
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")


def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + "-" + str(logs[key]) + "+"
    dir_name = "./logs/" + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name


def cal_loss_score(pred_score, gaze_score):
    return F.mse_loss(pred_score.flatten(), gaze_score.flatten())


def cal_loss_cam(cam, gold_gaze):
    return args.lamb1 * F.mse_loss(cam, gold_gaze)


def cal_loss_kl(sim_score, gaze_score_distribution):
    return args.lamb2 * F.kl_div(sim_score.log(), gaze_score_distribution, reduction='mean')


def run():
    if args.train:
        train_dataset = SelfDataset(data_dir=args.data_dir, split_str="train")
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    if args.eval:
        valid_dataset = SelfDataset(data_dir=args.data_dir, split_str="valid")
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.bs, shuffle=False)
    if args.test:
        test_dataset = SelfDataset(data_dir=args.data_dir, split_str="test")
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)

    best_srocc = 0.
    model = VisualSimilarityModel(device=args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        if args.train:
            model.train()
            tqdm_loader = tqdm.tqdm(train_loader)
            for (index, gaze_img, gaze_score, original_img, gaze_score_distribution) in tqdm_loader:
                original_img = original_img.to(device)
                gaze_img = gaze_img.float().to(device)
                gaze_score = gaze_score.float().to(device)
                gaze_score_distribution = gaze_score_distribution.to(device)
                y, cam, sim_score = model(original_img, gaze_score_distribution, device=device)
                loss_score = cal_loss_score(y, gaze_score)
                loss_kl = cal_loss_kl(sim_score, gaze_score_distribution)
                loss_cam = cal_loss_cam(cam, gaze_img)
                loss = loss_score + loss_kl + loss_cam
                try:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                except:
                    traceback.print_exc()
            torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint_" + str(epoch) + ".pth"))
            if args.eval:
                model.eval()
                tqdm_eval_loader = tqdm.tqdm(valid_loader)
                pred_list, gold_list = [], []
                for (index, gaze_img, gaze_score, original_img, gaze_score_distribution) in tqdm_eval_loader:
                    original_img = original_img.to(device)
                    gaze_score = gaze_score.float().to(device)
                    gaze_score_distribution = gaze_score_distribution.to(device)
                    y, _, _ = model(original_img, gaze_score_distribution, device)
                    pred_list += y.data.cpu().numpy().tolist()
                    gold_list += gaze_score.data.cpu().numpy().tolist()
                np.save(os.path.join(log_dir, f'valid_pred_{epoch}.npy'), np.array(pred_list))
                np.save(os.path.join(log_dir, f'valid_gold_{epoch}.npy'), np.array(gold_list))
                np.save(os.path.join(log_dir, 'valid_pred_latest.npy'), np.array(pred_list))
                np.save(os.path.join(log_dir, 'valid_gold_latest.npy'), np.array(gold_list))
                spearman, pearson, kendall, rmse = report(pred_list, gold_list)
                if spearman > best_srocc:
                    best_srocc = spearman
                    torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint_best.pth"))
                with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
                    f.write(
                        f'Epoch {epoch}:  valid spearman {spearman}, valid pearson {pearson}, valid kendall {kendall}, valid rmse {rmse} \n')
                print(
                    f'Epoch {epoch}:  valid spearman {spearman}, valid pearson {pearson}, valid kendall {kendall}, valid rmse {rmse} \n')
    if args.test:
        model.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint_best.pth")))
        model.eval()
        test_pred_list, test_gold_list = [], []
        tqdm_test_loader = tqdm.tqdm(test_loader)
        for (index, gaze_img, gaze_score, original_img, gaze_score_distribution) in tqdm_test_loader:
            original_img = original_img.to(device)
            gaze_score = gaze_score.to(device)
            gaze_score_distribution = gaze_score_distribution.to(device)
            y, _, _ = model(original_img, gaze_score_distribution, device)
            test_pred_list += y.data.cpu().numpy().tolist()
            test_gold_list += gaze_score.data.cpu().numpy().tolist()
        np.save(os.path.join(log_dir, 'test_pred.npy'), np.array(test_pred_list))
        np.save(os.path.join(log_dir, 'test_gold.npy'), np.array(test_gold_list))
        spearman, pearson, kendall, rmse = report(test_pred_list, test_gold_list)
        with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
            f.write(
                f'Final Test:  test spearman {spearman}, test pearson {pearson}, test kendall {kendall}, test rmse {rmse} \n')
        print(
            f'Final Test:  test spearman {spearman}, test pearson {pearson}, test kendall {kendall}, test rmse {rmse} \n')


if __name__ == "__main__":
    logs = {
        "epoch": args.epoch,
        "train": args.train,
        "eval": args.eval,
        "test": args.test
    }
    make_log_dir(logs)
    run()
