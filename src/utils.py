import cv2
import numpy as np
from scipy import stats


def report(pred_list, gold_list):
    pred_scores = pred_list
    gt_scores = gold_list
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    test_kendall, _ = stats.kendalltau(pred_scores, gt_scores)
    test_rmse = np.sqrt(np.mean((np.array(pred_scores) - np.array(gt_scores)) ** 2))
    return test_srcc, test_plcc, test_kendall, test_rmse


class CenterCropResize:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.h_w_ratio = 1.0 * self.output_size[0] / self.output_size[1]

    def __call__(self, image):
        crop_image = self.center_crop(image, self.h_w_ratio)
        image = cv2.resize(crop_image, self.output_size[::-1])
        return image

    @staticmethod
    def center_crop(image, h_w_ratio=1):
        h, w = image.shape[:2]
        crop_h = int(w * h_w_ratio)
        if crop_h < h:
            h_start = (h - crop_h) // 2
            h_end = h_start + crop_h
            crop_image = image[h_start:h_end, :, :]
        else:
            crop_w = int(h / h_w_ratio)
            w_start = (w - crop_w) // 2
            w_end = w_start + crop_w
            crop_image = image[:, w_start:w_end, :]
        return crop_image

