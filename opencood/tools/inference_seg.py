# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')
from pdb import set_trace as pause

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # v2xreal 验证v2v场景
    # if "v2xreal" in hypes['test_dir']:
    #     hypes['dataset_mode']='v2v' 
    
    if "noise_setting" in hypes:
        hypes['noise_setting']['add_noise'] = False

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")

    thre=0.5
    opt.note += f'{thre}' + f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4, 
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    
    infer_info = opt.note

    TP_sum, FP_sum, FN_sum, TN_sum = 0, 0, 0, 0

    for i, batch_data in enumerate(data_loader):

        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            logits = model(batch_data['ego'])
            probs = torch.sigmoid(logits)            # 转为概率
            preds = (probs > thre).long()            # 二分类预测 mask   

            preds_np = preds.cpu().numpy()      # [B, N]
            labels_np = batch_data['ego']['label_dict'].cpu().numpy()

            preds_flat = preds_np.reshape(-1)
            labels_flat = labels_np.reshape(-1)

            TP_sum += np.logical_and(preds_flat == 1, labels_flat == 1).sum()
            FP_sum += np.logical_and(preds_flat == 1, labels_flat == 0).sum()
            FN_sum += np.logical_and(preds_flat == 0, labels_flat == 1).sum()
            TN_sum += np.logical_and(preds_flat == 0, labels_flat == 0).sum()
        
        torch.cuda.empty_cache()
    

    accuracy = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum + 1e-6)
    precision = TP_sum / (TP_sum + FP_sum + 1e-6)
    recall = TP_sum / (TP_sum + FN_sum + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou_fg = TP_sum / (TP_sum + FP_sum + FN_sum + 1e-6)

    # 前景稀疏任务中 Accuracy 指标意义不大，重点看 IoU / F1 / Recall / Precision
    # 目标： Precision≥0.7 Recall≥0.8  F1≥0.7 IoU_fg ≥0.6

    print("Dataset Metrics:")
    print(f"Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU_fg: {iou_fg:.4f}")

    dump_dict = {}

    dump_dict.update({
        'Acc': round(float(accuracy), 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'F1': round(float(f1), 4),
        'IoU_fg': round(float(iou_fg), 4)
    })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, 'seg.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'seg_{infer_info}.yaml'))

if __name__ == '__main__':
    main()
