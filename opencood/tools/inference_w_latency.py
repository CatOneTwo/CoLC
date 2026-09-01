# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader
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
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='save how many numbers of visualization result?')
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
    left_hand = True if ("OPV2V" in hypes['test_dir'] or 'V2XSET' in hypes['test_dir']) else False

    # if 'dair_v2x' in hypes['test_dir']:
    #     opt.also_laplace = True

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']
    

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # add latency.
    latency_list = [100, 200, 300, 400, 500]

    AP30 = []
    AP50 = []
    AP70 = []
    for latency in latency_list:
        # setting latency
        np.random.seed(303)
        latency_setting = OrderedDict()
        latency_setting['add_latency'] = True
        latency_setting['time_delay'] = latency

        suffix = ""

        # build dataset for each latency setting
        print('Dataset Building')
        print(f"Latency Added: {latency}.")
        hypes.update({"latency_setting": latency_setting})
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        
        # Create the dictionary for evaluation
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        
        latency_level = f"{latency}_" + opt.fusion_method + suffix + opt.note

        for i, batch_data in enumerate(data_loader):
            print(f"{latency_level}_{i}")
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                
                if opt.fusion_method == 'late':
                    infer_result = inference_utils.inference_late_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'early':
                    infer_result = inference_utils.inference_early_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no_w_uncertainty':
                    infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']

                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)

            torch.cuda.empty_cache()

        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, latency_level)
        AP30.append(ap30)
        AP50.append(ap50)
        AP70.append(ap70)

        dump_dict = {'ap30': AP30 ,'ap50': AP50, 'ap70': AP70}
        yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'AP030507_latency{suffix}.yaml'))


if __name__ == '__main__':
    main()
