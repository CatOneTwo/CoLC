# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# inference for early fusion or CoLC model

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
                        default='cecooper',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--nei_model', type=str, default='',
                        help='load pretrained model')
    parser.add_argument('--vqvae_model', type=str, default='',
                        help='load vqvae model')
    parser.add_argument('--fg_ratio', type=float, default=1.0,
                        help='foreground_points_sample_ratio for neighbor')
    parser.add_argument('--bg_ratio', type=float, default=0.1,
                        help='foreground_points_sample_ratio for neighbor')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    # assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

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

    # neighbor point selection ratio
    fg_ratio = opt.fg_ratio
    bg_ratio = opt.bg_ratio

    print(fg_ratio, bg_ratio)
    if fg_ratio == 1.0:
        neighbor_args = {'nei_supply': True,
                    'score_threshold': 0.2,
                    'bg_rs': {'ratio': bg_ratio}}
    else:
        neighbor_args = {'nei_supply': True,
                        'score_threshold': 0.2,
                        'fg_fps': {'ratio': fg_ratio, 'min_points': 100},
                        'bg_rs': {'ratio': bg_ratio}}
    
    hypes['model']['args']['neighbor_points_selection']=neighbor_args
    
    # hypes['model']['args'].pop('neighbor_points_selection', None)
    
    # dataset 
    hypes['fusion']['args'] = {'proj_first': True}

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")


    opt.note += f"_epoch{resume_epoch}" + '_fg' +str(fg_ratio) + '_bg' +str(bg_ratio)
    
    # opt.note += f"_epoch{resume_epoch}" + '_early'


        
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    

    # nei model
    # nei_model = train_utils.create_model(hypes)
    nei_yaml_file = os.path.join(opt.nei_model, 'config.yaml')
    nei_hypes = yaml_utils.load_yaml(nei_yaml_file)
    nei_model = train_utils.create_model(nei_hypes)
    nei_model = train_utils.load_pretrained_model(opt.nei_model, nei_model) 
    for p in nei_model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        nei_model.to(device)
    nei_model.eval()
    model.nei_model = nei_model

    # load pretrained vq-based completion model
    if opt.vqvae_model:
        print('vqvae_model_dir:',opt.vqvae_model)
        vqvae_yaml_file = os.path.join(opt.vqvae_model, 'config.yaml')
        vavae_hypes = yaml_utils.load_yaml(vqvae_yaml_file)
        vqvae_model = train_utils.create_model(vavae_hypes)
        resume_epoch, vqvae_model = train_utils.load_saved_model(opt.vqvae_model, vqvae_model)
        for p in vqvae_model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            vqvae_model.to(device)
        vqvae_model.eval()
        model.vqvae = vqvae_model.vqvae
    else:
        opt.note += '_nolc'
    
    # setting noise
    # np.random.seed(303)


    # model size parameter
    # total_num = sum(p.numel() for p in model.nei_model.parameters())
    # print('Total:', total_num) # 420833

    # total_num = sum(p.numel() for p in model.vqvae.lidar_encoder.parameters())
    # print('Total:', total_num) # 1547360

    # total_num = sum(p.numel() for p in model.vqvae.vector_quantizer.parameters())
    # print('Total:', total_num) # 16384

    # total_num = sum(p.numel() for p in model.vqvae.lidar_decoder.parameters())
    # print('Total:', total_num) # 1548065

    # total_num = sum(p.numel() for p in model.fuse_layer.parameters())
    # print('Total:', total_num) # 21249
    
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
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    infer_info = opt.fusion_method + opt.note

    lidar_nums=[]
    # box_nums=[]
    
    for i, batch_data in enumerate(data_loader):

        print(f"{infer_info}_{i}")
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
            elif opt.fusion_method == 'cecooper':
                infer_result = inference_utils.inference_cecooper_fusion(batch_data,
                                                        model,
                                                        opencood_dataset,
                                                        device)
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
            
            # calculate communication for early fusion
            if 'origin_lidar_dict' in infer_result:
                lidar_num=[]
                for key, item in infer_result['origin_lidar_dict'].items():
                    if key != 'ego':
                        lidar_num.append(item.shape[0])
                lidar_nums += lidar_num

            # calculate communication for late fusion
            # if 'box_num' in infer_result:
            #     box_nums += infer_result['box_num']

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
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)

                if 'color' not in opt.note:
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
                else:
                    simple_vis.visualize_colorful(infer_result,
                                        infer_result['origin_lidar_dict'],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
                    
        torch.cuda.empty_cache()

    if len(lidar_nums) > 0:
        mean1 = np.mean(lidar_nums)   #mean
        max1 = np.max(lidar_nums)     #max
        print(mean1, max1) # Transmitted points number (mean and max)
    
        dump_dict = {}

        dump_dict.update({'mean': float(mean1),
                        'max': float(max1),
                        'log2max': float(np.log2(max1*16))
                        })
        if infer_info is None:
            yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, 'comm.yaml'))
        else:
            yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'comm_{infer_info}.yaml'))



    # if len(box_nums) > 0:
    #     mean1 = np.mean(box_nums)   #
    #     max1 = np.max(box_nums)     #
    #     print(mean1, max1)

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
