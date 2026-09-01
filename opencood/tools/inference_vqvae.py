# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib
# 测试点云重建效果

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
from opencood.visualization import simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')


from pdb import set_trace as pause

import cv2

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--initial_epoch', type=int, default=-1,
                        help='initial_epoch')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()


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
    resume_epoch, model = train_utils.load_saved_model(saved_path, model, initial_epoch=opt.initial_epoch)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
        
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # codebook usage 直方图和热力图
    # import matplotlib.pyplot as plt
    # # plt.figure(figsize=(7, 5))
    # plt.figure(figsize=(6, 4))
    # plt.style.use('ggplot')
    # plt.bar(range(512), model.vqvae.code_usage.cpu().numpy())
    # plt.xlabel("Code Index", fontsize=12, color='black')
    # plt.ylabel("Total Usage Count", fontsize=12, color='black')
    # plt.title("DAIR-V2X", fontsize=12, color='black')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("vq_codebook_usage_dairv2x.png")

    # pause()

    # K = 512  # codebook size
    # heatmap_data = model.vqvae.code_usage.cpu().numpy().reshape(32, 16)

    # plt.figure(figsize=(8, 6))
    # plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Usage Count")
    # plt.title("Codebook Usage Heatmap")
    # plt.xlabel("Code Index X")
    # plt.ylabel("Code Index Y")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("vq_codebook_usage_heatmap_trainset_opv2v.png")
    


    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=0,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
       
    infer_info = 'vqvae_colc' + opt.note

    t_all = []
    vis_save_path_root = os.path.join(opt.model_dir, f'voxel_{infer_info}')
    if not os.path.exists(vis_save_path_root):
        os.makedirs(vis_save_path_root)
    
    num = 0

    gt_hist_list = []
    rec_hist_list = []
    gt_hist_list_interval = []
    rec_hist_list_interval = []
    lidar_rec_fiou_list = []
    lidar_rec_iou_list = []
    lidar_rec_mse_list = []

    for i, batch_data in enumerate(data_loader):

        if batch_data is None:
            continue

        # if num > 1:
        #     break

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            

            print(f"{infer_info}_{i}")
            
            t1 = time.time()
            output_dict = model(batch_data['ego'])
            t2 = time.time()
            t_all.append(t2 - t1)

            voxels = output_dict['voxels']
            generated_voxel = output_dict['generated_voxel']

            lidar_rec_iou = output_dict['lidar_rec_iou']

            lidar_rec_iou_list.append(lidar_rec_iou.cpu())

            # lidar_rec_fiou = output_dict['lidar_rec_fiou']

            # lidar_rec_fiou_list.append(lidar_rec_fiou.cpu())


            occ_mask = output_dict['occ_mask']
            mse = ((generated_voxel - voxels*occ_mask) ** 2).mean()
            lidar_rec_mse_list.append(mse.item())


            if (i % opt.save_vis_interval == 0):

                num += 1
                draw_bev_lidar(voxels, "{}/No{}_gt.png".format(vis_save_path_root, i))
                draw_bev_lidar(generated_voxel, "{}/No{}_generated.png".format(vis_save_path_root, i))
                

                if model.vqvae.rec_mode == 's2d':
                # if model.mae.rec_mode == 's2d':
                    voxels_sparse = output_dict['voxels_sparse']
                    draw_bev_lidar(voxels_sparse, "{}/No{}_sparse.png".format(vis_save_path_root, i))
            
    
        torch.cuda.empty_cache()

    # lidar_rec_fiou_mean = np.array(lidar_rec_fiou_list).mean()
    # print('lidar_rec_fiou:',lidar_rec_fiou_mean)


    # calculate jsd and mmd

    print('Lidar completion evaluate:')
    
    lidar_rec_iou_mean = np.array(lidar_rec_iou_list).mean()

    print('lidar_rec_iou:',lidar_rec_iou_mean)
    lidar_rec_mse_mean = np.array(lidar_rec_mse_list).mean()
    print('lidar_rec_mse:',lidar_rec_mse_mean)
    
    dump_dict = {}

    print('average time:', np.mean(t_all) / 1)
    print('average fps:', 1 / np.mean(t_all))

    dump_dict.update({
        'iou': float(lidar_rec_iou_mean),
        'mse': float(lidar_rec_mse_mean),
        'fps': float(1 / np.mean(t_all))})
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, 'rec.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'rec_{infer_info}.yaml'))



def draw_bev_lidar(voxels, pth):
    cv2.imwrite(
        pth,
        voxels[0].max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    )

    # if voxels.size()[0]>1:
    #     cv2.imwrite(
    #         pth,
    #         voxels[1].max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    #     )


if __name__ == '__main__':
    main()
