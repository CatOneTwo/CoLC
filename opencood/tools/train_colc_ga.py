# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Communication-efficient Cooper

import faulthandler
faulthandler.enable()

import argparse
import os
import statistics
import importlib
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob
from icecream import ic
import tqdm
import numpy as np

import random

from pdb import set_trace as pause

def seed_torch(seed=3407):
    print('seed fixed!')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="cecooper",
                        help='passed to inference.')
    parser.add_argument('--log', type=str, default='',
                        help='log name suffix')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='load pretrained model')
    parser.add_argument('--nei_model', type=str, default='',
                        help='load nei model')
    parser.add_argument('--vqvae_model', type=str, default='',
                        help='load vqvae model')
    # parser.add_argument('--mae_model', type=str, default='',
    #                     help='load vqvae model')
    
    opt = parser.parse_args()
    return opt


def main():

    # seed_torch()

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # if 'neighbor_points_selection' in hypes['model']['args']:
    #     opt.log += 'nei_'
    #     hypes_tmp = hypes['model']['args']['neighbor_points_selection']
    #     if hypes_tmp['nei_supply']:
    #         if 'fg_fps' not in hypes_tmp and 'bg_rs' not in hypes_tmp:
    #             opt.log += 'fg_only'
    #         elif 'fg_fps' in hypes_tmp and 'bg_rs' not in hypes_tmp:
    #             fg_ratio = hypes_tmp['fg_fps']['ratio'] 
    #             opt.log += 'fg_fps' + str(fg_ratio)
    #         elif 'fg_fps' not in hypes_tmp and 'bg_rs' in hypes_tmp:
    #             bg_ratio = hypes_tmp['bg_rs']['ratio'] 
    #             opt.log += 'fg_bg_rs' + str(bg_ratio)
    #         elif 'fg_fps' in hypes_tmp and 'bg_rs' in hypes_tmp:
    #             fg_ratio = hypes_tmp['fg_fps']['ratio'] 
    #             bg_ratio = hypes_tmp['bg_rs']['ratio'] 
    #             opt.log += 'fg_fps'+str(fg_ratio) +'_bg_rs' + str(bg_ratio)
    #     else:
    #         opt.log += 'all'

    # if 'novqvae' in opt.log:
    #     # hypes['model']['args']['lidar_completion']=False
    #     hypes['model']['args'].pop('lidar_completion', None)

    if 'cav_lidar_sample' in hypes['fusion']['args']:
        hypes_tmp = hypes['fusion']['args']['cav_lidar_sample']
        opt.log += hypes_tmp['method']
        if hypes_tmp['method'] == 'RS':
            opt.log += str(hypes_tmp['ratio'])

    # hypes_model = hypes['model']['args']
    # if 'lidar_completion' in hypes_model and hypes_model['lidar_completion']:
    #     opt.log += '_'
    #     if 'fusion' not in hypes_model:
    #         opt.log += 'mean'
    #     else:
    #         opt.log += hypes_model['fusion']

    # hypes_loss = hypes['loss']['args']
    # if 'kd' in hypes_loss:
    #     opt.log += '_'
    #     if 'pillar_kd' in hypes_loss['kd']:
    #         opt.log += 'p'
    #     if 'feature_kd' in hypes_loss['kd']:
    #         opt.log += 'f'
    #     if 'decoder_kd' in hypes_loss['kd']:
    #         opt.log += 'd'

    hypes['log_suffix']=opt.log


    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

        # load pretrainde model if it exists.
        # TODO ego是否加载单车模型？
        if opt.pretrained_model:
            print('pretrained_model_dir:',opt.pretrained_model)
            model = train_utils.load_pretrained_model(opt.pretrained_model, model)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # 点云补全模型
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

    # if opt.mae_model:
    #     print('mae_model_dir:',opt.mae_model)
    #     mae_yaml_file = os.path.join(opt.mae_model, 'config.yaml')
    #     mae_hypes = yaml_utils.load_yaml(mae_yaml_file)
    #     mae_model = train_utils.create_model(mae_hypes)
    #     resume_epoch, mae_model = train_utils.load_saved_model(opt.mae_model, mae_model)
    #     for p in mae_model.parameters():
    #         p.requires_grad_(False)
    #     if torch.cuda.is_available():
    #         mae_model.to(device)
    #     mae_model.eval()
    #     model.mae = mae_model.mae

    # 近邻模型
    # print('nei_model_dir:',opt.nei_model)
    # nei_yaml_file = os.path.join(opt.nei_model, 'config.yaml')
    # nei_hypes = yaml_utils.load_yaml(nei_yaml_file)
    # nei_model = train_utils.create_model(nei_hypes)
    # nei_model = train_utils.load_pretrained_model(opt.nei_model, nei_model) 
    # for p in nei_model.parameters():
    #     p.requires_grad_(False)
    # if torch.cuda.is_available():
    #     nei_model.to(device)
    # nei_model.eval()
    # model.nei_model = nei_model

    ############ For DiscoNet ##############
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "opencood.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls
        
        teacher_model = teacher_model_class(teacher_model_config)
        # teacher_model.load_state_dict(torch.load(teacher_checkpoint_path), strict=False)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint_path, map_location='cuda:0'), strict=False)
        
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
            teacher_model.to(device)

        teacher_model.eval()
    else:
        kd_flag = False
    ########################################
    
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        
        accum_steps = 8

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue

            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            if (i + 1) % accum_steps == 0 or (i + 1 == len(train_loader)) == 0:
                optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            
            # Ego端融合近邻端传输的点云，并进行感知      
            ouput_dict = model(batch_data['ego'])

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)

            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            # criterion.logging(epoch, i, len(train_loader), writer)
            criterion.logging(epoch, i, len(train_loader), writer,pbar=pbar2)
            
            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                # criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", pbar=pbar2)
            
            pbar2.update(1)
            final_loss = final_loss / accum_steps

            # back-propagation
            final_loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1 == len(train_loader)) == 0:
                optimizer.step()

            # torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch

                    # 近邻端筛选点云
                    # batch_data = model.neighbor_points_selection(batch_data, opencood_validate_dataset, device)

                    # Ego端融合近邻端传输的点云，并进行感知
                    ouput_dict = model(batch_data['ego'])

                    if kd_flag:
                        teacher_output_dict = teacher_model(batch_data['ego'])
                        ouput_dict.update(teacher_output_dict)

                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss

                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        # if epoch % hypes['train_params']['save_freq'] == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(saved_path,
        #                             'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True    
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method

        if opt.vqvae_model:
            cmd = f"CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_colc.py --model_dir {saved_path} --fusion_method {opt.fusion_method} --vqvae_model {opt.vqvae_model} --nei_model {opt.nei_model} --note '_color'"
            print(f"Running command: {cmd}")
            os.system(cmd)
        else:
            cmd = f"CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_colc.py --model_dir {saved_path} --fusion_method {opt.fusion_method} --nei_model {opt.nei_model} --note '_color'"
            print(f"Running command: {cmd}")
            os.system(cmd)

if __name__ == '__main__':
    main()
