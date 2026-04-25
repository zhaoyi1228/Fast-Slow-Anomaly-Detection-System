# v3-1 
# date: 20241001
# 将causal及uncausal mlp与net利用同一个优化器训练。
# args.lr作用于optimizer。
# 利用StepLR逐渐降低学习率，在后期更好地拟合。

from setproctitle import setproctitle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import StepLR

from dataset import VideoAnomalyDataset_C3D
from models import model
# from models.Chroma_VAE_v2 import Encoder, Decoder
from models.CVAEAno_ver9 import Encoder, Decoder
from models.classifier_ver2 import Classifier

from models.perturbmodel import PerturbationModel

from models.loss_functions import VAE_LL_loss, AE_loss
# 1030重启原始val
# from train_ad_perturb_ver3_val import val

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc, remake_video_3d_output
from utils.vis_utils import visualize_sequences
import plotting
import perturb
import random

# torch.backends.cudnn.benchmark = False

root_dir = '/home/zhaoyi/media/data3/'
# Config
def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    # parser.add_argument("--val_step", type=int, default=200)
    # parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--gpu_id_train", type=str, default=0)
    # parser.add_argument("--gpu_id_eval", type=str, default=0)

    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--static_threshold", type=float, default=0.3)
    parser.add_argument("--sample_num", type=int, default=9)
    parser.add_argument("--filter_ratio", type=float, default=0.5)
    parser.add_argument("--vae_checkpoint", type=str, default="/home/smbu/mediadisk/Jigsaw-VAD-main/results/20231210-22:37:11_ped2_GCE_ver1_2_zdim128/model-96-loss240,best.pkl")
    parser.add_argument("--classifier_checkpoint", type=str, default="./pretrained_classifier/model_latest.pkl")
    parser.add_argument("--dataset", type=str, default="shanghaitech", choices=['shanghaitech', 'ped2', 'avenue','UBnormal'])
    parser.add_argument("--checkpoint",type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--triplet", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--spat_loss_weight",type=float,default=1.0)
    # ce params
    parser.add_argument("--Nalpha", type=int, default=25)
    parser.add_argument("--Nbeta", type=int, default=25)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--L", type=int, default=96)
    parser.add_argument("--M", type=int, default=2)     
    parser.add_argument("--usece", type=float, default=0.0001)
    parser.add_argument("--perturb_lambda", type=float, default=1.)

    parser.add_argument("--model_version", type=str, default="CVAEver9_trainv3")
    # parser.add_argument("--aug_type", type=str, default="new")
    # parser.add_argument("--perturb_type", type=str, default="MLP", choices=["causal", "uncausal", "random", "singledim", "MLP", "none"])
    args = parser.parse_args()
    args.z_dim = args.K + args.L

    args.device_train = torch.device("cuda:{}".format(args.gpu_id_train) if torch.cuda.is_available() else "cpu")
    # print("gpu:", args.device_train)
    print(torch.cuda.device_count())
    args.device_eval = torch.device("cuda:{}".format(args.gpu_id_train) if torch.cuda.is_available() else "cpu")

    if args.dataset in ['shanghaitech', 'avenue','UBnormal']:
        args.filter_ratio = 0.8
    elif args.dataset == 'ped2':
        args.filter_ratio = 0.5
    
    setproctitle("{}_{}".format(args.dataset, args.model_version))
    return args


def gen_spat_labels(spat_labels):
    return torch.any(spat_labels != torch.arange(9), axis=1)


def train(args):
    if not args.log_date:
        running_date = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_data : {}".format(running_date))
    for k,v in vars(args).items():
        print("-------------{} : {}".format(k, v))

    if args.dataset == "avenue":
        data_dir = f"{root_dir}/{args.dataset}/training/frames"
    elif args.dataset == "ped2" or args.dataset == "shanghaitech"or args.dataset =='UBnormal':
        data_dir = f"{root_dir}/{args.dataset}/training/frames"
    else:
        raise NotImplementedError("dataset not implemented yet")
    
    detect_pkl = f'detect/{args.dataset}_train_detect_result_yolov3.pkl'

    vad_dataset = VideoAnomalyDataset_C3D(data_dir, 
                                          dataset=args.dataset,
                                          detect_dir=detect_pkl,
                                          fliter_ratio=args.filter_ratio, 
                                          frame_num=args.sample_num,
                                          static_threshold=args.static_threshold)
    
    # if args.dataset == "shanghaitech" or args.dataset == "UBnormal":
    #     args.sample_num = 7
    if args.dataset == "UBnormal":
        args.sample_num = 7
        
    vad_dataloader = DataLoader(vad_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    net = model.WideBranchNet(time_length=args.sample_num, num_classes=[args.sample_num ** 2, 81]).cuda(args.device_train)
    causal_perturbMLP = PerturbationModel(latent_dim=args.K)
    uncausal_perturbMLP = PerturbationModel(latent_dim=args.L)

    if args.checkpoint is not None:
        if os.path.isdir(args.checkpoint):
            state = torch.load(os.path.join(args.checkpoint, "best.pth"), weights_only=True)
            net.load_state_dict(state, strict=True)
            net.cuda(args.device_train)
            perturb_state = torch.load(os.path.join(args.checkpoint, "best_perturbator.pth"), weights_only=True)
            causal_perturbMLP.load_state_dict(perturb_state["causal"], strict=True)
            causal_perturbMLP.cuda(args.device_train)
            uncausal_perturbMLP.load_state_dict(perturb_state["uncausal"], strict=True)
            uncausal_perturbMLP.cuda(args.device_train)
        else:
            state = torch.load(args.checkpoint, weights_only=True)
            print('load ' + args.checkpoint)
            net.load_state_dict(state, strict=True)
            net.cuda(args.device_train)
        # smoothed_auc, smoothed_auc_avg, _, _, _, eer = val(args, net)
        # print("eer: {:.4f}".format(eer))
        # exit(0)
    
    # if args.perturb_type == 'MLP':
    
    
    
    classifier = Classifier(num_classes = [2]).cuda(args.device_train)
    classifier_weight = torch.load(args.classifier_checkpoint, map_location=args.device_train, weights_only=True)
    classifier.load_state_dict(classifier_weight)
    classifier.eval()
    perturb_criterion = torch.nn.TripletMarginLoss(margin=args.triplet)
    mse_criterion = torch.nn.MSELoss()
    # perturb_optimizer = optim.Adam(params=perturb_params, lr=args.lr)

    encoder = Encoder(z_dim=args.z_dim, x_dim=64).cuda(args.device_train)
    decoder = Decoder(z_dim=args.z_dim, x_dim=64).cuda(args.device_train)
    vae_weight = torch.load(args.vae_checkpoint, map_location=args.device_train, weights_only=True)
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(vae_weight["encoder"])
    decoder.load_state_dict(vae_weight["decoder"])
    scaler = GradScaler()
    
    spat_loss_weight = args.spat_loss_weight

    net.cuda(args.device_train)
    net = net.train()
    causal_perturbMLP.cuda(args.device_train)
    uncausal_perturbMLP.cuda(args.device_train)
    # smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, net)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    params = list(causal_perturbMLP.parameters()) + list(uncausal_perturbMLP.parameters()) + list(net.parameters())
    optimizer = optim.Adam(params=params, lr=args.lr)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    # Train
    log_dir = './log/{}_{}_zdim{}_{}/'.format(running_date, args.dataset, args.z_dim, args.model_version)
    save_dir = './ADresults/{}_{}_zdim{}_{}/'.format(running_date, args.dataset, args.z_dim, args.model_version)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    global_step = 0
    cur_eer = 0.
    max_acc = -1
    timestamp_in_max = None
    with open(os.path.join(save_dir, "model_settings.txt"), "w") as f:
        f.writelines("The running date : {}\n".format(running_date))
        for k,v in vars(args).items():
            f.writelines("-------------{} : {}\n".format(k, v))
    f.close()
    start_time = time.time()

    for epoch in range(args.epochs):
        tbar = tqdm(enumerate(vad_dataloader))

        # changed_sample_num = 0
        for it, data in tbar:
            with autocast():
                video, obj, temp_labels, spat_labels, t_flag = data['video'], data['obj'], data['label'], data["trans_label"], data["temporal"]
                n_temp = t_flag.sum().item()

                spat_labels_org = spat_labels
                obj = obj.cuda(args.device_train, non_blocking=True).half()
                # print(obj.shape,'obj')
                # print(spat_labels.shape,'spat_labels')
                
                s_flag = torch.any(spat_labels != torch.arange(9), axis=1) # spat_labels为jigsaw修改过的，则将该obj的s_flag设置为True
                y_label = torch.zeros(obj.shape[0], 1).cuda(args.device_train)
                y_label[s_flag] = 1
              
                # ver3 modify, 将s_flag==1且yc==1的序列替换为单张生成图像重复:
                tmp_slice = obj[:,:,3,:,:]
                selected = tmp_slice[s_flag]
                z, _, _, res = encoder(selected)

                z_c = z.clone()
                z_n = z.clone()
                z_c[:, :args.K] = causal_perturbMLP(z[:, :args.K])
                z_n[:, args.K:] = uncausal_perturbMLP(z[:, args.K:])
                xc = decoder(z_c,res)
                xn = decoder(z_n,res)
                xhat = decoder(z,res)

                prob_xc, _ = classifier(xc)
                yc = np.argmax(prob_xc.detach().cpu().numpy(), axis=1)
                indices = torch.where(s_flag)[0]

                # print(len(indices))
                # print(yc.shape)
                for i in range(len(indices)):
                    index = indices[i]
                    if yc[i] == 1:
                        obj[index, :, :, :, :] = xc[i].unsqueeze(1).repeat(1, 7, 1, 1)
                        # changed_sample_num += 1
                # end

                temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device_train)
                spat_labels = spat_labels[~t_flag].long().view(-1).cuda(args.device_train)

                temp_logits, spat_logits = net(obj)

                temp_logits = temp_logits[t_flag].view(-1, args.sample_num)
                spat_logits = spat_logits[~t_flag].view(-1, 9)

                temp_loss = criterion(temp_logits, temp_labels)
                spat_loss = criterion(spat_logits, spat_labels)
                loss = temp_loss + spat_loss * spat_loss_weight

                prob_x, _ = classifier(xhat)
                y_ = np.argmax(prob_x.detach().cpu().numpy(), axis=1)
                
                prob_xn, _ = classifier(xn)
                yn = np.argmax(prob_xn.detach().cpu().numpy(), axis=1)
                perturb_loss = perturb_criterion(prob_x, prob_xc, prob_xn)
                
                # 控制z_c及z_n尽可能对z的影响小
                mse_loss = mse_criterion(z_c-z, torch.zeros_like(z_c))
                mse_loss += mse_criterion(z_n-z, torch.zeros_like(z_n))

                # 0920 modify,修改perturb mse的大小。
                loss += perturb_loss+ args.perturb_lambda * mse_loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  #
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20)
            # 1001 modify: add mlp backpropagation
            torch.nn.utils.clip_grad_norm_(causal_perturbMLP.parameters(), max_norm=20)
            torch.nn.utils.clip_grad_norm_(uncausal_perturbMLP.parameters(), max_norm=20)
            # scaler.step(perturb_optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


            
            writer.add_scalar('Train/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Train/Temporal', temp_loss.item(), global_step=global_step)
            writer.add_scalar('Train/Spatial', spat_loss.item(), global_step=global_step)
            
            tbar.set_description("[{}:{}/{}] loss: {:.6f} t_loss: {:.6f} s_loss: {:.6f} \ttime: {:.6f}s".\
                format(epoch, it + 1, len(vad_dataloader), loss.item(), temp_loss.item(), spat_loss.item(),  time.time() - t0))
        
            t0 = time.time()

            global_step += 1

        # 1004 fix bug(忘了这句了，直接没更新学习率)
        scheduler.step()

        # print("changed: {}".format(changed_sample_num))
        smoothed_auc, smoothed_auc_avg, temp_timestamp,  spat_auc, temp_auc, eer = val(args, net)
        writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
        writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)
        writer.add_scalar('Test/spatial_auc', spat_auc, global_step=global_step)
        writer.add_scalar('Test/temporal_auc', temp_auc, global_step=global_step)

        with open(os.path.join(save_dir, "model_settings.txt"), "a+") as f:
            f.writelines("cur test time {}: auc: {} auc_avg:{} \n".\
                format(temp_timestamp, smoothed_auc, smoothed_auc_avg))
        f.close()
        # end 
        
        # # visualize
        # num_visualize = 4
        # y = np.concatenate([y_[:num_visualize], \
        #     yn[:num_visualize], yc[:num_visualize]], axis=0)
        # visualize_sequences(
        #     spat_labels_org[s_flag][:num_visualize].numpy(),
        #     obj[s_flag][:num_visualize].detach().cpu().numpy(), 
        #     xn[:num_visualize].detach().cpu().numpy(), 
        #     xc[:num_visualize].detach().cpu().numpy(),
        #     y, 
        #     save_dir,
        #     temp_timestamp,
        #     return_fig=True,
        #     aug= True)
        if epoch % 5 == 0:
            if not os.path.exists(os.path.join(save_dir, "epoch")):
               os.makedirs(os.path.join(save_dir, "epoch"), )
            save_every = os.path.join(save_dir, "epoch", 'e{}.pth'.format(epoch))
            save_every_ = os.path.join(save_dir, "epoch", 'e{}_perturbator.pth'.format(epoch))
            torch.save(net.state_dict(), save_every)
            torch.save({
                    "causal": causal_perturbMLP.state_dict(),
                    "uncausal": uncausal_perturbMLP.state_dict(),
                    }, save_every_)

        if smoothed_auc > max_acc:
            max_acc = smoothed_auc
            timestamp_in_max = temp_timestamp
            cur_eer = eer
            save = os.path.join(save_dir, '{}.pth'.format('best'))
            save_ = os.path.join(save_dir, '{}.pth'.format('best_perturbator'))
            
            torch.save(net.state_dict(), save)
            torch.save({
                    "causal": causal_perturbMLP.state_dict(),
                    "uncausal": uncausal_perturbMLP.state_dict()
                    }, save_)
        
        # print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
        # net = net.train()
        print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max + ", relate eer:{:.4f}".format(cur_eer))
        net = net.train()

    end_time = time.time()
    print("Time used for AD model training: {:.2f} seconds".format(end_time-start_time))
    with open(os.path.join(save_dir, "model_settings.txt"), "a+") as f:
        f.writelines("Final max AUC: {:.4f}, relate EER: {:.4f}".format(max_acc, cur_eer))
        f.writelines("Time used for AD model training: {:.2f} seconds".format(end_time-start_time))
    f.close()
            
# 1030 重启val，以防二遍测试太复杂
def val(args, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    if args.dataset == "avenue":
        data_dir =  f"{root_dir}/{args.dataset}/testing/frames"
    elif args.dataset == "ped2" or args.dataset == "shanghaitech" or args.dataset == 'UBnormal':
        data_dir =  f"{root_dir}/{args.dataset}/testing/frames"
    else:
        raise NotImplementedError("dataset not implemented yet")
    
    detect_pkl = f'detect/{args.dataset}_test_detect_result_yolov3.pkl'

    if args.dataset == "shanghaitech" or args.dataset == "UBnormal":
        args.sample_num = 9
    testing_dataset = VideoAnomalyDataset_C3D(data_dir, 
                                              dataset=args.dataset,
                                              detect_dir=detect_pkl,
                                              fliter_ratio=args.filter_ratio,
                                              frame_num=args.sample_num)
    testing_data_loader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)

    if args.dataset == "shanghaitech" or args.dataset == "UBnormal":
        args.sample_num = 7
    net.eval()

    video_output = {}
    for data in tqdm(testing_data_loader):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].cuda(args.device_train)
    
        with torch.no_grad():
            temp_logits, spat_logits = net(obj)
            temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        
        for video_, frame_, s_score_, t_score_  in zip(videos, frames, scores, scores2):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([s_score_, t_score_])

    # micro_auc, macro_auc = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    # return micro_auc, macro_auc, running_date

    micro_auc, macro_auc, spat_auc, temp_auc, err = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    return micro_auc, macro_auc, running_date, spat_auc, temp_auc, err


def save_and_evaluate(video_output, running_date, dataset='shanghaitech'):
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    if dataset == 'shanghaitech' or dataset == 'UBnormal':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output, dataset=dataset)
    else:
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(video_output, dataset=dataset)

    spatial_res, _, _ = evaluate_auc(video_output_spatial, dataset=dataset)
    temporal_res, _, _ = evaluate_auc(video_output_temporal, dataset=dataset)
    smoothed_res, smoothed_auc_list, err = evaluate_auc(video_output_complete, dataset=dataset)
    # 6.11 modify end
    
    return smoothed_res.auc, np.mean(smoothed_auc_list), spatial_res.auc, temporal_res.auc, err
    # 6.12 modify



if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)
