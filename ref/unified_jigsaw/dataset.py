import os 
from numpy.random import f, permutation, rand
from PIL import Image
import time
import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
ROOT_DIR = "/home/zhaoyi/media/data3/"
# for all four dataset, sample num = 7

class VideoAnomalyDataset_C3D(Dataset):
    """Video Anomaly Dataset."""
    def __init__(self,
                 data_dir, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        assert dataset in ['shanghaitech', 'ped2', 'avenue', 'UBnormal'], 'wrong type of dataset.'
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.fliter_ratio = fliter_ratio
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()
        if 'train' in self.data_dir and self.dataset == 'UBnormal':
            # file_list = glob.glob(os.path.join(data_dir, 'normal*'))
            file_list = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder)) and not folder.startswith('abnormal')]

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2 # 3 or 4

        self.videos_list = []

        if('train' in data_dir):
            self.test_stage = False
        elif('test' in data_dir):
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'
        if not self.test_stage and self.dataset == 'shanghaitech':
            self.sample_step = 5
        else:
            self.sample_step = 1
        
        if detect_dir != None:
            with open(detect_dir, 'rb') as f:
                self.detect = pickle.load(f)
        else:
            self.detect = None

        self.objects_list = []
        # print(file_list)
        self._load_data(file_list)
    
    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        contain = 0
        total_small_ = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    # 测试阶段中间帧作为起始帧3/4，否则为最后一帧6/8
        for video_file in file_list:            # 遍历所有视频
            if "DS_Store" in video_file:
                continue
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)        # 所有视频帧
            # print(l)
            self.videos += 1
            length = len(l)
            
            total_frames += length
            # print("start_ind: {}, length: {}, self.sample_step: {}".format(start_ind, length,self.sample_step))
            # print("lenght", len(self.detect[video_file]))
            for frame in range(start_ind, length - start_ind, self.sample_step):        # frame从中间帧到
                # avenue/ped test: 3~ length-3 train: 6~ length-6
                # stc test: 4~length-4 train: 8~length-8(stride=5)
                # UBnormal test: 4~length-4 train: 8~length-8
                if self.detect is not None:
                    # print(video_file)
                    # print("video_file:",video_file,"\nframe:" ,frame)
                    # print(self.detect[video_file])
                    # print(self.detect[video_file])
                    # if video_file == '05':
                    #     print(video_file,frame,length)
                    detect_result = self.detect[video_file][frame]
                    detect_result = detect_result[detect_result[:, 4] > self.fliter_ratio, :]
                    object_num = detect_result.shape[0]
                else:
                    object_num = 1

                flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(object_num, 0)
                is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
                is_contain = is_contain.any(-1)
                if self.dataset == 'UBnormal':
                    detect_result[:,0] = detect_result[:,0] * 1104
                    detect_result[:,2] = detect_result[:,2] * 1104
                    detect_result[:,1] = detect_result[:,1] * 720
                    detect_result[:,3] = detect_result[:,3] * 720
                is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10     # 去掉小物体
                width = detect_result[:, 2] - detect_result[:, 0]
                height = detect_result[:, 3] - detect_result[:, 1]
                # aspect_ratio = np.minimum(width / height, height / width) 
                aspect_ratio = height / width
                for i in range(object_num):
                    if not is_contain[i]:
                        if not is_small[i]:
                            self.objects_list.append({"video_name":video_file, "frame":frame, "object": i, 
                                "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                        else:
                            total_small_ += 1
                    else:
                        contain += 1

        print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s."\
            .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))

    def __len__(self):
        return len(self.objects_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx): 
        temproal_flag = idx % 2 == 0 
        record = self.objects_list[idx]

        # 修改开始
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:               # 训练过程中，以0.9999的概率计算时序扰动结果（perm）。
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)

        if self.test_stage:
            perm = np.arange(7)
        else:               # 训练过程中，以0.9999的概率计算时序扰动结果（perm）。
            if random.random() < 0.0001:
                perm = np.arange(7)
            else:
                perm = np.random.permutation(7)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])
        # return obj
        # 修改结束

        if not temproal_flag and not self.test_stage:   # 训练且不进行时序扰动
            if random.random() < 0.0001:                # 以概率0.9999计算空间扰动结果（spatial_perm）
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        # 进行空间增强生成
        obj_copy = torch.from_numpy(np.copy(obj))

        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False) 
        obj = torch.from_numpy(obj)

        clip_id = str(record["frame"]) + '_' + str(record["object"])

        # 修改时序扰动的帧数为7
        # NOT permute clips containing static contents，如果obj运动太小，则不进行时序扰动
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(7)

        # 修改结束

        # 进行时序增强生成
        if temproal_flag:
            obj = obj[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)  #控制所有元素在[0,1]之间
        obj_copy = torch.clamp(obj_copy, 0., 1.) 
        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm, 
            "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
            "temporal": temproal_flag,
            "obj_copy": obj_copy,
            }
        return  ret
    

    def get_object(self, video_name, frame, obj_id):
        # change the root dir with the environment.
        if self.dataset == "shanghaitech":
            video_dir = os.path.join(ROOT_DIR, 'prefetched', self.dataset+"7", self.phase, video_name)
        else:
            video_dir = os.path.join(ROOT_DIR, 'prefetched', self.dataset, self.phase, video_name)
        # 1216 scr comment:
        # start scr
        # if self.dataset == 'shanghaitech':
        #     video_dir = os.path.join('/data/Part1/lizhaoyi/Jigsaw-VAD-main',self.dataset, self.phase, video_name) 
        # end scr
        obj = np.load(os.path.join(video_dir, str(frame) + '_' + str(obj_id) + '.npy'))   # (3, 7, 64, 64)
        # print("obj.shape:",obj.shape)

        #对stc和ubnormal的特殊处理，将sample数量从9变成7，去掉开头和结尾的两帧。（1003:已保存stc sample-num=7 版本，保存前七帧，后两帧忽略）
        # if self.dataset == "shanghaitech" or self.dataset == "UBnormal":
        #     obj = obj[:, 1:8,:,:]
        if self.dataset == "UBnormal":
            obj = obj[:, :7,:,:]

        if not self.test_stage:
            if random.random() < 0.5: #以0.5的概率将obj进行水平翻转
                obj = obj[:, :, :, ::-1]

        # print(obj.shape)
        return obj

    

    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list


    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        # 修改开始
        # clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        # 不同数据集都设置成7个sample
        clip = np.zeros((3, 7, 64, 64), dtype=np.float32)
        # 修改结束
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip


    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permuation=permuation, num=3, dropout=dropout)
        return clip


class VAD_C3D_wtrans(VideoAnomalyDataset_C3D):
    def __init__(self,
                 data_dir, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):
        super(VAD_C3D_wtrans, self).__init__(data_dir, dataset, detect_dir, fliter_ratio, frame_num, static_threshold)

    def irregular(self, clip):
        def rotate(clip):
            result_matrix = np.copy(clip)
    # angle_range = (90, 90)  # 可根据需求调整
            p = random.random()
            if 0 <= p <0.5:
                angle_ = -90
            else:
                angle_ = 90
            
            c,l,h,w = clip.shape
            rotate_transform = transforms.RandomRotation(degrees=(angle_, angle_)) #, interpolation=transforms.InterpolationMode.BILINEAR)
            
            for i in range(l):
                clip_torch = torch.from_numpy(clip[:,i,:,:].copy())
                # 下面这种写法会报错，加上.copy()以后就ok了
                # ValueError: At least one stride in the given numpy array is negative,
                #  and tensors with negative strides are not currently supported. (You can probably work 
                # around this by making a copy of your array  with array.copy().)
                # clip_torch = torch.from_numpy(clip[:,i,:,:])
                clip_aug = rotate_transform(clip_torch)
                
                result_matrix[:,i,:,:] = clip_aug.numpy()
            return result_matrix
            
        def flip(clip):
            result_matrix = np.copy(clip)
            p = 1 # flip概率
            c,l,h,w = clip.shape
            Hflip = transforms.RandomHorizontalFlip(p)
            Vflip = transforms.RandomVerticalFlip(p)
            # rotate_transform = transforms.RandomRotation(degrees=angle_range,interpolation=transforms.InterpolationMode.BILINEAR)
            phv = random.random()
            for i in range(l):
                clip_torch = torch.from_numpy(clip[:,i,:,:].copy())
                # if phv < 0.5:
                #     clip_aug = Hflip(clip_torch)
                # else:
                #     clip_aug = Vflip(clip_torch)
                clip_aug = Hflip(clip_torch)
                result_matrix[:,i,:,:] = clip_aug.numpy()
            return result_matrix
    
        def colorjitter(clip):
            result_matrix = np.copy(clip)
            c,l,h,w = clip.shape
            colorFunc = transforms.ColorJitter(brightness=[0.5,1.8],
                                                contrast=[1.,1.],
                                                saturation=0) 
            for i in range(l):
                clip_torch = torch.from_numpy(clip[:,i,:,:].copy())
                clip_aug = colorFunc(clip_torch)
                result_matrix[:,i,:,:] = clip_aug.numpy()
            return result_matrix    
        
        def gaussianblur(clip):
            result_matrix = np.copy(clip)
            c,l,h,w = clip.shape
            gaussianFunc = transforms.GaussianBlur(kernel_size=(7,7),
                                                sigma=(0.9,0.9))
            for i in range(l):
                clip_torch = torch.from_numpy(clip[:,i,:,:].copy())
                clip_aug = gaussianFunc(clip_torch)
                result_matrix[:,i,:,:] = clip_aug.numpy()
            return result_matrix

        def patchmask(clip):
            result_matrix = np.copy(clip)
            c,l,h,w = clip.shape
            patch_size = int(h * ((random.random()*7+3)/20))
            loc_x = random.randint(0,h-patch_size-1)
            loc_y = random.randint(0,w-patch_size-1)

            patch = np.zeros([c, 1, patch_size, patch_size])

            mean_ = np.mean(np.mean(np.mean(result_matrix[ :, :, loc_x:loc_x+patch_size, loc_y:loc_y+patch_size],axis=3),axis=2),axis=1)

            patch[0,:,:,:] = mean_[0]
            patch[1,:,:,:] = mean_[1]
            patch[2,:,:,:] = mean_[2]
            
            result_matrix[ :, :, loc_x:loc_x+patch_size, loc_y:loc_y+patch_size] = patch
            return result_matrix

        

        aug_list = ["gaussian", "rotate", "flip", "patchmask", "colorjitter"]

        # 随机选择，多种可重复，至少选择1种，至多选择 (n+1)//2种，其中n为aug_list的长度，目前为3种
        random_aug = random.choices(aug_list, k=random.randint(1, (len(aug_list)+1)//2))
        # 随机选择，多种不重复，至少选择1种，至多选择 (n+1)//2种，其中n为aug_list的长度，目前为3种
        # random_aug = random.sample(aug_list, k=random.randint(1, (len(aug_list)+1)//2))

        aug_prob = np.random.rand((len(random_aug)))
        random.shuffle(random_aug)
        
        if aug_prob.mean()<0.5:
            aug_prob[0] = aug_prob[0] + 0.5
        
        for idx, aug_type in enumerate(random_aug):
            if aug_prob[idx] > 0.5:
                # if aug_type == "replace":
                #     clip = replace(clip)
                if aug_type == 'rotate':
                    clip = rotate(clip)
                    continue
                if aug_type == 'flip':
                    clip = flip(clip)
                    continue
                if aug_type == 'colorjitter':
                    clip = colorjitter(clip)
                    continue
                if aug_type  == 'gaussian':
                    clip = gaussianblur(clip)
                    continue
                if aug_type == 'patchmask':
                    clip = patchmask(clip)
                    continue
        return clip 
        # return clip
    

    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0 
        record = self.objects_list[idx]
        
        if self.test_stage:
            perm = np.arange(7)
        else:               # 训练过程中，以0.9999的概率计算时序扰动结果（perm）。
            if random.random() < 0.0001:
                perm = np.arange(7)
            else:
                perm = np.random.permutation(7)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])
        # print(obj.shape)
        # print("obj.max():", obj.max())

        # modify object with random transformations
        if random.random() > 0.5:
            obj = self.irregular(obj)

        if not temproal_flag and not self.test_stage:   # 训练且不进行时序扰动
            if random.random() < 0.0001:                # 以概率0.9999计算空间扰动结果（spatial_perm）
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        # 进行空间增强生成
        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False) 
        obj = torch.from_numpy(obj)
        obj = torch.clamp(obj, 0., 1.)
        # clip_id = str(record["frame"]) + '_' + str(record["object"])

        # 修改时序扰动的帧数为7
        # NOT permute clips containing static contents，如果obj运动太小，则不进行时序扰动
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(7)

        # 修改结束


        # 进行时序增强生成
        if temproal_flag:
            obj = obj[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)  #控制所有元素在[0,1]之间

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm, 
            "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
            "temporal": temproal_flag}
        return  ret
    

class VideoAnomalyDataset_C3D_check(VideoAnomalyDataset_C3D):
    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0 
        record = self.objects_list[idx]
        
        obj = self.get_object(record["video_name"], record["frame"], record["object"])
        perm = np.arange(7)
        spatial_perm = np.random.permutation(9)
        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False) 
        obj = torch.from_numpy(obj)
        obj = torch.clamp(obj, 0., 1.)
        # clip_id = str(record["frame"]) + '_' + str(record["object"])

        # 修改时序扰动的帧数为7
        # NOT permute clips containing static contents，如果obj运动太小，则不进行时序扰动
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        obj = torch.clamp(obj, 0., 1.)  #控制所有元素在[0,1]之间

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm, 
            "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
            "temporal": temproal_flag}
        return  ret


class VAD_C3D_wtrans_after_aug(VideoAnomalyDataset_C3D):
    def __init__(self,
                 data_dir, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):
        super(VAD_C3D_wtrans_after_aug, self).__init__(data_dir, dataset, detect_dir, fliter_ratio, frame_num, static_threshold)

    def irregular(self, clip):
        
        # aug_list = ["gaussian", "rotate", "flip", "patchmask", "colorjitter"]
        aug_list = ["patchmask", "brightness", "snow", "impulse_noise"]
        from trans_utils import augmentation_functions
        # 随机选择，多种可重复，至少选择1种，至多选择 (n+1)//2种，其中n为aug_list的长度，目前为3种
        # random_aug = random.choices(aug_list, k=random.randint(1, (len(aug_list)+1)//2))
        # 随机选择，多种不重复，至少选择1种，至多选择 (n+1)//2种，其中n为aug_list的长度，目前为3种
        # random_aug = random.sample(aug_list, k=random.randint(1, (len(aug_list)+1)//2))

        aug_prob = np.random.rand((len(random_aug)))
        random.shuffle(random_aug)
        
        if aug_prob.mean()<0.5:
            aug_prob[0] = aug_prob[0] + 0.5
        
        for idx, aug_type in enumerate(random_aug):
            if aug_prob[idx] > 0.5:
                if aug_type in augmentation_functions:
                    clip = augmentation_functions[aug_type](clip, severity=1)
                else:
                    raise NotImplementedError("Augmentation type {} is not implenmented.".format(args.aug_type))
        
        # copy one image to all the 7 frames.
        clip = np.tile(clip[:,3,:,:][:,None], (1,7,1,1))
            # if aug_prob[idx] > 0.5:
            #     if aug_type == 'rotate':
            #         clip = rotate(clip)
            #         continue
            #     if aug_type == 'flip':
            #         clip = flip(clip)
            #         continue
            #     if aug_type == 'colorjitter':
            #         clip = colorjitter(clip)
            #         continue
            #     if aug_type  == 'gaussian':
            #         clip = gaussianblur(clip)
            #         continue
            #     if aug_type == 'patchmask':
            #         clip = patchmask(clip)
            #         continue
        return clip 
        # return clip
    

    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0 
        record = self.objects_list[idx]
        
        if self.test_stage:
            perm = np.arange(7)
        else:               # 训练过程中，以0.9999的概率计算时序扰动结果（perm）。
            if random.random() < 0.0001:
                perm = np.arange(7)
            else:
                perm = np.random.permutation(7)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])
        # print(obj.shape)
        # print("obj.max():", obj.max())


        if not temproal_flag and not self.test_stage:   # 训练且不进行时序扰动
            if random.random() < 0.0001:                # 以概率0.9999计算空间扰动结果（spatial_perm）
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        # 进行空间增强生成
        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False) 
        
        # modify object with random transformations after jigsaw
        # 1023 
        if not temproal_flag and random.random() > 0.5:
            obj = self.irregular(obj)

        obj = torch.from_numpy(obj)
        obj = torch.clamp(obj, 0., 1.)

        
        # clip_id = str(record["frame"]) + '_' + str(record["object"])

        # 修改时序扰动的帧数为7
        # NOT permute clips containing static contents，如果obj运动太小，则不进行时序扰动
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(7)

        # 修改结束


        # 进行时序增强生成
        if temproal_flag:
            obj = obj[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)  #控制所有元素在[0,1]之间

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm, 
            "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
            "temporal": temproal_flag}
        return  ret
    


class VAD_C3D_all_aug(VideoAnomalyDataset_C3D):
    def __init__(self,
                 data_dir, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):
        super(VAD_C3D_all_aug, self).__init__(data_dir, dataset, detect_dir, fliter_ratio, frame_num, static_threshold)
    
    def get_all_aug(self, obj):
        # aug_list = ["gaussian", "rotate", "flip", "patchmask", "colorjitter"]
        """
        a:brightness
        b:patchmask
        c:impulse_noise
        d:snow
        e:fog
        f:jpeg
        g:elastic
        h:defocus_blur
        i:gaussian_blur
        j:gaussian_noise
        k:motion_blur
        l:shot_noise
        m:speckle_noise
        """
        aug_list = ["none", "brightness", "patchmask", "impulse_noise", "snow", "fog", 
        "jpeg_compression", "elastictransform", "defocus_blur", "gaussian_blur",
        "gaussian_noise", "motion_blur", "shot_noise", "speckle_noise",]
        from trans_utils import augmentation_functions
        
        obj_list = []
        obj_raw = np.array(obj)
        for idx, aug_type in enumerate(aug_list):
            # if aug_prob[idx] > 0.5:
            if aug_type in augmentation_functions:
                # print("aug_type:{} obj_min:{} obj_max:{}".format(aug_type, np.min(obj), np.max(obj)))
                clip = augmentation_functions[aug_type](obj, severity=1)
                obj_list.append(clip[:,3,:,:])
            elif aug_type == "none":
                obj_list.append(obj[:,3,:,:])
            else:

                raise NotImplementedError("Augmentation type {} is not implenmented.".format(args.aug_type))
            
        # copy one image to all the 7 frames.
        # clip = np.tile(clip[:,3,:,:][:,None], (1,7,1,1))

        return {"obj_list":obj_list, "aug_list": aug_list}


    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0 
        record = self.objects_list[idx]
        
        if self.test_stage:
            perm = np.arange(7)
        else:               # 训练过程中，以0.9999的概率计算时序扰动结果（perm）。
            if random.random() < 0.0001:
                perm = np.arange(7)
            else:
                perm = np.random.permutation(7)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])
        # print(obj.shape)
        # print("obj.max():", obj.max())

        # modify object with random transformations
    
        aug_obj_dict = self.get_all_aug(obj)

        if not temproal_flag and not self.test_stage:   # 训练且不进行时序扰动
            if random.random() < 0.0001:                # 以概率0.9999计算空间扰动结果（spatial_perm）
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        # 进行空间增强生成
        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False) 
        obj = torch.from_numpy(obj)
        obj = torch.clamp(obj, 0., 1.)
        # clip_id = str(record["frame"]) + '_' + str(record["object"])

        # 修改时序扰动的帧数为7
        # NOT permute clips containing static contents，如果obj运动太小，则不进行时序扰动
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(7)

        # 修改结束


        # 进行时序增强生成
        if temproal_flag:
            obj = obj[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)  #控制所有元素在[0,1]之间

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm, 
            "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
            "temporal": temproal_flag, "aug_obj_dict":aug_obj_dict}
        return  ret
    