import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
import os
import random
import torch
import copy

class RISEV(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISEV, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.loss_fn = nn.CosineSimilarity(eps=1e-6)

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath, p1=0.1):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = p1
        

    def forward(self, image, text_features):
        N = self.N
        _, _, H, W = image.size()
        stack = torch.mul(self.masks, image.data)
        p = []
        for i in range(0, N, self.gpu_batch):
            image_features = self.model.get_image_features(stack[i:min(i + self.gpu_batch, N)])
            p.append(self.loss_fn(image_features, text_features).detach().cpu().unsqueeze(-1))
        p = torch.cat(p)
        sal = torch.matmul(p.data.transpose(0, 1).float(), self.masks.cpu().view(N, H * W))
        sal = sal.view((1, H, W))
        sal = sal / N / self.p1
        sal = sal.mean(0, keepdim=True)
        sal = (sal - sal.min()) / (sal.max() - sal.min())
        return sal
    
def rise_v(model,image,text_features):
    exp = RISEV(model, (224, 224), gpu_batch=20)
    if os.path.exists('masks.npy'):
        exp.load_masks('masks.npy')
    else:
        exp.generate_masks(6000, 8, 0.1, savepath='masks.npy')
    sal = exp(image, text_features)
    return sal



class RISET(nn.Module):
    def __init__(self, model):
        super(RISET, self).__init__()
        self.ori_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        self.loss_fn = nn.CosineSimilarity(eps=1e-6)

    def generate_masks(self,input_size, N, s, p1):
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size
        
        np.random.seed(0)

        grid = np.random.rand(N, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *input_size))

        for i in range(N):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        masks = masks.reshape(-1, 1, *input_size).squeeze()
        masks = torch.from_numpy(masks).float()
        masks = masks.cuda()
        N = N
        p1 = p1
        return masks, N, p1


    def forward(self, tid, image_features, N=200, s=8, p1=0.1):
        masks, N, p1 = self.generate_masks((tid.shape[-1],512), N, s, p1)
        sal = list()
        for i in range(0, N):
            self.model.text_model.embeddings.token_embedding.weight.data[tid,:] = self.ori_model.text_model.embeddings.token_embedding.weight.data[tid,:] * masks[i]
            text_features = self.model.get_text_features(tid)
            sal.append(self.loss_fn(image_features, text_features).detach().cpu().unsqueeze(-1) * masks[i].cpu())
        sal = torch.stack(sal)
        sal = sal.mean(0).sum(-1) / p1
        sal_min = sal.min()
        sal_max = sal.max()
        sal = (sal - sal_min) / (sal_max - sal_min)
        return sal



def rise_t(model,image,text_features):
    exp = RISET(model)
    sal = exp(image, text_features)
    return sal



def rise(model,image_feat,tids,image_features,text_features):
    saliency_v = []
    saliency_t = []
    for idx in range(image_feat.shape[0]):
        t_id = tids[idx]
        i_feat = image_feat[idx:idx+1]
        image_feature = image_features[idx:idx+1].cuda()
        text_feature = text_features[idx]
        vmap = rise_v(model,i_feat,text_feature)
        tmap = rise_t(model,t_id,image_feature)
        saliency_v.append(vmap)
        saliency_t.append(tmap)
    return np.stack(saliency_v, axis=0),saliency_t