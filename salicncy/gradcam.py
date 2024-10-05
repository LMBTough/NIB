import torch
import numpy as np
import torch.nn.functional as F

class GradCAMCLIP(object):
    def __init__(self,model,processor,is_text=False):
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output[0]
            return None

        if is_text:
            target_layer = model.text_model.encoder.layers[9]
        else:
            target_layer = model.vision_model.encoder.layers[9]
        

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.model = model
        self.is_text = is_text
        self.processor = processor


    def forward(self, img, prompt):
        inp = self.processor(
            text=[prompt],
            images=img,
            return_tensors="pt",
        )
        for k in inp:
            inp[k] = inp[k].to('cuda')
        out = self.model(**inp, output_attentions=True)
        self.model.zero_grad()
        if self.is_text:
            logit = out.logits_per_text[0, 0]
        else:
            logit = out.logits_per_image[0, 0]
        logit.backward()
        gradients = self.gradients['value']
        activations = self.activations['value']
        saliency_map = (gradients * activations).sum(-1)
        saliency_map = F.relu(saliency_map)
        saliency_map = saliency_map[:,1:] if not self.is_text else saliency_map
        if not self.is_text:
            saliency_map = saliency_map.view(1 , 1, 7, 7)
            saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        if saliency_map_max == saliency_map_min:
            return torch.zeros_like(saliency_map).cpu().detach().numpy()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).cpu().detach().numpy()
        return saliency_map
        

    def __call__(self, img,caption):
        return self.forward(img,caption)



def gradcam(model,processor,captions,image_feat):
    saliency_v = []
    saliency_t = []
    for idx in range(len(captions)):
        i_feat = image_feat[idx]
        caption = captions[idx]
        vamp = GradCAMCLIP(model,processor,is_text=False)
        tmap = GradCAMCLIP(model,processor,is_text=True)
        vmap = vamp(i_feat, caption)
        tmap = tmap(i_feat, caption)
        saliency_v.append(vmap.squeeze())
        saliency_t.append(tmap.squeeze())
    saliency_v = np.stack(saliency_v, axis=0)
    return saliency_v,saliency_t