import torch
import numpy as np

def saliencymap_vision(model,processor,img, prompt):
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to('cuda')
    out = model(**inp, output_attentions=True)
    model.zero_grad()    
    logit = out.logits_per_image[0, 0]
    A = out.vision_model_output.attentions[-1]
    grad = torch.autograd.grad(logit, A)[0].detach().abs()
    grad = grad[0, :, 0, 1:].mean(dim=0)
    grad = torch.nn.functional.interpolate(grad.reshape(1, 1, 7, 7), size=224, mode='bilinear')[0, 0].cpu().detach().numpy()
    heatmap = (grad - grad.min()) / (grad.max() - grad.min())
    return heatmap


def saliencymap_text(model,processor,img, prompt):
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to('cuda')
    out = model(**inp, output_attentions=True)
    model.zero_grad()
    logit = out.logits_per_text[0, 0]
    A = out.text_model_output.attentions[-1]
    grad = torch.autograd.grad(logit, A)[0].detach().abs()
    heatmap = (grad[0, :, -1]).mean(dim=0).cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap



def saliencymap(model,processor,captions,image_feat):
    saliency_v = []
    saliency_t = []
    for idx in range(len(captions)):
        i_feat = image_feat[idx]
        caption = captions[idx]
        vmap = saliencymap_vision(model,processor,i_feat, caption)
        tmap = saliencymap_text(model,processor,i_feat, caption)
        saliency_v.append(vmap)
        saliency_t.append(tmap)
    saliency_v = np.stack(saliency_v, axis=0)
    return saliency_v,saliency_t