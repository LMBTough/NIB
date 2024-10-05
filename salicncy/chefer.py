import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpret_vision(model,processor,img, prompt):
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
    grad = torch.autograd.grad(logit, A)[0].detach()
    R = (grad[0, :, 0, 1:] * A.detach()[0, :, 0, 1:]).clamp(min=0).mean(dim=0)
    heatmap  = torch.nn.functional.interpolate(R.reshape(1, 1, 7, 7), size=224, mode='bilinear')[0, 0].cpu().detach().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    return heatmap


def interpret_text(model,processor,img, prompt):
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to(device)
    out = model(**inp, output_attentions=True)
    model.zero_grad()    
    logit = out.logits_per_text[0, 0]
    A = out.text_model_output.attentions[-1]
    grad = torch.autograd.grad(logit, A)[0].detach()
    
    heatmap = (grad[0, :, -1] * A.detach()[0, :, -1]).clamp(min=0).mean(dim=0).cpu().detach().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    return heatmap

def chefer(model,processor,captions,image_feat):
    saliency_v = []
    saliency_t = []
    for idx in range(len(captions)):
        i_feat = image_feat[idx]
        caption = captions[idx]
        vmap = interpret_vision(model,processor,i_feat, caption)
        tmap = interpret_text(model,processor,i_feat, caption)
        saliency_v.append(vmap)
        saliency_t.append(tmap)
    saliency_v = np.stack(saliency_v, axis=0)
    return saliency_v,saliency_t