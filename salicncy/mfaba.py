import torch
import numpy as np
import copy

def mfaba_vision(model,processor,img, prompt):
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to('cuda')
    grads = list()
    hats = [inp['pixel_values'].clone().detach().cpu()]
    for _ in range(10):
        inp['pixel_values'] = torch.autograd.Variable(inp['pixel_values'], requires_grad=True)
        out = model(**inp, output_attentions=True)
        model.zero_grad()
        logit = out.logits_per_image[0, 0]
        grad = torch.autograd.grad(logit, inp['pixel_values'])[0]
        grads.append(grad.cpu().detach())
        inp['pixel_values'] = inp['pixel_values'] - 0.01 * grad.sign()
        hats.append(inp['pixel_values'].clone().detach().cpu())
    
    hats = torch.stack(hats)
    hats = hats[1:] - hats[:-1]
    grads = torch.stack(grads)
    heatmap = -torch.sum(hats * grads, dim=0).squeeze().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def mfaba_text(model,processor,img, prompt):
    model2 = copy.deepcopy(model)
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to('cuda')
    
    hats = [model2.text_model.embeddings.token_embedding.weight[inp['input_ids'][0]].cpu()]
    grads = list()
    for _ in range(10):
        out = model2(**inp, output_attentions=True)
        model2.zero_grad()  
        logit = out.logits_per_text[0, 0]
        grad = torch.autograd.grad(logit, model2.text_model.embeddings.token_embedding.weight)[0]
        grads.append(grad.cpu().detach()[inp['input_ids'][0].cpu()])
        model2.text_model.embeddings.token_embedding.weight.data = model2.text_model.embeddings.token_embedding.weight.data - 0.01 * grad.sign()
        hats.append(model2.text_model.embeddings.token_embedding.weight[inp['input_ids'][0]].cpu())
    grads = torch.stack(grads)
    hats = torch.stack(hats)
    hats = hats[1:] - hats[:-1]
    attribution = -torch.sum(hats * grads, dim=0).squeeze()
    attribution = torch.nansum(attribution, dim=-1)
    heatmap = attribution.cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def mfaba(model,processor,captions,image_feat):
    saliency_v = []
    saliency_t = []
    for idx in range(len(captions)):
        i_feat = image_feat[idx]
        caption = captions[idx]
        vmap = mfaba_vision(model,processor,i_feat, caption)
        tmap = mfaba_text(model,processor,i_feat, caption)
        saliency_v.append(vmap)
        saliency_t.append(tmap)
    saliency_v = np.stack(saliency_v, axis=0)
    return saliency_v,saliency_t