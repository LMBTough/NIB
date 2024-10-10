import torch
import numpy as np
import copy
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


def get_hs(model, image_feats):
    hidden_states = model.vision_model.embeddings(image_feats)
    return hidden_states

def get_output(model, hidden_states):
    output_attentions = False
    output_hidden_states = True
    return_dict = True
    hidden_states = model.vision_model.pre_layrnorm(hidden_states)

    attention_mask = None
    causal_attention_mask = None
    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for idx in range(len(model.vision_model.encoder.layers)):
        encoder_layer = model.vision_model.encoder.layers[idx]
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        layer_outputs = encoder_layer(
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = layer_outputs[0]

    if output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)


    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    encoder_outputs = BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = model.vision_model.post_layernorm(pooled_output)

    vision_outputs = BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
    pooled_output = vision_outputs[1]
    image_features = model.visual_projection(pooled_output)
    return image_features
    

def mfaba_vision(model,processor,img, prompt):
    loss_fn = torch.nn.CosineSimilarity()
    inp = processor(
        text=[prompt],
        images=img,
        return_tensors="pt",
    )
    for k in inp:
        inp[k] = inp[k].to('cuda')
    grads = list()
    hs = get_hs(model, inp['pixel_values'])
    text_features = model.get_text_features(inp['input_ids'])
    hats = [hs]
    grads = list()
    for _ in range(10):
        hs = torch.autograd.Variable(hs, requires_grad=True)
        image_features = get_output(model, hs)
        model.zero_grad()
        loss = loss_fn(image_features, text_features).mean()
        grad = torch.autograd.grad(loss, hs)[0]
        hs = hs - 0.01 * grad.sign()
        hats.append(hs)
        grads.append(grad.detach())
        
    hats = torch.stack(hats)
    hats = hats[1:] - hats[:-1]
    grads = torch.stack(grads)
    heatmap = -torch.sum(hats * grads, dim=0)
    saliency = torch.nansum(heatmap, -1)[:, 1:]
    dim = 7
    saliency = saliency.reshape(saliency.shape[0], 1, dim, dim)
    saliency = torch.nn.functional.interpolate(
        saliency, size=224, mode='bilinear')
    saliency = saliency.cpu().detach().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency


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