from scripts.methods import vision_heatmap_iba,text_heatmap_iba
import numpy as np

def m2ib(model,text_ids,image_feat,beta=0.1):
    saliency_v = []
    saliency_t = []
    for idx in range(image_feat.shape[0]):
        t_id = text_ids[idx]
        i_feat = image_feat[idx:idx+1]
        vmap = vision_heatmap_iba(t_id, i_feat, model, 9, beta, 1)
        tmap = text_heatmap_iba(t_id, i_feat, model, 9, beta, 1)
        saliency_v.append(vmap)
        saliency_t.append(tmap)
    return np.stack(saliency_v, axis=0),saliency_t