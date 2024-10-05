import os
import torch
import numpy as np
import sys
import copy
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric
sys.path.append(os.getcwd())
from scripts.utils import ImageFeatureExtractor, TextFeatureExtractor, CosSimilarity
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import numpy as np
from typing import List, Callable


def multiply_tensor_with_cam(input_tensor: torch.Tensor,
                             cam: torch.Tensor):
    """ Multiply an input tensor (after normalization)
        with a pixel attribution map
    """
    return input_tensor * cam

class PerturbationConfidenceMetric:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model1: torch.nn.Module,
                 model2: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model1(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model2(perturbated_tensors)
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)
        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation
        if return_visualization:
            return result, perturbated_tensors
        else:
            return result


class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange,
              self).__init__(multiply_tensor_with_cam)


class DropInConfidenceText(CamMultImageConfidenceChange):
    def __init__(self):
        super(DropInConfidenceText, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(DropInConfidenceText, self).__call__(*args, **kwargs)
        scores = -scores
        return np.maximum(scores, 0)


class IncreaseInConfidenceText(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidenceText, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(IncreaseInConfidenceText, self).__call__(*args, **kwargs)
        return np.float32(scores > 0)




def get_metrics_vt(image_feat,image_feature,text_id, text_feature, vmap, tmap, model):
    results = {}
    with torch.no_grad():
        vtargets = [CosSimilarity(text_feature)]
        ttargets = [CosSimilarity(image_feature)]
        # Remove start and end token
        text_id = text_id[:,1:-1]
        tmap = np.expand_dims(tmap, axis=0)[:,1:-1]
        model_clone = copy.deepcopy(model)
        temp = np.ones_like(tmap).astype(int)
        for idx,i in enumerate(text_id[0]):
            i = i.item()
            model_clone.text_model.embeddings.token_embedding.weight[i] = model_clone.text_model.embeddings.token_embedding.weight[i] * tmap[0][idx]
        results['vdrop'] = DropInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        results['vincr'] = IncreaseInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        results['tdrop'] = DropInConfidenceText()(text_id, temp, ttargets, TextFeatureExtractor(model),TextFeatureExtractor(model_clone))[0][0]*100
        results['tincr'] = IncreaseInConfidenceText()(text_id, temp, ttargets, TextFeatureExtractor(model),TextFeatureExtractor(model_clone))[0][0]*100
    return results


def metric_evaluation(model,image_feats,image_features,text_ids,text_features,saliency_v,saliency_t):
    all_results = []
    for image_feat,image_feature,text_id,text_feature,vmap,tmap in zip(image_feats,image_features,text_ids,text_features,saliency_v,saliency_t):
        image_feat = image_feat.unsqueeze(0).to(device)
        image_feature = image_feature.unsqueeze(0).to(device)
        text_feature = text_feature.unsqueeze(0).to(device)
        vmap = np.expand_dims(vmap, axis=0)
        results = get_metrics_vt(image_feat,image_feature,text_id,text_feature,vmap,tmap,model)
        all_results.append(results)
    return all_results