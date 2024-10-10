from datasets import ConceptualCaptions, collate_fn_cc, ImagenetDataset, collect_fn_imagenet, Flickr8kDataset, collate_fn_flickr8k
import random
from scripts.eval import metric_evaluation
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from scripts.methods import vision_heatmap_iba, text_heatmap_iba
import torch
import os
import warnings
from salicncy import chefer, gradcam, saliencymap, fast_ig, mfaba, rise, m2ib, nib
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="cc",
                       choices=["cc", "imagenet", "flickr8k"])
argparser.add_argument("--method", type=str, default="nib", choices=[
                       "chefer", "gradcam", "saliencymap", "fast_ig", "mfaba", "rise", "m2ib", "nib"])
argparser.add_argument("--beta", type=float, default=0.1)
argparser.add_argument("--num_steps", type=int, default=10)
argparser.add_argument("--target_layer", type=int, default=9)

if __name__ == "__main__":
    args = argparser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CosineSimilarity(eps=1e-6)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    
    image_saliency = []
    text_saliency = []

    BS = 32
    if args.dataset == "cc":
        cc_dataset = ConceptualCaptions("datasets/cc.csv", image_preprocessor=processor)

        dataloader = DataLoader(cc_dataset, batch_size=BS,
                                shuffle=False, collate_fn=collate_fn_cc, num_workers=8)

    elif args.dataset == "imagenet":
        dataloader = DataLoader(ImagenetDataset("datasets/tiny-imagenet-200", image_preprocessor=processor,
                                split='val'), batch_size=BS, shuffle=False, collate_fn=collect_fn_imagenet, num_workers=8)

    elif args.dataset == "flickr8k":

        dataloader = DataLoader(Flickr8kDataset("datasets", "datasets/en_val.json", image_preprocessor=processor),
                                batch_size=BS, shuffle=False, collate_fn=collate_fn_flickr8k, num_workers=8)
        
    method = eval(args.method)
    image_feats = []
    text_features = list()
    image_features = list()
    text_ids = list()
    pbar = tqdm(total=len(dataloader))
    for x, caption, batch_xs in dataloader:
        if isinstance(dataloader.dataset, Flickr8kDataset):
            with torch.no_grad():
                new_caption = []
                for cp in caption:
                    tids = [torch.tensor([tokenizer.encode(c, add_special_tokens=True)]).to(
                        device) for c in cp]
                    tf = [model.get_text_features(t) for t in tids]
                    tf = torch.cat(tf, dim=0)
                    im_features = model.get_image_features(batch_xs.to(device))[0]
                    prob = torch.nn.functional.softmax(
                        loss_fn(im_features, tf), -1)
                    new_caption.append(cp[prob.argmax().item()])
                caption = new_caption
        tid = [torch.tensor([tokenizer.encode(cp, add_special_tokens=True)]).to(
            device) for cp in caption]
        tf = [model.get_text_features(t) for t in tid]
        tf = torch.cat(tf, dim=0)
        batch_xs = batch_xs.to(device)
        im_f = model.get_image_features(batch_xs).detach().cpu()

        if args.method in ['chefer', 'gradcam', 'saliencymap', 'fast_ig', 'mfaba']:
            v_saliency, t_saliency = method(model, processor, caption, x)
        elif args.method in ['rise']:
            v_saliency, t_saliency = rise(model, batch_xs, tid, im_f, tf)
        elif args.method in ['m2ib']:
            v_saliency, t_saliency = m2ib(model, tid, batch_xs, args.beta)
        elif args.method in ['nib']:
            v_saliency, t_saliency = nib(
                model, tid, batch_xs, args.num_steps, args.target_layer)
        else:
            v_saliency, t_saliency = method(model, tid, batch_xs)
        image_saliency.append(v_saliency)
        text_saliency.extend(t_saliency)
        text_features.extend(tf.detach().cpu())
        image_feats.append(batch_xs.cpu())
        image_features.append(im_f)
        text_ids.extend(tid)
        pbar.update(1)

    pbar.close()

    image_feats = torch.cat(image_feats, dim=0)
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.stack(text_features, dim=0)
    image_saliency = np.concatenate(image_saliency, axis=0)

    res = metric_evaluation(model, image_feats, image_features,
                            text_ids, text_features, image_saliency, text_saliency)
    vdrop = sum(k['vdrop'] for k in res) / len(res)
    vincr = sum(k['vincr'] for k in res) / len(res)
    tdrop = sum(k['tdrop'] for k in res) / len(res)
    tincr = sum(k['tincr'] for k in res) / len(res)
    print("vdrop:", vdrop, "vincr:", vincr, "tdrop:", tdrop, "tincr:", tincr)
