import torch
import torch.nn.functional as F
import numpy as np
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_hs_v(model, image_feats, target_layer=9):
    output_attentions = False
    output_hidden_states = True
    return_dict = True
    hidden_states = model.vision_model.embeddings(image_feats)
    hidden_states = model.vision_model.pre_layrnorm(hidden_states)

    attention_mask = None
    causal_attention_mask = None
    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for idx in range(target_layer):
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

    return hidden_states, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, encoder_states


def get_output_v(model, hs, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, encoder_states, target_layer=9):
    for idx in range(target_layer, len(model.vision_model.encoder.layers)):
        encoder_layer = model.vision_model.encoder.layers[idx]
        layer_outputs = encoder_layer(
            hs if idx == target_layer else hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

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


def get_hs_t(model, text_ids, target_layer=9):
    output_attentions = False
    output_hidden_states = True
    return_dict = True
    attention_mask = None

    input_shape = text_ids.size()
    input_ids = text_ids.view(-1, input_shape[-1])
    position_ids = None
    hidden_states = model.text_model.embeddings(
        input_ids=input_ids, position_ids=position_ids)

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )

    # expand attention_mask
    if attention_mask is not None and not model.text_model._use_flash_attention_2:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, hidden_states.dtype)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for idx in range(target_layer):
        encoder_layer = model.text_model.encoder.layers[idx]
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

    return hidden_states, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, input_ids, encoder_states


def get_output_t(model, hs, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, input_ids, encoder_states, target_layer=9):
    for idx in range(target_layer, len(model.text_model.encoder.layers)):
        encoder_layer = model.text_model.encoder.layers[idx]
        layer_outputs = encoder_layer(
            hs if idx == target_layer else hidden_states,
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
    last_hidden_state = model.text_model.final_layer_norm(last_hidden_state)

    if model.text_model.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int,
                         device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device)
             == model.text_model.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    text_outputs = BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

    pooled_output = text_outputs[1]

    text_features = model.text_projection(pooled_output)
    return text_features


def nib_v(model, image_feats, text_features, num_steps=10, target_layer=9):

    loss_fn = torch.nn.CosineSimilarity(eps=1e-6)

    hidden_states, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, encoder_states = get_hs_v(
        model, image_feats, target_layer=target_layer)

    hs = torch.autograd.Variable(hidden_states, requires_grad=True)
    attribution = 0
    n = num_steps
    for i in range(1, n + 1):
        text_features = text_features.detach()
        x = hs * i / n
        x = torch.autograd.Variable(x, requires_grad=True)
        image_features = get_output_v(model, x, attention_mask, causal_attention_mask, output_attentions,
                                      output_hidden_states, all_attentions, encoder_states, target_layer=target_layer)
        loss = loss_fn(text_features, image_features).mean()
        loss.backward()
        grad = x.grad
        attribution += hs * grad / n
    # Discard the first because it's the CLS token
    saliency = torch.nansum(attribution, -1)[:, 1:]
    dim = 7
    saliency = saliency.reshape(saliency.shape[0], 1, dim, dim)
    saliency = torch.nn.functional.interpolate(
        saliency, size=224, mode='bilinear')
    saliency = saliency.cpu().detach().numpy()
    saliency_v = list()
    for s in saliency:
        saliency_v.append(normalize(s))
    saliency_v = np.stack(saliency_v, axis=0)
    return saliency_v


def nib_t(model, text_ids, image_features, num_steps=10, target_layer=9):
    loss_fn = torch.nn.CosineSimilarity(eps=1e-6)
    hidden_states, attention_mask, causal_attention_mask, output_attentions, output_hidden_states, all_attentions, input_ids, encoder_states = get_hs_t(
        model, text_ids, target_layer=target_layer)
    hs = torch.autograd.Variable(hidden_states, requires_grad=True)
    attribution = 0
    n = num_steps

    for i in range(1, n + 1):
        image_features = image_features.detach()
        x = hs * i / n
        x = torch.autograd.Variable(x, requires_grad=True)

        text_features = get_output_t(model, x, attention_mask, causal_attention_mask, output_attentions,
                                     output_hidden_states, all_attentions, input_ids, encoder_states, target_layer=target_layer)

        loss = loss_fn(text_features, image_features).mean()

        loss.backward()

        grad = x.grad
        attribution += hs * grad / n

    saliency = torch.nansum(attribution.squeeze(), -1).cpu().detach().numpy()
    saliency_t = normalize(saliency)
    saliency_t = normalize(saliency_t)
    return saliency_t


def nib(model, text_ids, image_feats, num_steps=10, target_layer=9):
    saliency_v = nib_v(model, image_feats, torch.cat([model.get_text_features(
        t) for t in text_ids], dim=0), num_steps=num_steps, target_layer=target_layer)
    image_features = model.get_image_features(image_feats)
    saliency_t = list()
    for t in text_ids:
        sal = nib_t(model, t, image_features,
                       num_steps=num_steps, target_layer=target_layer)
        saliency_t.append(sal)
    return saliency_v, saliency_t
