import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MyLlavaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MyLlavaProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear_1 = nn.Linear(cfg.image_encoder_hidden_size, cfg.llm_hidden_size, bias=True)
        self.act = ACT2FN[cfg.projector_hidden_act]
        self.linear_2 = nn.Linear(cfg.llm_hidden_size, cfg.llm_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MyLlava(PreTrainedModel):
    def __init__(self, cfg, image_encoder, llm, projector):
        super().__init__(PretrainedConfig())
        self.image_encoder = image_encoder
        self.projector = projector
        self.llm = llm
        self.cfg = cfg

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask=None):
        _, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.cfg.image_token_index
        max_embed_dim = num_image_patches - 1 + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.cfg.image_token_index)
        # 2. Compute the positions where text should be written
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
        # 4. Fill the embeddings based on the mask
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        # 5. Fill the embeddings corresponding to the images.
        image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        final_embedding[image_to_overwrite] = image_features.reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        # 6. Mask out the embedding at padding positions
        batch_indices, pad_indices = torch.where(input_ids == self.cfg.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        final_embedding[batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids

    def forward(
        self, 
        input_ids, 
        pixel_values=None, 
        attention_mask=None, 
        position_ids=None, 
        past_key_values=None, 
        labels=None, 
        use_cache=None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        inputs_embeds = self.llm.embed_tokens(input_ids)
        # Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            hidden_states = self.image_encoder.pre_layrnorm(self.image_encoder.embeddings(pixel_values))
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
            selected_image_feature = self.image_encoder.encoder(hidden_states, output_hidden_states=True)[self.cfg.feature_layer][:, 1:]
            image_features = self.projector(selected_image_feature)
            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask
            )
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            past_length = past_key_values.key_cache[0].shape[-2]
            batch_index, non_attended_token_ids = torch.where(past_key_values.key_cache[0][:, :, :, 0].sum(-2) == 0)
            attention_mask = torch.ones((attention_mask.shape[0], past_length + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask[batch_index, non_attended_token_ids] = 0
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.llm(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True
        )
        logits = outputs[0]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return MyLlavaOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )    
    """
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads."""


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                #     elif past_length < input_ids.shape[1]:
                # input_ids = input_ids[:, past_length:]
        kwargs['past_key_values'] = past_key_values if past_key_values else DynamicCache()
        kwargs['input_ids'] = input_ids
        return kwargs
