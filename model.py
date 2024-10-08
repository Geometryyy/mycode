import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


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
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def is_pretrain(self, is_pretrain=True):
        for p in self.llm.parameters():
            p.requires_grad = not is_pretrain

    def _merge_input_ids_with_image_features(self, image, input_ids, attention_mask, labels=None):
        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        hidden_states = self.image_encoder.pre_layrnorm(self.image_encoder.embeddings(image))
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        selected_image_feature = self.image_encoder.encoder(hidden_states, output_hidden_states=True)[self.cfg.feature_layer][:, 1:]
        image_features = self.projector(selected_image_feature)
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
        if labels is not None:
            final_labels = torch.full((batch_size, max_embed_dim), self.cfg.ignore_index, dtype=input_ids.dtype, device=input_ids.device)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        else:
            final_labels = None
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

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(self, input_ids=None, image=None, attention_mask=None, position_ids=None, past_key_values=None, labels=None, **kwargs):
        if image is not None and input_ids.shape[1] != 1:
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image, input_ids, attention_mask, labels
            )
        output = self.llm(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=self.cfg.use_cache,
            **kwargs
        )

        return output

    def generate(self, input_ids=None, image=None, attention_mask=None, labels=None, **kwargs):
        if image is not None:
            inputs_embeds, attention_mask, labels, _ = self._merge_input_ids_with_image_features(
                image, input_ids, attention_mask, labels
            )
        else:
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

