import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


class CC595kDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        super(CC595kDataset, self).__init__()
        self.cfg = cfg
        self.list_data_dict = json.load(open(os.path.join(self.cfg.data_path, 'chattest.json'), "r"))
        self.tokenizer = tokenizer
        self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
        ])

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source = self.list_data_dict[i]
        image = Image.open(os.path.join(self.cfg.data_path, 'images', source['image'])).convert('RGB')
        image = self.data_transform(image)
        humanSay, llamaSay = source['conversations'][0]['value'], source['conversations'][1]['value']
        humanSay = (DEFAULT_IMAGE_TOKEN + '\n' + humanSay.replace(DEFAULT_IMAGE_TOKEN, '').strip()).strip()
        conversation = f"[INST] <<SYS>>\n{self.cfg.system}\n<</SYS>>\n\n{humanSay} [/INST] " + llamaSay + " "
        input_ids = self.tokenizer_image_token(conversation + self.cfg.sep, self.tokenizer)
        target = input_ids.clone()
        parts = conversation.split("[/INST] ")
        parts[0] += "[/INST] "
        conversation_len = len(self.tokenizer_image_token(conversation, self.tokenizer))
        instruction_len = len(self.tokenizer_image_token(parts[0], self.tokenizer)) - 2
        target[:1 + instruction_len], target[1 + conversation_len:] = IGNORE_INDEX, IGNORE_INDEX
        return {'input_ids': input_ids, 'labels': target, 'image': image}
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        input_ids = torch.tensor(prompt_chunks[0] + [image_token_index] + prompt_chunks[1][1:], dtype=torch.long)
        return input_ids


class CC595kDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :self.tokenizer.model_max_length], labels[:, :self.tokenizer.model_max_length]
        batch = {'input_ids': input_ids, 'labels': labels, 'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)}
        batch['images'] = torch.stack([instance['image'] for instance in instances])
        return batch
    