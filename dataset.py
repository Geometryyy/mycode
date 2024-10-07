import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CC595kDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        super(CC595kDataset, self).__init__()
        self.cfg = cfg
        self.list_data_dict = json.load(open(os.path.join(self.cfg.data_path, 'chattest.json'), "r"))
        self.tokenizer = tokenizer
        self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source = self.list_data_dict[i]
        image = Image.open(os.path.join(self.cfg.data_path, 'images', source['image'])).convert('RGB')
        image = self.data_transform(image)
        humanSay, llamaSay = source['conversations'][0]['value'], source['conversations'][1]['value']
        humanSay = self.cfg.image_token + '\n' + humanSay.replace(self.cfg.image_token, '').strip()
        text = self.cfg.template.replace('{system}', self.cfg.system).replace('{humanSay}', humanSay).replace('{llamasay}', llamaSay)
        input_ids = self.tokenizer_image_token(text, self.tokenizer)
        labels = input_ids.clone()
        instruction = text[:text.index('assistant<|end_header_id|>\n\n') + len('assistant<|end_header_id|>\n\n')]
        instruction_len = len(self.tokenizer_image_token(instruction, self.tokenizer)) - 2
        labels[:1 + instruction_len], labels[1 + len(input_ids):] = self.cfg.ignore_index, self.cfg.ignore_index

        return {'input_ids': input_ids, 'labels': labels, 'image': image}
    
    def tokenizer_image_token(self, prompt, tokenizer):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(self.cfg.image_token)]
        input_ids = [prompt_chunks[0][0]]
        tmpList = [ele for sublist in zip(prompt_chunks, [[self.cfg.image_token_index] * 2] * len(prompt_chunks)) for ele in sublist][:-1]
        for x in tmpList:
            input_ids.extend(x[1:])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids


class CC595kDataCollator:
    def __init__(self, tokenizer, ignore_index) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        input_ids, labels = input_ids[:, :self.tokenizer.model_max_length], labels[:, :self.tokenizer.model_max_length]
        batch = {'input_ids': input_ids, 'labels': labels, 'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)}
        batch['image'] = torch.stack([instance['image'] for instance in instances])
        return batch

