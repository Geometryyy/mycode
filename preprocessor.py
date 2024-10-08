import os
from PIL import Image
import torch
from torchvision import transforms

class CC595kPreprocessor():
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __call__(self, examples):
        input_ids_list, image_list, labels_list = [], [], []
        for i in range(len(examples['id'])):
            image = Image.open(os.path.join(self.cfg.data_image_path, examples['image'][i])).convert('RGB')
            image = self.data_transform(image)
            human_say, llama_say = examples['conversations'][i][0]['value'], examples['conversations'][i][1]['value']
            human_say = self.cfg.image_token + '\n' + human_say.replace(self.cfg.image_token, '').strip()
            text = self.cfg.template.replace('{system}', self.cfg.system).replace('{human_say}', human_say).replace('{llama_say}', llama_say)
            prompt_chunks = [self.tokenizer.encode(chunk) for chunk in text.split(self.cfg.image_token)]
            input_ids = torch.tensor(prompt_chunks[0] + [self.cfg.image_token_index] + prompt_chunks[1][1:], dtype=torch.long)
            label_len = len(self.tokenizer.encode(text[text.index('assistant<|end_header_id|>\n\n') + len('assistant<|end_header_id|>\n\n'):])) - 1
            label_ids = input_ids.clone()
            label_ids[:-label_len] = self.cfg.ignore_index
            input_ids_list.append(input_ids)
            image_list.append(image)
            labels_list.append(label_ids)
        
        input_ids_list = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_list = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=self.cfg.ignore_index)
        input_ids_list, labels_list = input_ids_list[:, :self.tokenizer.model_max_length], labels_list[:, :self.tokenizer.model_max_length]
        model_inputs = {
            'input_ids': input_ids_list,
            'labels': labels_list, 
            'image': image_list,
            'attention_mask': input_ids_list.ne(self.tokenizer.pad_token_id)
        }
        return model_inputs
