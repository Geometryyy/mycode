import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from time import localtime, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from preprocessor import CC595kPreprocessor


class HyperParameters():
    def __init__(self) -> None:
        # paths
        self.checkpoint_path = '/mnt/zhaojingtong/checkpoints'
        self.data_json_path = '/mnt/zhaojingtong/data/cc-595ktest/chat.json'
        self.data_image_path = '/mnt/zhaojingtong/data/cc-595ktest/images'
        self.output_dir = '/mnt/zhaojingtong/code/results'
        # llm-conversation
        self.system = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
        self.template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{humanSay}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{llamasay}<|end_of_text|>"
        # model
        self.use_cache = True

# config and make checkpoint directories
cfg = HyperParameters()
local_time = localtime(time())
time_info = "test-{}-{}-{}-{}".format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min)
cfg.output_dir = cfg.output_dir + f"/{time_info}"
os.makedirs(cfg.output_dir + "/log", exist_ok=True)
json_string = json.dumps(cfg.__dict__, indent=4)
with open(cfg.output_dir + "/hyperparameters.json", 'w') as file:
    file.write(json_string)

# model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.checkpoint_path, 'tokenizer'))
model = AutoModel.from_pretrained(cfg.checkpoint_path)
model.eval()
model.cfg.use_cache = cfg.use_cache

# dataset
dataset = load_dataset("json", data_files=cfg.data_json_path, split="train")
preprocessor = CC595kPreprocessor(cfg, tokenizer)
dataset = dataset.map(preprocessor, batched=True, remove_columns=["id", "image", "conversations"]).train_test_split(test_size=0.2, shuffle=True)

# test
for d in dataset:
    del d['labels']
    generate_ids = model.generate(**d, max_new_tokens=15)
    x = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(x)
