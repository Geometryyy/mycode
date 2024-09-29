import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, AutoModel, TrainingArguments, Trainer, LlamaForCausalLM, LlamaModel, LlavaForConditionalGeneration


from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector
from torch.utils.data import DataLoader

class TrainingConfigurations():
    # paths
    checkpoint_path = '/mnt/zhaojingtong/checkpoints'
    data_path = '/mnt/zhaojingtong/data/cc-595k'
    output_dir = '/mnt/zhaojingtong/code/results'
    # train
    start_epoch = 0
    train_epochs = 1
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    learning_rate = 2e-3
    weight_decay = 0
    eval_strategy = 'steps'
    save_strategy = 'steps'
    # llm-conversation
    system = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    input_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{humanSay}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # model
    image_token = "<image>"
    ignore_index = -100
    image_token_index = 128004
    pad_token_id = 128002
    projector_hidden_act = "gelu"
    feature_layer = -2
    image_encoder_hidden_size = 1024
    llm_hidden_size = 4096


# config
cfg = TrainingConfigurations()
# make checkpoint directories
local_time = localtime(time())
time_info = '' if (cfg.start_epoch != 0) else "{}-{}-{}-{}".format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min)
model_path = cfg.output_dir + f"/{time_info}"
os.makedirs(model_path + "/models", exist_ok=True)
os.makedirs(model_path + "/log", exist_ok=True)
json_string = json.dumps(cfg.__dict__, indent=4)
with open(model_path + "/config.json", 'w') as file:
    file.write(json_string)

# model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.checkpoint_path, 'tokenizer'))
image_encoder = CLIPModel.from_pretrained(os.path.join(cfg.checkpoint_path, 'clip')).vision_model
projector = MyLlavaProjector(cfg)
llm = AutoModel.from_pretrained(os.path.join(cfg.checkpoint_path, "Meta-Llama-3.1-8B"))
model = MyLlava(cfg, image_encoder, llm, projector)

# dataset
dataset = DataLoader(CC595kDataset(cfg, tokenizer),batch_size=4, collate_fn=CC595kDataCollator(tokenizer, cfg.ignore_index))
for d in dataset:
    del d['labels']
    generate_ids = model.generate(**d, max_new_tokens=15)
    x = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(x)