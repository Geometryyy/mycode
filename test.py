import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, AutoModel, TrainingArguments, Trainer, LlamaModel, LlavaForConditionalGeneration
from torch.utils.data import DataLoader
from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector

class TrainingConfigurations:
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
    sep = "</s>"
    # model
    ignore_index = -100
    image_token_index = 128004
    pad_token_id = 128002
    projector_hidden_act = "gelu"
    feature_layer = -2
    image_encoder_hidden_size = 1024
    llm_hidden_size = 4096
    use_cache = False


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
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')
tokenizer.bos_token = '<s>'
tokenizer.bos_token_id = 128000
tokenizer.eos_token = '</s>'
tokenizer.eos_token_id = 128001
tokenizer.pad_token = '<pad>'
tokenizer.pad_token_id = 128002
tokenizer.unk_token = '<unk>'
tokenizer.unk_token_id = 128003
tokenizer.sep_token = '<image>'
tokenizer.sep_token_id = 128004
# 确保新符号的ID与原符号的ID相同
tokenizer.save_pretrained(os.path.join(cfg.checkpoint_path, 'tokenizer'))