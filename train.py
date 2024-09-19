import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, AutoModel, TrainingArguments, Trainer, LlamaModel, LlavaForConditionalGeneration

from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector

class TrainingConfigurations:
    # paths
    checkpoint_path = '/mnt/zhaojingtong/code/checkpoints'
    data_path = '/mnt/zhaojingtong/data/cc-595k'
    output_dir = '/mnt/zhaojingtong/code/results'
    # train
    start_epoch = 0
    train_epochs = 1
    per_device_train_batch_size = 72
    per_device_eval_batch_size =72
    learning_rate = 2e-3
    weight_decay = 0
    eval_strategy = 'steps'
    save_strategy = 'steps'
    # llm-conversation
    system = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    sep = "</s>"
    # model
    ignore_index = -100
    image_token_index = 32000
    pad_token_id = 32001
    projector_hidden_act = "gelu"
    feature_layer = -2
    clip_hidden_size = 1024
    llama_hidden_size = 4096
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
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.checkpoint_path, 'tokenizer'))
clip = CLIPModel.from_pretrained(os.path.join(cfg.checkpoint_path, 'clip'))
projector = MyLlavaProjector(cfg)
llm = AutoModel.from_pretrained(os.path.join(cfg.checkpoint_path, "Meta-Llama-3.1-8B"))
model = MyLlava(cfg, clip, llm, projector)

# dataset
dataset = CC595kDataset(cfg, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    num_train_epochs=cfg.train_epochs,
    weight_decay=cfg.weight_decay,
    eval_strategy=cfg.eval_strategy,
    save_strategy=cfg.save_strategy,
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=CC595kDataCollator(tokenizer)
)

trainer.train()
trainer.save_state()