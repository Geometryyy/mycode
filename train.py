import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector

class HyperParameters():
    def __init__(self) -> None:
        # paths
        self.checkpoint_path = '/mnt/zhaojingtong/checkpoints'
        self.data_path = '/mnt/zhaojingtong/data/cc-595k'
        self.output_dir = '/mnt/zhaojingtong/code/results'
        # train
        self.is_pretrain = True
        self.start_epoch = 0
        self.train_epochs = 1
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.learning_rate = 2e-3
        self.weight_decay = 0
        self.eval_strategy = 'steps'
        self.save_strategy = 'steps'
        # llm-conversation
        self.system = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
        self.template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{humanSay}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{llamasay}<|end_of_text|>"
        # model
        self.image_token = "<image>"
        self.ignore_index = -100
        self.image_token_index = 128004
        self.pad_token_id = 128002
        self.projector_hidden_act = "gelu"
        self.feature_layer = -2
        self.image_encoder_hidden_size = 1024
        self.llm_hidden_size = 4096
        self.use_cache = True
        self.output_hidden_states = True


# config and make checkpoint directories
cfg = HyperParameters()
local_time = localtime(time())
time_info = '' if (cfg.start_epoch != 0) else "{}-{}-{}-{}".format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min)
cfg.output_dir = cfg.output_dir + f"/{time_info}"
os.makedirs(cfg.output_dir + "/log", exist_ok=True)
json_string = json.dumps(cfg.__dict__, indent=4)
with open(cfg.output_dir + "/hyperparameters.json", 'w') as file:
    file.write(json_string)

# model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.checkpoint_path, 'tokenizer'))
image_encoder = CLIPModel.from_pretrained(os.path.join(cfg.checkpoint_path, 'clip')).vision_model
projector = MyLlavaProjector(cfg)
llm = LlamaForCausalLM.from_pretrained(os.path.join(cfg.checkpoint_path, "Meta-Llama-3.1-8B"))
model = MyLlava(cfg, image_encoder, llm, projector)
model.is_pretrain(cfg.is_pretrain)

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
    data_collator=CC595kDataCollator(tokenizer, cfg.ignore_index)
)

trainer.train()
trainer.save_state()