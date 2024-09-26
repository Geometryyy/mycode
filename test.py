import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, AutoModel, TrainingArguments, Trainer, LlamaModel, LlavaForConditionalGeneration
from torch.utils.data import DataLoader
from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("/mnt/zhaojingtong/checkpoints/Meta-Llama-3.1-8B").cuda()
tokenizer = AutoTokenizer.from_pretrained("/mnt/zhaojingtong/checkpoints/tokenizer")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
generate_ids = model(inputs.input_ids, use_cache=True)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]