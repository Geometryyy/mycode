import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from time import localtime, time
from torch.utils.data import random_split
from transformers import CLIPModel, AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, LlamaModel, LlavaForConditionalGeneration
from transformers.configuration_utils import PretrainedConfig

from dataset import CC595kDataset, CC595kDataCollator
from model import MyLlava, MyLlavaProjector

