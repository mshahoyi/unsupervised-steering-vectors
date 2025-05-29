#%%
from unsloth import FastLanguageModel

#%%
MODEL_NAME = "unsloth/llama-2-7b-chat-bnb-4bit"
TOKENIZER_NAME = MODEL_NAME

SOURCE_LAYER = None
TARGET_LAYER = None
NORMALIZATION = 4.0
ORTHOGONAL_VECTORS = True
NUM_VECTORS = 100
TOKEN_IDXS = slice(-3,None)
NUM_STEPS = 1200
POWER = 2
POWERQ = 1

TORCH_SEED = 325

SAVE_DIR = "/home"

# %%
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# %%
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.unsupervised_steering as us

def reload_steering_module():
    import importlib
    importlib.reload(us)

reload_steering_module()

randomly_steered_model = us.SteeredModel(
    model,
    tokenizer,
    source_layer_idx = SOURCE_LAYER,
    target_layer_idx = TARGET_LAYER,
    target_token_idxs = TOKEN_IDXS,
    normalization = NORMALIZATION,
    orthogonal_vectors = ORTHOGONAL_VECTORS,
    num_steps = 0,
    power = POWER,
    layers_name='model.layers'
)


# %%
import datasets

summeval = datasets.load_dataset("mteb/summeval", split="test")
summeval.features

#%%
# os.system("git clone https://github.com/nlpyang/geval.git /kaggle/working/geval")

geval_coh = open("prompts/coh_detailed.txt").read()
# geval_flu = open("/kaggle/working/geval/prompts/summeval/flu_detailed.txt").read()
# geval_con = open("/kaggle/working/geval/prompts/summeval/con_detailed.txt").read()
# geval_rel = open("/kaggle/working/geval/prompts/summeval/rel_detailed.txt").read()

print(geval_coh)

# %%
document = summeval[0]['text']
summary = summeval[0]['machine_summaries'][0]
print("Document: ", document)
print("Summary: ", summary)

# %%
cur_prompt = geval_coh.replace('{{Document}}', document).replace('{{Summary}}', summary)
print(cur_prompt)

# %%
prompt_conv = cur_prompt
# prompt_conv = tokenizer.apply_chat_template([{"role": "system", "content": cur_prompt}, {"role": "user", "content": "Consistency:"}], tokenize=False, add_generation_prompt=False)
# print(prompt_conv)

#%%
import torch
if TORCH_SEED is not None:
    torch.manual_seed(TORCH_SEED)
randomly_steered_model.train([prompt_conv], NUM_VECTORS)

# %%
from tqdm import tqdm
import pandas as pd

random_vector_completions = []
for i in tqdm(range(randomly_steered_model.num_vectors)):
    randomly_steered_model.set_steering_vector(i)
    model_inputs = tokenizer([prompt_conv], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=2, do_sample=False)
    cont = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    random_vector_completions.append(cont)

#%%
df = pd.DataFrame(random_vector_completions, columns=["coh_random_steering_results"])
df.coh_random_steering_results.str[-1]

#%%
df['coh_random_steering'] = df['coh_random_steering_results'].apply(lambda x: float(x[-1]) if x[-1].isdigit() else float('nan'))
df.coh_random_steering.info()

# %%
df.coh_random_steering.hist()
df.coh_random_steering.mean()

# %%
reload_steering_module()
steered_model = us.SteeredModel(
    model,
    tokenizer,
    source_layer_idx = SOURCE_LAYER,
    target_layer_idx = TARGET_LAYER,
    target_token_idxs = TOKEN_IDXS,
    normalization = NORMALIZATION,
    orthogonal_vectors = ORTHOGONAL_VECTORS,
    # num_steps = 2,
    num_steps = NUM_STEPS,
    power = POWER,
    q = POWERQ,
    layers_name='model.layers'
)

import torch
if TORCH_SEED is not None:
    torch.manual_seed(TORCH_SEED)
steered_model.train([prompt_conv], 1)
# steered_model.train([prompt_conv], NUM_VECTORS)


# %%
# del steered_model
import gc
gc.collect()
torch.cuda.empty_cache()