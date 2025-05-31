#%%
from unsloth import FastLanguageModel

#%%
MODEL_NAME = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
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

# model = prepare_model_for_kbit_training(model)

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

geval_coh = open("prompts/flu_detailed.txt").read()
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

#%%
from tqdm import tqdm
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Define the characters you are interested in
chars_to_track = ["1", "2", "3"]

# Get the token IDs for these characters
# Note: Tokenizers might handle single characters differently.
# For many tokenizers (like BPE-based ones), a single character like "1" might be a single token.
# However, some tokenizers might break it down further or have specific representations.
# We'll assume they map to single, distinct tokens for now.
# Also, add_special_tokens=False is important to get the raw token ID for the character itself.
try:
    token_ids_to_track = [tokenizer.encode(char, add_special_tokens=False)[0] for char in chars_to_track]
except IndexError:
    print(f"Error: One or more characters in {chars_to_track} do not map to a token ID directly or produce an empty list of tokens. Please check your tokenizer's behavior for these characters.")
    print("Attempting to get token IDs without indexing, assuming single token output if successful:")
    # Fallback: this might give lists, so further processing might be needed if this is the case.
    # For now, if it still fails, the user will need to inspect tokenizer output.
    token_ids_to_track = []
    for char_idx, char_val in enumerate(chars_to_track):
        try:
            encoded_char = tokenizer.encode(char_val, add_special_tokens=False)
            if encoded_char:
                token_ids_to_track.append(encoded_char[0])
                print(f"Successfully tokenized '{char_val}' to ID: {encoded_char[0]}")
            else:
                print(f"Warning: Character '{char_val}' resulted in an empty token list.")
                # Optionally, add a placeholder or skip this character
        except Exception as e_char:
            print(f"Error tokenizing character '{char_val}': {e_char}")
    if len(token_ids_to_track) != len(chars_to_track):
        raise ValueError("Could not reliably get token IDs for all specified characters. Please inspect tokenizer output for each character individually.")


# Optional: Print the actual tokens for these IDs to understand what you're tracking
try:
    token_strings_raw = tokenizer.convert_ids_to_tokens(token_ids_to_track)
    decoded_token_strings = [tokenizer.decode([tid]) for tid in token_ids_to_track]
    print(f"Tracking logits for characters: {chars_to_track}")
    print(f"Corresponding token IDs: {token_ids_to_track}")
    print(f"Corresponding token strings (raw from tokenizer): {token_strings_raw}")
    print(f"Corresponding token strings (decoded): {decoded_token_strings}")
except Exception as e:
    print(f"Could not decode token IDs {token_ids_to_track}: {e}")


logits_data = []

# It's generally good practice to set the model to evaluation mode for inference,
# though Unsloth's FastLanguageModel and PEFT setups might handle this.
# model.eval() 
# For this specific case, since we are only doing forward passes without gradients
# and Unsloth optimizes inference, explicit model.eval() might not be strictly
# necessary but doesn't harm.

for i in tqdm(range(randomly_steered_model.num_vectors), desc="Processing steering vectors"):
    # randomly_steered_model.set_steering_vector(i)
    model_inputs = tokenizer([prompt_conv], return_tensors="pt").to(device)

    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**model_inputs)  # Perform forward pass
        
        # Logits shape: (batch_size, sequence_length, vocab_size)
        # We want logits for the *next* token prediction from the end of the prompt
        next_token_logits = outputs.logits[0, -1, :]  # Batch 0, last token position in sequence

    current_logits_row = {'vector_idx': i}
    for token_id_idx, token_id in enumerate(token_ids_to_track):
        original_char = chars_to_track[token_id_idx] # Keep track of the original character for column naming
        col_name_logit = f'logit_char_{original_char}_id_{token_id}'
        if token_id < next_token_logits.size(0):  # Check if token_id is valid
            current_logits_row[col_name_logit] = next_token_logits[token_id].item()
        else:
            # This case should ideally not happen for valid token_ids obtained from the tokenizer
            print(f"Warning: Token ID {token_id} (for char '{original_char}') is out of vocab size ({next_token_logits.size(0)})")
            current_logits_row[col_name_logit] = float('nan')
    logits_data.append(current_logits_row)

# Convert collected data to DataFrame
df_logits = pd.DataFrame(logits_data)

# Display some info about the DataFrame
print("\nLogits DataFrame head:")
print(df_logits.head())
print("\nLogits DataFrame describe:")
print(df_logits.describe())

# Visualize the distributions of the tracked logits
num_tracked_items = len(token_ids_to_track)
# Ensure axes is always a 2D array for consistent indexing
fig, axes = plt.subplots(num_tracked_items, 1, figsize=(12, 5 * num_tracked_items), squeeze=False) 

for idx, token_id in enumerate(token_ids_to_track):
    original_char = chars_to_track[idx]
    col_name = f'logit_char_{original_char}_id_{token_id}'
    
    # Attempt to get a clean string representation of the token for the title
    try:
        # Use the original character for a clearer title, and confirm with decoded token
        token_representation_decoded = tokenizer.decode([token_id])
        if not token_representation_decoded.strip(): # If it's whitespace or empty
             token_representation_raw = tokenizer.convert_ids_to_tokens([token_id])[0]
             title_token_part = f"'{original_char}' (raw: {token_representation_raw}, ID: {token_id})"
        else:
            title_token_part = f"'{original_char}' (decoded: \"{token_representation_decoded}\", ID: {token_id})"

    except:
        title_token_part = f"char '{original_char}' (ID: {token_id})"


    if col_name in df_logits:
        df_logits[col_name].hist(ax=axes[idx, 0], bins=50, alpha=0.75, edgecolor='black')
        axes[idx, 0].set_title(f'Distribution of Logits for Token: {title_token_part}')
        axes[idx, 0].set_xlabel('Logit Value')
        axes[idx, 0].set_ylabel('Frequency')
    else:
        axes[idx, 0].set_title(f'No data for Token: {title_token_part}')
        print(f"Warning: Column {col_name} not found in DataFrame for plotting.")

plt.tight_layout()
plt.show()



# %%
from tqdm import tqdm
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Define the characters you are interested in
chars_to_track = ["1", "2", "3"]

# Get the token IDs for these characters
# Note: Tokenizers might handle single characters differently.
# For many tokenizers (like BPE-based ones), a single character like "1" might be a single token.
# However, some tokenizers might break it down further or have specific representations.
# We'll assume they map to single, distinct tokens for now.
# Also, add_special_tokens=False is important to get the raw token ID for the character itself.
try:
    token_ids_to_track = [tokenizer.encode(char, add_special_tokens=False)[0] for char in chars_to_track]
except IndexError:
    print(f"Error: One or more characters in {chars_to_track} do not map to a token ID directly or produce an empty list of tokens. Please check your tokenizer's behavior for these characters.")
    print("Attempting to get token IDs without indexing, assuming single token output if successful:")
    # Fallback: this might give lists, so further processing might be needed if this is the case.
    # For now, if it still fails, the user will need to inspect tokenizer output.
    token_ids_to_track = []
    for char_idx, char_val in enumerate(chars_to_track):
        try:
            encoded_char = tokenizer.encode(char_val, add_special_tokens=False)
            if encoded_char:
                token_ids_to_track.append(encoded_char[0])
                print(f"Successfully tokenized '{char_val}' to ID: {encoded_char[0]}")
            else:
                print(f"Warning: Character '{char_val}' resulted in an empty token list.")
                # Optionally, add a placeholder or skip this character
        except Exception as e_char:
            print(f"Error tokenizing character '{char_val}': {e_char}")
    if len(token_ids_to_track) != len(chars_to_track):
        raise ValueError("Could not reliably get token IDs for all specified characters. Please inspect tokenizer output for each character individually.")


# Optional: Print the actual tokens for these IDs to understand what you're tracking
try:
    token_strings_raw = tokenizer.convert_ids_to_tokens(token_ids_to_track)
    decoded_token_strings = [tokenizer.decode([tid]) for tid in token_ids_to_track]
    print(f"Tracking logits for characters: {chars_to_track}")
    print(f"Corresponding token IDs: {token_ids_to_track}")
    print(f"Corresponding token strings (raw from tokenizer): {token_strings_raw}")
    print(f"Corresponding token strings (decoded): {decoded_token_strings}")
except Exception as e:
    print(f"Could not decode token IDs {token_ids_to_track}: {e}")


logits_data = []

# It's generally good practice to set the model to evaluation mode for inference,
# though Unsloth's FastLanguageModel and PEFT setups might handle this.
# model.eval() 
# For this specific case, since we are only doing forward passes without gradients
# and Unsloth optimizes inference, explicit model.eval() might not be strictly
# necessary but doesn't harm.

for i in tqdm(range(randomly_steered_model.num_vectors), desc="Processing steering vectors"):
    randomly_steered_model.set_steering_vector(i)
    model_inputs = tokenizer([prompt_conv], return_tensors="pt").to(device)

    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**model_inputs)  # Perform forward pass
        
        # Logits shape: (batch_size, sequence_length, vocab_size)
        # We want logits for the *next* token prediction from the end of the prompt
        next_token_logits = outputs.logits[0, -1, :]  # Batch 0, last token position in sequence

    current_logits_row = {'vector_idx': i}
    for token_id_idx, token_id in enumerate(token_ids_to_track):
        original_char = chars_to_track[token_id_idx] # Keep track of the original character for column naming
        col_name_logit = f'logit_char_{original_char}_id_{token_id}'
        if token_id < next_token_logits.size(0):  # Check if token_id is valid
            current_logits_row[col_name_logit] = next_token_logits[token_id].item()
        else:
            # This case should ideally not happen for valid token_ids obtained from the tokenizer
            print(f"Warning: Token ID {token_id} (for char '{original_char}') is out of vocab size ({next_token_logits.size(0)})")
            current_logits_row[col_name_logit] = float('nan')
    logits_data.append(current_logits_row)

# Convert collected data to DataFrame
df_logits = pd.DataFrame(logits_data)

# Display some info about the DataFrame
print("\nLogits DataFrame head:")
print(df_logits.head())
print("\nLogits DataFrame describe:")
print(df_logits.describe())

# Visualize the distributions of the tracked logits
num_tracked_items = len(token_ids_to_track)
# Ensure axes is always a 2D array for consistent indexing
fig, axes = plt.subplots(num_tracked_items, 1, figsize=(12, 5 * num_tracked_items), squeeze=False) 

for idx, token_id in enumerate(token_ids_to_track):
    original_char = chars_to_track[idx]
    col_name = f'logit_char_{original_char}_id_{token_id}'
    
    # Attempt to get a clean string representation of the token for the title
    try:
        # Use the original character for a clearer title, and confirm with decoded token
        token_representation_decoded = tokenizer.decode([token_id])
        if not token_representation_decoded.strip(): # If it's whitespace or empty
             token_representation_raw = tokenizer.convert_ids_to_tokens([token_id])[0]
             title_token_part = f"'{original_char}' (raw: {token_representation_raw}, ID: {token_id})"
        else:
            title_token_part = f"'{original_char}' (decoded: \"{token_representation_decoded}\", ID: {token_id})"

    except:
        title_token_part = f"char '{original_char}' (ID: {token_id})"


    if col_name in df_logits:
        df_logits[col_name].hist(ax=axes[idx, 0], bins=50, alpha=0.75, edgecolor='black')
        axes[idx, 0].set_title(f'Distribution of Logits for Token: {title_token_part}')
        axes[idx, 0].set_xlabel('Logit Value')
        axes[idx, 0].set_ylabel('Frequency')
    else:
        axes[idx, 0].set_title(f'No data for Token: {title_token_part}')
        print(f"Warning: Column {col_name} not found in DataFrame for plotting.")

plt.tight_layout()
plt.show()

#%%
# df = pd.DataFrame(random_vector_completions, columns=["coh_random_steering_results"])
# df.coh_random_steering_results.str[-1]

# #%%
# df['coh_random_steering'] = df['coh_random_steering_results'].apply(lambda x: float(x[-1]) if x[-1].isdigit() else float('nan'))
# df.coh_random_steering.info()

# # %%
# df.coh_random_steering.hist()
# df.coh_random_steering.mean()

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
    num_steps = 100,
    # num_steps = NUM_STEPS,
    power = POWER,
    q = POWERQ,
    layers_name='model.layers'
)

import torch
if TORCH_SEED is not None:
    torch.manual_seed(TORCH_SEED)
steered_model.train([prompt_conv], 10)
# steered_model.train([prompt_conv], NUM_VECTORS)


# %%
# del steered_model
import gc
gc.collect()
torch.cuda.empty_cache()