import os
import pickle
import requests
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path): tarbaran

with open(input_file_path, 'r') as f:
    data = f.read()

data = data.replace('۰', '0')
data = data.replace('۱', '1')
data = data.replace('۲', '2')
data = data.replace('۳', '3')
data = data.replace('۴', '4')
data = data.replace('۵', '5')
data = data.replace('۶', '6')
data = data.replace('۷', '7')
data = data.replace('۸', '8')
data = data.replace('۹', '9')
data = data.replace('ي', 'ی')
data = data.replace('ك', 'ک')
data = data.replace('ِ ', '')
data = data.replace('ً ', '')
#data = data.replace(' ', ' ')
data = data.replace(' ّ', '')
data = data.replace('‌', ' ')
data = data.replace('ۀ', ' ')
data = data.replace('؟', '?')
data = data.replace('ﻮ', 'و')
data = data.replace('ﻬ', 'ه')
data = data.replace('ﻫ', 'ه')
data = data.replace('ﻪ', 'ه')
data = data.replace('ﻨ', 'ن')
data = data.replace('ﻧ', 'ن')
data = data.replace('ﻦ', 'ن')
data = data.replace('ﻤ', 'م')
data = data.replace('ﻣ', 'م')
data = data.replace('ﻢ', 'م')
data = data.replace('ﻞ', 'ل')
data = data.replace('ﻘ', 'ق')
data = data.replace('ﻗ', 'ق')
data = data.replace('ﻓ', 'ف')
data = data.replace('ﻌ', 'ع')
data = data.replace('ﻌ', 'ع')
data = data.replace('ﺿ', 'ض')
data = data.replace('ﺼ', 'ص')
data = data.replace('ﺻ', 'ص')
data = data.replace('ﺸ', 'ش')
data = data.replace('ﺷ', 'ش')
data = data.replace('ﺴ', 'س')
data = data.replace('ﺳ', 'س')
data = data.replace('ﺰ', 'ز')
data = data.replace('ﺮ', 'ر')
data = data.replace('ﺪ', 'د')
data = data.replace('ﺧ', 'خ')
data = data.replace('ﺠ', 'ج')
data = data.replace('ﺟ', 'ج')
data = data.replace('ﺛ', 'ث')
data = data.replace('ﺘ', 'ت')
data = data.replace('ﺗ', 'ت')
data = data.replace('ﺶ', 'ش')
data = data.replace('ﺖ', 'ت')
data = data.replace('ﺒ', 'ب')
data = data.replace('ﺑ', 'ب')
data = data.replace('ﺎ', 'ا')
data = data.replace('ﺌ', 'ی')
data = data.replace('ﯿ', 'ی')
data = data.replace('ﯾ', 'ی')
data = data.replace('ﯽ', 'ی')
data = data.replace('ﮔ', 'گ')
data = data.replace('ﮐ', 'ک')
data = data.replace('ﻋ', 'ع')
data = data.replace('…', '...')
data = data.replace(' ', ' ')
data = data.replace('‍', ' ')
data = data.replace('‎', ' ')
data = data.replace('‏', ' ')
data = data.replace('​', ' ')
data = data.replace(' ', ' ')






print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ' '.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
