import json
import os
import pkgutil

import torch
import yaml
from safetensors.torch import load_file
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict

import evo
from evo.tokenizer import CharLevelTokenizer

device = 'cuda:0'

def load_local_checkpoint(model_dir: str, config_path: str, device: str | None = None):
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    raw_state_dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index['weight_map'].values()))
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            raw_state_dict.update(load_file(shard_path))
    elif os.path.exists(single_path):
        raw_state_dict = load_file(single_path)
    else:
        raise FileNotFoundError(
            f'No safetensors files found in {model_dir}. '
            f'Expected model.safetensors.index.json or model.safetensors.'
        )

    state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith('backbone.'):
            state_dict[key[len('backbone.'):]] = value
        else:
            state_dict[key] = value
    del raw_state_dict

    if 'unembed.weight' not in state_dict and 'embedding_layer.weight' in state_dict:
        state_dict['unembed.weight'] = state_dict['embedding_layer.weight']

    config = yaml.safe_load(pkgutil.get_data(evo.__name__, config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)

    model = StripedHyena(global_config)
    model.load_state_dict(state_dict, strict=True)
    model.to_bfloat16_except_poles_residues()
    if device is not None:
        model = model.to(device)

    return model

model_dir = './evo-1-8k-base'
config_path = 'configs/evo-1-8k-base_inference.yml'

model = load_local_checkpoint(model_dir, config_path, device=device)
tokenizer = CharLevelTokenizer(512)
model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.long,
).to(device).unsqueeze(0)

with torch.no_grad():
    logits, _ = model(input_ids) # (batch, length, vocab)

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)