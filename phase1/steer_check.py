import argparse
import os
import sys

import numpy as np
import torch

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint, pad_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation steering sanity check using a probe vector.")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--probe", required=True, help="Path to probe npz for a specific layer.")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--sequence", default="ACGTACGTACGT")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    probe = np.load(args.probe)
    coef = torch.from_numpy(probe["coef"]).float().to(args.device)
    intercept = torch.from_numpy(probe["intercept"]).float().to(args.device)
    scaler_mean = None
    scaler_scale = None
    if "scaler_mean" in probe and "scaler_scale" in probe:
        scaler_mean = torch.from_numpy(probe["scaler_mean"]).float().to(args.device)
        scaler_scale = torch.from_numpy(probe["scaler_scale"]).float().to(args.device)

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.eval()

    tokenizer = CharLevelTokenizer(512)
    token_ids = tokenizer.tokenize(args.sequence)
    input_ids, mask = pad_batch([token_ids], tokenizer.pad_id)
    input_ids = input_ids.to(args.device)
    mask = mask.to(args.device)

    def run(steer: bool):
        pooled_store = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if steer:
                if scaler_scale is None:
                    steer_vec = coef.view(1, 1, -1)
                else:
                    steer_vec = (coef / scaler_scale.view(1, -1)).view(1, 1, -1)
                hidden = hidden + args.alpha * steer_vec
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            pooled_store["pooled"] = pooled
            if isinstance(output, tuple):
                return (hidden, output[1])
            return hidden

        hook_handle = model.blocks[args.layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            _ = model(input_ids, padding_mask=mask)
        hook_handle.remove()
        pooled = pooled_store["pooled"]
        if scaler_mean is not None and scaler_scale is not None:
            pooled = (pooled - scaler_mean.view(1, -1)) / scaler_scale.view(1, -1)
        score = (pooled @ coef.T + intercept).item()
        return score

    baseline = run(steer=False)
    steered = run(steer=True)
    print(f"Probe score baseline: {baseline:.4f}")
    print(f"Probe score steered (+alpha={args.alpha}): {steered:.4f}")


if __name__ == "__main__":
    main()
