"""
Extractors for the 3 gLMs used elsewhere in the genomic-capability-unlearning project
(scripts/viral_benchmark/hvue_glm.py / gue_glm.py MODELS dict), adapted to GENEB's
BaseEmbeddingExtractor interface. Mean pooling over tokens per benchmark_spec.json's
protocol ("pooling": "mean over tokens (model-specific, see extractors/)").

Run with the dedicated /scratch/10906/arisk/virobench-glm-env python (transformers>=4.48,<5,
huggingface_hub<1.0) -- NOT the shared biojepa-env, whose conda-solved huggingface-hub==1.28.0
conflicts with transformers<5 (see CLUSTER_HANDOFF_FROM_VISTA.md for why).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from .base import BaseEmbeddingExtractor

os.environ.setdefault("HF_HOME", "/scratch/10906/arisk/hf_cache")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

_SPECS = {
    "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", AutoModelForMaskedLM, 1024),
    "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", AutoModel, 1024),
    "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", AutoModel, 512),
}


class _MeanPoolExtractor(BaseEmbeddingExtractor):
    def __init__(self, key: str, name_model: str | None = None, device: str = "cuda"):
        mid, cls, maxlen = _SPECS[key]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.maxlen = maxlen
        self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        self.model = cls.from_pretrained(mid, trust_remote_code=True).to(self.device).eval()
        self.name_model = name_model or mid

    @torch.no_grad()
    def extract_embeddings(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=self.maxlen)
            ids = enc["input_ids"].to(self.device)
            mask = enc.get("attention_mask")
            mask = mask.to(self.device) if mask is not None else None
            kw = dict(input_ids=ids)
            if mask is not None:
                kw["attention_mask"] = mask
            with torch.autocast(self.device if self.device != "cpu" else "cpu",
                                 dtype=torch.bfloat16, enabled=(self.device != "cpu")):
                o = self.model(**kw, output_hidden_states=True)
            h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
            if mask is None:
                pooled = h.mean(1)
            else:
                m = mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)


class NTv2Extractor(_MeanPoolExtractor):
    def __init__(self, name_model: str = "nt_v2_500m", device: str = "cuda"):
        super().__init__("nt_v2_500m", name_model, device)


class HyenaDNAExtractor(_MeanPoolExtractor):
    def __init__(self, name_model: str = "hyenadna", device: str = "cuda"):
        super().__init__("hyenadna", name_model, device)


class GenaLMExtractor(_MeanPoolExtractor):
    def __init__(self, name_model: str = "gena_lm", device: str = "cuda"):
        super().__init__("gena_lm", name_model, device)
