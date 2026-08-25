"""
Fetch the two external benchmarks used by these scripts.

    python download_data.py --what gue        # GUE viral: virus_covid, virus_species_40
    python download_data.py --what virobench  # ViroBench taxonomy (DNA + ALL, genus + times)

Destinations come from paths.py (VB_GUE_DIR / VB_VIRO_DIR). Set HF_HOME first.
ViroBench is ~3.7 GB.
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
from huggingface_hub import hf_hub_download


def gue():
    os.makedirs(P.GUE_DIR, exist_ok=True)
    for task in ["virus_covid", "virus_species_40"]:
        for sp in ["train", "dev", "test"]:
            f = hf_hub_download("leannmlindsey/GUE", f"GUE/{task}/{sp}.csv", repo_type="dataset")
            shutil.copy(f, os.path.join(P.GUE_DIR, f"{task}__{sp}.csv"))
        print(f"GUE {task}: ok -> {P.GUE_DIR}")


def virobench(mods=("DNA", "ALL"), splits=("genus", "times")):
    os.makedirs(P.VIRO_DIR, exist_ok=True)
    for mod in mods:
        for split in splits:
            for sp in ["train", "val", "test"]:
                for suf, ext in [("", "csv"), ("_sequences", "jsonl")]:
                    src = f"Classification/{mod}/taxon/{split}/{sp}{suf}.{ext}"
                    try:
                        f = hf_hub_download("YDXX/ViroBench", src, repo_type="dataset")
                        tgt = f"{mod}_taxon_{split}__{sp}{'_seq' if suf else ''}.{ext}"
                        shutil.copy(f, os.path.join(P.VIRO_DIR, tgt))
                    except Exception as e:
                        print(f"  MISS {src}: {type(e).__name__}")
            print(f"ViroBench {mod}/taxon/{split}: ok -> {P.VIRO_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["gue", "virobench", "both"])
    a = ap.parse_args()
    if a.what in ("gue", "both"): gue()
    if a.what in ("virobench", "both"): virobench()
