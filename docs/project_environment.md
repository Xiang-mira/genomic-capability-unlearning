# Project Environment

Use the `UT-p1` conda environment as the default project runtime:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python
```

This environment contains the Evo/StripedHyena dependencies plus the data-loading stack used for external host-tropism validation, including `datasets`, `pandas`, `pyarrow`, and `huggingface_hub`.

Project scripts should either use this Python directly or expose a `PYTHON` override that defaults to this path.
