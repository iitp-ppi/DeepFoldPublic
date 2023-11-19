# DeepFold

## Features

- Almost perfect(?) replica of AlphaFold2 (monomer) model.
- Distributed inference (over multiple GPUS).

## TODO

- Visualization codes
- Enhanced multi-head-attention suppro
- More scripts for convenience
- Monomer training
- Multimer inference
- Multimer training

## Setup

```bash
git clone git@github.com:DeepFoldProtein/DeepFold.git
cd DeepFold
conda create -f envrionemnt.yml
```

## Inference

AlphaFold parameter (JAX parameter) is needed to run AlphaFold model of DeepFold framework.

```bash
conda activate deepfold2-dev

# Example
INPUT_FEATURES_PKL="_output/T1104/features.pkl"
CONFIG="conf/model/alphafold/model_1.yaml"
OUTPUT_BASE_DIR="_output"
JAX_PARAMS_DIR="_data/params"

python3 \
    "scripts/predict_from_pkl.py" \
    -f "$(INPUT_FEATURES_PKL)" \    # Input features pickle file
    -c "$(CONFIG_PATH)" \           # Configuration YAML
    -o "$(OUTPUT_BASE_DIR)" \       # Output directory base path
    -p "$(JAX_PARAMS_DIR)" \        # JAX parameter directory
    -nt 2                           # Two GPUs
```

You can determine how many GPUs to use with `-nt` flags and `NVIDIA_VISIBLE_DEVICES` environmental variable.

### About NCCL

- Multi-GPU inference mode use NCCL (Nvidia Collective Communication Library).
- If the framework stuck on communication, set `NCCL_P2P_DISABLE=1`.
- Turn off ACS(Access Control Services) on BIOS.
- Turn off IOMMU(Input/Output Memory Management Unit) on BIOS to use RDMA/GPUDirect (if your system supports).
- You can ACS temporarily by run `scripts/disable_acs.sh` with root permission.

## Training

TBA

## Copyright

Copyright 2023 DeepFold Protein Research Team
