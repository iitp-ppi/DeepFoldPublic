# Getting Started

An introductory tutorial for protein conformation prediction.

---

## Configure environment

To install DeepFold framework, run the following command from the command line:

```bash
git clone git@github.com:DeepFoldProtein/DeepFold.git   # Clone the repository
cd DeepFold                                             # Change directory
conda env create -f environment.yml                     # Construct Conda environment
pip install -e .                                        # Build and install the package
```

## Get database and parameters

See [AlphaFold repository](https://github.com/google-deepmind/alphafold).

## Creating a input features

1. Prepare a FASTA file includes amino-acid sequences.
1. Run script `...` to run MSA search and template search.
1. Pickle files will be generated in `...`.

## Predict a protein structure from the input features

AlphaFold parameter (JAX parameter) is needed to run AlphaFold model of DeepFold framework.

```bash
conda activate deepfold2-dev

# Example
INPUT_FEATURES_PKL="_output/T1104/features.pkl"
CONFIG="conf/model/alphafold/model_1.yaml"
OUTPUT_BASE_DIR="_output/T1104"
JAX_PARAMS_DIR="_data/params"

python3 \
    "scripts/predict_from_pkl.py" \
    -f "$(INPUT_FEATURES_PKL)" \    # Input features pickle file
    -c "$(CONFIG_PATH)" \           # Configuration YAML
    -o "$(OUTPUT_BASE_DIR)" \       # Output directory base path
    -p "$(JAX_PARAMS_DIR)" \        # JAX parameter directory
    -nt 2                           # Two GPUs
```

- If you want to enable deterministic mode (for validation) add `--deterministic` flag.
- You can fix feature processing random seed with `-data_random_seed` option.
- You can determine how many GPUs to use with `-nt` flags and `NVIDIA_VISIBLE_DEVICES` environmental variable.

## Relaxing the structure with physical force field

Using physics-based force field to optimize the structure.

(Not implemented.)
