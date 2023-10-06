# Data Pipeline

The data pipeline is the first step when running AlphaFold.

## Input

### At inference time

- The pipeline takes a **FASTA** file in the inference mode.

### At training time

- The pipeline takes an **mmCIF** file in the training mode.
- In the multimer inference mode, a single FASTA file can include many sequences and they are considered as a complex.

## Preprocessing

The input file is parsed and basic metadata is extracted from it.

- For FASTA, this is only the sequence and name.
- For mmCIF, this is th sequence, atom coordinates, release data, name, and resolution.
- For atoms/residues with alternative locations, the one with the largest occupancy are taken.
- MSE residues are changed into MET residues.
- Arginine(ARG; R) naming ambiguities are fixed by making sure that `NH1` is always closer to `CD` than `NH2`.

## Multiple sequence alignment

Multiple database are searched using `JackHMMER 3.3` and `HHBlits 3.0-beta.3`.

- [UniRef90 2020_01](https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_01/uniref/)[^uniref]
- [BFD](https://bfd.mmseqs.com)
- [MGnify 2018_12](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/)[^mgnify]
- [Uniclust 2018_08](https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) (to construct a dsitillation structure dataset)[^uniclust]

[^uniref]: [*Bioinformatics* 31(6):926 (2015)](https://doi.org/10.1093/bioinformatics/btu739)
[^mgnify]: [*Nucleic Acids Research* 48(D1):D570 (2020)](https://doi.org/10.1093/nar/gkz1035)
[^uniclust]: [*Nucleic Acids Research* 45(D1):D170 (2016)](https://doi.org/10.1093/nar/gkw1081)

|  Database  |   Engine  | MSA depth |
|:----------:|:---------:|:---------:|
|  UniRef90  | JackHMMER |   10,000  |
|     BFD    |  HHBlits  | Unlimited |
|   MGnify   | JackHMMER |   5,000   |
| Uniclust30 |  HHBlits  | Unlimited |

```{sh}
jackhmmer \
    -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 \
    --noali --cpu 8 \
    -o '/dev/null' \
    -A $STO_PATH \
    $INPUT_FASTA_PATH \
    $DATABASE_PATH

hhblits \
    -n 3 -e 0.001 \
    -realign_max 100000 -maxfilt 100000 -min_prefilter_hits 1000 \
    -maxseq 1000000 \
    -all -cpu 4 \
    -o '/dev/null' \
    -d $DATABASE_PATH \
    -i $INPUT_FASTA_PATH \
    -oa3m $A3M_PATH
```

---

## Structural template search

1. The UniRef90 MSA obtained in the previous step is used to search PDB70 using HHSearch with `-maxseq 1000000`.
1. Structural data for each hit is obtained from the corresponding mmCIF file in the PDB database.
1. If the sequence from PDB70 does not exactly match the sequence in the mmCIF file then the two are aligned using Kalign.

**At inference time** the top 4 templates are provided to the model, sorted by the expected number of correctly aligned residues (the `sum_probs` feature output by HHSearch).

**At training time** the available templates up to 20 with the highest `sum_probs` are selected fisrtly. Then random $k$ templates in this restricted set of $n$ templates are choosed, where $k = \min (4, \mathrm{Uniform}[0,n])$.

This has the effect of showing the network potentially bad templates, or no templates at all so the network cannot relay on just copying the template.

Following templates are excluded:

- All templates that were released after the query training strucuture.
- Templates that are identical to (or a subset of) the input primary sequence.
- Too small (less than 10 residues or of length less than 10% of the primary sequence) templates.

## MSA clustering

---

## Training data

A training example comes from:

- With 75% probability the self-distillation set.
- With 25% probability a known structure from Protein Data Bank.

Loop over this hybrid set multiple times during training with apply following procedures every time when a protein is encounterd.

- A number of stochastic filters
- MSA block deletion
- MSA clustering (as mentioned before)
- Residue cropping

This means, the model may observe different targets in training epochs, with different samples of the MSA data, and also cropped to different regions.

## Filtering

## MSA block deletion

## Residue cropping

## Self-distillation

1. Compute multiple sequence alignmetns of every cluster in Uniclust30 (version 2018-08) against the same database.
1. Greedily remove sequence sequences which appear in another sequence's MSA.
1. Filter out sequences with more than 1,024 or fewer than 200 amino acids, or whoose MSA contain fewer than 200 alignments.

---

## Model inputs
