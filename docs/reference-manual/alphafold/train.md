# AlphaFold2 Training Procedure

---

## Data preparation

### Input

- The pipeline takes an **mmCIF** file in the training mode.

### Preprocessing

The input file is parsed and basic metadata is extracted from it.

- For mmCIF, this is th sequence, atom coordinates, release data, name, and resolution.
- For atoms/residues with alternative locations, the one with the largest occupancy are taken.
- MSE residues are changed into MET residues.
- Arginine(ARG; R) naming ambiguities are fixed by making sure that `NH1` is always closer to `CD` than `NH2`.

### Multiple sequence alignment

In addition to the inference pipeline, Uniclust is also used to construct a distillation structure dataset.

### Self-distillation set

To create the sequecne dataset:

1. Compute multiple sequence alignmetns of every cluster in Uniclust30 (version 2018-08) against the same database.
1. Greedily remove sequence sequences which appear in another sequence's MSA.
1. Filter out sequences with more than 1,024 or fewer than 200 amino acids, or whoose MSA contain fewer than 200 alignments.

For building the distillation set of predicted structures:

1. Train a teacher model with (just) the PDB dataset.
1. Using this teacher model, predict the structure of every sequence in the set constructed above.
1. For every pair of predicted residues, calculate a confidence metric by computing the Kullback-Leibler divergnece between the pairwise distance distribtuion and a reference distribution for that residue separation.
1. The pairwise metric $c_{ij}$ is then averaged over $j$ corresponding to a maximum sequence separation of $\pm 128$ and excluding $i=j$ to give a per-residue confidence metric $c_i$.

The confidence metric is calculated as:

$$ c_{ij} = D_{KL} \left( p^\mathrm{ref}_{|i-j|}(X) \middle\| p^\mathrm{pred}_{ij}(X) \right) $$

The reference distribution is computed by taking distance distribution predictions for 1,000 randomly sampled sequences in Uniclust30 and computing the mean distribution for a given sequence separation.

Extra augmentation is added to distillation dataset examples by uniformly sampling the MSA to 1,000 sequences without replacement (this is on top of any sampling that happens in the data pipeline).

### Structure template search

**At training time** the available templates up to 20 with the highest `Sum_probs` are selected fisrtly. Then random $k$ templates in this restricted set of $n$ templates are choosed, where $k = \min (4, \mathrm{Uniform}[0,n])$.

This has the effect of showing the network potentially bad templates, or no templates at all so the network cannot relay on just copying the template.

Following templates are excluded:

- All templates that were released after the query training strucuture.
- Templates that are identical to (or a subset of) the input primary sequence.
- Too small (less than 10 residues or of length less than 10% of the primary sequence) templates.

## Training data

A training example comes from:

- With 75% probability the self-distillation set.
- With 25% probability a known structure from Protein Data Bank. (Extra augmentation.)

Loop over this hybrid set multiple times during training with apply following procedures every time when a protein is encounterd.

- A number of stochastic filters
- MSA block deletion
- MSA clustering (as mentioned before)
- Residue cropping

This means, the model may observe different targets in training epochs, with different samples of the MSA data, and also cropped to different regions.

### Filtering

The following filters are applied to the training data:

- Input mmCIFs are restricted to have resolution less than 9 Å.
- Protein chains are accepted with probability $\frac{1}{512} \max ( \min (N_\mathrm{res} , 512 ) , 256)$, where $N_\mathrm{res}$ is the length of the chain.[^1]
- Sequences filtered out when any singe amino acid accounts for more than 80% of the input primary sequence.
- Protein chains are accepted with the probabilty inverse to the size of the cluster[^2] that this chain falls into.

[^1]: This re-balanced the length distribution and makes the network train on crops from the longer chains more often.
[^2]: Use 40% sequence identity clusters of the Protein Data Bank clustered with MMSeqs2.

### MSA block deletion

During training contiguous blocks of sequences are deleted from the MSA.
The MSA is grouped by tool and ordered by the normal output of each tool, typically e-value.
This means that similar sequences are more likely to be adjacent in the MSA and block deletions are more likely to generate diversity that removes whole branches of the phylogeny[^phylogeny].

[^phylogeny]: Phylogeny, the history of the evolution of a species or group, especially in reference to lines of descent and relationships among broad groups of organisms.

1. Select a MSA.
1. Select five block starting points.
1. Delete $\lfloor 0.3 \cdot N_\mathrm{seq} \rfloor$ entries starting from the starting points. (Each interval can be overlapped.)

### Residue cropping

During training the residue dimension in all data is cropped in the following way.

---

## Loss

---

## Training

### Self-distillation
