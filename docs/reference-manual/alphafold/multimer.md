# Caparison between AlphaFold2 and AlphaFold-Multimer

## Data processing and input features

### Multi-chain permutation alignment

### MSA construction

### Cross-chain genetics

### Multi-chain cropping

Uniformly sample a single bio-assembly from all the availiable bio-assemblies.
The MSA features are merged ...

Uses only per-chain templates.

### Extra features

Extra positional encodings denoting whether a given pair of amino acids corresponds to different chains and whether they belong to different homomer or heteromer chains are added.
Therefore, the features input to the multimer model are identical to monomer model with three extra features:

- `asym_id`
- `entity_id`
- `sym_id`

## Network

### Multimer input embedder

- Extra positional encodings with relative position encoding are introduced.

### Template stack

- The order of the attention and triangular multiplicative update layers in the template is stack swapped relative to the AlphaFold model. (It consistent with the order in the Evoformer stack.)
- The template unit vectors are enabled. (They are not set to zero but computed from the template coordinates.)
- The template embeddings are aggregated by simply averaging the template embeddings. (Template pointwise attention removed.)

### Evoformer stack

- Moved the outer product mean to the start of the Evoformer block. (We do not need to keep all of the MSA stack activations and all the pair stack activations in memory at the same time during backpropagation.)

## Losses

### Mixed FAPE clamping

For the **inter**-chain pairs, clamping cutoff length is changed from $10$ Å to $30$ Å.
This provides a better gradient signal for incorrect interfaces.
(For the **intra**-chain pairs of the complex they keep the same $10$ Å clamping.)

### Chain center-of-mass loss

To help prevent the model from predicting overlapping chains when it is highly uncertain we introduce an additional chain center-of-mass loss term such that chains are pushed apart if they are closer than in the ground truth, defined as:

$$\mathcal{L}_\mathrm{com} = \frac{1}{N(N-1)} \sum _i \sum _j \min \left[ \frac{1}{20} \left( \| c^{pred}_i - c^{pred}_j \| - \| c^{gt}_i - c^{gt}_j \| + 4\right) , 0 \right]^2$$

where $N$ is the number of chains, $c^{pred}_i$ is the center of mass of the $C_\alpha$ atoms for predicted chain $i$ and $c^{gt}_i$ is the center of mass of the $C_\alpha$ atoms for ground truth chain $i$.
The loss is clamped if the error is $-4$ Å or greater to prevent slight model uncertainty pushing the chains apart.

### Clash loss

**multimer_v1** doesn't include this term.

In order to reduce the number of violations they modified the violation los by changing the term penalizing stereic clashes of non-bonded atoms to avaerge over clashing atoms rather than summing across all atoms.

$$\mathcal{L}_\mathrm{clash} = \frac{1}{N_\mathrm{clash}} \sum ^{N_\mathrm{nbpairs}}_{i=1} \max \left[ d^i_\mathrm{lit} - \tau - d^i_\mathrm{pred} , 0 \right]$$

where $N_\mathrm{clash} = \mathrm{count}_i \left( d^i_\mathrm{lit} - \tau > d^i_\mathrm{pred} \right)$.
Here the sum runs over nonbonded atom pairs, $d^i_\mathrm{lit}$ is the minimum distance between atoms derived from the relevant van der Waals radii. $\tau$ is a tolerance factor that is set to $1.5$ Å.
$d^i_\mathrm{pred}$ is the distance between the atom in pair $i$ in the predicted structure.

The weight of the violation losses is $0.03$ and the weight of the bond-angle term is $0.3$.

## Heads

### Inteface pTM

$$\mathrm{ipTM} = \max _i \frac{1}{|D \setminus C_i|} \sum _{j \in D \setminus C_i} E \left[ \frac{1}{1 + \left( \frac{e_{ij}}{d_0 (| D \setminus C_i |) )} \right)^2} \right]$$

## Outputs

### Inference regimen

- All $5$ trained models are run once and select the best model on each target according to model confidnece.
- Run each model $5$ times with different random seeds. (25 predictions.)

### Model confidence

$$\mathrm{confidence} = 0.8 \cdot \mathrm{ipTM} + 0.2 \cdot \mathrm{pTM}$$
