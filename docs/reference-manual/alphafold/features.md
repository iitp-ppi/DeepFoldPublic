# Data Pipeline

The data pipeline is the first step when running AlphaFold.

## Input

- The pipeline takes a **FASTA** file in the inference mode.
- In the multimer inference mode, a single FASTA file can include many sequences and they are considered as a complex.

The input file is parsed and basic metadata is extracted from it.

- For FASTA, this is only the sequence and name.

## Genetic search methods

### Multiple sequence alignment

Multiple database are searched using `JackHMMER 3.3` and `HHBlits 3.0-beta.3`.

- [UniRef90 2020_01](https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_01/uniref/)[^uniref]
- [BFD](https://bfd.mmseqs.com)
- [MGnify 2018_12](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/)[^mgnify]
- [Uniclust 2018_08](https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) (to construct a distillation structure dataset)[^uniclust]

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

### Structural template search

1. The **UniRef90 MSA** obtained in the previous step is used to search PDB70 using **hhsearch** with `-maxseq 1000000`.
1. Structural data for each hit is obtained from the corresponding mmCIF file in the **PDB database**.
1. If the sequence from PDB70 does not exactly match the sequence in the mmCIF file then the two are aligned using **Kalign**.

```{sh}
hhsearch \
    -cpu 4 \
    -maxseq 1000000 \
    -i $INPUT_A3M_PATH \
    -d $DATABASE_PATH \
    -o $HHR_PATH

# $DATABASE_PATH should be the common prefix for the database files
# i.e. up to but not including_hhm.ffindex etc.

kalign \
    -i $INPUT_FASTA_PATH \
    -o $OUTPUT_A3M_PATH \
    -format fasta
```

**At inference time** the top 4 templates are provided to the model, sorted by the expected number of correctly aligned residues (the `Sum_probs` feature output by hhsearch).

#### Parsing Stockholm format

1. Search UniRef90 with JackHMMER.
1. Remove dupliate sequences (ignoring insertions w.r.t. query). (`deepfold.data.parsers.deduplicate_stockholm_msa`)
1. Remove empty columns (dashes-only). (`deepfold.data.parsers.remove_empty_columns_from_stockholm_msa`)
1. Ready to search templates with `HHSearch` or `hmmsearch`.

For the **AlphaFold-Multimer** the template search is similar to above with few differences.

- The AlphaFold-Multimer system uses only per-chain templates.
- The template search is simiilar to the monomer template search, except that it uses **hmmsearch** and **hmmbuild** instead of hhsearch.
    1. The UniRef90 MSA obtained in the MSA search is converted to an HMM using hmmbuild.
    1. hmmbuild is then used to search for matches of the HMM against `pdb_seqres.txt`[^seqres].
    1. The number of templates are limited to 20.
- Further processing is as described above.
<!-- 1. Any structure released after 2018-04-30 is excluded from training. -->

```{sh}
hmmsearch \
    -F1 0.1 -F2 0.1 -F3 0.1 -incE 100 -E 100 -domE 100 -incdomE 100 \
    ...

hmmbuild \
    ...
```

[^seqres]: <https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt> (on 2020-05-14)

---

## Pipeline

After runs alignment tools on the input sequence, features are generated with following procedure.

### Summary

#### Sequence features

- `aatype`
- `between_segment_residues`
- `domain_name`
- `residue_index`
- `seq_length`
- `sequence`

#### MSA features

- `deletion_matrix_int`
- `msa`
- `num_alignments`
- `msa_species_identifiers`

#### Template features

- `template_aatype`
- `template_all_atom_masks`
- `template_all_atom_positions`
- `template_domain_names`
- `template_sequences`
- `template_sum_probs`

### MSA clustering

The computational and peak memory cost of the main Evoformer module scales as $N_\mathrm{seq}^2 \times N_\mathrm{res}$, so it is highly desirable to reduce the number of sequences used in the main Evoformer module for purely computational reasons.
However, randomly chosing a fixed-size subset of sequences without replacement has a problem that sequences not included in the random subset have no influence on the prediction.

---

## Model inputs

For monomer models

- `target_feat` : $[N_\mathrm{res}, 21]$
- `residue_index` : $[N_\mathrm{res}]$
- `msa_feat` : $[N_\mathrm{clust}, N_\mathrm{res}, 49]$
- `extra_msa_feat` : $[N_\mathrm{extra_seq}, N_\mathrm{res}, 25]$
- `template_pair_feat` : $[N_\mathrm{templ}, N_\mathrm{res}, N_\mathrm{res}, 88]$
- `template_angle_feat` : $[N_\mathrm{templ}, N_\mathrm{res}, 57]$

The features input to the multimer model are identical to monomer model with three extra features:

- `asym_id` : a unique integer per chain indicating the chain number. $[N_\mathrm{res}]$
- `entity_id` : a unique integer for each set of identical chains. $[N_\mathrm{res}]$
- `sym_id` : a unique integer within a set of identical chains. $[N_\mathrm{res}]$

For example in an A3B2 stoichiometry complex all the A chains would have the same `entity_id` and have `sym_id`s 0, 1, 2.

```text
Chain ID    : A B C D E
asym_id     : 0 1 2 3 4
entity_id   : 0 0 0 1 1
sym_id      : 0 1 2 0 1
```
