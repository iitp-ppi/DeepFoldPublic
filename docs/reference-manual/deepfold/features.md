# Monomer Input Features

## Input sequence

- FASTA format includes exactly one amino-acid sequence.

```text
>sp|P06213|INSR_HUMAN Insulin receptor OS=Homo sapiens OX=9606 GN=INSR PE=1 SV=4
MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHL
QILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYAL
VIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNE
...
```

## Genetic search

### Multiple sequence alignment (MSA)

(...)

### Structural template search

(...)

## Processing features

## Summary: `features.pkl`

### One-hot residue types

HHBlits convention with extra tokens.
See section *representations*.

### Sequence features

- `aatype: int32[N, 21]` Target sequence in integer representation.
- `between_segment_residues: int32[N]` Currently, zero.
- `domain_name: str[1]` Name of the domain.
- `residue_index: int32[N]` Residue index start from zero.
- `seq_length: int32[N]` Currently `[num_res] * num_res`.
- `sequence: str[1]` One-symbol sequence of the target.

### MSA features

- `deletion_matrix_int: int32[S, N]` Deletion counters for each residues.
- `msa: int32[S, N]` MSA without deletions.
- `num_alignments: int32[N]` Currently, `[num_seq] * num_res`.
- `msa_species_identifiers: str[S]` Species identifiers
- `msa_uniprot_accession_identifiers: str[S]` Uniprot accession identifiers (not in AF).

### Template features

- `template_aatype: float32[N_t, N, 22]` One-hot encoded residue types.
- `template_all_atom_masks: float32[N_t, N, 37]` Atom masks for templates.
- `template_all_atom_positions: float32[N_t, N, 37, 3]` Atom positions for templates. Encoded with `atom37` representation.
- `template_domain_names: str[N_t]` Domain names.
- `template_sequence: str[N_t]` One-symbol sequences.
- `template_sum_probs: float32[N_t, 1]` Output from the template aligner. Higher templates are selected.
