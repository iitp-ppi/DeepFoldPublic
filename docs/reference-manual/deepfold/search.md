# Genetic Search

## Input sequence

- FASTA format includes exactly one amino-acid sequence.

```text
>sp|P06213|INSR_HUMAN Insulin receptor OS=Homo sapiens OX=9606 GN=INSR PE=1 SV=4
MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHL
QILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYAL
VIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNE
...
```

## Multiple sequecne alignment (MSA)

### Databases

### Tools

## Structural template search

### Monomer (hhsearch)

### Multimer (hmmsearch)

## Implementation Notes

- `DataPipeline` excludes `uniprot.sto` and `hmm_output.sto` when creates MSA features.

## Summary: `features.pkl`

### One-hot residue types

HHBlits convention with extra tokens.
See section *representations*.

### Sequence features

- `aatype: int[N, 21]` Target sequence in integer representation.
- `between_segment_residues: int[N]` Currently, zero. *Deprecated*
- `domain_name: object[1]` Name of the domain.
- `residue_index: int[N]` Residue index start from zero.
- `seq_length: int[N]` Currently `[num_res] * num_res`.
- `sequence: object[1]` One-symbol sequence of the target.

### MSA features

- `deletion_matrix_int: int[S, N]` Deletion counters for each residues.
- `msa: int[S, N]` MSA without deletions.
- `num_alignments: int[N]` Currently, `[num_seq] * num_res`.
- `msa_species_identifiers: object[S]` Species identifiers
- `msa_uniprot_accession_identifiers: object[S]` Uniprot accession identifiers (not in AF).

### Template features

- `template_aatype: float[N_t, N, 22]` One-hot encoded residue types.
- `template_all_atom_mask: float[N_t, N, 37]` Atom masks for templates.
- `template_all_atom_positions: float[N_t, N, 37, 3]` Atom positions for templates. Encoded with `atom37` representation.
- `template_domain_names: object[N_t]` Domain names.
- `template_sequence: object[N_t]` One-symbol sequences.
- `template_sum_probs: float[N_t, 1]` Output from the template aligner. Higher templates are selected.
