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

- UniRef90
- MGnify
- BFD
- UniRef30 (FKA UniClust30)
- UniProt (TrEMBL + Swiss-Prot)

### Tools

- JackHMMER
- HHBlits

## Structural template search

### Monomer

- hhsearch
- PDB70 database

### Multimer (hmmsearch)

- hmmbuild
- hhmsearch
- PDB seqres database

## Summary (Single chain `features.pkl`)

- aatype
- between_segment_residues
- domain_name
- residue_index
- seq_length
- sequence
- deletion_matrix_int
- msa
- num_alignments
- msa_species_identifiers
- template_aatype
- template_all_atom_masks
- template_all_atom_positions
- template_domain_names
- template_sequence
- template_sum_probs
