# Configuration

- Configuration files are written in YAML format.
- [OmegaConf](https://github.com/omry/omegaconf) is used to load and save configurations.
- OmegaConf support variable interpolation and it is lazy.

## `globals`

Global options

## `data`

### `common`

#### `features`

List of feature entry shapes.

- `aatype`
- `all_atom_mask`
- `all_atom_positions`
- `alt_chi_angles`
- `atom14_alt_gt_exists`
- `atom14_alt_gt_positions`
- `atom14_atom_exists`
- `atom14_atom_is_ambiguous`
- `atom14_gt_exists`
- `atom14_gt_positions`
- `atom37_atom_exists`
- `backbone_affine_mask`
- `backbone_affine_tensor`
- `bert_mask`
- `chi_angles`
- `chi_mask`
- `extra_deletion_value`
- `extra_has_deletion`
- `extra_msa`
- `extra_msa_mask`
- `extra_msa_row_mask`
- `is_distillation`
- `msa_feat`
- `msa_mask`
- `msa_row_mask`
- `pseudo_beta`
- `pseudo_beta_mask`
- `random_crop_to_size_seed`
- `residue_index`
- `residx_atom14_to_atom37`
- `residx_atom37_to_atom14`
- `resolution`
- `rigidgroups_alt_gt_frames`
- `rigidgroups_group_exists`
- `rigidgroups_group_is_ambiguous`
- `rigidgroups_gt_exists`
- `rigidgroups_gt_frames`
- `seq_length`
- `seq_mask`
- `target_feat`
- `template_aatype`
- `template_all_atom_masks`
- `template_all_atom_positions`
- `template_backbone_affine_mask`
- `template_backbone_affine_tensor`
- `template_mask`
- `template_pseudo_beta`
- `template_pseudo_beta_mask`
- `template_sum_probs`
- `true_msa`

#### `masked_msa` (Common)

- See AF2 supplementary 1.2.7 MSA clustering.
- `uniform_prob := 0.1` are replaced with a uniformly sampled amino acids.
- `profile_prob := 0.1` are replaced with an amino acid sampled from the MSA profile.
- `same_prob := 0.1` are not replaced.
- Otherwise amino acids are replaced with a `masked_msa_token`.

#### `max_extra_msa` (Common)

- Maximum number of extra MSA sequences.

#### `block_delete_msa` (Training)

- See AF supplementary 1.2.6. MSA block deletion.
- `msa_fraction_per_block := 0.3`
- `randomize_num_blocks := False`
- `num_blocks := 5`
- `min_num_msa := 16`

#### `max_cluster_features`

#### `num_recycle`

#### `reduce_msa_clusters_by_max_templates`

#### `resample_msa_in_recycling`

#### `use_templates`

---

### Mode specific options

#### `fixed_size`

#### `subsample_templates`

#### `masked_msa_replace_fraction`

#### `max_msa_clusters`

#### `max_templates`

#### `num_ensemble`

---

### `predict`

- `fixed_size: True`
- `subsample_templates: True`
- `masked_msa_replace_fraction: 0.15`
- `max_msa_clusters: 512`
- `max_templates: 4`
- `num_ensemble: 1` *Deprecated*

### `eval`

### `train`

## `model`

## `loss`
