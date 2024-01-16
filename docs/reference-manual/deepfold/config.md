# Configuration

- Configuration files are written in YAML format.
- [OmegaConf](https://github.com/omry/omegaconf) is used to load and save configurations.
- OmegaConf support variable interpolation and it is lazy.

## `globals`

Global options

## `data`

### `common`

#### `features`

List of feature entries.

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
- `chi_angles_sin_cos` (`chi_angles`)
- `chi_mask`
- `extra_msa_deletion_value` (`extra_deletion_value`)
- `extra_msa_has_deletion` (`extra_has_deletion`)
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
- `template_all_atom_mask` (`template_all_atom_masks`)
- `template_all_atom_positions`
- `template_backbone_affine_mask`
- `template_backbone_affine_tensor`
- `template_mask`
- `template_pseudo_beta`
- `template_pseudo_beta_mask`
- `template_torsion_angles_mask`
- `template_torsion_angles_sin_cos`
- `template_alt_torsion_angles_sin_cos`
- `template_sum_probs`
- `true_msa`

Followings are additional features for multimer models.

- `assembly_num_chains`
- `asym_id`
- `asym_len`
- `cluster_bias_mask`
- `cluster_profile`
- `cluster_deletion_mean`
- `deletion_matrix`
- `deletion_mean`
- `entity_id`
- `entity_mask`
- `extra_deletion_matrix`
- `msa`
- `msa_profile`
- `num_alignments`
- `num_sym`
- `num_templates`
- `sym_id`

#### `masked_msa`

- See AF2 supplementary 1.2.7 MSA clustering.
- `uniform_prob := 0.1` are replaced with a uniformly sampled amino acids.
- `profile_prob := 0.1` are replaced with an amino acid sampled from the MSA profile.
- `same_prob := 0.1` are not replaced.
- Otherwise amino acids are replaced with a `masked_msa_token`.

#### `max_msa_entries`

- Maximum number of extra MSA sequences.

#### `block_delete_msa`

- During *training* contiguous blocks of sequences are deleted from the MSA.
- See AF supplementary 1.2.6 MSA block deletion.
- `msa_fraction_per_block := 0.3` Ratio of block size to the sequence length.
- `num_blocks := 5` How many deletion blocks will be sampled.
- `randomize_num_blocks := False` If true, the number of blocks is randomly sampled between once and `num_blocks`.
- `min_num_msa := 16` If the number of sequences are less than the value, it will not block deletion.

#### `max_msa_entry`

Randomly delete MSA sequences to reduce the cost of MSA features.

#### `v2_feature`

Use data operations introduced with AlphaFold-Multimer model.

#### `gumbel_sample`

Sample MSA with Gumbel sampling otherwise sample MSA uniformly.
It is the default for multimer models.

The first row is always skipped. It's center must be the target sequences.

#### `max_cluster_features`

If true, it will assign each extra MSA sequence to its nearest neighbor in sampled MSA.

#### `reduce_msa_clusters_by_max_templates`

Replace last MSA clusters by the number of templates to the template sequences.

#### `resample_msa_in_recycling`

If true, it willi resample MSA in recycling.
Actually, sampling is done during procesesing input features.

#### `use_templates`

If true, the model will use distogram from the structural templates (when it is possible).

#### `use_template_torsion_angles`

If true, the model will use torsion angles from the structural templates (when it is possible).

#### `is_multimer`

If true, the multimer model is used.

#### `max_recycling_iters`

Maximum number of recycling iterations.
In the original implementation it is equivalent to `num_recycle`.

### Mode specific options

These options are used for prediction, evaluation and training.

#### `fixed_size`

Fix the shape of representations for the performance during training.

#### `subsample_templates`

See the third process of AF2 supplementary 1.2.3 Template search.

#### `block_delete_msa` <!-- markdownlint-disable-line no-duplicate-heading -->

Whether do MSA block deletion or not.

#### `random_delete_msa`

Randomly remove MSA sequences more than `max_msa_entries`.

#### `masked_msa_replace_fraction`

Used to create data for BERT on raw MSA.

#### `max_msa_clusters`

Maximum number of MSA clusters.

#### `max_templates`

The maximum number of templates.

#### `num_ensemble`

(*Deprecated*) The number of ensembles to be averaged.

#### `crop`

Whether crop the sequences or not.

#### `crop_size`

The size of each cropped sequences.

#### `spatial_crop_prob`

If the randomly generated real number from the unit interval is larger than the probability, then the spatial crop is done. Otherwise the contiguous crop is done.

#### `ca_ca_threshold`

The threshold value used to find interfaces.

#### `supervised`

Proceess the supervised features:

- `atom14_*`
- `rigidgroups_*`
- `pseudo_beta` and `pseudo_beta_mask`
- `chi_angles_sin_cos` and `chi_mask`

#### `clamp_prob`

The value is considered almost sure for the multimer models. (`clamp_prob := 1.0`)

See AF2 supplementary 1.11.5 Loss clamping details.

#### `max_distillation_msa_clusters`

See the last line of AF2 supplementary 1.3 Self-distillation dataset.

#### `share_mask`

Entity-sharing MSA mask.

## `model`

## `loss`
