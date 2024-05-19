import numpy as np


def crop_features(feats: dict, start: int, end: int):
    """
    Args:
        [start, end] -> range(start - 1, end)
    """
    mask = np.arange(start - 1, end, dtype="int")
    new_feats = {}

    new_feats["domain_name"] = feats["domain_name"].copy()
    new_feats["template_domain_names"] = feats["template_domain_names"].copy()
    new_feats["template_sum_probs"] = feats["template_sum_probs"].copy()

    new_feats["aatype"] = feats["aatype"][mask, :]
    new_feats["between_segment_residues"] = feats["between_segment_residues"][mask]
    new_feats["residue_index"] = feats["residue_index"][mask]
    new_feats["deletion_matrix_int"] = feats["deletion_matrix_int"][:, mask]
    new_feats["msa"] = feats["msa"][:, mask]
    new_feats["num_alignments"] = feats["num_alignments"][mask]
    new_feats["template_aatype"] = feats["template_aatype"][:, mask, :]
    new_feats["template_all_atom_positions"] = feats["template_all_atom_positions"][:, mask, :, :]
    new_feats["template_all_atom_mask"] = feats["template_all_atom_mask"][:, mask, :]

    new_feats["seq_length"] = feats["seq_length"][mask]
    new_feats["seq_length"].fill(end - start + 1)

    new_feats["sequence"] = np.array([feats["sequence"].item()[start - 1 : end]], dtype=np.object_)

    if isinstance(feats["template_sequence"], np.ndarray):
        new_feats["template_sequence"] = [s.item()[start - 1 : end] for s in feats["template_sequence"]]
    else:
        new_feats["template_sequence"] = [s[start - 1 : end] for s in feats["template_sequence"]]

    return new_feats
