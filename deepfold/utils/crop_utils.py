from typing import Dict

import numpy as np


def unpad_to_schema_shape_(
    outputs: Dict[str, np.ndarray],
    output_schema_shapes: Dict[str, tuple],
    num_residues: int,
    num_clustered_msa_seq: int,
) -> Dict[str, np.ndarray]:

    shape_map = {
        "NUM_RES": num_residues,
        "NUM_MSA_SEQ": num_clustered_msa_seq,
    }

    for key, arr in outputs.items():
        array_shape = list(arr.shape)
        schema_shape = output_schema_shapes[key]
        assert len(array_shape) == len(schema_shape), key

        out_shape = tuple(shape_map.get(dim_schema, dim_size) for dim_schema, dim_size in zip(schema_shape, array_shape))
        slices = tuple(slice(0, size) for size in out_shape)

        outputs[key] = arr[slices]

    return outputs
