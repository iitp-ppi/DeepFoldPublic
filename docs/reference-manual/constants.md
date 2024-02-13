# Internal Representations

## Constants

Some module attributes are frequently used in the package.
`deepfold.common.residue_constants` is the module where they are declared.

There are 20 different amino acids that are used for building proteins.
You can think of each residue as a token, and every type of amino acid as a different kind of token.
However, sometimes we were unable to identify the exact type of some residues.
In this case, we give those residues an `UNK` (or `X`) token.
There are 21 distinct types of `aatype` for this reason.

In the case of template pair features, it has additional `GAP` (or `-`) token.
This token indicates that there is no structural template for the corresponding residue.
Therefore, there are 22 distinct types for template features.

MSA featuers has one more token named `MASKED_MSA_TOKEN`.
This token is for BERT training proceudre.
Therefore, there are 23 distinct types for MSA features.

## One-hot residue type representation

The one-hot representation can be inversed with following table.
See `deepfold.common.residue_constants.restypes_with_x_and_gap`.

| Code | 1 |  3  | Name             |
|:----:|:-:|:---:|------------------|
|   0  | A | ALA | Alanine          |
|   1  | R | ARG | Arginine         |
|   2  | N | ASN | Asparagine       |
|   3  | D | ASP | Aspartate        |
|   4  | C | CYS | Cysteine         |
|   5  | Q | GLN | Glutamine        |
|   6  | E | GLU | Glutamate        |
|   7  | G | GLY | Glycine          |
|   8  | H | HIS | Histidine        |
|   9  | I | ILE | Isoleucine       |
|  10  | L | LEU | Leucine          |
|  11  | M | MET | Methionine       |
|  12  | K | LYS | Lysine           |
|  13  | F | PHE | Phenylalanine    |
|  14  | P | PRO | Proline          |
|  15  | S | SER | Serine           |
|  16  | T | THR | Threonine        |
|  17  | W | TRP | Tryptophan       |
|  18  | Y | TYR | Tyrosine         |
|  19  | V | VAL | Valine           |
|  20  | X | UNK | Unknown          |
|  21  | - | GAP | Gap              |
|  22  |   |     | Maksed MSA token |

## Atom representation

- See `deepfold.common.residue_constants.restype_name_to_atom14_names`.
- See `deepfold.common.residue_constants.RESTYPE_ATOM14_TO_ATOM37`.
- See `deepfold.common.residue_constants.RESTYPE_ATOM37_TO_ATOM14`.
