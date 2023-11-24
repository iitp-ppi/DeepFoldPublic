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

We follow HHBlits convention.
The one-hot representation can be inversed with following table.

| Code | 1 |  3  | Name             |
|:----:|:-:|:---:|------------------|
|   0  | A | ALA | Alanine          |
|   1  | C | CYS | Cysteine         |
|   2  | D | ASP | Aspartate        |
|   3  | E | GLU | Glutamate        |
|   4  | F | PHE | Phenylalanine    |
|   5  | G | GLY | Glycine          |
|   6  | H | HIS | Histidine        |
|   7  | I | ILE | Isoleucine       |
|   8  | K | LYS | Lysine           |
|   9  | L | LEU | Leucine          |
|  10  | M | MET | Methionine       |
|  11  | N | ASN | Asparagine       |
|  12  | P | PRO | Proline          |
|  13  | Q | GLN | Glutamine        |
|  14  | R | ARG | Arginine         |
|  15  | S | SER | Serine           |
|  16  | T | THR | Threonine        |
|  17  | V | VAL | Valine           |
|  18  | W | TRP | Tryptophan       |
|  19  | Y | TYR | Tyrosine         |
|  20  | X | UNK | Unknown          |
|  21  | - | GAP | Gap              |
|  22  |   |     | Maksed MSA token |

## `atom37` representation

## `atom14` representation
