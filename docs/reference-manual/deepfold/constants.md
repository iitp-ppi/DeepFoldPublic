# Constants

---

Some module attributes are frequently used in the package.
`deepfold.common.residue_constants` is the module where they are declared.

There are 20 different amino acids that are used for building proteins.
You can think of each residue as a token, and every type of amino acid as a different kind of token.
However, sometimes we were unable to identify the exact type of some residues.
In this case, we give those residues an `UNK` token.
There are 21 distinct types of `aatype` for this reason.

In the case of template pair features, it has additional `GAP` token.
This token indicates that there is no structural template for the corresponding residue.
Therefore, there are 22 distinct types for template features.

MSA featuers has one more token named `MASKED_MSA_TOEKN`.
This token is for BERT training proceudre.
Therefore, there are 23 distinct types for MSA features.

