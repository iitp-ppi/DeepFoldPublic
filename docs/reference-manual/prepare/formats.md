# Formats

## `FASTA`

The FASTA format is a text-based format for representing either nucleotide sequences or amino acid (protein) sequences, in which nucleotides or amino acids are represented using single-letter codes.
It originated from the FASTA software package[^1], but has now become a near universal standard in the field of bioinformatics.

[^1]: The commonly used form is different from the original format.

A sequence in FASTA format begins with a single-line description, followed by lines of sequence data.
The description line (defline) is distinguished from the sequence data by a greater-than (“>”) symbol at the beginning.
It is recommended that all lines of text be shorter than 80 characters in length.

For example:

```{plain text}
>1HDD_3|Chains C, D|PROTEIN (ENGRAILED HOMEODOMAIN)|Drosophila melanogaster (7227)
MDEKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKKS
```

```{plain text}
>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP
AVHASLDKFLASVSTVLTSKYR
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens OX=9606 GN=HBB PE=1 SV=2
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG
KEFTPPVQAAYQKVVAGVANALAHKYH
```

## `PDB`

## `mmCIF`

## `A3M`

## `STO`

## `HHR`
