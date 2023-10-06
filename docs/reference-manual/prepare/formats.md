# Formats

## `FASTA`

The FASTA format is a text-based format for representing either nucleotide sequences or amino acid (protein) sequences, in which nucleotides or amino acids are represented using single-letter codes.
It originated from the FASTA software package[^fasta], but has now become a near universal standard in the field of bioinformatics.

[^fasta]: The commonly used form is different from the original format.

A sequence in FASTA format begins with a single-line description, followed by lines of sequence data.
The description line (defline) is distinguished from the sequence data by a greater-than (“>”) symbol at the beginning.
It is recommended that all lines of text be shorter than 80 characters in length.

Example:

```{plain text}
>1HDD_3|Chains C, D|PROTEIN (ENGRAILED HOMEODOMAIN)|Drosophila melanogaster (7227)
MDEKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKKS
```

A multiple sequence FASTA format would be obtained by concatenating several single sequence FASTA files in a common file.

Example:

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

Blank lines are not allowed in the middle of FASTA input.

Sequences are expected to be represented in the standard IUB/IUPAC amino acid and nucleic acid codes, with these exceptions: lower-case letters are accepted and are mapped into upper-case[^a3m][^blast].

Amino acid codes:

```{plain text}
A  alanine               P  proline
B  aspartate/asparagine  Q  glutamine
C  cystine               R  arginine
D  aspartate             S  serine
E  glutamate             T  threonine
F  phenylalanine         U  selenocysteine
G  glycine               V  valine
H  histidine             W  tryptophan
I  isoleucine            Y  tyrosine
K  lysine                Z  glutamate/glutamine
L  leucine               X  any
M  methionine            *  translation stop
N  asparagine            -  gap of indeterminate length
```

[^a3m]:
    In A3M sequences, lowercase characters are taken to mean insertions, and "`-`" indicates deletions.
    See [HH-suite repository](https://github.com/soedinglab/hh-suite/blob/4c0cd66434ce0b83ccd247053f57989fdd53d82b/scripts/reformat.pl#L44) for further informations.
[^blast]:
    The BLAST webpage will not accept "`-`" in the query.
    To represent gaps, use a string of "`X`" instead.

The [NCBI](https://blast.ncbi.nlm.nih.gov/doc/blast-topics/#fasta) defined a standard for the unique identifier used for the sequence in the header line.

## `A3M`

The A3M format consists of aligned fasta, in which alignments are shown with inserts as lower case characters, matches as upper characters, deletions as "`0`", and gaps aligned to insert as "`.`".
Note that gaps aligned to inserts can be omitted in the A3M format.

In the standard A3M format, sequences are separated by "`>`".

## `STO`

[The STO (Stockholm) format](https://sonnhammer.sbc.su.se/Stockholm.html) is a MSA format used by probablistic database search tools like HMMER (including `jackhmmer`).
A stockholm file consists of a header line with a format and verison identifier; mark-up lines starting with "`#=GF`", "`#=GC`", "`#=GS`" or `"#=GR`"; alignment lines with the sequence name and aligned sequence; a "`//`" line indicating the end of the alignment.
Alignments are shown with inserts as lower case characters, matches as upper case characters, and gaps as '`.`' or '`-`'.

Example:

```{plain text}
# STOCKHOLM 1.0
#=GF ID sp|P10636-7|TAU_HUMAN-i1
#=GF DE Isoform Tau-E of Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT
#=GF AU jackhmmer (HMMER 3.3.2)

#=GS sp|P10636-7|TAU_HUMAN            DE Isoform Tau-E of Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT
#=GS UniRef90_P10636-7/1-412          DE [subseq from] Isoform Tau-E of Microtubule-associated protein tau n=74 Tax=Boreoeutheria TaxID=1437010 RepID=P10636-7
#=GS UniRef90_UPI000387CAEE/1-85      DE [subseq from] microtubule-associated protein tau isoform X7 n=5 Tax=Catarrhini TaxID=9526 RepID=UPI000387CAEE
#=GS UniRef90_UPI000387CAEE/101-187   DE [subseq from] microtubule-associated protein tau isoform X7 n=5 Tax=Catarrhini TaxID=9526 RepID=UPI000387CAEE
(...)

sp|P10636-7|TAU_HUMAN                    MAEPRQEFEVMEDHAG-T--------YGL---------GDRKD---Q--GGYTM--H--Q-D-Q-EGDT--D-A-G-L-K-E-S-------P---L--Q-TP--T---E--D----G---S-------E--E---P-G---S---E--T-S--D-A---K-------ST--P-----T--A--E-A--------E--E--
UniRef90_P10636-7/1-412                  MAEPRQEFEVMEDHAG-T--------YGL---------GDRKD---Q--GGYTM--H--Q-D-Q-EGDT--D-A-G-L-K-E-S-------P---L--Q-TP--T---E--D----G---S-------E--E---P-G---S---E--T-S--D-A---K-------ST--P-----T--A--E-A--------E--E--
#=GR UniRef90_P10636-7/1-412          PP 89**************.*........***.........*****...*..*****..*..*.*.*.****..*.*.*.*.*.*.*.......*...*..*.**..*...*..*....*...*.......*..*...*.*...*...*..*.*..*.*...*.......**..*.....*..*..*.*........*..*..
UniRef90_UPI000387CAEE/1-85              MAEPRQEFEVMEDHAG-T--------YGL---------GDRKD---Q--GGYTM--H--Q-D-Q-EGDT--D-A-G-L-K-E-S-------P---L--Q-TP--T---E--D----G---S-------E--E---P-G---S---E--T-S--D-A---K-------ST--P-----T--A--E-D--------V--T--
#=GR UniRef90_UPI000387CAEE/1-85      PP 89**************.*........***.........*****...*..*****..*..*.*.*.****..*.*.*.*.*.*.*.......*...*..*.**..*...*..*....*...*.......*..*...*.*...*...*..*.*..*.*...*.......**..*.....*..*..*.9........8..8..
UniRef90_UPI000387CAEE/101-187           ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------T--T-A--------E--E--
#=GR UniRef90_UPI000387CAEE/101-187   PP ....................................................................................................................................................................................4..5.8........9..*..
(...)
//
```

## `HHR`

## `PDB`

## `mmCIF`
