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
>tr|E1L4V1|E1L4V1_9FIRM 4-oxalocrotonate tautomerase family enzyme OS=Veillonella atypica ACS-049-V-Sch6 OX=866776 GN=dmpI PE=3 SV=1
MPLIHVELVEGRTKEQKKQLGEAITKATVDIVKVPAEAVKVIFVDMKKDEFMEGGILRSER
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

The A3M format consists of aligned fasta, in which alignments are shown with inserts as lower case characters, matches as upper characters, deletions as "`-`".

In the standard A3M format, sequences are separated by "`>`".

Example:

```{plain text}
>tr|E1L4V1|E1L4V1_9FIRM 4-oxalocrotonate tautomerase family enzyme OS=Veillonella atypica ACS-049-V-Sch6 OX=866776 GN=dmpI PE=3 SV=1
MPLIHVELVEGRTKEQKKQLGEAITKATVDIVKVPAEAVKVIFVDMKKDEFMEGGILRSER
>tr|A0A1H7KCH6|A0A1H7KCH6_9NOCA 4-oxalocrotonate tautomerase family enzyme OS=Rhodococcus maanshanensis OX=183556 GN=SAMN05444583_10410 PE=3 SV=1
MPMVTIKTMQGKSPEAIRKTMSDVGRVVAENLGYDAAHVMVFVEEVADTHFLTAGRTWAEL
>tr|A0A2N6EKZ9|A0A2N6EKZ9_9DELT Tautomerase OS=Desulfuromonas sp. OX=892 GN=C0619_11485 PE=3 SV=1
MPVIIVRTVEGVTTQQKEQLIEQFTRTMKEVMGKNPEATHIVIEEIPAENWGIRGRTVAAI
>tr|C5T5R3|C5T5R3_ACIDE 4-oxalocrotonate tautomerase OS=Acidovorax delafieldii 2AN OX=573060 GN=AcdelDRAFT_2243 PE=4 SV=1
MPTYHIEMMEGRAIEQTTKRGAEVNRASAPILGCSPDSMDMVVTGAERESRATDGALRPQP
>tr|A0A1S2C447|A0A1S2C447_9BACI 4-oxalocrotonate tautomerase OS=Bacillus aryabhattai OX=412384 GN=BCV52_11195 PE=3 SV=1
MPIVQVHLLEGREEKDIKNMLVEVTEALSNSLHVEKDRVRIIVNEVPSSHWGIAGVSRKEI
>tr|A0A2E0IFC6|A0A2E0IFC6_9RHOB Tautomerase OS=Ahrensia sp. OX=1872419 GN=CL534_01550 PE=3 SV=1
MPIAEILIFEGRTDDQKREIMREVTDALARSMDAEPERIRVILKEIPTNHFSVAGVSIADS
>SRR5262249_49533381
VSDWSSDVCSSDLIEYRKTIGEIVYHAMRDVIDVPKDDKFQIIAEHPGEAFNVS-------
>AntAceMinimDraft_17_1070374.scaffolds.fasta_scaffold12881_1 # 3 # 206 # 1 # ID=12881_1;partial=10;start_type=Edge;rbs_motif=None;rbs_spacer=None;gc_cont=0.564
-PSHTLEIINLIKQQHIKAIVMEPyfDRKTPDFIGAKTGAKVVVLYPSVGGKAGLDD--Y---
>SRR5216683_115507
--LVEVTRYQDHADRDPAGKLGPArrhagFRNRP-RLQDrrlairrrADEDIRLLDFDNRSQ------------
>SRR5262249_33036445
--LIQITLNAGRSLEQKKAFYKRIVDDMHTQLNVRPQEVVINLWKSRRKTGRSVAALR---
>SRR5438270_10750673
--LIQITLNTDRRDEEGVLQKDRRRPPcsleraarrrghqpcrgRQGKLVVRRRHRAIRGVEEVAGRLRGKX------
>SRR3974390_3476593
MPLIRISLLTGNPDDDRRAIAENLYASLREPFNVPENDFFAPVDELEPRDFIYDRK-----
>ERR1700722_9471896
NAaHSHLTarRKAGSLpAgDLRQP----LPR---DARSVERG-RRRSVHDH---------------
>SRR3954471_21681111
RHtLATGQS---KTGDSVRVRFNSKKAGNPATVTAPS-QSR--PASSPRGMRvM---------
```

## `STO`

[The STO (Stockholm) format](https://sonnhammer.sbc.su.se/Stockholm.html) is a MSA format used by probablistic database search tools like HMMER (including `jackhmmer`).
A stockholm file consists of a header line with a format and verison identifier; mark-up lines starting with "`#=GF`", "`#=GC`", "`#=GS`" or `"#=GR`"; alignment lines with the sequence name and aligned sequence; a "`//`" line indicating the end of the alignment.
Alignments are shown with inserts as lower case characters, matches as upper case characters, and gaps as '`.`' or '`-`'.

See [HMMER Documentation](http://hmmer.org/documentation.html).

Example:

```{plain text}
   1 # STOCKHOLM 1.0
   2 #=GF ID tr|E1L4V1|E1L4V1_9FIRM-i1
   3 #=GF DE 4-oxalocrotonate tautomerase family enzyme OS=Veillonella atypica ACS-049-V-Sch6 OX=866776 GN=dmpI PE=3 SV=1
   4 #=GF AU jackhmmer (HMMER 3.3.2)
   5
   6 #=GS tr|E1L4V1|E1L4V1_9FIRM        DE 4-oxalocrotonate tautomerase family enzyme OS=Veillonella atypica ACS-049-V-Sch6 OX=866776 GN=dmpI PE=3 SV=1
   7 #=GS UniRef90_X8HJZ3/1-61          DE [subseq from] 4-oxalocrotonate tautomerase family enzyme n=2 Tax=Veillonella TaxID=29465 RepID=X8HJZ3_9FIRM
   8 #=GS UniRef90_C4FQC3/5-65          DE [subseq from] 4-oxalocrotonate tautomerase family enzyme n=36 Tax=root TaxID=1 RepID=C4FQC3_9FIRM
   9 #=GS UniRef90_UPI0013897EF9/1-61   DE [subseq from] 4-oxalocrotonate tautomerase n=1 Tax=Veillonella sp. R32 TaxID=2021312 RepID=UPI0013897EF9
  10 #=GS UniRef90_A0A096AMA4/1-61      DE [subseq from] 4-oxalocrotonate tautomerase n=3 Tax=Veillonella montpellierensis TaxID=187328 RepID=A0A096AMA4_9FIRM
(...)
1609 #=GS UniRef90_UPI001CB4C3EC/1-58   DE [subseq from] tautomerase family protein n=1 Tax=Paraburkholderia tropica TaxID=92647 RepID=UPI001CB4C3EC
1610 #=GS UniRef90_A0A2V6XG00/14-72     DE [subseq from] 4-oxalocrotonate tautomerase n=9 Tax=Bacteria TaxID=2 RepID=A0A2V6XG00_9BACT
(...)
1611
1612 tr|E1L4V1|E1L4V1_9FIRM                MPLIHVELVEG-RT-K-EQKKQLGEAITKATVDIVK-VPAEA--VKVIFVDMKK-DE-F-M-EGGILRSER
1613 UniRef90_X8HJZ3/1-61                  MPLIHVELVEG-RT-K-EQKKQLGEAITKATVEIVK-VPVEA--VKVIFVDMKK-DE-F-M-EGGILRSER
1614 #=GR UniRef90_X8HJZ3/1-61          PP 8**********.**.*.*******************.*****..**********.**.*.*.*******98
1615 UniRef90_C4FQC3/5-65                  MPLIHVELVEG-RT-K-EQKKQLGEAITKATVDIVN-VPADA--VKVIFVDMKK-DD-Y-M-EGGILRSER
1616 #=GR UniRef90_C4FQC3/5-65          PP 8**********.**.*.*******************.*****..**********.**.*.*.*******98
1617 UniRef90_UPI0013897EF9/1-61           MPIIHVELVEG-RT-F-EQKKQLGEAITKAAVDIVK-VPADA--VKVVFVDMKK-DN-Y-M-EGGVMRSEK
1618 #=GR UniRef90_UPI0013897EF9/1-61   PP 8**********.**.*.*******************.*****..**********.**.*.*.*******96
1619 UniRef90_A0A096AMA4/1-61              MPMIHVELVEG-RT-K-EQKKELASAITKATVDIIG-VPVEA--VKVMFVDLKA-DE-F-M-EGGVLRSER
1620 #=GR UniRef90_A0A096AMA4/1-61      PP 8**********.**.*.*******************.*****..**********.**.*.*.*******98
1621 UniRef90_A0A380NKA4/1-61              MPIIHVELVEG-RT-F-EQKKELGEVITKATVDIIK-VPKEA--VKVIFTDMKK-DN-F-M-EAGVMRSEK
1622 #=GR UniRef90_A0A380NKA4/1-61      PP 8**********.**.*.*******************.*****..**********.**.*.*.*******96
(...)
4817 UniRef90_UPI001CB4C3EC/1-58           MPIIRLEMLTG-RT-H-AQKAELAEVLTRETARIAK-CPLSD--VQLVMTEVER-SM-W-S-VGGTLA---
4818 #=GR UniRef90_UPI001CB4C3EC/1-58   PP 8**********.**.*.*******************.*****..**********.**.*.9.999885...
4819 UniRef90_A0A2V6XG00/14-72             MPNITIQWYAG-RT-Q-QQKRELTQAITEAMVKIGK-TTADQ--VHIVFQDVEK-AN-W-G-YNGKLAS--
4820 #=GR UniRef90_A0A2V6XG00/14-72     PP 78999999***.**.*.*******************.*****..**********.98.8.8.7777665..
4821 #=GC PP_cons                          89*********.**.*.*******************.*****..**********.*9.9.9.999988876
4822 #=GC RF                               xxxxxxxxxxx.xx.x.xxxxxxxxxxxxxxxxxxx.xxxxx..xxxxxxxxxx.xx.x.x.xxxxxxxxx
4823 //
```

## `HHR`

The HHR format is generated by HHsearch or HHblits in HH-suite for hidden Markov models.
An HHR format contains multiple pairwise alignments for a single query sequence.

## `PDB`

## `mmCIF`
