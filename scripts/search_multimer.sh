#/bin/bash -ex

FASTA_PATH=$1
OUTPUT_DIR=$2
DB_BASE=${DATABASE_BASE:-"/scratch/database/casp16"}

NUM_CPUS=${NUM_CPUS:-SLURM_CPUS_PER_TASK}
OMP_NUM_THREADS=$NUM_CPUS

function run_jackhmmer {
    local NAME=$1
    local INPUT_FASTA_PATH=$2
    local DB_PATH=$3
    local STO_PATH="${OUTPUT_DIR}/${NAME}_hits.sto"

    jackhmmer \
        -o /dev/null \
        -A $STO_PATH --noali \
        --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 \
        -N 1 -E 0.0001 \
        --cpu $NUM_CPUS \
        $INPUT_FASTA_PATH $DB_PATH 2>&1 >"$NAME.log"
}

function run_hhblits {
    local NAME=$1
    local INPUT_FASTA_PATH=$2
    local DB_PATH=$3
    local A3M_PATH="${OUTPUT_DIR}/${NAME}_hits.a3m"

    hhblits \
        -i $INPUT_FASTA_PATH \
        -cpu $NUM_CPUS \
        -oa3m $A3M_PATH \
        -o /dev/null \
        -n 3 -e 0.0001 \
        -realign_max 100000 \
        -maxfilt 100000 \
        -min_prefilter_hits 1000 \
        -maxseq 1000000 -d \
        $DB_PATH 2>&1 >"$NAME.log"
}

if ! command -v hhblits &>/dev/null; then
    echo "Cannot find HHBlits..."
    exit 1
else
    echo "HHBLITS_PATH='$(which hhblits)'"
fi

if ! command -v jackhmmer &>/dev/null; then
    echo "Cannot find JackHMMER..."
    exit 1
else
    echo "JACKHMMER_PATH='$(which jackhmmer)'"
fi

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters: $# != 2"
fi

# Check input
if ! [ -f "$FASTA_PATH" ]; then
    echo "Cannot find input: '$FASTA_PATH'"
    exit 1
fi

# Prepare output directory
mkdir -p $OUTPUT_DIR

# UniRef90
run_jackhmmer "uniref90" $INPUT_FASTA_PATH "${DB_BASE}/uniref90/uniref90.fasta"

# MGnify
run_jackhmmer "mgnify" $INPUT_FASTA_PATH "${DB_BASE}/mgnify/mgy_clusters_2022_05.fa"

# UniProt
run_jackhmmer "uniprot" $INPUT_FASTA_PATH "${DB_BASE}/uniprot/uniprot.fasta"

# BFD
run_hhblits "bfd" $INPUT_FASTA_PATH "${DB_BASE}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

# UniRef30
run_hhblits "uniref30" $INPUT_FASTA_PATH "${DB_BASE}/uniref30/UniRef30_2021_03/UniRef30_2021_03"

# PDB70
# run_hhblits "pdb70" $INPUT_FASTA_PATH "${DB_BASE}/pdb70/pdb70"

# UniClust30
# run_jachmmer "uniclust30" $INPUT_FASTA_PATH "${DB_BASE/uniclust/...}"

exit 0
