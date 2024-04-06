#!/bin/bash

#SBATCH --job-name=msa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal

FASTA_PATH=$1
OUTPUT_DIR=$2
DATABASE_BASE=${DATABASE_BASE:-"/scratch/database/casp16"}
REFORMAT=${REFORMAT:-"$(which reformat.pl)"}

date
echo "IN=$FASTA_PATH"
echo "OUT=$OUTPUT_DIR"
echo "DATABASE_BASE=$DATABASE_BASE"
echo "HOSTNAME=$(hostname)"
echo

NUM_CPUS=16
NUM_CPUS=${SLURM_CPUS_PER_TASK:-$NUM_CPUS}
OMP_NUM_THREADS=$NUM_CPUS

echo "NUM_CPUS=$NUM_CPUS"

function run_jackhmmer() {
    local NAME=$1
    local INPUT_FASTA_PATH=$2
    local DB_PATH=$3
    local STO_PATH="${OUTPUT_DIR}/${NAME}_hits.sto"

    echo "TOOL=JACKHMMER"
    echo "DB=$DB_PATH"

    if [ -s $STO_PATH ]; then
        echo "File exists and is not empty: $STO_PATH"
        echo
        return
    fi

    time jackhmmer \
        --cpu $NUM_CPUS \
        -o /dev/null \
        -A $STO_PATH --noali \
        --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 \
        -N 1 -E 0.0001 \
        $INPUT_FASTA_PATH $DB_PATH >/dev/null 2>&1

    echo
}

function run_hhblits() {
    local NAME=$1
    local INPUT_FASTA_PATH=$2
    local DB_PATH=$3
    local A3M_PATH="${OUTPUT_DIR}/${NAME}_hits.a3m"

    echo "TOOL=HHBLITS"
    echo "DB=$DB_PATH"

    if [ -s $A3M_PATH ]; then
        echo "File exists and is not empty: $A3M_PATH"
        echo
        return
    fi

    time hhblits \
        -cpu $NUM_CPUS \
        -i $INPUT_FASTA_PATH \
        -oa3m $A3M_PATH \
        -o /dev/null \
        -n 3 -e 0.0001 \
        -realign_max 100000 \
        -maxfilt 100000 \
        -min_prefilter_hits 1000 \
        -maxseq 1000000 \
        -cpu $NUM_CPUS \
        -d $DB_PATH >/dev/null 2>&1

    echo
}

function run_hmmsearch() {
    local INPUT_STO_PATH=$1
    local DB_PATH=$2
    local HMM_PATH="${OUTPUT_DIR}/output.hmm"
    local STO_PATH="${OUTPUT_DIR}/pdb_hits.sto"

    echo "TOOL=HMMSEARCH"
    echo "QUERY=$INPUT_STO_PATH"
    echo "DB=$DB_PATH"

    if [ -s $STO_PATH ]; then
        echo "File exists and is not empty: $STO_PATH"
        echo
        return
    fi

    time hmmbuild \
        --hand --amino \
        $HMM_PATH \
        $INPUT_STO_PATH >/dev/null 2>&1

    time hmmsearch \
        --noali --cpu $NUM_CPUS \
        --F1 0.1 --F2 0.1 --F3 0.1 \
        --incE 100 -E 100 --domE 100 --incdomE 100 \
        -A $STO_PATH \
        $HMM_PATH \
        $DB_PATH >/dev/null 2>&1

    rm -f $HMM_PATH

    echo
}

function run_hhsearch() {
    local INPUT_STO_PATH=$1
    local DB_PATH=$2
    local HHR_PATH="${OUTPUT_DIR}/pdb_hits.hhr"
    local A3M_PATH="${OUTPUT_DIR}/query.a3m"

    echo "TOOL=HHSEARCH"
    echo "DB=$DB_PATH"

    if [ -s $HHR_PATH ]; then
        echo "File exists and is not empty: $HHR_PATH"
        echo
        return
    fi

    $REFORMAT sto a3m $INPUT_STO_PATH $A3M_PATH

    time hhsearch \
        -i $A3M_PATH \
        -o $HHR_PATH \
        -maxseq 1000000 \
        -cpu $NUM_CPUS \
        -d $DB_PATH >/dev/null 2>&1

    rm -f $A3M_PATH

    echo
}

# Check arguments
if [ $# -ne 1 ]; then
    echo "Illegal number of parameters: $# != 1"
fi

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

# Check input
if ! [ -f "$FASTA_PATH" ]; then
    echo "Cannot find input: '$FASTA_PATH'"
    exit 1
fi

echo

# Prepare output directory
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# UniRef30
run_hhblits "uniref30" $FASTA_PATH "${DATABASE_BASE}/uniref30/UniRef30_2021_03/UniRef30_2021_03"

# UniRef90
run_jackhmmer "uniref90" $FASTA_PATH "${DATABASE_BASE}/uniref90/uniref90.fasta"

# UniProt
run_jackhmmer "uniprot" $FASTA_PATH "${DATABASE_BASE}/uniprot/uniprot.fasta"

# MGnify
run_jackhmmer "mgnify" $FASTA_PATH "${DATABASE_BASE}/mgnify/mgy_clusters_2022_05.fa"

# BFD
run_hhblits "bfd" $FASTA_PATH "${DATABASE_BASE}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

# HMM
run_hmmsearch "${OUTPUT_DIR}/uniref90_hits.sto" "${DATABASE_BASE}/pdb/pdb_seqres.txt"

# PDB70
run_hhsearch "${OUTPUT_DIR}/uniref90_hits.sto" "${DATABASE_BASE}/pdb70/pdb70"

exit 0
