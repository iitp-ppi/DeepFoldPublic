#!/bin/bash -x

#SBATCH --job-name=mmseqs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --time=UNLIMITED
#SBATCH --partition=normal
#SBATCH --output="slurm-%J.out"

[ ! "$#" -eq 2 ] && echo "Please provide base directory and input!" && exit 1
[ ! -d "$1" ] && "Base directory $1 not found!" && exit 1
[ ! -d "$2" ] && "Input $2 not found!" && exit 1

CONDA_HOME="${HOME}/conda"
source "${CONDA_HOME}/etc/profile.d/conda.sh"

DBBASE="/gpfs/database/colabfold"
BASE="$1"
INPUT="$2"
UNIREF_DB="uniref30_2302_db"
# TEMPLATE_DB
METAGENOMIC_DB="colabfold_envdb_202108_db"
MMSEQS="$(which mmseqs)"

FILTER="1" # 0 for False
EXPAND_EVAL="inf"
ALIGN_EVAL="10"
DIFF="3000"
QSC="-20.0"
MAX_ACCEPT="1000000"
PREFILTER_MODE="0"
S="8"
DB_LOAD_MODE="2"
THREADS="64"

if [[ $FILTER -ne 0 ]]; then
    ALIGN_EVAL="10"
    QSC="0.8"
    MAX_ACCEPT="100000"
fi

DB=$UNIREF_DB
if [[ ! -e "${DBBASE}/${DB}.dbtype" ]]; then
    echo "Database $DB does not exist"
fi
if [[ ! -e "${DBBASE}/${DB}.idx" ]] && [[ ! -e "${DBBASE}/${DB}.idx.index" ]]; then
    echo "Search does not use index"
    DB_LOAD_MODE="0"
    DB_SUFFIX_1="_seq"
    DB_SUFFIX_2="_aln"
    DB_SUFFIX_3=""
else
    DB_SUFFIX_1=""
    DB_SUFFIX_2=""
    DB_SUFFIX_3=""
fi

SEARCH_PARAM="--num-iterations 3 --db-load-mode $DB_LOAD_MODE -a -e 0.1 --max-seqs 10000 --prefilter-mode ${PREFILTER_MODE}"
if [[ "S" -ne "" ]]; then
    SEARCH_PARAM="${SEARCH_PARAM} -s $S"
else
    SEARCH_PARAM="${SEARCH_PARAM} --k-score 'seq:96,prof:80'"
fi

FILTER_PARAM="--filter-msa $FILTER --filter-min-enable 1000 --diff $DIFF --qid '0.0,0.2,0.4,0.6,0.8,1.0' --qsc 0 --max-seq-id 0.95"
EXPAND_PARAM="--expansion-mode 0 -e $EXPAND_EVAL --expand-filter-clusters $FILTER --max-seq-id 0.95"

$MMSEQS lndb $INPUT $BASE/qdb
$MMSEQS search $BASE/qdb $DBBASE/$UNIREF_DB $BASE/res $BASE/tmp --threads $THREADS $SEARCH_PARAM || exit 1
$MMSEQS mvdb $BASE/tmp/latest/profile_1 $BASE/prof_res || exit 1
$MMSEQS lndb $BASE/qdb_h $BASE/prof_res_h || exit 1
$MMSEQS expandaln $BASE/qdb $DBBASE/$UNIREF_DB$DB_SUFFIX_1 $BASE/res $DBBASE/$UNIREF_DB$DB_SUFFIX_2 $BASE/res_exp --db-load-mode $DB_LOAD_MODE --threads $THREADS $EXPAND_PARAM || exit 1
$MMSEQS align $BASE/prof_res $DBBASE/$UNIREF_DB$DB_SUFFIX_1 $BASE/res_exp $BASE/res_exp_realign --db-load-mode $DB_LOAD_MODE -e $ALIGN_EVAL --max-accept $MAX_ACCEPT --threads $THREADS --alt-ali 10 -a || exit 1
$MMSEQS filterresult $BASE/qdb $DBBASE/$UNIREF_DB$DB_SUFFIX_1 $BASE/res_exp_realign $BASE/res_exp_realign_filter --db-load-mode $DB_LOAD_MODE --qid 0 --qsc $QSC --diff 0 --threads $THREADS --max-seq-id 1.0 --filter-min-enable 100 || exit 1
$MMSEQS result2msa $BASE/qdb $DBBASE/$UNIREF_DB$DB_SUFFIX_1 $BASE/res_exp_realign_filter $BASE/uniref.a3m --msa-format-mode 6 --db-load-mode $DB_LOAD_MODE --threads $THREADS $FILTER_PARAM || exit 1
$MMSEQS rmdb $BASE/res_exp_realign || exit 1
$MMSEQS rmdb $BASE/res_exp || exit 1
$MMSEQS rmdb $BASE/res || exit 1
$MMSEQS rmdb $BASE/res_exp_realign_filter || exit 1

if [[ ! -z $METAGENOMIC_DB ]]; then
    USE_ENV="1"
fi

if [[ ! -z $USE_ENV ]]; then
    $MMSEQS search $BASE/prof_res $DBBASE/$METAGENOMIC_DB $BASE/res_env $BASE/tmp3 --threads $THREADS $SEARCH_PARAM || exit 1
    $MMSEQS expandaln $BASE/prof_res $DBBASE/$METAGENOMIC_DB$DB_SUFFIX_1 $BASE/res_env $DBBASE/$METAGENOMIC_DB$DB_SUFFIX_2 $BASE/res_env_exp -e $EXPAND_EVAL --expansion-mode 0 --db-load-mode $DB_LOAD_MODE --threads $THREADS || exit 1
    $MMSEQS align $BASE/tmp3/latest/profile_1 $DBBASE/METAGENOMIC_DB$DB_SUFFIX_1 $BASE/res_env_exp $BASE/res_env_exp_realign --db-load-mode $DB_LOAD_MODE -e $ALIGN_EVAL --max-accept $MAX_ACCEPT --threads $THREADS --alt-ali 10 -a || exit 1
    $MMSEQS filterresult $BASE/qdb $DBBASE/$METAGENOMIC_DB$DB_SUFFIX_1 $BASE/res_env_exp_realign $BASE/res_env_exp_realign_filter --db-load-mode $DB_LOAD_MODE --qid 0 --qsc $QSC --diff 0 --max-seq-id 1.0 --threads $THREADS --filter-min-enable 100 || exit 1
    $MMSEQS result2msa $BASE/qdb $DBBASE/$METAGENOMIC_DB$DB_SUFFIX_1 $BASE/res_env_exp_realign_filter $BASE/bfd.mgnify30.metaeuk30.smag30.a3m --msa-format-mode 6 --db-load-mode $DB_LOAD_MODE --threads $THREADS $FILTER_PARAM || exit 1

    $MMSEQS rmdb $BASE/res_env_exp_realign_filter || exit 1
    $MMSEQS rmdb $BASE/res_env_exp_realign || exit 1
    $MMSEQS rmdb $BASE/res_env_exp || exit 1
    $MMSEQS rmdb $BASE/res_env || exit 1

    $MMSEQS mergedbs $BASE/qdb $BASE/final.a3m $BASE/uniref.a3m BASE/bfd.mgnify30.metaeuk30.smag30.a3m || exit 1
    $MMSEQS rmdb $BASE/bfd.mgnify30.metaeuk30.smag30.a3m || exit 1
else
    $MMSEQS mvdb $BASE/uniref.a3m $BASE/final.a3m || exit 1
fi

$MMSEQS unpackdb $BASE/final.a3m $BASE/msas --unpack-name-mode 0 --unpack-suffix ".a3m" || exit 1
$MMSEQS rmdb $BASE/final.a3m || exit 1
$MMSEQS rmdb $BASE/uniref.a3m || exit 1
$MMSEQS rmdb $BASE/res || exit 1

for FILE in $BASE/prof_res*; do
    rm $FILE
done
rm -rf $BASE/tmp
# rm -rf $BASE/tmp2
if [[ ! -z $USE_ENV ]]; then
    rm -rf $BASE/tmp3
fi
