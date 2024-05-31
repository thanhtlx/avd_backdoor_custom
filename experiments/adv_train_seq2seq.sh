GPU=${1} \
ATTACK_VERSION=${2} \
N_ALT_ITERS=${3} \
Z_OPTIM=${4} \
U_OPTIM=${7} \
Z_INIT=${5} \
U_PGD_EPOCHS=${8} \
U_ACCUMULATE_BEST_REPLACEMENTS=${9} \
USE_LOSS_SMOOTHING=${11} \
U_RAND_UPDATE_PGD=${10} \
Z_EPSILON=${6} \
U_LEARNING_RATE=${14} \
Z_LEARNING_RATE=${15} \
SMOOTHING_PARAM=${16} \
VOCAB_TO_USE=${18} \
EXACT_MATCHES=${21} \
SHORT_NAME="${12}" \
DATASET=${17} \
TRANSFORMS=${13} \
NUM_REPLACEMENTS=${19} \
  ./scripts/per-epoch-adv-train-seq2seq.sh
