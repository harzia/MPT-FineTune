#!/bin/bash

set -e

if [[ -z "${DATA_PATH}" ]]; then
    echo "Error: The DATA_PATH environment variable is not set."
    exit 1
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
    echo "Error: The OUTPUT_PATH environment variable is not set."
    exit 1
fi

echo "args: $@"

DATADIR="${DATA_PATH}/TopTagging/TopLandscape"
OUTPUT_VOL_DIR="${OUTPUT_PATH}"

# set a comment via `COMMENT`
suffix=${COMMENT}

modelopts="model/example_MPT.py --use-amp --optimizer-option weight_decay 0.01"
lr="1e-4"
extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none"

# "kin", "kinpid", "kinpidplus"
FEATURE_TYPE="kin"

NUM_HEADS=8
NUM_EXPERTS=8
TOP_K=2
CAPACITY_FACTOR=1.5
AUX_LOSS_COEF=0.01
ROUTER_JITTER=0.01

WEAVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-heads) NUM_HEADS="$2"; shift 2 ;;
        --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --capacity-factor) CAPACITY_FACTOR="$2"; shift 2 ;;
        --aux-loss-coef) AUX_LOSS_COEF="$2"; shift 2 ;;
        --router-jitter) ROUTER_JITTER="$2"; shift 2 ;;
        *) WEAVER_ARGS+=("$1"); shift ;;
    esac
done

NETWORK_OPTIONS=""
NETWORK_OPTIONS+=" --network-option num_heads ${NUM_HEADS}"
NETWORK_OPTIONS+=" --network-option moe_num_experts ${NUM_EXPERTS}"
NETWORK_OPTIONS+=" --network-option moe_top_k ${TOP_K}"
NETWORK_OPTIONS+=" --network-option moe_capacity_factor ${CAPACITY_FACTOR}"
NETWORK_OPTIONS+=" --network-option moe_aux_loss_coef ${AUX_LOSS_COEF}"
NETWORK_OPTIONS+=" --network-option moe_router_jitter ${ROUTER_JITTER}"

modelopts+=" --load-model-weights ${PRETRAINED_PATH}"

mkdir -p "${OUTPUT_VOL_DIR}/training" "${OUTPUT_VOL_DIR}/logs" "${OUTPUT_VOL_DIR}/tensorboard" "${OUTPUT_VOL_DIR}/results"

ln -sfn "${OUTPUT_VOL_DIR}/tensorboard" runs

weaver \
    --data-train "${DATADIR}/train_file.parquet" \
    --data-val "${DATADIR}/val_file.parquet" \
    --data-test "${DATADIR}/test_file.parquet" \
    --data-config dataset/TopLandscape/top_kin.yaml --network-config $modelopts \
    --model-prefix ${OUTPUT_VOL_DIR}/training/TopLandscape/$FEATURE_TYPE/MPT/{auto}${suffix}/net $NETWORK_OPTIONS \
    --num-workers 1 --fetch-step 1 --in-memory \
    --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 20 --gpus 0 \
    --start-lr $lr --optimizer ranger --log ${OUTPUT_VOL_DIR}/logs/TopLandscape_${FEATURE_TYPE}_MPT_{auto}${suffix}.log \
    --predict-output ${OUTPUT_VOL_DIR}/results/TopLandscape_${FEATURE_TYPE}_MPT${suffix}/pred.root \
    --tensorboard TopLandscape_${FEATURE_TYPE}_MPT${suffix} \
    ${extraopts} "${WEAVER_ARGS[@]}"