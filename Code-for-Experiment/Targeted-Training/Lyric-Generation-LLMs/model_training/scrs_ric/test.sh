export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

echo "DIR="$DIR

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

NNODES=${NNODES:-1}

NODE_RANK=${NODE_RANK:-0}

MASTER_ADDR=${MASTER_ADDR:-localhost}

MASTER_PORT=${MASTER_PORT:-6001}

MODEL="Qwen/Qwen-7B-Chat-Int4"
DATA="path_to_data"

function usage() {
    echo '
Usage: bash finetune/finetune_qlora_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "DISTRIBUTED_ARGS="$DISTRIBUTED_ARGS