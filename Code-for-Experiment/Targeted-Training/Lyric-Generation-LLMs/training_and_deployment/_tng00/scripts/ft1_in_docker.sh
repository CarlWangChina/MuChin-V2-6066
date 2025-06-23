SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PARENT_DIR/.env.config"

TRNG_DATA_FN="rdy_jb_231226112506-0.json"
CF_TRNG_DATA="$CP_TRNG_DATA/$TRNG_DATA_FN"
HP_CKPT_OP_D="$HP_FT_OUTPUT/ckpt1002"

GPU_DEVICE=all

mkdir -p ${HP_CKPT_OP_D}

docker run --gpus ${GPU_DEVICE} --rm --name qwen-ft1 \
    --mount type=bind,source=${H_BASE_MODEL},target=${C_BASE_MODEL} \
    --mount type=bind,source=${HP_TRNG_DATA},target=${CP_TRNG_DATA} \
    --mount type=bind,source=${HP_CKPT_OP_D},target=${CP_FT_OUTPUT} \
    --shm-size=2gb -itd ${DEF_DOCKER_IMG} \
    bash finetune/finetune_qlora_ds.sh -m ${C_BASE_MODEL}/ -d ${CF_TRNG_DATA}