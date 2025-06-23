IMAGE_NAME=qwenllm/qwen:cu117
BASE_MODEL_NAME=Qwen-14B-Chat-Int4
BASE_MODEL_HDIR=/home/carl/model/qwen/${BASE_MODEL_NAME}
BASE_MODEL_CDIR=/data/shared/Qwen/${BASE_MODEL_NAME}
TRNG_DATA_HDIR=/home/carl/mproj/qwen-trn/data/ftip
TRNG_DATA_CDIR=/data/shared/Qwen/data

TRNG_DATA_FNAME=rdy_231226112506-0.json
TRNG_DATA_CFILE=$TRNG_DATA_CDIR/$TRNG_DATA_FNAME
OUTPUT_CDIR=/data/shared/Qwen/output_qwen
OUTPUT_HDIR=/home/carl/mproj/qwen-trn/data/ftop/ckpt2001

GPU_DEVICE=all

mkdir -p ${OUTPUT_HDIR}

docker run --gpus ${GPU_DEVICE} --rm --name qwen \
    --mount type=bind,source=${BASE_MODEL_HDIR},target=${BASE_MODEL_CDIR} \
    --mount type=bind,source=${TRNG_DATA_HDIR},target=${TRNG_DATA_CDIR} \
    --mount type=bind,source=${OUTPUT_HDIR},target=${OUTPUT_CDIR} \
    -v /home/carl/mproj/qwen-trn/dm-scripts:/data/shared/Qwen/c-scripts \
    --shm-size=2gb \
    -itd ${IMAGE_NAME} \
    bash c-scripts/ft2_qlora_ds.sh -m ${BASE_MODEL_CDIR}/ -d ${TRNG_DATA_CFILE}