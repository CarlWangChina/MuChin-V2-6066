. /home/carl/mproj/qwen-trn/.env.config

dimg_name=$DEF_DOCKER_IMG
cntr_name="qwen-ckpt-lrc-infer"
ckpt_name=$DEF_BASE_MODEL
timeseach=3

file_name=""

function usage() {
    echo 'Usage: read the source code.'
}

while [ "$
    case $1 in
        -i | --image-name )
            dimg_name=$2
            shift 2
            ;;
        -n | --cntnr-name )
            cntr_name=$2
            shift 2
            ;;
        -c | --checkpoint )
            ckpt_name=$2
            shift 2
            ;;
        -x | --times-each )
            timeseach=$2
            shift 2
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            if [ -z "$file_name" ]; then
                file_name=$1
            else
                echo "Ignoring extra argument: $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$file_name" ]; then
    echo "Error: Required positional arg 'file_name' is missing."
    exit 1
fi

if [[ "$timeseach" =~ ^[0-9]+$ ]]; then
    timeseach=$(( (timeseach < 1) ? 1 : (timeseach > 20) ? 20 : timeseach ))
else
    echo "Error: -x|--times-each must be a integer number."
    exit 1
fi

bmdl_hdir=$MODEL_BP_HMC/$DEF_BASE_MODEL
bmdl_cdir=$MODEL_MP_ONC/$DEF_BASE_MODEL

ckpt_cdir=$MODEL_MP_ONC/$ckpt_name

if [ "$ckpt_name" != "$DEF_BASE_MODEL" ]; then
    ckpt_hdir=$HP_FT_OUTPUT/$ckpt_name
    extra_mount="--mount type=bind,source=\"$ckpt_hdir\",target=\"$ckpt_cdir\""
else
    extra_mount=""
fi

docker run --gpus all --rm --name "$cntr_name" \
    --mount type=bind,source="$bmdl_hdir",target="$bmdl_cdir" \
    $extra_mount \
    -v "$HP_CUST_SCRS":"$CP_CUST_SCRS" \
    -v "$HP_CKPT_IFIO":"$CP_CKPT_IFIO" \
    -it "$dimg_name" \
    python "${CP_CUST_SCRS
    -b "$bmdl_cdir" -r "$CP_CKPT_IFIO" -f "$file_name" -x "$timeseach"