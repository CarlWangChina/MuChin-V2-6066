SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PARENT_DIR/.env.config"

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

ckpt_cdir="/data/shared/Qwen/"$ckpt_name

if [ "$ckpt_name" != "$DEF_BASE_MODEL" ]; then
    ckpt_hdir=$HP_FT_OUTPUT/$ckpt_name
    extra_mount="-v $ckpt_hdir:$ckpt_cdir"
else
    ckpt_cdir=$C_BASE_MODEL
    extra_mount=""
fi

hp_cust_scrs="$PROJECT_ROOT/_tng00/cp_eval/exec_ic"

GPU_DEVICE='"device=0,1,2,3"'

docker run --gpus "$GPU_DEVICE" --rm --name "$cntr_name" \
    -v "$H_BASE_MODEL":"$C_BASE_MODEL" \
    $extra_mount \
    -v "$hp_cust_scrs":"$CP_CUST_SCRS" \
    -v "$HP_CKPT_IFIO":"$CP_CKPT_IFIO" \
    -it "$dimg_name" \
    python "${CP_CUST_SCRS
    -b "$C_BASE_MODEL" -r "$CP_CKPT_IFIO" -f "$file_name" -x "$timeseach"