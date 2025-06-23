proj_root="/export/efs/projects/mb-lyric"

pys_fname="revannmss_2_inm.py"
py_script="$proj_root/mb_lyric/$pys_fname"

log_fname="r${1}_m${2}-${3}.log"
log_pdirp="$proj_root/log/rs2_inm_ps"
log_rfile="$log_pdirp/$log_fname"

mkdir -p $log_pdirp/

if pgrep -f $pys_fname > /dev/null; then
    msg="$(date +%Y-%m-%d_%H:%M:%S)|WARNNING: $pys_fname is ALREADY running, "
    msg+="hope you know exactly what you're doing."
    echo $msg >> $log_rfile
    echo $msg
else
    cd "$proj_root"
    msg="$(date +%Y-%m-%d_%H:%M:%S): launching '$py_script' by venv..."
    echo $msg >> $log_rfile
    echo $msg

    nohup .venv/bin/python -u $py_script $1 $2 $3 > >(tee -a $log_rfile) 2>&1 &
    py_script_pid=$!
    disown
    echo "$(date +%Y-%m-%d_%H:%M:%S): GREAT! Start Logging:" >> $log_rfile
    echo "$py_script is running in the background with PID: $py_script_pid."

fi