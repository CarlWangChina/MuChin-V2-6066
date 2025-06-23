count=`ps -fe | grep "Code-for-Experiment/Targeted-Training/Lyric-Generation-LLMs/lyric_cleaning/mb_lyric/revprompt_1.py" | grep -v "grep" | wc -l`
echo "now has "$count" process running revpormpt_1"
project_dir="/export/efs/projects/mb-lyric"
mkdir -p $project_dir/log/

if [ $count -lt 1 ]; then
    echo "restarting"
    msg=$(date +%Y-%m-%d_%H:%M:%S)" try to start or restart..."
    cd "$project_dir"
    echo $msg >> log/rp1_status_m${1}t${2}r${3}.log
    .venv/bin/python -u mb_lyric/Code-for-Experiment/Targeted-Training/Lyric-Generation-LLMs/lyric_cleaning/mb_lyric/revprompt_1.py $1 $2 $3
else
    echo "no need to restart"
    msg=$(date +%Y-%m-%d_%H:%M:%S)" state looks good!"
    echo $msg >> $project_dir/log/rp1_status_m${1}t${2}r${3}.log
fi