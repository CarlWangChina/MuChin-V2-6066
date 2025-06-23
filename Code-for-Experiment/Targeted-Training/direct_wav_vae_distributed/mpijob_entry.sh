if [ "$OMPI_COMM_WORLD_LOCAL_RANK" -eq 0 ]; then
    ray start --num-cpus=16 --num-gpus=0 --head --disable-usage-stats
else
    echo "Wait for head node to start..."
    sleep 10
fi

python scripts/train.py

if [ "$OMPI_COMM_WORLD_LOCAL_RANK" -eq 0 ]; then
    ray stop
fi