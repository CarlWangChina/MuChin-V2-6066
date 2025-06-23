build:
	docker build -t clap-server-docker .
run:
	docker run \
	--name clap-server-docker \
	--gpus "device=6" \
	-v /export/data/home/feee/.cache/huggingface/hub:/cache \
	-v /export/data/datasets-mp3/cb/:/export/data/datasets-mp3/cb/ \
	--net host \
	-d clap-server-docker