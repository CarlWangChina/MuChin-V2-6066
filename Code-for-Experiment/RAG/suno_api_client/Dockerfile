ARG CUDA_VERSION=12.1
# ARG from=pytorch/pytorch:2.1.2-cuda${CUDA_VERSION}-cudnn8-runtime
ARG from=pytorch/pytorch:2.1.2-cuda${CUDA_VERSION}-cudnn8-devel
FROM ${from} as base

ARG from
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_OFFLINE=1
# 设置工作目录
WORKDIR /app

# 复制项目文件到镜像中的工作目录
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt &&


# 启动应用程序
CMD ["python", "app.py"]

