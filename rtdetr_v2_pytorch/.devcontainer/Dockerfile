FROM nvcr.io/nvidia/pytorch:24.04-py3

ARG DEBIAN_FRONTEND=noninteractive

USER root

WORKDIR /home/data

RUN pip install \
        plotly \
		ujson \
        jupyterlab-language-pack-es-ES \
		dash \
		dash-core-components \
		dash-html-components \
		networkx \
		pycocotools \
		PyYAML \
		tensorboard
    
EXPOSE 8888