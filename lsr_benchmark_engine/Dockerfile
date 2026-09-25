# linux/amd64 only: TIRA does not run other platforms.
# Requesting a GPU is a separate step from submitting the image: choose an
# a100-resources-gpu or h100-resources-gpu tier when starting the run.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y python3 python3-pip git \
    && apt-get clean \
    && pip3 cache purge

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm -Rf /tmp/requirements.txt \
    && pip3 cache purge

# GPUSparse kernels
ADD gpusparse_src /gpusparse
ENV GPUSPARSE_SRC=/gpusparse

ADD gpusparse_retrieval.py /index-and-retrieve.py
