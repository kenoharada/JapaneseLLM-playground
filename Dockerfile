FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y
RUN pip install setuptools
RUN pip install setuptools
RUN pip install wandb transformers datasets tqdm tiktoken openai sentencepiece
# for JGLUE
RUN pip install bs4 zenhan mecab-python3 pyknp
# for cyber
RUN pip install accelerate