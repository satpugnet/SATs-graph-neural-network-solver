FROM nvcr.io/nvidia/pytorch:19.12-py3

COPY ./src /SATs-graph-neural-network-solver/src
COPY ./scripts /SATs-graph-neural-network-solver/scripts
COPY ./requirements.txt /SATs-graph-neural-network-solver/requirements.txt

#RUN pip3 install torch
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /SATs-graph-neural-network-solver

RUN pip install --no-cache-dir -r requirements.txt
RUN ./scripts/setup.sh
RUN apt-get update
RUN apt-get install vim -y
