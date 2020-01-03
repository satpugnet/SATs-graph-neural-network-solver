FROM pytorch/manylinux-cuda100

COPY ./src /SATs-graph-neural-network-solver/src
COPY ./scripts /SATs-graph-neural-network-solver/scripts
COPY ./requirements.txt /SATs-graph-neural-network-solver/requirements.txt

WORKDIR /SATs-graph-neural-network-solver

RUN pip3 install -r requirements.txt
RUN ./scripts/setup.sh

CMD [ "python", "./src/main.py" ]
