# Default
FROM continuumio/miniconda3

WORKDIR .

COPY . .

# Install requirements 
RUN apt-get update && apt-get install libglib2.0-0
RUN conda env create -f ./etc/env.txt
RUN chmod 755 train.sh inference.sh
