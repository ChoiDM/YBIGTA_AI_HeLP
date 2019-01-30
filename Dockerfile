# Default
FROM continuumio/miniconda3

WORKDIR .

COPY . .

# Install requirements 
RUN conda env create -f ./etc/env.txt
RUN chmod 755 train.sh inference.sh
