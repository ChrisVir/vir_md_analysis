FROM rapidsai/miniforge-cuda

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root

WORKDIR /app

RUN apt-get update \
    && apt-get -y install nano sudo wget zip

RUN mamba install  -c conda-forge \
          openmm openmmtools cudatoolkit openmmforcefields mdtraj \
          python=3.12 numpy pandas sarge typer mdtraj awscli -y \
    && mamba clean -a -y

RUN pip3 install torch  

COPY . vir_openmm_md/

RUN cd vir_openmm_md/ \
    && pip install -e .

#USER app_user