FROM rapidsai/miniforge-cuda

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root

WORKDIR /app

RUN apt-get update \
    && apt-get -y install nano sudo wget zip

RUN mamba install  -c conda-forge -c bioconda \
        python=3.10 numpy scipy pandas biopython abnumber \ 
        ambertools mdtraj typer \
    && mamba clean -a -y

COPY . vir_md_analysis/

RUN cd vir_md_analysis/ \
    && pip install -e .

#USER app_user