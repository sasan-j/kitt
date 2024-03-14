# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

ARG USERNAME=kitt
ARG USER_UID=1005
ARG USER_GID=100


FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.10

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

RUN echo $USERNAME
RUN echo $USER_UID
RUN echo $USER_GID

# Create the user
RUN groupadd -f --gid 100 users \
    && useradd --uid 1005 --gid 100 -m kitt -s /bin/bash

RUN mkdir /opt/conda \
    && chown -R kitt:users /opt/conda

USER kitt

# Install Miniconda and create main env
ADD --chown=kitt:users --chmod=744 https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda3.sh
RUN ls -laxoh .
RUN /bin/bash /tmp/miniconda3.sh -b -u -p /opt/conda \
    && rm /tmp/miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda \
    python=$PYTHON_VERSION \
    && /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda config --set ssl_verify False \
    && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
    # && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# Install requirements
COPY --chown=kitt:users requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

CMD ["/bin/bash"]
