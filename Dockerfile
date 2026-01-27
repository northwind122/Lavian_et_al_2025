FROM mambaorg/micromamba:1.5.8

ARG ENV_FILE=environment.docker.yml
COPY ${ENV_FILE} /tmp/environment.yml
RUN micromamba create -y -n vis_nav -f /tmp/environment.yml \
    && micromamba clean -a -y

ENV MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /work
COPY . /work

RUN micromamba run -n vis_nav pip install -r /work/requirements.txt \
    && micromamba run -n vis_nav pip install -e .

COPY docker/start.sh /usr/local/bin/start.sh

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/start.sh"]
