FROM debian:trixie-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        libreoffice-calc \
        python3 \
        python3-defusedxml \
        python3-numpy \
        python3-openpyxl \
        python3-tqdm \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp/home
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER 65534:65534
WORKDIR /work
