FROM python:3.9-slim

# ENV GRANT_SUDO=yes
# USER root

# aptの取得先を日本に変更(ビルドスピード高速化のため) https://c-a-p-engineer.github.io/tech/2022/09/25/docker-speedup/#apt-のパッケージ取得先を日本のサーバーに変更
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y \
    cron \
    tzdata \
    vim \
    gcc \
    make \
    git \
    wget \
    # opencvで必要
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./docker/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -U pip

WORKDIR /home/rl_practice

RUN useradd -m -d /home/rl_practice -s /bin/bash torikky

RUN echo 'LANG="ja_JP.UTF-8"' >> /etc/default/locale
RUN echo 'LANGUAGE="ja_JP:ja"' >> /etc/default/locale

# USER torikky