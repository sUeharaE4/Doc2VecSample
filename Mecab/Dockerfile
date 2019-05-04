FROM python:3.7
USER root
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get -y install locales \
    vim less \
    libboost-all-dev \
    mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 \
    sudo
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

### create and into workdir
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
# ADD . /code/

### install dictionally NEologd
WORKDIR /code
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /code/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -a -y
RUN mecab -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/


WORKDIR /code
RUN mkdir pythonScripts
WORKDIR /code/pythonScripts
RUN wget -q http://www.rondhuit.com/download/ldcc-20140209.tar.gz
RUN tar xfz ldcc-20140209.tar.gz
RUN rm ldcc-20140209.tar.gz
ADD ./pythonScripts/ /code/pythonScripts/
WORKDIR /code/pythonScripts/model
RUN unzip -q lyrics2vec.zip && rm lyrics2vec.zip
WORKDIR /code/pythonScripts
RUN python sample.py

WORKDIR /code
# RUN django-admin startproject mysite .
# RUN python manage.py startapp doc2vec_sample
