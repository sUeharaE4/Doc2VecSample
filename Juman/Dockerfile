FROM python:3.7
USER root
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN apt-get install -y libboost-all-dev
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

### create and into workdir
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
# ADD . /code/

### install JUMAN++
WORKDIR /code
RUN wget -q http://lotus.kuee.kyoto-u.ac.jp/nl-resource/jumanpp/jumanpp-1.01.tar.xz \
    && tar Jxf jumanpp-1.01.tar.xz
WORKDIR /code/jumanpp-1.01
RUN ./configure \
    && make -s \
    && make -s install \
    && rm /code/jumanpp-1.01.tar.xz

### install JUMAN
WORKDIR /code
RUN wget -q http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/juman/juman-7.01.tar.bz2 \
    && tar xf juman-7.01.tar.bz2
WORKDIR /code/juman-7.01
RUN ./configure \
    && make -s \
    && make -s install \
    && rm /code/juman-7.01.tar.bz2

### install KNP
### if wget return 404, please check latest version of KNP and change later commands
WORKDIR /code
RUN wget -q http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/knp/knp-4.19.tar.bz2 \
    && tar xf knp-4.19.tar.bz2
WORKDIR /code/knp-4.19
RUN ./configure \
    && make -s \
    && make -s install \
    && rm /code/knp-4.19.tar.bz2

### install PyKNP
WORKDIR /code
RUN wget -q http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/knp/pyknp-0.3.tar.gz \
    && tar xfz pyknp-0.3.tar.gz
WORKDIR /code/pyknp-0.3
RUN python setup.py install \
    && rm /code/pyknp-0.3.tar.gz

WORKDIR /code
RUN mkdir pythonScripts
WORKDIR /code/pythonScripts
COPY ./pythonScripts/sample.py .
RUN wget -q http://www.rondhuit.com/download/ldcc-20140209.tar.gz
RUN tar xfz ldcc-20140209.tar.gz
RUN rm ldcc-20140209.tar.gz
RUN python sample.py


WORKDIR /code
RUN django-admin startproject mysite .
RUN python manage.py startapp doc2vec_sample