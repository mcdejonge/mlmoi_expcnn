FROM python:3.9.2



RUN apt update && apt upgrade -y


RUN pip install --upgrade pip

RUN pip install poetry && pip install mypy

RUN poetry config virtualenvs.create false



COPY ./pyproject.toml /root

WORKDIR /root/

RUN poetry install && poetry update



RUN apt install -y zsh



# RUN apt install -y gdebi-core htop

# RUN curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb

# RUN gdebi --non-interactive quarto-linux-amd64.deb && rm quarto-linux-amd64.deb
