# valnet
Valnet is a microservice to validate any kind of address. It's using internally a neural network with word embedding to validate a address and the service itself is exposed by a django rest api.

## Backend requirements

* [tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone
* [docker](https://www.docker.com/) - Build, Manage and Secure Your Apps Anywhere. Your Way.
* [docker-compose](https://docs.docker.com/compose/) - Compose is a tool for defining and running multi-container Docker applications. 
* [python](https://www.python.org/) - Python is a programming language that lets you work quickly and integrate systems more effectively.
* [tensorflow keras](https://www.tensorflow.org/guide/keras) - Keras is a high-level API to build and train deep learning models. It's used for fast prototyping, advanced research, and production

### Versions requirements

* Docker **>=18.09.2**
* Docker-compose **>=1.21.0**
* python == **3.7.X**
* tensorflow **== 2.0** 

## Example request call

```bash
curl -X POST http://localhost:8000/core/validate -H 'Content-Type: application/json' -d '{ "address": "Slack Technologies Limited 4th Floor, One Park Place Hatch Street Upper Dublin 2, Irlanda" }'
```

Example response payload

```bash 
{"valid":true}
```

## Setup locally

```bash 
python3 -m venv env 
source env/bin/activate 
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Run locally 

```bash
docker compose up 
python3 manage.py runserver 
```

## Train model & run tensorboard

```bash
source env/bin/activate 
python3 train_model.py
tensorboard --logdir logs/search
```

## Format of data

All files for training this model are located in the data directory.
Each line in such a file contains 2 values separated by a comma.

![data_format](./images/data_format.png)

## Docker build & run locally 

To run valnet locally in container and attached to your local network, you need to execute all these statement.

```bash
sudo docker build -t=valnet . 
sudo docker run --network="host" valnet
```



