# https://github.com/Thomj-Dev/SEMBAS/blob/main/examples/exploring_bnn_perf/client/sembas_api.py

import socket
import struct
from time import sleep
import json
import torch


def wait_until_open(client: socket.socket, max_attempts: int | None = 10, delay: float = 0.1):
    "Provides time for the server to begin before giving up."
    success = False
    i = 0
    while not success and (max_attempts is None or i < max_attempts):
        try:
            client.connect(("127.0.0.1", 2000))
            success = True
        except:
            sleep(delay)

        i += 1


def setup_socket(ndim: int, fail_on_refuse=False):
    "Create the FUT's connection to SEMBAS"
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if fail_on_refuse:
        client.connect(("127.0.0.1", 2000))
    else:
        wait_until_open(client)

    # send config to remote classifier
    try:
        ndim_packed = struct.pack("!q", ndim)
        client.sendall(ndim_packed)

        msg = client.recv(1024).decode("utf-8")
        if msg != "OK\n":
            raise Exception("Invalid number of dimensions?")
    except Exception as e:
        client.close()
        raise e

    return client


def receive_request(client: socket.socket, ndim: int) -> torch.Tensor:
    "Receives a request from SEMBAS, i.e. an input to classify."
    data_size = ndim * 8  # ndim * size(f64)
    data = client.recv(data_size)
    return torch.tensor(struct.unpack(f"{ndim}d", data))


def send_response(client: socket.socket, cls: bool):
    "Sends a response to SEMBAS, i.e. the class of the input it requested."
    bool_byte = int(cls).to_bytes(1, byteorder="big")
    client.sendall(bool_byte)

def send_config(client: socket.socket, config: dict):
    config_json = json.dumps(config).encode("utf-8")
    length = len(config_json)
    client.sendall(struct.pack("!I", length))  # 4-byte length prefix
    client.sendall(config_json)

def setup_socket_dict(config: dict, fail_on_refuse=False):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if fail_on_refuse:
        client.connect(("127.0.0.1", 2000))
    else:
        wait_until_open(client)

    try:
        send_config(client, config)  # << inject your full param dict here

        msg = client.recv(1024).decode("utf-8")
        if msg != "OK\n":
            raise Exception("Server rejected config")
    except Exception as e:
        client.close()
        raise e

    return client
