import socket
import struct
import matplotlib.pyplot as plt
from time import sleep
import torch


def wait_until_open(
    client: socket.socket, max_attempts: int | None = 10, delay: float = 0.1
):
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


def send_message(client: socket.socket, msg: str):
    print(f"Sending msg {msg}")
    data = f"{msg}\n".encode("utf-8")
    client.sendall(data)


def receive_message(client: socket.socket) -> str:
    buffer = bytearray()
    while True:
        chunk = client.recv(1)
        if not chunk:
            raise ConnectionError("Socket closed while reading message")
        if chunk == b"\n":
            break

        buffer.extend(chunk)

    msg = buffer.decode("utf-8")
    print(f"Recieved message {msg}")
    return msg


def setup_socket(ndim, max_attempts: int = None, fail_on_refuse=False):
    "Create the FUT's connection to SEMBAS"
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if fail_on_refuse:
        client.connect(("127.0.0.1", 2000))
    else:
        wait_until_open(client, max_attempts)

    # send config to remote classifier
    try:
        ndim_packed = struct.pack("!q", ndim)
        client.sendall(ndim_packed)

        msg = receive_message(client)
        if msg != "OK":
            raise Exception(
                f"Invalid number of dimensions? Expected {ndim} and got N={msg[:-1]}"
            )
    except Exception as e:
        client.close()
        raise e

    return client


def receive_request(client: socket.socket, ndim: int) -> torch.Tensor:
    "Receives a request from SEMBAS, i.e. an input to classify."
    data_size = ndim * 8  # ndim * size(f64)
    print(f"Expecting {data_size} bytes")
    data = client.recv(data_size)
    _msg = None
    try:
        _msg = data.decode("utf-8")
    except:
        pass

    print(f"Got request of length {len(data)} bytes. Read: {_msg}")
    return torch.tensor(struct.unpack(f"{ndim}d", data))


def send_response(client: socket.socket, cls: bool):
    "Sends a response to SEMBAS, i.e. the class of the input it requested."
    bool_byte = int(cls).to_bytes(1, byteorder="big")
    client.sendall(bool_byte)


class SembasSession:
    """
    A SEMBAS exploration session.

    ### Phased-Mode
    This mode of session follows a number rules:
    - After setup, server will send message regarding the current SEMBAS phase (str)
    - After each request, server will send message regarding the current SEMBAS phase (str)

    These rules must also be upheld by the server's configuration.

    Phases are non-fixed. You can define your own phases. However, there are three pre-
    defined phases that you can use for convenience:
    - GS = global search
    - SS = surface search
    - BS = boundary search

    This is slower, technically, than the default mode, but provides a means for the
    client to know exactly what phase it is in. That way, intelligent decisions about
    how to execute the system can be made. However, often the bottleneck is not comms
    nor SEMBAS, but rather the FUT; therefore, the slowdown is often a non-issue.

    ### Directed-Mode
    (Is not mutally exclusive with Phased-Mode.)
    Directed-mode allows for intermediate messaging from the client back to SEMBAS to allow
    for control from the client to the server. For example, if the FUT has undergone a change,
    (such as the ingestion of new training data for a model) a message could be sent by the
    client to inform the server to re-acquire the boundary.

    This mode follows a simple pattern:
    Before any request has been made, String messages can be sent.
    For a request to be made, the keyword "CONT" (meaning to continue) is sent, after which
    the request will be made by SEMBAS.
    After the request is completed with a response from the client, the process repeats,
    waiting for the "CONT" message once again.

    The SembasSession object handles the intermediate messaging, so all is neccessary is
    the execution of request, response, and message sending methods.

    """

    PHASE_GLOBAL_SEARCH = "GS"
    PHASE_SURFACE_SEARCH = "SS"
    PHASE_BOUNDARY_EXPL = "BE"
    PHASE_REACQUIRE = "RE"

    MSG_CONTINUE = "CONT"

    STEP_REQ = "REQ"
    STEP_RES = "RES"
    STEP_MSG = "MSG"

    def __init__(
        self,
        bounds: tuple[torch.Tensor, torch.Tensor],
        max_attempts: int = None,
        plot_samples=False,
        dim_names=None,
    ):
        print("Init")
        assert (
            bounds[0].shape == bounds[1].shape and len(bounds[0].shape) == 1
        ), "Incorrect bounds shapes"
        assert not plot_samples or plot_samples and bounds[0].shape[0] <= 2
        self._ndim = bounds[0].shape[0]
        self.socket = setup_socket(ndim=self._ndim, max_attempts=max_attempts)
        self.lo, self.hi = bounds

        self._phase_retrieved = False

        self._phase = receive_message(self.socket)
        self._step = self.STEP_MSG

        self.plot_samples = plot_samples
        if plot_samples:
            assert len(self.lo) <= 3, "Cannot visualize anything above 3 dimensions"
            self._is3d = len(self.lo) == 3
            if self._is3d:
                self._fig = plt.figure()
                self._ax = self._fig.add_subplot(111, projection="3d")
                self._ax.set_zlim(self.lo[2], self.hi[2])
                if dim_names is not None:
                    self._ax.set_zlabel(dim_names[2])
            else:
                self._fig, self._ax = plt.subplots()
                self._ax.set_ylabel("Angle")

            self._ax.set_xlim(self.lo[0], self.hi[0])
            self._ax.set_ylim(self.lo[1], self.hi[1])
            self._ax.set_title("Samples")
            if dim_names is not None:
                self._ax.set_xlabel(dim_names[0])
                self._ax.set_ylabel(dim_names[1])

    @property
    def prev_known_phase(self):
        return self._phase

    @property
    def step(self):
        """
        The communiction follows two steps: request and response.
        - Request: Ready state for receiving a request from SEMBAS
        - Response: Ready state for sending a response to SEMBAS
        """
        return self._step

    @property
    def ndim(self):
        return self._ndim

    def receive_request(self) -> torch.Tensor:
        print("receive_request")
        self._lazily_update_phase()
        self.send_message(self.MSG_CONTINUE)
        assert (
            self._step == self.STEP_REQ
        ), f"Must first send pending response? step: {self._step}"
        self._lazily_update_phase()
        self._step = self.STEP_RES
        print("Receiving request")
        req = self.map_sembas(receive_request(self.socket, self.ndim))
        self._prev_req = req
        print("Request received")
        return req

    def send_response(self, cls: bool):
        print(f"sending_response({cls})")
        assert (
            self._step == self.STEP_RES
        ), "Must first receive request prior to response!"

        if self.plot_samples:
            self._ax.scatter(*self._prev_req, color="red" if cls else "blue")
            plt.pause(0.01)

        send_response(self.socket, cls)
        print("(Response Sent)")
        self._phase_retrieved = False

        self._step = self.STEP_MSG
        print("\n")

    def send_message(self, msg: str):
        print(f"send_message({msg})")
        assert (
            self._step == self.STEP_MSG
        ), f"Must complete request in order to send messages! step: {self._step}"
        if msg == self.MSG_CONTINUE:
            self._step = self.STEP_REQ

        self._lazily_update_phase()
        print("Sending message")
        send_message(self.socket, msg)
        print("(Msg Sent)")
        self._phase_retrieved = False

    def map_sembas(self, x: tuple) -> torch.Tensor:
        x = torch.tensor(x)
        return x * (self.hi - self.lo) + self.lo

    def force_continue(self):
        print("force_continue")
        self._lazily_update_phase()
        self.send_message(self.MSG_CONTINUE)

    def expect_phase(self) -> str:
        "Fetches phase for when in messaging step."
        print("expect_phase")
        assert (
            self._step == self.STEP_MSG
        ), "Cannot expect phase unless in messaging step"

        self._lazily_update_phase()

        return self._phase

    def _lazily_update_phase(self):
        if not self._phase_retrieved:
            print("Receiving phase...")
            self._phase = receive_message(self.socket)
            print(f"Updated phase: {self._phase}")
            self._phase_retrieved = True
