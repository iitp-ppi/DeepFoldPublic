import faulthandler
import functools
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import traceback
import types
import unittest
import time
from enum import Enum
from typing import Any, Callable, TypeVar

import torch
import torch.multiprocessing as mp

import deepfold.distributed.core as dist
from deepfold.distributed.tensor.device_mesh import DeviceMesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_WORLD_SIZE: int = 4

NUM_DEVICES: int = DEFAULT_WORLD_SIZE
DEVICE_TYPE: str = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

TIMEOUT_DEFAULT: int = int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "300"))
TIMEOUT_OVERRIDE = {}


def get_timeout(test_id: str) -> int:
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)


T = TypeVar("T")


def init_multigpu_helper(world_size: int, backend: str) -> int:
    """
    Simulate the multi-nodes with multiple GPUs on each node.
    """
    n_gpus = torch.cuda.device_count()
    visible_devices = list(range(n_gpus))

    if backend == "nccl":
        os.environ["NCCL_MAX_NRINGS"] = str(1)

    n_gpus_per_proc = 1
    if world_size > n_gpus:
        n_gpus_per_proc = n_gpus // world_size
    rank_to_gpu = {i: list(visible_devices[i * n_gpus_per_proc : (i + 1) * n_gpus_per_proc]) for i in range(world_size)}

    return rank_to_gpu


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROC_RANK: int = -1

    class ExitCode(Enum):
        ERROR_EXIT = 10
        NO_DEVICE = 20

    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    def join_or_run(self, fn):
        @functools.wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROC_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        fn = getattr(self, methodName)
        setattr(self, methodName, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROC_RANK
        # PID to pipe
        self.pid_to_pipe = {}

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        return self.id().split(".")[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(self.world_size):
            parent_conn, child_conn = mp.Pipe()
            process = proc(
                target=self.__class__._run,
                name=f"Rank {rank}",
                args=(rank, self._current_test_name(), child_conn),
            )
            process.start()
            logger.info(f"Start process {rank} with pid {process.pid}")
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = mp.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int) -> None:
        logger.info(f"Start event listener thread for rank {rank}")
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:
                if parent_pipe.closed:
                    logger.info(f"Pipe closed for rank {rank}, stopping the event listener thread")
                    return

                event = parent_pipe.recv()
                logger.info(f"Received event {event} on rank {rank}")

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process
                    with tempfile.NamedTemporaryFile("r+") as fp:
                        faulthandler.dump_traceback(fp)
                        fp.flush()
                        fp.seek(0)
                        parent_pipe.send(fp.read())
                        logger.info(f"Rank {rank} sent traceback")

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, parent_pipe) -> None:
        self = cls(test_name)
        self.rank = rank
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, paraent_pipe) -> None:
        # Start event listener thread
        signal_recv_pipe, signal_send_pipe = mp.Pipe(duplex=False)
        event_listner_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(paraent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listner_thread.start()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)

        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(f"Process {self.rank} skipping test {test_name} for following reason: {se}")
            sys.exit(MultiProcessTestCase.ExitCode.ERROR_EXIT)
        except Exception as e:
            logger.error(
                f"Caught exception: \n{traceback.format_exc()} exiting"
                f" process {self.rank} with exit code: {MultiProcessTestCase.ExitCode.ERROR_EXIT}"
            )
            paraent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.ExitCode.ERROR_EXIT)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listner_thread is not None
            event_listner_thread.join()
            paraent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(f"Encountered error while trying to get traceback for rank {i}: {e}")

        # Wait for results
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(f"Pipe closed for process {rank}, cannot retrieve traceback")
                        continue

                    traceback = pipe.recv()
                    logger.error(f"Rank {rank} timed out with traceback: \n\n{traceback}")
                else:
                    logger.error(f"Could not retrieve traceback for timed out rank: {rank}")
            except ConnectionError as e:
                logger.error(f"Encountered error while trying to get traceback for rank {rank}: {e}")

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # Check if any subprocess exited with an error
                for i, p in enumerate(self.processes):
                    if p.exitcode == MultiProcessTestCase.ExitCode.ERROR_EXIT:
                        print(f"Rank {i} terminated with exit code {p.exitcode}, terminating remaining processes")
                        active_children = mp.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly
                if all(p.exitcode is not None for p in self.processes):
                    break
                # Check if time out
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    logger.info(f"Time out after {timeout} seconds and kill subprocesses")
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep
                time.sleep(0.5)

            elapsed_time = time.time() - start_time

            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            # Close all pipes
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        # Check no errors thrown in the child processes.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Rank {i} time out after {elapsed_time} seconds")
            self.assertNotEqual(self.ExitCode.ERROR_EXIT, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Check that the return codes of all spawned processes match.
        """
        if not self.processes:
            logger.warning("No subprocesses were spawned")
            return

        first_process = self.processes[0]
        errored_processes = [
            (i, p) for i, p in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.ExitCode.ERROR_EXIT
        ]
        if errored_processes:
            error = ""
            for i, process in enumerate(self.processes):
                error_message = self.pid_to_pipe[process.id].recv()
                error += f"Rank {i} exited with error code {MultiProcessTestCase.ExitCode.ERROR_EXIT} and exception: \n{error_message}\n"
            raise RuntimeError(error)

        # If no process exited uncleanly, check timeouts
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Rank {i} terminated or timed out after {elapsed_time} seconds")
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg=f"Expect rank{i} exit code to match rank 0 exit code of {first_process.exitcode}, but got {p.exitcode}",
            )

        self.assertEqual(
            first_process.exitcode,
            0,
            msg=f"Expect zero exit code but got {first_process.exitcode} for pid: {first_process.pid}",
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0


class DTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(DEVICE_TYPE, list(range(NUM_DEVICES)))

    def init_pg(self, backend: str = "nccl") -> None:
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(MultiProcessTestCase.ExitCode.NO_DEVICE)

        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
        )

        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


TestFunc = Callable[[Any], Any]


def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @functools.wraps(func)
    def wrapper(self: DTensorTestBase, *args, **kwargs) -> None:
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        pg_backend = "nccl" if self.device_type == "cuda" else "gloo"

        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(MultiProcessTestCase.ExitCode.NO_DEVICE)

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(12345)

        self.init_pg(backend=pg_backend)
        func(self, *args, **kwargs)
        self.destroy_pg()

    return wrapper
