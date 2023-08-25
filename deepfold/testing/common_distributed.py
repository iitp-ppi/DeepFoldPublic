import enum
import faulthandler
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from functools import wraps
from typing import NamedTuple

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TIMEOUT_DEFAULT = int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "300"))
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


def get_timeout(test_id) -> int:
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)


class TestSkip(NamedTuple):
    exit_code: int
    message: str


TEST_SKIPS = {
    "backend_unavailable": TestSkip(72, "Skipped because distributed backend is not available."),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "odd_worldsize": TestSkip(87, "Skipped due to odd world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu-1": TestSkip(75, "Need at least 1 CUDA device"),
    "multi-gpu-2": TestSkip(77, "Need at least 2 CUDA devices"),
    "multi-gpu-3": TestSkip(80, "Need at least 3 CUDA devices"),
    "multi-gpu-4": TestSkip(81, "Need at least 4 CUDA devices"),
    "multi-gpu-5": TestSkip(82, "Need at least 5 CUDA devices"),
    "multi-gpu-6": TestSkip(83, "Need at least 6 CUDA devices"),
    "multi-gpu-7": TestSkip(84, "Need at least 7 CUDA devices"),
    "multi-gpu-8": TestSkip(85, "Need at least 8 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no GPU peer access"),
    "generic": TestSkip(86, "Test skipped at subprocess level, look at subprocess log for skip reason"),
    "importerror": TestSkip(88, "Test skipped due to missing import"),
}


DEFAULT_WORLD_SIZE = 4


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK = -1

    TEST_ERROR_EXIT_CODE = 10

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
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
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name=f"processes {rank}",
                args=(rank, self._current_test_name(), self.file_name, child_conn),
            )
            process.start()
            logger.info(f"Started process {rank} with pid {process.pid}")
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(enum.Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info(f"Starting event listener thread for rank {rank}")
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe is ready_pipes:
                if parent_pipe.closed:
                    logger.info(f"Pipe closed for process {rank}, stopping event listener thread")
                    return

                event = parent_pipe.recv()
                logger.info(f"Received event {event} on process {rank}")

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info(f"Process {rank} sent traceback")

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(f"Process {self.rank} skipping test {test_name} for following reason: {se}")
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception as e:
            logger.error(
                f"Caught exception \n{traceback.format_exc()}"
                f" exiting process {self.rank} with exit code:"
                f" {MultiProcessTestCase.TEST_ERROR_EXIT_CODE}"
            )
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)
            assert event_listener_thread is not None
            event_listener_thread.join()
            parent_pipe.close()

    def _get_timeout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(f"Encountered error while trying to get traceback for process {i}: {e}")

        for rank, pipe in pipes:
            try:
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(f"Pipe closed for process {rank}, cannot retrieve traceback")
                        continue

                    traceback = pipe.recv()
                    logger.error(f"Process {rank} timed out with traceback: \n\n{traceback}")
                else:
                    logger.error(f"Could not retrieve traceback for timed out process: {rank}")
            except ConnectionError as e:
                logger.error(f"Encountered error while trying to get raceback for process {rank}: {e}")

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False

        try:
            while True:
                for i, p in enumerate(self.processes):
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        logger.info(
                            f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes"
                        )
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break

                if all(p.exitcode is not None for p in self.processes):
                    break

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timeout_process_traceback()
                    logger.info(f"Timed out after {timeout} seconds and killing subprocesses")

                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling
                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)

        finally:
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        Checks that there is any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Process {i} timed out after {elapsed_time} seconds")
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match.
        Skips tests if they returned a return code indicating a skipping condition.
        """

        if not self.processes:
            logger.warning("No subprocesses were spawned")
            return

        first_process = self.processes[0]

        errored_processes = [
            (i, p) for i, p in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]

        if errored_processes:
            error = ""
            for i, process in errored_processes:
                error_message = self.pid_to_pipe[process.pid].recv()
                error += f"Process {i} exited with error code {MultiProcessTestCase.TEST_ERROR_EXIT_CODE}"
                error += f" and exception:\n{error_message}\n"
            raise RuntimeError(error)

        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Process {i} terminated or timed out after {elapsed_time} seconds")
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg=f"Expect process {i} exit code to match Process 0 exit code of {first_process.exitcode}, but got {p.exitcode}",
            )

        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)

        self.assertEqual(
            first_process.exitcode,
            0,
            msg=f"Expected zero exit code but got {first_process.exitcode} for pid: {first_process.pid}",
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0
