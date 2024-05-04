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
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Union

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def skip_if_no_gpu(func):
    """Skips if the world size exceeds the number of GPUs."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit()
        world_size = int(os.getenv("WORLD_SIZE"))
        if torch.cuda.device_count() < world_size:
            sys.exit()

        return func(*args, **kwargs)

    return wrapper


def skip_if_lt_x_gpu(n: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= n:
                return func(*args, **kwargs)
            sys.exit()

        return wrapper

    return decorator


def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset TORCH_NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of TORCH_NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both TORCH_NCCL_BLOCKING_WAIT and
    TORCH_NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save and unset TORCH_NCCL_ASYNC_ERROR_HANDLING
        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
            del os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # TORCH_NCCL_ASYNC_ERROR_HANDLING was unset
            cached_nccl_async_error_handling = None

        # Save val of TORCH_NCCL_BLOCKING_WAIT and set it
        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ["TORCH_NCCL_BLOCKING_WAIT"]
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            # restore old values
            if cached_nccl_async_error_handling is not None:
                os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = cached_nccl_async_error_handling

            if cached_nccl_blocking_wait is not None:
                os.environ["TORCH_NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

    return wrapper


DEFAULT_WORLD_SIZE: int = 4
DEFAULT_TIME_OUT: int = 300


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK: int = -1
    TEST_ERROR_EXIT_CODE: int = 10

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

    # The main process spawns N subprocesses that run the test
    def __init__(self, meethod_name: str = "runTest") -> None:
        super().__init__(meethod_name)
        fn = getattr(self, meethod_name)
        setattr(self, meethod_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.processes: List[multiprocessing.Process] = []
        self.rank: int = self.MAIN_PROCESS_RANK
        self.file_name: str = tempfile.NamedTemporaryFile(delete=False).name
        self.pid_to_pipe: Dict[int, multiprocessing.connection.Connection] = {}

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        return self.id().split(".")[-1]

    def _start_processes(self, proc: Any) -> None:
        self.processes = []
        for rank in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name=f"process {rank}",
                args=(rank, self._current_test_name(), self.file_name, child_conn),
            )
            process.start()
            logger.info(f"Started process {rank} with pid {process.pid}")
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info(f"Starting event listener thread for rank {rank}")
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])
            if parent_pipe in ready_pipes:
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
        signal_recv_pipe, singal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
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
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        except Exception as e:
            logger.error(f"Caught exception: \n{traceback.format_exc()} exiting process {self.rank}")
            # Send error to parent process
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if singal_send_pipe is not None:
                singal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done
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

        # Wait for results
        for rank, pipe in pipes:
            try:
                # Wait
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(f"Pipe closed for process {rank}, cannot retrieve traceback")
                        continue

                    traceback = pipe.recv()
                    logger.error(f"Process {rank} timed out with traceback: \n\n{traceback}")
                else:
                    logger.error("Could not retrieve traceback for timed out process: {rank}")
            except ConnectionError as e:
                logger.error(f"Encountered error while trying to get traceback for process {rank}: {e}")

    def _join_processes(self, fn) -> None:
        timeout = DEFAULT_TIME_OUT
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # Check if any subprocess exited with an error early
                for i, p in enumerate(self.processes):
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        logger.error(f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes")
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid exitcode
                if all(p.exitcode is not None for p in self.processes):
                    break
                # Check if we should time out the test
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timeout_process_traceback()
                    logger.error(f"Time out after {timeout} seconds and killing subprocesses")
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid busy polling
                time.sleep(0.1)

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
        """Checks that we didn't have any errors thrown in the child processes."""

        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Process {i} timed out after {elapsed_time} seconds")
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        # If no processes are spawned, there is nothing to check
        if not self.processes:
            logger.warning("Note: no subprocesses were spawned, test was likely skipped.")
            return

        first_process = self.processes[0]

        # TODO: Enhance
        errored_processes = [(i, p) for i, p in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE]
        if errored_processes:
            error = ""
            for i, proc in errored_processes:
                error_msg = self.pid_to_pipe[proc.pid].recv()
                error += f"Process {i} exited with error code {MultiProcessTestCase.TEST_ERROR_EXIT_CODE} and exception:\n{error_msg}\n"
            raise RuntimeError(error)

        # If no process exited uncleanly, check timeouts and then exit each process cleanly
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f"Process {i} terminated or timed out after {elapsed_time} seconds")
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg=f"Expect process {i} exit code to match process 0 exit code of {first_process.exitcode} but got {p.exitcode}",
            )
        self.assertEqual(
            first_process.exitcode,
            0,
            msg=f"Expected zero exit code but got {first_process.exitcode} for pid: {first_process.pid}",
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0
