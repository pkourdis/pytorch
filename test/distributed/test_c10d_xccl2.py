# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree xccl2 backend.

import gc
import os
import subprocess
import sys
import time
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    requires_xccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_FBCODE,
    IS_SANDCASTLE,
    run_tests,
    TEST_XPU,
    TestCase,
)


class ProcessGroupXCCL2Test(MultiProcContinuousTest):
    hw_classification = HardwareClassification.XPU

    @classmethod
    def backend_str(cls) -> str:
        return "xccl2"

    @classmethod
    def device_type(cls) -> str:
        return "xpu"

    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.xpu.set_device(self.rank)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.xpu.synchronize()
        time.sleep(2)
        dist.barrier()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast(self) -> None:
        t = torch.full((10,), float(self.rank + 1), device=self.device)
        dist.broadcast(t, src=0)
        self.assertEqual(t, torch.full((10,), 1.0, device=self.device))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_allreduce(self) -> None:
        t = torch.full((10,), float(self.rank + 1), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(1, self.world_size + 1)))
        self.assertEqual(t, torch.full((10,), expected, device=self.device))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        out = [torch.empty((4,), device=self.device) for _ in range(self.world_size)]
        dist.all_gather(out, t)
        for r in range(self.world_size):
            self.assertEqual(out[r], torch.full((4,), float(r), device=self.device))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        out = torch.empty((4 * self.world_size,), device=self.device)
        dist.all_gather_into_tensor(out, t)
        expected = torch.cat(
            [
                torch.full((4,), float(r), device=self.device)
                for r in range(self.world_size)
            ]
        )
        self.assertEqual(out, expected)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor(self) -> None:
        inp = torch.arange(4 * self.world_size, dtype=torch.float32, device=self.device)
        out = torch.empty((4,), device=self.device)
        dist.reduce_scatter_tensor(out, inp)
        base = torch.arange(
            4 * self.world_size, dtype=torch.float32, device=self.device
        )
        chunk = base[self.rank * 4 : (self.rank + 1) * 4]
        self.assertEqual(out, chunk * self.world_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self) -> None:
        inp = torch.full((self.world_size,), float(self.rank), device=self.device)
        out = torch.empty((self.world_size,), device=self.device)
        dist.all_to_all_single(out, inp)
        # Rank r receives, at position s, the value sent by rank s -> float(s).
        self.assertEqual(
            out,
            torch.arange(self.world_size, dtype=torch.float32, device=self.device),
        )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_barrier(self) -> None:
        # Bind this rank's device first (barrier has no tensor to infer it from).
        torch.xpu.set_device(self.rank)
        dist.barrier()  # should not hang or raise

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_send_recv(self) -> None:
        # Ring: rank r sends to r+1, receives from r-1. With an odd world size two
        # neighbours both send first, and this hangs on xccl2, whose blocking
        # send/recv all share one oneCCL communicator; the legacy xccl backend,
        # which uses a 2-rank communicator per peer pair, completes it.
        if self.world_size % 2 != 0:
            self.skipTest("xccl2 blocking send/recv hangs on an odd-sized ring")
        send_t = torch.full((8,), float(self.rank), device=self.device)
        recv_t = torch.empty((8,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        self.assertEqual(recv_t, torch.full((8,), float(prev), device=self.device))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_batch_isend_irecv(self) -> None:
        # Mixed batch on a single PG (the PP 1F1B pattern that needs coalescing).
        send_t = torch.full((1,), float(self.rank), device=self.device)
        recv_t = torch.empty((1,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        ops = [
            dist.P2POp(dist.isend, send_t, nxt),
            dist.P2POp(dist.irecv, recv_t, prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        self.assertEqual(recv_t, torch.full((1,), float(prev), device=self.device))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_options_not_shared_with_legacy_backend(self) -> None:
        # nccl2 reuses core's ProcessGroupNCCL.Options, but the legacy XCCL
        # backend lives in torch-xpu-ops and carries an unrelated Options, so
        # xccl2 binds its own. Guard against the two being conflated.
        self.assertIsNot(dist.ProcessGroupXCCL2.Options, dist.ProcessGroupXCCL.Options)
        opts = dist.ProcessGroupXCCL2.Options()
        self.assertFalse(opts.is_high_priority_stream)
        self.assertEqual(opts.hints, {})
        opts.is_high_priority_stream = True
        opts.hints = {"key": "value"}
        self.assertTrue(opts.is_high_priority_stream)
        self.assertEqual(opts.hints, {"key": "value"})

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_identity(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertIsInstance(backend, dist.ProcessGroupXCCL2)
        self.assertEqual(dist.get_backend(), "xccl2")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_version_reported_after_bootstrap(self) -> None:
        # backend_version_ is filled in by initXcclResources, so force the lazy
        # bootstrap before reading it.
        dist.all_reduce(torch.ones(1, device=self.device))
        backend = dist.get_backend_impl(device=self.device)
        self.assertTrue(backend.backend_version.isdigit())

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_work_carries_group_sequence_number(self) -> None:
        pg = dist.distributed_c10d._get_default_group()
        t = torch.ones(4, device=self.device)
        seqs = []
        for _ in range(2):
            work = dist.all_reduce(t, async_op=True)
            seqs.append(work._get_sequence_number())
            self.assertEqual(seqs[-1], pg._get_sequence_number_for_group())
            work.wait()
        self.assertEqual(seqs[1], seqs[0] + 1)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_coalescing_window_single_coalesced_op(self) -> None:
        pg = dist.distributed_c10d._get_default_group()
        tensors = [torch.ones(4, device=self.device) for _ in range(2)]
        pg._start_coalescing(self.device)
        dist.all_reduce_coalesced(tensors, async_op=True)
        work = pg._end_coalescing(self.device)
        # The window returns the op's coalesced work, not a combined one.
        self.assertEqual(work.result(), tensors)
        work.wait()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_unsupported_low_precision_dtypes(self) -> None:
        # Unlike nccl2, xccl2 maps no Float8 or Float4 type onto oneCCL.
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = torch.ones(4, device=self.device).to(dtype)
            with self.assertRaisesRegex(RuntimeError, "Unsupported tensor data type"):
                dist.all_reduce(tensor)

        tensor = torch.empty(4, dtype=torch.float4_e2m1fn_x2, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Unsupported tensor data type"):
            dist.all_reduce(tensor)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather_zero_sized(self) -> None:
        # All zero-sized outputs
        input_empty = torch.empty(0, device=self.device)
        outputs_empty = [
            torch.empty(0, device=self.device) for _ in range(self.world_size)
        ]
        dist.all_gather(outputs_empty, input_empty)
        for out in outputs_empty:
            self.assertEqual(out.numel(), 0)

        # Uneven gather where the first rank(s) have zero-sized tensors.
        # Zero-sized outputs all have a null data_ptr and must not be
        # misclassified as aliased.
        sizes = [0, 4] if self.world_size == 2 else [0, 0] + [4] * (self.world_size - 2)
        input_tensor = (
            torch.empty(0, device=self.device)
            if sizes[self.rank] == 0
            else torch.full((sizes[self.rank],), float(self.rank), device=self.device)
        )
        outputs = [torch.empty(s, device=self.device) for s in sizes]
        dist.all_gather(outputs, input_tensor)
        for r, s in enumerate(sizes):
            if s == 0:
                self.assertEqual(outputs[r].numel(), 0)
            else:
                self.assertEqual(
                    outputs[r], torch.full((s,), float(r), device=self.device)
                )


class ProcessGroupXCCL2WorkLifetimeTest(MultiProcessTestCase):
    hw_classification = HardwareClassification.XPU

    @property
    def world_size(self) -> int:
        return 2

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_work_outlives_process_group(self) -> None:
        device = torch.device("xpu", self.rank)
        torch.xpu.set_device(device)
        dist.init_process_group(
            "xccl2",
            world_size=self.world_size,
            rank=self.rank,
            store=dist.FileStore(self.file_name, self.world_size),
            device_id=device,
        )
        work = dist.all_reduce(torch.ones(4, device=device), async_op=True)
        work.wait()

        dist.destroy_process_group()
        del work
        gc.collect()


class _ProcessGroupXCCL2OptionsTest(MultiProcContinuousTest):
    """Base for groups initialized with backend specific options."""

    @classmethod
    def backend_str(cls) -> str:
        return "xccl2"

    @classmethod
    def device_type(cls) -> str:
        return "xpu"

    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.xpu.set_device(self.rank)

    def _check_all_reduce(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(self.world_size)))
        self.assertEqual(t, torch.full((4,), expected, device=self.device))


class ProcessGroupXCCL2HintsTest(_ProcessGroupXCCL2OptionsTest):
    hw_classification = HardwareClassification.XPU

    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupXCCL2.Options()
        opts.is_high_priority_stream = True
        opts.hints = {"is_high_priority_stream": "1"}
        return opts

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_options(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertTrue(backend.options.is_high_priority_stream)
        self.assertEqual(backend.options.hints, {"is_high_priority_stream": "1"})
        self._check_all_reduce()


class ProcessGroupXCCL2EagerNewGroupTest(_ProcessGroupXCCL2OptionsTest):
    hw_classification = HardwareClassification.XPU

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        store = dist.FileStore(rdvz_file, world_size)
        dist.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
            device_id=torch.device("xpu", rank),
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_new_group_with_nonmembers(self) -> None:
        ranks = [0]
        group = dist.new_group(ranks=ranks)
        if self.rank in ranks:
            tensor = torch.tensor([self.rank + 1], device=self.device)
            dist.all_reduce(tensor, group=group)
            self.assertEqual(tensor, torch.ones_like(tensor))
            dist.destroy_process_group(group)
        else:
            self.assertEqual(group, dist.GroupMember.NON_GROUP_MEMBER)

        store = dist.distributed_c10d._get_default_store()
        store.set(f"eager_new_group/{self.rank}", b"1")
        store.wait([f"eager_new_group/{rank}" for rank in range(self.world_size)])
        self._check_all_reduce()


class ProcessGroupXCCL2DestroyTest(_ProcessGroupXCCL2OptionsTest):
    hw_classification = HardwareClassification.XPU

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_destroy_process_group_returns(self) -> None:
        # Tear down a throwaway subgroup so the class-wide default group
        # survives to prove the process is still healthy.
        pg = dist.new_group(
            backend="xccl2",
            use_local_synchronization=True,
            timeout=timedelta(seconds=60),
        )
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t, group=pg)
        self.assertEqual(t, torch.full_like(t, self.world_size))

        dist.destroy_process_group(pg)
        self._check_all_reduce()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_group_desc_reaches_backend(self) -> None:
        pg = dist.new_group(
            backend="xccl2",
            use_local_synchronization=True,
            group_desc="xccl2_desc",
        )
        backend = pg._get_backend(self.device)
        self.assertEqual(backend.options.group_desc, "xccl2_desc")
        dist.destroy_process_group(pg)


class ProcessGroupXCCL2SplitTest(_ProcessGroupXCCL2OptionsTest):
    hw_classification = HardwareClassification.XPU

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_split_group_child_global_ranks(self) -> None:
        pg = dist.distributed_c10d._get_default_group()
        # Bring the parent communicator up before splitting it.
        dist.all_reduce(torch.ones(4, device=self.device))
        half = self.world_size // 2
        halves = (
            list(range(half))
            if self.rank < half
            else list(range(half, self.world_size))
        )
        child = pg.split_group(halves, group_name=f"halves{halves}")
        child_options = child._get_backend(self.device).options
        self.assertEqual(child_options.global_ranks_in_group, halves)

        # Splitting the child again must map its ranks through the parent's map
        # rather than treating them as world ranks.
        mine = [halves.index(self.rank)]
        grandchild = child.split_group(mine, group_name=f"single{self.rank}")
        self.assertEqual(
            grandchild._get_backend(self.device).options.global_ranks_in_group,
            [self.rank],
        )

        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t, group=child)
        expected = float(sum(halves))
        self.assertEqual(t, torch.full_like(t, expected))

        # ProcessGroup.split_group does not register into _world, so
        # destroy_process_group cannot reach these groups.
        grandchild.shutdown()
        child.shutdown()


_UNINITIALIZED_XPU_SCRIPT = """\
import torch
import torch.distributed as dist

# Nothing above this point may touch torch.xpu: the caching allocator is
# brought up lazily by the Python torch.xpu layer.
dist.init_process_group(
    "cpu:gloo,xpu:xccl2",
    rank=0,
    world_size=1,
    store=dist.HashStore(),
    device_id={device_id},
)
assert not torch.xpu.is_initialized(), "creating a process group initialized XPU"
{extra}
dist.destroy_process_group()
"""


class ProcessGroupXCCL2UninitializedXpuTest(TestCase):
    """Constructing a process group must not need the XPU caching allocator up.

    Eager init allocates the barrier buffer before a process necessarily makes
    a torch.xpu call. Runs in a subprocess because the harness calls
    torch.xpu.set_device in setUp, which hides uninitialized-allocator bugs.
    """

    hw_classification = HardwareClassification.XPU

    def _run_child(
        self, extra: str = "", device_id: str = 'torch.device("xpu:0")'
    ) -> None:
        try:
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    _UNINITIALIZED_XPU_SCRIPT.format(device_id=device_id, extra=extra),
                ],
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.realpath(__file__)),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.fail("child process timed out")
        except subprocess.CalledProcessError as e:
            self.fail(f"child process failed with:\n{e.output.decode()}")

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_eager_init(self) -> None:
        self._run_child()

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_eager_init_without_device_id(self) -> None:
        self._run_child(device_id="None")

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_creator_without_process_group(self) -> None:
        self._run_child(
            """
opts = torch._C._distributed_c10d._DistributedBackendOptions()
opts.store = dist.HashStore()
opts.group_rank = 0
opts.group_size = 1
opts.timeout = dist.constants.default_pg_timeout
opts.group_id = "standalone"
opts.global_ranks_in_group = [0]
backend = dist.distributed_c10d._create_xccl2_process_group(opts, None)
backend.shutdown()
"""
        )

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_barrier_after_init(self) -> None:
        # The allocation is merely deferred, so barrier() is the second entrance
        # into the same allocator.
        self._run_child("dist.barrier()")


if __name__ == "__main__":
    if TEST_XPU:
        run_tests()
