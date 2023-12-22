
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_youkaichao/5e/c5e7zvlxdhbfmtgajg33udh7oqj5amzgtqaikvsv7qdk4qgkvdpx.py
# Source Nodes: [clone, decoder_input_ids, eq, masked_fill_, setitem, setitem_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.fill, aten.lift_fresh, aten.masked_fill, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
# clone => clone
# decoder_input_ids => full
# eq => eq
# masked_fill_ => full_default_1, where
# setitem => copy, slice_scatter
# setitem_1 => copy_1, full_default, select_scatter
triton_poi_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tl.load(in_ptr0 + ((-1) + x0), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tl.full([1], 2, tl.int64)
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp12 = tl.full([1], -100, tl.int64)
    tmp13 = tmp11 == tmp12
    tmp14 = tl.where(tmp13, tmp3, tmp11)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [clone, decoder_input_ids, eq, masked_fill_, setitem, setitem_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.fill, aten.lift_fresh, aten.masked_fill, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0.run(arg0_1, buf0, 1024, grid=grid(1024), stream=stream0)
        del arg0_1
        return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
