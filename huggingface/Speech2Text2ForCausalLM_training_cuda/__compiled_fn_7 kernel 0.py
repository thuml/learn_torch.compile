
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lxxcktaofbyxibvcliuk3lkog5pj2nxsyn5vc7qawagfsam3pn.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ni/cni7ciqt2tal42td4xfupljviw64occr5p4uvdh3y42kcuuvuqro.py
# Source Nodes: [detach], Original ATen: [aten.detach]
# detach => alias_1
triton_poi_fused_detach_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tmp10 + 1026
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tl.device_assert((0 <= tmp13) & (tmp13 < 1026), "index out of bounds: 0 <= tmp13 < 1026")
    tmp14 = tl.load(in_ptr2 + (x0 + (256*tmp13)), None)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 256), (256, 1))
    assert_size_stride(arg1_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(arg1_1, buf0, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty((1, 128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [detach], Original ATen: [aten.detach]
        triton_poi_fused_detach_1.run(buf2, arg1_1, arg0_1, buf3, 32768, grid=grid(32768), stream=stream0)
        del arg0_1
        del arg1_1
        return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('Speech2Text2ForCausalLM', benchmark_compiled_module)
