
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


# kernel path: /tmp/torchinductor_youkaichao/qg/cqg4llif7gpuh2nnjhmiwrcgdwiqbllijfexaeca57osrwbczed4.py
# Source Nodes: [attention_mask, cumsum], Original ATen: [aten._to_copy, aten.cumsum]
# attention_mask => convert_element_type
# cumsum => cumsum
triton_poi_fused__to_copy_cumsum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhbqj6bbbebgcs2jahqhxfgngmens3ya7bmonkbkb5zuvwjzvma.py
# Source Nodes: [add, attention_mask, mul, positions], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sub]
# add => add
# attention_mask => convert_element_type
# mul => mul
# positions => sub
triton_poi_fused__to_copy_add_mul_sub_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 + tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjq5yemqtj3vpd5zu3w7dmed7z65gjlx6qkgwkxxdbxkcyoxy4g.py
# Source Nodes: [embedding], Original ATen: [aten.embedding]
# embedding => embedding
triton_poi_fused_embedding_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 2050
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 2050), "index out of bounds: 0 <= tmp3 < 2050")
    tmp4 = tl.load(in_ptr1 + (x0 + (768*tmp3)), None)
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (2050, 768), (768, 1))
    assert_size_stride(primals_2, (1, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 2048), device='cuda', dtype=torch.int64)
        # Source Nodes: [attention_mask, cumsum], Original ATen: [aten._to_copy, aten.cumsum]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_0.run(primals_2, buf0, 2048, grid=grid(2048), stream=stream0)
        # Source Nodes: [attention_mask, cumsum], Original ATen: [aten._to_copy, aten.cumsum]
        buf1 = aten.cumsum(buf0, 1)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [add, attention_mask, mul, positions], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sub]
        triton_poi_fused__to_copy_add_mul_sub_1.run(buf3, primals_2, 2048, grid=grid(2048), stream=stream0)
        del primals_2
        buf4 = empty((1, 2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [embedding], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_2.run(buf3, primals_1, buf4, 1572864, grid=grid(1572864), stream=stream0)
        del primals_1
        return (buf4, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2050, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
