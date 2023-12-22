
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


# kernel path: /tmp/torchinductor_youkaichao/om/combm7fjy3xmbhryi2khkv43m3m5gt4lj4aezcmfkjibs33udljc.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 768.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tl.full([1], False, tl.int1)
    tmp26 = 0.0
    tmp27 = tl.where(tmp25, tmp26, tmp24)
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 == tmp29
    tmp31 = tl.where(tmp30, tmp26, tmp24)
    tmp33 = tmp32 == tmp29
    tmp34 = tl.where(tmp33, tmp26, tmp24)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp34, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ap/cap4rx5ajnpqe32gmdx7u7mdot5sfrjjioipfpk42xag2cq4eooc.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnna3khxhinjn45efsfqniaeu4nwuwualevwk2ykhx43lwh5crr.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqk74azukczvmpdvynyjkxm7desrfrzh6eqiv54xfocl6zkvscco.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3147264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6hbn767le3t3emw3wiw7qrogig4gasygaxi37pp5ft7joqm7wb.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38603520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_6, full_default, add, mul_2, getitem_3, div, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_6, (1, 1024), (1024, 1))
    assert_size_stride(full_default, (1, 1024), (1024, 1))
    assert_size_stride(add, (1, 1024), (1024, 1))
    assert_size_stride(mul_2, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf6 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf10 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf14 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, getitem_3, primals_4, mul_2, div, add, primals_6, buf6, buf10, buf14, 1024, 768, grid=grid(1024), stream=stream0)
        del div
        del primals_4
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        buf4 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_1.run(tangents_1, getitem_3, mul_2, buf3, buf4, 768, 1024, grid=grid(768), stream=stream0)
        del getitem_3
        del mul_2
        del tangents_1
        buf5 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_2.run(buf5, 768, grid=grid(768), stream=stream0)
        aten.index_put_(buf5, [full_default], buf6, True)
        del buf6
        del full_default
        buf9 = empty((4098, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_3.run(buf9, 3147264, grid=grid(3147264), stream=stream0)
        aten.index_put_(buf9, [add], buf10, True)
        del add
        del buf10
        buf13 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_4.run(buf13, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf13, [primals_6], buf14, True)
        del buf14
        del primals_6
        return (buf13, buf9, buf5, buf3, buf4, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    full_default = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    add = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    mul_2 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    div = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_6, full_default, add, mul_2, getitem_3, div, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
