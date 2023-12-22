
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


# kernel path: /tmp/torchinductor_youkaichao/qf/cqffvw36jofkexpfub77ct3ycb3gsw4fcllhrtfnmoskfibeqjru.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16398336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/oj/cojib5cjtva2qww365umgv3zoxwow4fqvapujgxd7ejhr7zbxrys.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmnkdgjcltwrrb4duvesoo5azl4juci3yxkwh7g6jl5b6beoxg4.py
# Source Nodes: [masked_fill_], Original ATen: [aten._log_softmax_backward_data, aten.masked_fill, aten.nll_loss_backward]
# masked_fill_ => full_default_1
triton_red_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32028
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (32028*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxcshlojd75kpoeptxjnffel3dimyszdvd6quala6i7vduegw3b.py
# Source Nodes: [masked_fill_], Original ATen: [aten._log_softmax_backward_data, aten.masked_fill, aten.nll_loss_backward]
# masked_fill_ => full_default_1
triton_per_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czguz4277ihrd4cmyr4pbokbhe7kykzrmrrwjh3bgwohl3ftkltf.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16398336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128112)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 - tmp16
    tmp18 = tmp0 + tmp17
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgy7dhqg36xe27p3ogmfofpqxbqvz2rf47ablvr6dig3zje2e5ao.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckd4otsf2o4r7y4ilhkruka6szf2mrdwhpqpyl47jtz37rv2vqlh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2smpat56a7k47qbvozfjcbqlbsp2cuxyejrcxukuoclhila6rq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6m3jzqc46hbpsxjdl46fgriukyg7omdqvdkm6nvonus7ihvjsh4.py
# Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
# masked_fill_ => full_default_1
triton_poi_fused_masked_fill_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3yy5ffypc3lxx57nh7qlijzhhmiqex7geykiwn5t6lyhb3xckg.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pq6jex4xgqm7pvpxurpinx23aqcozpavxdkktiwvjtl2fsxomn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crswlsba5nfwnx6g7j6uvvx3cuaqv7zsfj7vdtysn5pk5gh5fvbx.py
# Source Nodes: [attn_weights_95], Original ATen: [aten._softmax, aten._softmax_backward_data]
# attn_weights_95 => div_35, exp_35, sub_95
triton_per_fused__softmax__softmax_backward_data_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvncswfzznbe3l4hompnfrjvr34hqagbtjap66xeyko4h6t3ffy.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (1024*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dxtnrufhd6xtlznlza2c7yhgkk6oavltbdg6f6ncqx5sbbcl7g.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2c5h5p2h6tabnsaji4dh6vofxzfwsrydc4mj3e4wyxxsdw6rgd.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (128*x2) + (8192*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y1) + (1024*y0)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqqfp6ipfog7lx3qfps5q7tiww5nxekyocqj4wwfkkwo3mvy2jk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6eopsn5miwkmqkblqge4trmpecg5eljxi5vqnb4c5idlfhlobe.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2cincwm5jfhmvgcxzfefm3uj3t5xb6rpcff7etitl4ciqzqsce.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = 1024.0
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 - tmp10
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctn22ha2rgoqi6hu6mafypxybzegsdq4z3ooaodvbaaklxjpmtgt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqyyfjnwfstjkttphob3sl43b6v3egmjcw52vhcqkjlei7t7pyd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = 1024.0
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 - tmp10
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewxq2njjpvwjweaqkxodq636medt64cocdsjquywdjypgmip6tb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ee/cee2bsbtanli3hzrcm53j7sjufs2ph3x35hynoorljhf5ju3qtoc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: 'i32', 30: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(29, 30))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr9 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr11 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr12 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr13 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr14 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr15 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr16 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr17 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr18 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp39 = tl.load(in_ptr19 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr20 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp43 = tl.load(in_ptr21 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr22 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp47 = tl.load(in_ptr23 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp49 = tl.load(in_ptr24 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.load(in_ptr25 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp61 = tl.load(in_ptr26 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp44 = tmp42 + tmp43
    tmp46 = tmp44 + tmp45
    tmp48 = tmp46 + tmp47
    tmp50 = tmp48 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = tl.where(rmask & xmask, tmp51, 0)
    tmp54 = triton_helpers.promote_to_tensor(tl.sum(tmp53, 0))
    tmp56 = tmp50 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = tl.where(rmask & xmask, tmp57, 0)
    tmp60 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp62 = 1024.0
    tmp63 = tmp50 * tmp62
    tmp64 = tmp63 - tmp54
    tmp65 = tmp55 * tmp60
    tmp66 = tmp64 - tmp65
    tmp67 = tmp61 * tmp66
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp67, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4viw4phs7kzgf6rv3kd7m3pvvls7nh76jlmryhdfx3okcgpxvd.py
# Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
# masked_fill_ => full_default_1
triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = 1024.0
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 - tmp10
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp17 + tmp24
    tmp27 = tl.full([1], 1, tl.int64)
    tmp28 = tmp26 == tmp27
    tmp29 = 32.0
    tmp30 = tmp25 * tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp28, tmp31, tmp30)
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp32, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/au/cauevsqrlthnzku67mty5cibnuxmg6jgtzce6l7nhe7kvsdxd6bz.py
# Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill, aten.mul]
# masked_fill_ => full_default_1
triton_poi_fused_embedding_dense_backward_masked_fill_mul_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_masked_fill_mul_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131186688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgn46wnuxy2hvkpf3yl2x2keh6s2hl6gokzb6uswcau7tb3gk5h4.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsvycqui2oppbx3kxlqat2zfepdlusexzgq6ryjgt4xa3mo7pg2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_12, primals_18, primals_28, primals_34, primals_44, primals_50, primals_60, primals_66, primals_76, primals_82, primals_92, primals_98, primals_108, primals_114, primals_124, primals_130, primals_140, primals_146, primals_156, primals_162, primals_172, primals_178, primals_188, primals_194, primals_197, primals_207, primals_217, primals_223, primals_233, primals_243, primals_249, primals_259, primals_269, primals_275, primals_285, primals_295, primals_301, primals_311, primals_321, primals_327, primals_337, primals_347, primals_353, primals_363, primals_373, primals_379, primals_389, primals_399, primals_405, primals_415, primals_425, primals_431, primals_441, primals_451, primals_457, primals_467, primals_477, primals_483, primals_493, primals_503, primals_509, primals_514, view, mul_2, view_3, bmm, amax, sum_1, view_17, mul_5, view_19, view_21, mul_7, view_23, bmm_2, amax_1, sum_2, view_37, mul_10, view_39, view_41, mul_12, view_43, bmm_4, amax_2, sum_3, view_57, mul_15, view_59, view_61, mul_17, view_63, bmm_6, amax_3, sum_4, view_77, mul_20, view_79, view_81, mul_22, view_83, bmm_8, amax_4, sum_5, view_97, mul_25, view_99, view_101, mul_27, view_103, bmm_10, amax_5, sum_6, view_117, mul_30, view_119, view_121, mul_32, view_123, bmm_12, amax_6, sum_7, view_137, mul_35, view_139, view_141, mul_37, view_143, bmm_14, amax_7, sum_8, view_157, mul_40, view_159, view_161, mul_42, view_163, bmm_16, amax_8, sum_9, view_177, mul_45, view_179, view_181, mul_47, view_183, bmm_18, amax_9, sum_10, view_197, mul_50, view_199, view_201, mul_52, view_203, bmm_20, amax_10, sum_11, view_217, mul_55, view_219, view_221, mul_57, view_223, bmm_22, amax_11, sum_12, view_237, mul_60, view_239, view_241, mul_62, view_243, mul_66, view_247, view_263, mul_69, view_265, view_267, bmm_26, amax_13, sum_14, view_279, mul_72, view_281, view_283, mul_74, view_285, view_301, mul_77, view_303, bmm_30, amax_15, sum_16, view_317, mul_80, view_319, view_321, mul_82, view_323, view_339, mul_85, view_341, bmm_34, amax_17, sum_18, view_355, mul_88, view_357, view_359, mul_90, view_361, view_377, mul_93, view_379, bmm_38, amax_19, sum_20, view_393, mul_96, view_395, view_397, mul_98, view_399, view_415, mul_101, view_417, bmm_42, amax_21, sum_22, view_431, mul_104, view_433, view_435, mul_106, view_437, view_453, mul_109, view_455, bmm_46, amax_23, sum_24, view_469, mul_112, view_471, view_473, mul_114, view_475, view_491, mul_117, view_493, bmm_50, amax_25, sum_26, view_507, mul_120, view_509, view_511, mul_122, view_513, view_529, mul_125, view_531, bmm_54, amax_27, sum_28, view_545, mul_128, view_547, view_549, mul_130, view_551, view_567, mul_133, view_569, bmm_58, amax_29, sum_30, view_583, mul_136, view_585, view_587, mul_138, view_589, view_605, mul_141, view_607, bmm_62, amax_31, sum_32, view_621, mul_144, view_623, view_625, mul_146, view_627, view_643, mul_149, view_645, bmm_66, amax_33, sum_34, view_659, mul_152, view_661, view_663, mul_154, view_665, view_681, mul_157, view_683, bmm_70, amax_35, sum_36, view_697, mul_160, view_699, view_701, mul_162, view_703, sub_99, convert_element_type_6, permute_375, div_38, permute_377, le, permute_381, div_39, permute_385, permute_390, permute_391, permute_392, permute_393, permute_397, permute_402, permute_406, div_40, permute_410, permute_415, permute_416, alias_68, permute_417, permute_418, permute_422, permute_427, permute_431, div_41, permute_435, le_1, permute_439, div_42, permute_443, permute_448, permute_449, permute_450, permute_451, permute_455, permute_460, permute_464, div_43, permute_468, permute_473, permute_474, alias_71, permute_475, permute_476, permute_480, permute_485, permute_489, div_44, permute_493, le_2, permute_497, div_45, permute_501, permute_506, permute_507, permute_508, permute_509, permute_513, permute_518, permute_522, div_46, permute_526, permute_531, permute_532, alias_74, permute_533, permute_534, permute_538, permute_543, permute_547, div_47, permute_551, le_3, permute_555, div_48, permute_559, permute_564, permute_565, permute_566, permute_567, permute_571, permute_576, permute_580, div_49, permute_584, permute_589, permute_590, alias_77, permute_591, permute_592, permute_596, permute_601, permute_605, div_50, permute_609, le_4, permute_613, div_51, permute_617, permute_622, permute_623, permute_624, permute_625, permute_629, permute_634, permute_638, div_52, permute_642, permute_647, permute_648, alias_80, permute_649, permute_650, permute_654, permute_659, permute_663, div_53, permute_667, le_5, permute_671, div_54, permute_675, permute_680, permute_681, permute_682, permute_683, permute_687, permute_692, permute_696, div_55, permute_700, permute_705, permute_706, alias_83, permute_707, permute_708, permute_712, permute_717, permute_721, div_56, permute_725, le_6, permute_729, div_57, permute_733, permute_738, permute_739, permute_740, permute_741, permute_745, permute_750, permute_754, div_58, permute_758, permute_763, permute_764, alias_86, permute_765, permute_766, permute_770, permute_775, permute_779, div_59, permute_783, le_7, permute_787, div_60, permute_791, permute_796, permute_797, permute_798, permute_799, permute_803, permute_808, permute_812, div_61, permute_816, permute_821, permute_822, alias_89, permute_823, permute_824, permute_828, permute_833, permute_837, div_62, permute_841, le_8, permute_845, div_63, permute_849, permute_854, permute_855, permute_856, permute_857, permute_861, permute_866, permute_870, div_64, permute_874, permute_879, permute_880, alias_92, permute_881, permute_882, permute_886, permute_891, permute_895, div_65, permute_899, le_9, permute_903, div_66, permute_907, permute_912, permute_913, permute_914, permute_915, permute_919, permute_924, permute_928, div_67, permute_932, permute_937, permute_938, alias_95, permute_939, permute_940, permute_944, permute_949, permute_953, div_68, permute_957, le_10, permute_961, div_69, permute_965, permute_970, permute_971, permute_972, permute_973, permute_977, permute_982, permute_986, div_70, permute_990, permute_995, permute_996, alias_98, permute_997, permute_998, permute_1002, permute_1007, permute_1011, div_71, permute_1015, le_11, permute_1019, div_72, permute_1023, permute_1028, permute_1029, permute_1030, permute_1031, permute_1035, permute_1040, permute_1044, div_73, permute_1048, permute_1053, permute_1054, alias_101, permute_1055, permute_1056, permute_1060, permute_1065, permute_1069, div_74, div_75, permute_1073, le_12, permute_1077, div_76, permute_1081, permute_1086, permute_1087, permute_1088, permute_1089, permute_1093, permute_1098, permute_1102, div_77, permute_1106, le_13, permute_1110, div_78, permute_1114, permute_1119, permute_1120, permute_1121, permute_1122, permute_1126, permute_1131, permute_1135, div_79, permute_1139, le_14, permute_1143, div_80, permute_1147, permute_1152, permute_1153, permute_1154, permute_1155, permute_1159, permute_1164, permute_1168, div_81, permute_1172, le_15, permute_1176, div_82, permute_1180, permute_1185, permute_1186, permute_1187, permute_1188, permute_1192, permute_1197, permute_1201, div_83, permute_1205, le_16, permute_1209, div_84, permute_1213, permute_1218, permute_1219, permute_1220, permute_1221, permute_1225, permute_1230, permute_1234, div_85, permute_1238, le_17, permute_1242, div_86, permute_1246, permute_1251, permute_1252, permute_1253, permute_1254, permute_1258, permute_1263, permute_1267, div_87, permute_1271, le_18, permute_1275, div_88, permute_1279, permute_1284, permute_1285, permute_1286, permute_1287, permute_1291, permute_1296, permute_1300, div_89, permute_1304, le_19, permute_1308, div_90, permute_1312, permute_1317, permute_1318, permute_1319, permute_1320, permute_1324, permute_1329, permute_1333, div_91, permute_1337, le_20, permute_1341, div_92, permute_1345, permute_1350, permute_1351, permute_1352, permute_1353, permute_1357, permute_1362, permute_1366, div_93, permute_1370, le_21, permute_1374, div_94, permute_1378, permute_1383, permute_1384, permute_1385, permute_1386, permute_1390, permute_1395, permute_1399, div_95, permute_1403, le_22, permute_1407, div_96, permute_1411, permute_1416, permute_1417, permute_1418, permute_1419, permute_1423, permute_1428, permute_1432, div_97, permute_1436, le_23, permute_1440, div_98, permute_1444, permute_1449, permute_1450, permute_1451, permute_1452, permute_1456, permute_1461, permute_1465, div_99, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51 = args
    args.clear()
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_28, (1024, ), (1, ))
    assert_size_stride(primals_34, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_76, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, ), (1, ))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_98, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_124, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_172, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_188, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_285, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_301, (1024, ), (1, ))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_353, (1024, ), (1, ))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_389, (1024, ), (1, ))
    assert_size_stride(primals_399, (1024, ), (1, ))
    assert_size_stride(primals_405, (1024, ), (1, ))
    assert_size_stride(primals_415, (1024, ), (1, ))
    assert_size_stride(primals_425, (1024, ), (1, ))
    assert_size_stride(primals_431, (1024, ), (1, ))
    assert_size_stride(primals_441, (1024, ), (1, ))
    assert_size_stride(primals_451, (1024, ), (1, ))
    assert_size_stride(primals_457, (1024, ), (1, ))
    assert_size_stride(primals_467, (1024, ), (1, ))
    assert_size_stride(primals_477, (1024, ), (1, ))
    assert_size_stride(primals_483, (1024, ), (1, ))
    assert_size_stride(primals_493, (1024, ), (1, ))
    assert_size_stride(primals_503, (1024, ), (1, ))
    assert_size_stride(primals_509, (1024, ), (1, ))
    assert_size_stride(primals_514, (1, 128), (128, 1))
    assert_size_stride(view, (1, 128), (128, 1))
    assert_size_stride(mul_2, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_3, (128, 1024), (1024, 1))
    assert_size_stride(bmm, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_1, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_17, (128, 1024), (1024, 1))
    assert_size_stride(mul_5, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_19, (128, 1024), (1024, 1))
    assert_size_stride(view_21, (128, 4096), (4096, 1))
    assert_size_stride(mul_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_23, (128, 1024), (1024, 1))
    assert_size_stride(bmm_2, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_1, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_2, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_37, (128, 1024), (1024, 1))
    assert_size_stride(mul_10, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_39, (128, 1024), (1024, 1))
    assert_size_stride(view_41, (128, 4096), (4096, 1))
    assert_size_stride(mul_12, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_43, (128, 1024), (1024, 1))
    assert_size_stride(bmm_4, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_2, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_3, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_57, (128, 1024), (1024, 1))
    assert_size_stride(mul_15, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_59, (128, 1024), (1024, 1))
    assert_size_stride(view_61, (128, 4096), (4096, 1))
    assert_size_stride(mul_17, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_63, (128, 1024), (1024, 1))
    assert_size_stride(bmm_6, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_3, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_4, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_77, (128, 1024), (1024, 1))
    assert_size_stride(mul_20, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_79, (128, 1024), (1024, 1))
    assert_size_stride(view_81, (128, 4096), (4096, 1))
    assert_size_stride(mul_22, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_83, (128, 1024), (1024, 1))
    assert_size_stride(bmm_8, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_4, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_5, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_97, (128, 1024), (1024, 1))
    assert_size_stride(mul_25, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_99, (128, 1024), (1024, 1))
    assert_size_stride(view_101, (128, 4096), (4096, 1))
    assert_size_stride(mul_27, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_103, (128, 1024), (1024, 1))
    assert_size_stride(bmm_10, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_5, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_6, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_117, (128, 1024), (1024, 1))
    assert_size_stride(mul_30, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_119, (128, 1024), (1024, 1))
    assert_size_stride(view_121, (128, 4096), (4096, 1))
    assert_size_stride(mul_32, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_123, (128, 1024), (1024, 1))
    assert_size_stride(bmm_12, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_6, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_7, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_137, (128, 1024), (1024, 1))
    assert_size_stride(mul_35, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_139, (128, 1024), (1024, 1))
    assert_size_stride(view_141, (128, 4096), (4096, 1))
    assert_size_stride(mul_37, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_143, (128, 1024), (1024, 1))
    assert_size_stride(bmm_14, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_7, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_8, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_157, (128, 1024), (1024, 1))
    assert_size_stride(mul_40, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_159, (128, 1024), (1024, 1))
    assert_size_stride(view_161, (128, 4096), (4096, 1))
    assert_size_stride(mul_42, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_163, (128, 1024), (1024, 1))
    assert_size_stride(bmm_16, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_8, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_9, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_177, (128, 1024), (1024, 1))
    assert_size_stride(mul_45, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_179, (128, 1024), (1024, 1))
    assert_size_stride(view_181, (128, 4096), (4096, 1))
    assert_size_stride(mul_47, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_183, (128, 1024), (1024, 1))
    assert_size_stride(bmm_18, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_9, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_10, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_197, (128, 1024), (1024, 1))
    assert_size_stride(mul_50, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_199, (128, 1024), (1024, 1))
    assert_size_stride(view_201, (128, 4096), (4096, 1))
    assert_size_stride(mul_52, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_203, (128, 1024), (1024, 1))
    assert_size_stride(bmm_20, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_10, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_11, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_217, (128, 1024), (1024, 1))
    assert_size_stride(mul_55, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_219, (128, 1024), (1024, 1))
    assert_size_stride(view_221, (128, 4096), (4096, 1))
    assert_size_stride(mul_57, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_223, (128, 1024), (1024, 1))
    assert_size_stride(bmm_22, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_11, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_12, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_237, (128, 1024), (1024, 1))
    assert_size_stride(mul_60, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_239, (128, 1024), (1024, 1))
    assert_size_stride(view_241, (128, 4096), (4096, 1))
    assert_size_stride(mul_62, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_243, (1, 128), (128, 1))
    assert_size_stride(mul_66, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_247, (128, 1024), (1024, 1))
    assert_size_stride(view_263, (128, 1024), (1024, 1))
    assert_size_stride(mul_69, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_265, (128, 1024), (1024, 1))
    assert_size_stride(view_267, (128, 1024), (1024, 1))
    assert_size_stride(bmm_26, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_13, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_14, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_279, (128, 1024), (1024, 1))
    assert_size_stride(mul_72, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_281, (128, 1024), (1024, 1))
    assert_size_stride(view_283, (128, 4096), (4096, 1))
    assert_size_stride(mul_74, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_285, (128, 1024), (1024, 1))
    assert_size_stride(view_301, (128, 1024), (1024, 1))
    assert_size_stride(mul_77, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_303, (128, 1024), (1024, 1))
    assert_size_stride(bmm_30, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_15, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_16, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_317, (128, 1024), (1024, 1))
    assert_size_stride(mul_80, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_319, (128, 1024), (1024, 1))
    assert_size_stride(view_321, (128, 4096), (4096, 1))
    assert_size_stride(mul_82, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_323, (128, 1024), (1024, 1))
    assert_size_stride(view_339, (128, 1024), (1024, 1))
    assert_size_stride(mul_85, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_341, (128, 1024), (1024, 1))
    assert_size_stride(bmm_34, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_17, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_18, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_355, (128, 1024), (1024, 1))
    assert_size_stride(mul_88, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_357, (128, 1024), (1024, 1))
    assert_size_stride(view_359, (128, 4096), (4096, 1))
    assert_size_stride(mul_90, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_361, (128, 1024), (1024, 1))
    assert_size_stride(view_377, (128, 1024), (1024, 1))
    assert_size_stride(mul_93, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_379, (128, 1024), (1024, 1))
    assert_size_stride(bmm_38, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_19, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_20, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_393, (128, 1024), (1024, 1))
    assert_size_stride(mul_96, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_395, (128, 1024), (1024, 1))
    assert_size_stride(view_397, (128, 4096), (4096, 1))
    assert_size_stride(mul_98, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_399, (128, 1024), (1024, 1))
    assert_size_stride(view_415, (128, 1024), (1024, 1))
    assert_size_stride(mul_101, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_417, (128, 1024), (1024, 1))
    assert_size_stride(bmm_42, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_21, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_22, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_431, (128, 1024), (1024, 1))
    assert_size_stride(mul_104, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_433, (128, 1024), (1024, 1))
    assert_size_stride(view_435, (128, 4096), (4096, 1))
    assert_size_stride(mul_106, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_437, (128, 1024), (1024, 1))
    assert_size_stride(view_453, (128, 1024), (1024, 1))
    assert_size_stride(mul_109, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_455, (128, 1024), (1024, 1))
    assert_size_stride(bmm_46, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_23, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_24, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_469, (128, 1024), (1024, 1))
    assert_size_stride(mul_112, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_471, (128, 1024), (1024, 1))
    assert_size_stride(view_473, (128, 4096), (4096, 1))
    assert_size_stride(mul_114, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_475, (128, 1024), (1024, 1))
    assert_size_stride(view_491, (128, 1024), (1024, 1))
    assert_size_stride(mul_117, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_493, (128, 1024), (1024, 1))
    assert_size_stride(bmm_50, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_25, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_26, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_507, (128, 1024), (1024, 1))
    assert_size_stride(mul_120, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_509, (128, 1024), (1024, 1))
    assert_size_stride(view_511, (128, 4096), (4096, 1))
    assert_size_stride(mul_122, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_513, (128, 1024), (1024, 1))
    assert_size_stride(view_529, (128, 1024), (1024, 1))
    assert_size_stride(mul_125, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_531, (128, 1024), (1024, 1))
    assert_size_stride(bmm_54, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_27, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_28, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_545, (128, 1024), (1024, 1))
    assert_size_stride(mul_128, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_547, (128, 1024), (1024, 1))
    assert_size_stride(view_549, (128, 4096), (4096, 1))
    assert_size_stride(mul_130, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_551, (128, 1024), (1024, 1))
    assert_size_stride(view_567, (128, 1024), (1024, 1))
    assert_size_stride(mul_133, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_569, (128, 1024), (1024, 1))
    assert_size_stride(bmm_58, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_29, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_30, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_583, (128, 1024), (1024, 1))
    assert_size_stride(mul_136, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_585, (128, 1024), (1024, 1))
    assert_size_stride(view_587, (128, 4096), (4096, 1))
    assert_size_stride(mul_138, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_589, (128, 1024), (1024, 1))
    assert_size_stride(view_605, (128, 1024), (1024, 1))
    assert_size_stride(mul_141, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_607, (128, 1024), (1024, 1))
    assert_size_stride(bmm_62, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_31, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_32, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_621, (128, 1024), (1024, 1))
    assert_size_stride(mul_144, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_623, (128, 1024), (1024, 1))
    assert_size_stride(view_625, (128, 4096), (4096, 1))
    assert_size_stride(mul_146, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_627, (128, 1024), (1024, 1))
    assert_size_stride(view_643, (128, 1024), (1024, 1))
    assert_size_stride(mul_149, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_645, (128, 1024), (1024, 1))
    assert_size_stride(bmm_66, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_33, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_34, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_659, (128, 1024), (1024, 1))
    assert_size_stride(mul_152, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_661, (128, 1024), (1024, 1))
    assert_size_stride(view_663, (128, 4096), (4096, 1))
    assert_size_stride(mul_154, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_665, (128, 1024), (1024, 1))
    assert_size_stride(view_681, (128, 1024), (1024, 1))
    assert_size_stride(mul_157, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_683, (128, 1024), (1024, 1))
    assert_size_stride(bmm_70, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_35, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_36, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_697, (128, 1024), (1024, 1))
    assert_size_stride(mul_160, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_699, (128, 1024), (1024, 1))
    assert_size_stride(view_701, (128, 4096), (4096, 1))
    assert_size_stride(mul_162, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_703, (128, 1024), (1024, 1))
    assert_size_stride(sub_99, (128, 128112), (128112, 1))
    assert_size_stride(convert_element_type_6, (), ())
    assert_size_stride(permute_375, (128112, 1024), (1024, 1))
    assert_size_stride(div_38, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_377, (1024, 4096), (4096, 1))
    assert_size_stride(le, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_381, (4096, 1024), (1024, 1))
    assert_size_stride(div_39, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_385, (1024, 1024), (1024, 1))
    assert_size_stride(permute_390, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_391, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_392, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_393, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_397, (1024, 1024), (1024, 1))
    assert_size_stride(permute_402, (1024, 1024), (1024, 1))
    assert_size_stride(permute_406, (1024, 1024), (1024, 1))
    assert_size_stride(div_40, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_410, (1024, 1024), (1024, 1))
    assert_size_stride(permute_415, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_416, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_68, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_417, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_418, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_422, (1024, 1024), (1024, 1))
    assert_size_stride(permute_427, (1024, 1024), (1024, 1))
    assert_size_stride(permute_431, (1024, 1024), (1024, 1))
    assert_size_stride(div_41, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_435, (1024, 4096), (4096, 1))
    assert_size_stride(le_1, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_439, (4096, 1024), (1024, 1))
    assert_size_stride(div_42, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_443, (1024, 1024), (1024, 1))
    assert_size_stride(permute_448, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_449, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_450, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_451, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_455, (1024, 1024), (1024, 1))
    assert_size_stride(permute_460, (1024, 1024), (1024, 1))
    assert_size_stride(permute_464, (1024, 1024), (1024, 1))
    assert_size_stride(div_43, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_468, (1024, 1024), (1024, 1))
    assert_size_stride(permute_473, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_474, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_71, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_475, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_476, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_480, (1024, 1024), (1024, 1))
    assert_size_stride(permute_485, (1024, 1024), (1024, 1))
    assert_size_stride(permute_489, (1024, 1024), (1024, 1))
    assert_size_stride(div_44, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_493, (1024, 4096), (4096, 1))
    assert_size_stride(le_2, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_497, (4096, 1024), (1024, 1))
    assert_size_stride(div_45, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_501, (1024, 1024), (1024, 1))
    assert_size_stride(permute_506, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_507, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_508, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_509, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_513, (1024, 1024), (1024, 1))
    assert_size_stride(permute_518, (1024, 1024), (1024, 1))
    assert_size_stride(permute_522, (1024, 1024), (1024, 1))
    assert_size_stride(div_46, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_526, (1024, 1024), (1024, 1))
    assert_size_stride(permute_531, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_532, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_74, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_533, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_534, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_538, (1024, 1024), (1024, 1))
    assert_size_stride(permute_543, (1024, 1024), (1024, 1))
    assert_size_stride(permute_547, (1024, 1024), (1024, 1))
    assert_size_stride(div_47, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_551, (1024, 4096), (4096, 1))
    assert_size_stride(le_3, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_555, (4096, 1024), (1024, 1))
    assert_size_stride(div_48, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_559, (1024, 1024), (1024, 1))
    assert_size_stride(permute_564, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_565, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_566, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_567, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_571, (1024, 1024), (1024, 1))
    assert_size_stride(permute_576, (1024, 1024), (1024, 1))
    assert_size_stride(permute_580, (1024, 1024), (1024, 1))
    assert_size_stride(div_49, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_584, (1024, 1024), (1024, 1))
    assert_size_stride(permute_589, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_590, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_77, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_591, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_592, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_596, (1024, 1024), (1024, 1))
    assert_size_stride(permute_601, (1024, 1024), (1024, 1))
    assert_size_stride(permute_605, (1024, 1024), (1024, 1))
    assert_size_stride(div_50, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_609, (1024, 4096), (4096, 1))
    assert_size_stride(le_4, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_613, (4096, 1024), (1024, 1))
    assert_size_stride(div_51, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_617, (1024, 1024), (1024, 1))
    assert_size_stride(permute_622, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_623, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_624, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_625, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_629, (1024, 1024), (1024, 1))
    assert_size_stride(permute_634, (1024, 1024), (1024, 1))
    assert_size_stride(permute_638, (1024, 1024), (1024, 1))
    assert_size_stride(div_52, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_642, (1024, 1024), (1024, 1))
    assert_size_stride(permute_647, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_648, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_80, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_649, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_650, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_654, (1024, 1024), (1024, 1))
    assert_size_stride(permute_659, (1024, 1024), (1024, 1))
    assert_size_stride(permute_663, (1024, 1024), (1024, 1))
    assert_size_stride(div_53, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_667, (1024, 4096), (4096, 1))
    assert_size_stride(le_5, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_671, (4096, 1024), (1024, 1))
    assert_size_stride(div_54, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_675, (1024, 1024), (1024, 1))
    assert_size_stride(permute_680, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_681, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_682, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_683, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_687, (1024, 1024), (1024, 1))
    assert_size_stride(permute_692, (1024, 1024), (1024, 1))
    assert_size_stride(permute_696, (1024, 1024), (1024, 1))
    assert_size_stride(div_55, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_700, (1024, 1024), (1024, 1))
    assert_size_stride(permute_705, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_706, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_83, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_707, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_708, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_712, (1024, 1024), (1024, 1))
    assert_size_stride(permute_717, (1024, 1024), (1024, 1))
    assert_size_stride(permute_721, (1024, 1024), (1024, 1))
    assert_size_stride(div_56, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_725, (1024, 4096), (4096, 1))
    assert_size_stride(le_6, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_729, (4096, 1024), (1024, 1))
    assert_size_stride(div_57, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_733, (1024, 1024), (1024, 1))
    assert_size_stride(permute_738, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_739, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_740, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_741, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_745, (1024, 1024), (1024, 1))
    assert_size_stride(permute_750, (1024, 1024), (1024, 1))
    assert_size_stride(permute_754, (1024, 1024), (1024, 1))
    assert_size_stride(div_58, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_758, (1024, 1024), (1024, 1))
    assert_size_stride(permute_763, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_764, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_86, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_765, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_766, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_770, (1024, 1024), (1024, 1))
    assert_size_stride(permute_775, (1024, 1024), (1024, 1))
    assert_size_stride(permute_779, (1024, 1024), (1024, 1))
    assert_size_stride(div_59, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_783, (1024, 4096), (4096, 1))
    assert_size_stride(le_7, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_787, (4096, 1024), (1024, 1))
    assert_size_stride(div_60, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_791, (1024, 1024), (1024, 1))
    assert_size_stride(permute_796, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_797, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_798, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_799, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_803, (1024, 1024), (1024, 1))
    assert_size_stride(permute_808, (1024, 1024), (1024, 1))
    assert_size_stride(permute_812, (1024, 1024), (1024, 1))
    assert_size_stride(div_61, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_816, (1024, 1024), (1024, 1))
    assert_size_stride(permute_821, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_822, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_89, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_823, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_824, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_828, (1024, 1024), (1024, 1))
    assert_size_stride(permute_833, (1024, 1024), (1024, 1))
    assert_size_stride(permute_837, (1024, 1024), (1024, 1))
    assert_size_stride(div_62, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_841, (1024, 4096), (4096, 1))
    assert_size_stride(le_8, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_845, (4096, 1024), (1024, 1))
    assert_size_stride(div_63, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_849, (1024, 1024), (1024, 1))
    assert_size_stride(permute_854, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_855, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_856, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_857, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_861, (1024, 1024), (1024, 1))
    assert_size_stride(permute_866, (1024, 1024), (1024, 1))
    assert_size_stride(permute_870, (1024, 1024), (1024, 1))
    assert_size_stride(div_64, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_874, (1024, 1024), (1024, 1))
    assert_size_stride(permute_879, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_880, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_92, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_881, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_882, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_886, (1024, 1024), (1024, 1))
    assert_size_stride(permute_891, (1024, 1024), (1024, 1))
    assert_size_stride(permute_895, (1024, 1024), (1024, 1))
    assert_size_stride(div_65, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_899, (1024, 4096), (4096, 1))
    assert_size_stride(le_9, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_903, (4096, 1024), (1024, 1))
    assert_size_stride(div_66, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_907, (1024, 1024), (1024, 1))
    assert_size_stride(permute_912, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_913, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_914, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_915, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_919, (1024, 1024), (1024, 1))
    assert_size_stride(permute_924, (1024, 1024), (1024, 1))
    assert_size_stride(permute_928, (1024, 1024), (1024, 1))
    assert_size_stride(div_67, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_932, (1024, 1024), (1024, 1))
    assert_size_stride(permute_937, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_938, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_95, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_939, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_940, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_944, (1024, 1024), (1024, 1))
    assert_size_stride(permute_949, (1024, 1024), (1024, 1))
    assert_size_stride(permute_953, (1024, 1024), (1024, 1))
    assert_size_stride(div_68, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_957, (1024, 4096), (4096, 1))
    assert_size_stride(le_10, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_961, (4096, 1024), (1024, 1))
    assert_size_stride(div_69, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_965, (1024, 1024), (1024, 1))
    assert_size_stride(permute_970, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_971, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_972, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_973, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_977, (1024, 1024), (1024, 1))
    assert_size_stride(permute_982, (1024, 1024), (1024, 1))
    assert_size_stride(permute_986, (1024, 1024), (1024, 1))
    assert_size_stride(div_70, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_990, (1024, 1024), (1024, 1))
    assert_size_stride(permute_995, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_996, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_98, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_997, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_998, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1002, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1007, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1011, (1024, 1024), (1024, 1))
    assert_size_stride(div_71, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1015, (1024, 4096), (4096, 1))
    assert_size_stride(le_11, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1019, (4096, 1024), (1024, 1))
    assert_size_stride(div_72, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1023, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1028, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1029, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1030, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1031, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1035, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1040, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1044, (1024, 1024), (1024, 1))
    assert_size_stride(div_73, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1048, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1053, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1054, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_101, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_1055, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1056, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1060, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1065, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1069, (1024, 1024), (1024, 1))
    assert_size_stride(div_74, (1, 128, 1), (128, 1, 1))
    assert_size_stride(div_75, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1073, (1024, 4096), (4096, 1))
    assert_size_stride(le_12, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1077, (4096, 1024), (1024, 1))
    assert_size_stride(div_76, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1081, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1086, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1087, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1088, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1089, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1093, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1098, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1102, (1024, 1024), (1024, 1))
    assert_size_stride(div_77, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1106, (1024, 4096), (4096, 1))
    assert_size_stride(le_13, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1110, (4096, 1024), (1024, 1))
    assert_size_stride(div_78, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1114, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1119, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1120, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1121, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1122, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1126, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1131, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1135, (1024, 1024), (1024, 1))
    assert_size_stride(div_79, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1139, (1024, 4096), (4096, 1))
    assert_size_stride(le_14, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1143, (4096, 1024), (1024, 1))
    assert_size_stride(div_80, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1147, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1152, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1153, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1154, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1155, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1159, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1164, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1168, (1024, 1024), (1024, 1))
    assert_size_stride(div_81, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1172, (1024, 4096), (4096, 1))
    assert_size_stride(le_15, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1176, (4096, 1024), (1024, 1))
    assert_size_stride(div_82, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1180, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1185, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1186, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1187, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1188, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1192, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1197, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1201, (1024, 1024), (1024, 1))
    assert_size_stride(div_83, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1205, (1024, 4096), (4096, 1))
    assert_size_stride(le_16, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1209, (4096, 1024), (1024, 1))
    assert_size_stride(div_84, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1213, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1218, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1219, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1220, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1221, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1225, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1230, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1234, (1024, 1024), (1024, 1))
    assert_size_stride(div_85, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1238, (1024, 4096), (4096, 1))
    assert_size_stride(le_17, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1242, (4096, 1024), (1024, 1))
    assert_size_stride(div_86, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1246, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1251, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1252, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1253, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1254, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1258, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1263, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1267, (1024, 1024), (1024, 1))
    assert_size_stride(div_87, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1271, (1024, 4096), (4096, 1))
    assert_size_stride(le_18, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1275, (4096, 1024), (1024, 1))
    assert_size_stride(div_88, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1279, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1284, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1285, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1286, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1287, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1291, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1296, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1300, (1024, 1024), (1024, 1))
    assert_size_stride(div_89, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1304, (1024, 4096), (4096, 1))
    assert_size_stride(le_19, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1308, (4096, 1024), (1024, 1))
    assert_size_stride(div_90, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1312, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1317, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1318, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1319, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1320, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1324, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1329, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1333, (1024, 1024), (1024, 1))
    assert_size_stride(div_91, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1337, (1024, 4096), (4096, 1))
    assert_size_stride(le_20, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1341, (4096, 1024), (1024, 1))
    assert_size_stride(div_92, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1345, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1350, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1351, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1352, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1353, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1357, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1362, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1366, (1024, 1024), (1024, 1))
    assert_size_stride(div_93, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1370, (1024, 4096), (4096, 1))
    assert_size_stride(le_21, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1374, (4096, 1024), (1024, 1))
    assert_size_stride(div_94, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1378, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1383, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1384, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1385, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1386, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1390, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1395, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1399, (1024, 1024), (1024, 1))
    assert_size_stride(div_95, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1403, (1024, 4096), (4096, 1))
    assert_size_stride(le_22, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1407, (4096, 1024), (1024, 1))
    assert_size_stride(div_96, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1411, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1416, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1417, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1418, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1419, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1423, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1428, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1432, (1024, 1024), (1024, 1))
    assert_size_stride(div_97, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1436, (1024, 4096), (4096, 1))
    assert_size_stride(le_23, (1, 128, 4096), (524288, 4096, 1))
    assert_size_stride(permute_1440, (4096, 1024), (1024, 1))
    assert_size_stride(div_98, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1444, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1449, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1450, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1451, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_1452, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_1456, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1461, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1465, (1024, 1024), (1024, 1))
    assert_size_stride(div_99, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 128112), (16398336, 128112, 1))
    assert_size_stride(tangents_3, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_4, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_5, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_6, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_7, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_8, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_9, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_10, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_11, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_12, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_13, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_14, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_15, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_16, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_17, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_18, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_19, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_20, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_21, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_22, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_23, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_24, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_25, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_26, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_27, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_28, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_29, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_30, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_31, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_32, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_33, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_34, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_35, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_36, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_37, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_38, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_39, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_40, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_41, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_42, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_43, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_44, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_45, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_46, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_47, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_48, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_49, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_50, (1, 16, 128, 64), (131072, 8192, 64, 1))
    assert_size_stride(tangents_51, (1, 128, 1024), (131072, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 128112), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 16398336, grid=grid(16398336), stream=stream0)
        buf1 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_514, buf1, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_fill_], Original ATen: [aten._log_softmax_backward_data, aten.masked_fill, aten.nll_loss_backward]
        triton_red_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_2.run(buf0, primals_514, tangents_1, convert_element_type_6, buf4, 512, 32028, grid=grid(512), stream=stream0)
        buf5 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_fill_], Original ATen: [aten._log_softmax_backward_data, aten.masked_fill, aten.nll_loss_backward]
        triton_per_fused__log_softmax_backward_data_masked_fill_nll_loss_backward_3.run(buf4, buf5, 128, 4, grid=grid(128), stream=stream0)
        del buf4
        buf3 = empty((128, 128112), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf3, (1, 128, 128112), (16398336, 128112, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf6, tangents_2, buf0, primals_514, tangents_1, convert_element_type_6, sub_99, buf5, 16398336, grid=grid(16398336), stream=stream0)
        del buf0
        del buf5
        del convert_element_type_6
        del primals_514
        del sub_99
        del tangents_1
        del tangents_2
        buf7 = empty((128112, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128112, 128), (1, 128112), 0), view_703, out=buf7)
        del view_703
        buf8 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 128112), (128112, 1), 0), permute_375, out=buf8)
        del buf6
        del permute_375
        buf11 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf8, primals_509, mul_162, div_38, buf11, 128, 1024, grid=grid(128), stream=stream0)
        del div_38
        del primals_509
        buf12 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf13 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf8, mul_162, buf12, buf13, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_162
        buf14 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (128, 1024), (1024, 1), 0), permute_377, out=buf14)
        del permute_377
        buf15 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (1024, 128), (1, 1024), 0), view_701, out=buf15)
        del view_701
        buf16 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf11, buf16, 1024, 128, grid=grid(1024), stream=stream0)
        buf17 = reinterpret_tensor(buf14, (1, 128, 4096), (524288, 4096, 1), 0); del buf14  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf17, le, 524288, grid=grid(524288), stream=stream0)
        del le
        buf18 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128, 4096), (4096, 1), 0), permute_381, out=buf18)
        del permute_381
        buf19 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (4096, 128), (1, 4096), 0), view_699, out=buf19)
        del view_699
        buf20 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf17, buf20, 4096, 128, grid=grid(4096), stream=stream0)
        buf25 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf25, buf18, primals_503, mul_160, div_39, 128, 1024, grid=grid(128), stream=stream0)
        del div_39
        del primals_503
        buf23 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf24 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf18, mul_160, buf23, buf24, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_160
        buf26 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 1024), (1024, 1), 0), permute_385, out=buf26)
        del permute_385
        buf27 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1024, 128), (1, 1024), 0), view_697, out=buf27)
        del view_697
        buf28 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf25, buf28, 1024, 128, grid=grid(1024), stream=stream0)
        buf29 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_390, reinterpret_tensor(buf26, (16, 128, 64), (64, 1024, 1), 0), out=buf29)
        del permute_390
        buf30 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (16, 128, 64), (64, 1024, 1), 0), permute_391, out=buf30)
        del permute_391
        buf32 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_95], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf30, bmm_70, amax_35, sum_36, buf32, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_35
        del bmm_70
        del sum_36
        buf33 = reinterpret_tensor(buf26, (16, 64, 128), (8192, 128, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_392, buf32, out=buf33)
        del permute_392
        buf34 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf32, permute_393, out=buf34)
        del permute_393
        buf35 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_50, buf29, buf35, 131072, grid=grid(131072), stream=stream0)
        del tangents_50
        buf36 = reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 1024), (1024, 1), 0), permute_397, out=buf36)
        del permute_397
        buf37 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (1024, 128), (1, 1024), 0), view_267, out=buf37)
        buf38 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf35, buf38, 1024, 128, grid=grid(1024), stream=stream0)
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_49, buf33, buf39, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_49
        buf40 = reinterpret_tensor(buf33, (128, 1024), (1024, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (128, 1024), (1024, 1), 0), permute_402, out=buf40)
        del permute_402
        buf41 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 128), (1, 1024), 0), view_267, out=buf41)
        buf42 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf39, buf42, 1024, 128, grid=grid(1024), stream=stream0)
        buf43 = reinterpret_tensor(buf39, (128, 1024), (1024, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf34, buf43, 131072, grid=grid(131072), stream=stream0)
        buf44 = reinterpret_tensor(buf34, (128, 1024), (1024, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf43, permute_406, out=buf44)
        del permute_406
        buf45 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (1024, 128), (1, 1024), 0), view_683, out=buf45)
        del view_683
        buf46 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf43, buf46, 1024, 128, grid=grid(1024), stream=stream0)
        buf51 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf51, buf44, primals_493, mul_157, div_40, 128, 1024, grid=grid(128), stream=stream0)
        del div_40
        del primals_493
        buf49 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf50 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf44, mul_157, buf49, buf50, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_157
        buf52 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 1024), (1024, 1), 0), permute_410, out=buf52)
        del permute_410
        buf53 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 128), (1, 1024), 0), view_681, out=buf53)
        del view_681
        buf54 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf51, buf54, 1024, 128, grid=grid(1024), stream=stream0)
        buf55 = reinterpret_tensor(buf43, (16, 128, 64), (8192, 64, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_415, reinterpret_tensor(buf52, (16, 128, 64), (64, 1024, 1), 0), out=buf55)
        del permute_415
        buf56 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (16, 128, 64), (64, 1024, 1), 0), permute_416, out=buf56)
        del permute_416
        buf58 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf56, alias_68, buf58, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_68
        buf59 = reinterpret_tensor(buf52, (16, 64, 128), (8192, 128, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_417, reinterpret_tensor(buf58, (16, 128, 128), (16384, 128, 1), 0), out=buf59)
        del permute_417
        buf60 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (16, 128, 128), (16384, 128, 1), 0), permute_418, out=buf60)
        del permute_418
        buf61 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_48, buf55, buf61, 131072, grid=grid(131072), stream=stream0)
        del tangents_48
        buf62 = reinterpret_tensor(buf55, (128, 1024), (1024, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (128, 1024), (1024, 1), 0), permute_422, out=buf62)
        del permute_422
        buf63 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 128), (1, 1024), 0), view_665, out=buf63)
        buf64 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf61, buf64, 1024, 128, grid=grid(1024), stream=stream0)
        buf65 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_47, buf59, buf65, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_47
        buf66 = reinterpret_tensor(buf59, (128, 1024), (1024, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (128, 1024), (1024, 1), 0), permute_427, out=buf66)
        del permute_427
        buf67 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (1024, 128), (1, 1024), 0), view_665, out=buf67)
        buf68 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf65, buf68, 1024, 128, grid=grid(1024), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (128, 1024), (1024, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf60, buf69, 131072, grid=grid(131072), stream=stream0)
        buf70 = reinterpret_tensor(buf60, (128, 1024), (1024, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf69, permute_431, out=buf70)
        del permute_431
        buf71 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 128), (1, 1024), 0), view_665, out=buf71)
        del view_665
        buf72 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf69, buf72, 1024, 128, grid=grid(1024), stream=stream0)
        buf77 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf77, buf62, buf66, buf70, primals_483, mul_154, div_41, 128, 1024, grid=grid(128), stream=stream0)
        del div_41
        del primals_483
        buf75 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf76 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf62, buf66, buf70, mul_154, buf75, buf76, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_154
        buf78 = reinterpret_tensor(buf17, (128, 4096), (4096, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (128, 1024), (1024, 1), 0), permute_435, out=buf78)
        del permute_435
        buf79 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1024, 128), (1, 1024), 0), view_663, out=buf79)
        del view_663
        buf80 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf77, buf80, 1024, 128, grid=grid(1024), stream=stream0)
        buf81 = reinterpret_tensor(buf78, (1, 128, 4096), (524288, 4096, 1), 0); del buf78  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf81, le_1, 524288, grid=grid(524288), stream=stream0)
        del le_1
        buf82 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 4096), (4096, 1), 0), permute_439, out=buf82)
        del permute_439
        buf83 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (4096, 128), (1, 4096), 0), view_661, out=buf83)
        del view_661
        buf84 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf81, buf84, 4096, 128, grid=grid(4096), stream=stream0)
        buf89 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf89, buf82, primals_477, mul_152, div_42, 128, 1024, grid=grid(128), stream=stream0)
        del div_42
        del primals_477
        buf87 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf88 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf82, mul_152, buf87, buf88, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_152
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (128, 1024), (1024, 1), 0), permute_443, out=buf90)
        del permute_443
        buf91 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (1024, 128), (1, 1024), 0), view_659, out=buf91)
        del view_659
        buf92 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf89, buf92, 1024, 128, grid=grid(1024), stream=stream0)
        buf93 = reinterpret_tensor(buf66, (16, 128, 64), (8192, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_448, reinterpret_tensor(buf90, (16, 128, 64), (64, 1024, 1), 0), out=buf93)
        del permute_448
        buf94 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (16, 128, 64), (64, 1024, 1), 0), permute_449, out=buf94)
        del permute_449
        buf96 = buf56; del buf56  # reuse
        # Source Nodes: [attn_weights_89], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf94, bmm_66, amax_33, sum_34, buf96, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_33
        del bmm_66
        del sum_34
        buf97 = reinterpret_tensor(buf90, (16, 64, 128), (8192, 128, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_450, buf96, out=buf97)
        del permute_450
        buf98 = reinterpret_tensor(buf62, (16, 128, 64), (8192, 64, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf96, permute_451, out=buf98)
        del permute_451
        buf99 = reinterpret_tensor(buf69, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_46, buf93, buf99, 131072, grid=grid(131072), stream=stream0)
        del tangents_46
        buf100 = reinterpret_tensor(buf93, (128, 1024), (1024, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 1024), (1024, 1), 0), permute_455, out=buf100)
        del permute_455
        buf101 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 128), (1, 1024), 0), view_267, out=buf101)
        buf102 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf99, buf102, 1024, 128, grid=grid(1024), stream=stream0)
        buf103 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_45, buf97, buf103, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_45
        buf104 = reinterpret_tensor(buf97, (128, 1024), (1024, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), permute_460, out=buf104)
        del permute_460
        buf105 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (1024, 128), (1, 1024), 0), view_267, out=buf105)
        buf106 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf103, buf106, 1024, 128, grid=grid(1024), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf98, buf107, 131072, grid=grid(131072), stream=stream0)
        buf108 = reinterpret_tensor(buf98, (128, 1024), (1024, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf107, permute_464, out=buf108)
        del permute_464
        buf109 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 128), (1, 1024), 0), view_645, out=buf109)
        del view_645
        buf110 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf107, buf110, 1024, 128, grid=grid(1024), stream=stream0)
        buf115 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf115, buf108, primals_467, mul_149, div_43, 128, 1024, grid=grid(128), stream=stream0)
        del div_43
        del primals_467
        buf113 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf114 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf108, mul_149, buf113, buf114, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_149
        buf116 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (128, 1024), (1024, 1), 0), permute_468, out=buf116)
        del permute_468
        buf117 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1024, 128), (1, 1024), 0), view_643, out=buf117)
        del view_643
        buf118 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf115, buf118, 1024, 128, grid=grid(1024), stream=stream0)
        buf119 = reinterpret_tensor(buf107, (16, 128, 64), (8192, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_473, reinterpret_tensor(buf116, (16, 128, 64), (64, 1024, 1), 0), out=buf119)
        del permute_473
        buf120 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (16, 128, 64), (64, 1024, 1), 0), permute_474, out=buf120)
        del permute_474
        buf122 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf120, alias_71, buf122, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_71
        buf123 = reinterpret_tensor(buf116, (16, 64, 128), (8192, 128, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_475, reinterpret_tensor(buf122, (16, 128, 128), (16384, 128, 1), 0), out=buf123)
        del permute_475
        buf124 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (16, 128, 128), (16384, 128, 1), 0), permute_476, out=buf124)
        del permute_476
        buf125 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_44, buf119, buf125, 131072, grid=grid(131072), stream=stream0)
        del tangents_44
        buf126 = reinterpret_tensor(buf119, (128, 1024), (1024, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 1024), (1024, 1), 0), permute_480, out=buf126)
        del permute_480
        buf127 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1024, 128), (1, 1024), 0), view_627, out=buf127)
        buf128 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf125, buf128, 1024, 128, grid=grid(1024), stream=stream0)
        buf129 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_43, buf123, buf129, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_43
        buf130 = reinterpret_tensor(buf123, (128, 1024), (1024, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (128, 1024), (1024, 1), 0), permute_485, out=buf130)
        del permute_485
        buf131 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (1024, 128), (1, 1024), 0), view_627, out=buf131)
        buf132 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf129, buf132, 1024, 128, grid=grid(1024), stream=stream0)
        buf133 = reinterpret_tensor(buf129, (128, 1024), (1024, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf124, buf133, 131072, grid=grid(131072), stream=stream0)
        buf134 = reinterpret_tensor(buf124, (128, 1024), (1024, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf133, permute_489, out=buf134)
        del permute_489
        buf135 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (1024, 128), (1, 1024), 0), view_627, out=buf135)
        del view_627
        buf136 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf133, buf136, 1024, 128, grid=grid(1024), stream=stream0)
        buf141 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf141, buf126, buf130, buf134, primals_457, mul_146, div_44, 128, 1024, grid=grid(128), stream=stream0)
        del div_44
        del primals_457
        buf139 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf140 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf126, buf130, buf134, mul_146, buf139, buf140, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_146
        buf142 = reinterpret_tensor(buf81, (128, 4096), (4096, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (128, 1024), (1024, 1), 0), permute_493, out=buf142)
        del permute_493
        buf143 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1024, 128), (1, 1024), 0), view_625, out=buf143)
        del view_625
        buf144 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf141, buf144, 1024, 128, grid=grid(1024), stream=stream0)
        buf145 = reinterpret_tensor(buf142, (1, 128, 4096), (524288, 4096, 1), 0); del buf142  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf145, le_2, 524288, grid=grid(524288), stream=stream0)
        del le_2
        buf146 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (128, 4096), (4096, 1), 0), permute_497, out=buf146)
        del permute_497
        buf147 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 128), (1, 4096), 0), view_623, out=buf147)
        del view_623
        buf148 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf145, buf148, 4096, 128, grid=grid(4096), stream=stream0)
        buf153 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf153, buf146, primals_451, mul_144, div_45, 128, 1024, grid=grid(128), stream=stream0)
        del div_45
        del primals_451
        buf151 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf152 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf146, mul_144, buf151, buf152, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_144
        buf154 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0), permute_501, out=buf154)
        del permute_501
        buf155 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1024, 128), (1, 1024), 0), view_621, out=buf155)
        del view_621
        buf156 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf153, buf156, 1024, 128, grid=grid(1024), stream=stream0)
        buf157 = reinterpret_tensor(buf130, (16, 128, 64), (8192, 64, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_506, reinterpret_tensor(buf154, (16, 128, 64), (64, 1024, 1), 0), out=buf157)
        del permute_506
        buf158 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (16, 128, 64), (64, 1024, 1), 0), permute_507, out=buf158)
        del permute_507
        buf160 = buf120; del buf120  # reuse
        # Source Nodes: [attn_weights_83], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf158, bmm_62, amax_31, sum_32, buf160, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_31
        del bmm_62
        del sum_32
        buf161 = reinterpret_tensor(buf154, (16, 64, 128), (8192, 128, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_508, buf160, out=buf161)
        del permute_508
        buf162 = reinterpret_tensor(buf126, (16, 128, 64), (8192, 64, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf160, permute_509, out=buf162)
        del permute_509
        buf163 = reinterpret_tensor(buf133, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_42, buf157, buf163, 131072, grid=grid(131072), stream=stream0)
        del tangents_42
        buf164 = reinterpret_tensor(buf157, (128, 1024), (1024, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (128, 1024), (1024, 1), 0), permute_513, out=buf164)
        del permute_513
        buf165 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1024, 128), (1, 1024), 0), view_267, out=buf165)
        buf166 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf163, buf166, 1024, 128, grid=grid(1024), stream=stream0)
        buf167 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_41, buf161, buf167, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_41
        buf168 = reinterpret_tensor(buf161, (128, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (128, 1024), (1024, 1), 0), permute_518, out=buf168)
        del permute_518
        buf169 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1024, 128), (1, 1024), 0), view_267, out=buf169)
        buf170 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf167, buf170, 1024, 128, grid=grid(1024), stream=stream0)
        buf171 = reinterpret_tensor(buf167, (128, 1024), (1024, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf162, buf171, 131072, grid=grid(131072), stream=stream0)
        buf172 = reinterpret_tensor(buf162, (128, 1024), (1024, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf171, permute_522, out=buf172)
        del permute_522
        buf173 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 128), (1, 1024), 0), view_607, out=buf173)
        del view_607
        buf174 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf171, buf174, 1024, 128, grid=grid(1024), stream=stream0)
        buf179 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf179, buf172, primals_441, mul_141, div_46, 128, 1024, grid=grid(128), stream=stream0)
        del div_46
        del primals_441
        buf177 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf178 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf172, mul_141, buf177, buf178, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_141
        buf180 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 1024), (1024, 1), 0), permute_526, out=buf180)
        del permute_526
        buf181 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (1024, 128), (1, 1024), 0), view_605, out=buf181)
        del view_605
        buf182 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf179, buf182, 1024, 128, grid=grid(1024), stream=stream0)
        buf183 = reinterpret_tensor(buf171, (16, 128, 64), (8192, 64, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_531, reinterpret_tensor(buf180, (16, 128, 64), (64, 1024, 1), 0), out=buf183)
        del permute_531
        buf184 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (16, 128, 64), (64, 1024, 1), 0), permute_532, out=buf184)
        del permute_532
        buf186 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf184, alias_74, buf186, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_74
        buf187 = reinterpret_tensor(buf180, (16, 64, 128), (8192, 128, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_533, reinterpret_tensor(buf186, (16, 128, 128), (16384, 128, 1), 0), out=buf187)
        del permute_533
        buf188 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (16, 128, 128), (16384, 128, 1), 0), permute_534, out=buf188)
        del permute_534
        buf189 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_40, buf183, buf189, 131072, grid=grid(131072), stream=stream0)
        del tangents_40
        buf190 = reinterpret_tensor(buf183, (128, 1024), (1024, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (128, 1024), (1024, 1), 0), permute_538, out=buf190)
        del permute_538
        buf191 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (1024, 128), (1, 1024), 0), view_589, out=buf191)
        buf192 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf189, buf192, 1024, 128, grid=grid(1024), stream=stream0)
        buf193 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_39, buf187, buf193, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_39
        buf194 = reinterpret_tensor(buf187, (128, 1024), (1024, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (128, 1024), (1024, 1), 0), permute_543, out=buf194)
        del permute_543
        buf195 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1024, 128), (1, 1024), 0), view_589, out=buf195)
        buf196 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf193, buf196, 1024, 128, grid=grid(1024), stream=stream0)
        buf197 = reinterpret_tensor(buf193, (128, 1024), (1024, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf188, buf197, 131072, grid=grid(131072), stream=stream0)
        buf198 = reinterpret_tensor(buf188, (128, 1024), (1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf197, permute_547, out=buf198)
        del permute_547
        buf199 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1024, 128), (1, 1024), 0), view_589, out=buf199)
        del view_589
        buf200 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf197, buf200, 1024, 128, grid=grid(1024), stream=stream0)
        buf205 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf205, buf190, buf194, buf198, primals_431, mul_138, div_47, 128, 1024, grid=grid(128), stream=stream0)
        del div_47
        del primals_431
        buf203 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf204 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf190, buf194, buf198, mul_138, buf203, buf204, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_138
        buf206 = reinterpret_tensor(buf145, (128, 4096), (4096, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (128, 1024), (1024, 1), 0), permute_551, out=buf206)
        del permute_551
        buf207 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (1024, 128), (1, 1024), 0), view_587, out=buf207)
        del view_587
        buf208 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf205, buf208, 1024, 128, grid=grid(1024), stream=stream0)
        buf209 = reinterpret_tensor(buf206, (1, 128, 4096), (524288, 4096, 1), 0); del buf206  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf209, le_3, 524288, grid=grid(524288), stream=stream0)
        del le_3
        buf210 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (128, 4096), (4096, 1), 0), permute_555, out=buf210)
        del permute_555
        buf211 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (4096, 128), (1, 4096), 0), view_585, out=buf211)
        del view_585
        buf212 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf209, buf212, 4096, 128, grid=grid(4096), stream=stream0)
        buf217 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf217, buf210, primals_425, mul_136, div_48, 128, 1024, grid=grid(128), stream=stream0)
        del div_48
        del primals_425
        buf215 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf216 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf210, mul_136, buf215, buf216, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_136
        buf218 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 1024), (1024, 1), 0), permute_559, out=buf218)
        del permute_559
        buf219 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1024, 128), (1, 1024), 0), view_583, out=buf219)
        del view_583
        buf220 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf217, buf220, 1024, 128, grid=grid(1024), stream=stream0)
        buf221 = reinterpret_tensor(buf194, (16, 128, 64), (8192, 64, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_564, reinterpret_tensor(buf218, (16, 128, 64), (64, 1024, 1), 0), out=buf221)
        del permute_564
        buf222 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf218, (16, 128, 64), (64, 1024, 1), 0), permute_565, out=buf222)
        del permute_565
        buf224 = buf184; del buf184  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf222, bmm_58, amax_29, sum_30, buf224, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_29
        del bmm_58
        del sum_30
        buf225 = reinterpret_tensor(buf218, (16, 64, 128), (8192, 128, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_566, buf224, out=buf225)
        del permute_566
        buf226 = reinterpret_tensor(buf190, (16, 128, 64), (8192, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf224, permute_567, out=buf226)
        del permute_567
        buf227 = reinterpret_tensor(buf197, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_38, buf221, buf227, 131072, grid=grid(131072), stream=stream0)
        del tangents_38
        buf228 = reinterpret_tensor(buf221, (128, 1024), (1024, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (128, 1024), (1024, 1), 0), permute_571, out=buf228)
        del permute_571
        buf229 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1024, 128), (1, 1024), 0), view_267, out=buf229)
        buf230 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf227, buf230, 1024, 128, grid=grid(1024), stream=stream0)
        buf231 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_37, buf225, buf231, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_37
        buf232 = reinterpret_tensor(buf225, (128, 1024), (1024, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (128, 1024), (1024, 1), 0), permute_576, out=buf232)
        del permute_576
        buf233 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 128), (1, 1024), 0), view_267, out=buf233)
        buf234 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf231, buf234, 1024, 128, grid=grid(1024), stream=stream0)
        buf236 = reinterpret_tensor(buf231, (128, 1024), (1024, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf226, buf236, 131072, grid=grid(131072), stream=stream0)
        buf237 = reinterpret_tensor(buf226, (128, 1024), (1024, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf236, permute_580, out=buf237)
        del permute_580
        buf244 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf244, buf237, primals_415, mul_133, div_49, 128, 1024, grid=grid(128), stream=stream0)
        del div_49
        del primals_415
        buf245 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (128, 1024), (1024, 1), 0), permute_584, out=buf245)
        del permute_584
        buf248 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_589, reinterpret_tensor(buf245, (16, 128, 64), (64, 1024, 1), 0), out=buf248)
        del permute_589
        buf254 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_36, buf248, buf254, 131072, grid=grid(131072), stream=stream0)
        del tangents_36
        buf255 = reinterpret_tensor(buf248, (128, 1024), (1024, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (128, 1024), (1024, 1), 0), permute_596, out=buf255)
        del permute_596
        buf249 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (16, 128, 64), (64, 1024, 1), 0), permute_590, out=buf249)
        del permute_590
        buf251 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf249, alias_77, buf251, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_77
        buf252 = reinterpret_tensor(buf245, (16, 64, 128), (8192, 128, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_591, reinterpret_tensor(buf251, (16, 128, 128), (16384, 128, 1), 0), out=buf252)
        del permute_591
        buf258 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_35, buf252, buf258, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_35
        buf259 = reinterpret_tensor(buf252, (128, 1024), (1024, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (128, 1024), (1024, 1), 0), permute_601, out=buf259)
        del permute_601
        buf253 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf251, (16, 128, 128), (16384, 128, 1), 0), permute_592, out=buf253)
        del permute_592
        buf262 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf253, buf262, 131072, grid=grid(131072), stream=stream0)
        buf263 = reinterpret_tensor(buf253, (128, 1024), (1024, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf262, permute_605, out=buf263)
        del permute_605
        buf270 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf255, buf259, buf263, primals_405, mul_130, buf244, div_50, buf270, 128, 1024, grid=grid(128), stream=stream0)
        del div_50
        del primals_405
        buf271 = reinterpret_tensor(buf209, (128, 4096), (4096, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (128, 1024), (1024, 1), 0), permute_609, out=buf271)
        del permute_609
        buf274 = reinterpret_tensor(buf271, (1, 128, 4096), (524288, 4096, 1), 0); del buf271  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf274, le_4, 524288, grid=grid(524288), stream=stream0)
        del le_4
        buf275 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 4096), (4096, 1), 0), permute_613, out=buf275)
        del permute_613
        buf282 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf275, primals_399, mul_128, buf270, div_51, buf282, 128, 1024, grid=grid(128), stream=stream0)
        del div_51
        del primals_399
        buf283 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (128, 1024), (1024, 1), 0), permute_617, out=buf283)
        del permute_617
        buf286 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_622, reinterpret_tensor(buf283, (16, 128, 64), (64, 1024, 1), 0), out=buf286)
        del permute_622
        buf292 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_34, buf286, buf292, 131072, grid=grid(131072), stream=stream0)
        del tangents_34
        buf293 = reinterpret_tensor(buf286, (128, 1024), (1024, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 1024), (1024, 1), 0), permute_629, out=buf293)
        del permute_629
        buf287 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (16, 128, 64), (64, 1024, 1), 0), permute_623, out=buf287)
        del permute_623
        buf289 = buf249; del buf249  # reuse
        # Source Nodes: [attn_weights_71], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf287, bmm_54, amax_27, sum_28, buf289, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_27
        del bmm_54
        del sum_28
        buf290 = reinterpret_tensor(buf283, (16, 64, 128), (8192, 128, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_624, buf289, out=buf290)
        del permute_624
        buf296 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_33, buf290, buf296, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_33
        buf297 = reinterpret_tensor(buf290, (128, 1024), (1024, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (128, 1024), (1024, 1), 0), permute_634, out=buf297)
        del permute_634
        buf291 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf289, permute_625, out=buf291)
        del permute_625
        buf300 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf291, buf300, 131072, grid=grid(131072), stream=stream0)
        buf301 = reinterpret_tensor(buf291, (128, 1024), (1024, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf300, permute_638, out=buf301)
        del permute_638
        buf308 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf301, primals_389, mul_125, buf282, div_52, buf308, 128, 1024, grid=grid(128), stream=stream0)
        del div_52
        del primals_389
        buf309 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (128, 1024), (1024, 1), 0), permute_642, out=buf309)
        del permute_642
        buf312 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_647, reinterpret_tensor(buf309, (16, 128, 64), (64, 1024, 1), 0), out=buf312)
        del permute_647
        buf318 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_32, buf312, buf318, 131072, grid=grid(131072), stream=stream0)
        del tangents_32
        buf319 = reinterpret_tensor(buf312, (128, 1024), (1024, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf318, (128, 1024), (1024, 1), 0), permute_654, out=buf319)
        del permute_654
        buf313 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (16, 128, 64), (64, 1024, 1), 0), permute_648, out=buf313)
        del permute_648
        buf315 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf313, alias_80, buf315, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_80
        buf316 = reinterpret_tensor(buf309, (16, 64, 128), (8192, 128, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_649, reinterpret_tensor(buf315, (16, 128, 128), (16384, 128, 1), 0), out=buf316)
        del permute_649
        buf322 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_31, buf316, buf322, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_31
        buf323 = reinterpret_tensor(buf316, (128, 1024), (1024, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 1024), (1024, 1), 0), permute_659, out=buf323)
        del permute_659
        buf317 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (16, 128, 128), (16384, 128, 1), 0), permute_650, out=buf317)
        del permute_650
        buf326 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf317, buf326, 131072, grid=grid(131072), stream=stream0)
        buf327 = reinterpret_tensor(buf317, (128, 1024), (1024, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf326, permute_663, out=buf327)
        del permute_663
        buf334 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf319, buf323, buf327, primals_379, mul_122, buf308, div_53, buf334, 128, 1024, grid=grid(128), stream=stream0)
        del div_53
        del primals_379
        buf335 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (128, 1024), (1024, 1), 0), permute_667, out=buf335)
        del permute_667
        buf338 = reinterpret_tensor(buf335, (1, 128, 4096), (524288, 4096, 1), 0); del buf335  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf338, le_5, 524288, grid=grid(524288), stream=stream0)
        del le_5
        buf339 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (128, 4096), (4096, 1), 0), permute_671, out=buf339)
        del permute_671
        buf346 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf339, primals_373, mul_120, buf334, div_54, buf346, 128, 1024, grid=grid(128), stream=stream0)
        del div_54
        del primals_373
        buf347 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (128, 1024), (1024, 1), 0), permute_675, out=buf347)
        del permute_675
        buf350 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_680, reinterpret_tensor(buf347, (16, 128, 64), (64, 1024, 1), 0), out=buf350)
        del permute_680
        buf356 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_30, buf350, buf356, 131072, grid=grid(131072), stream=stream0)
        del tangents_30
        buf357 = reinterpret_tensor(buf350, (128, 1024), (1024, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 1024), (1024, 1), 0), permute_687, out=buf357)
        del permute_687
        buf351 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf347, (16, 128, 64), (64, 1024, 1), 0), permute_681, out=buf351)
        del permute_681
        buf353 = buf313; del buf313  # reuse
        # Source Nodes: [attn_weights_65], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf351, bmm_50, amax_25, sum_26, buf353, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_25
        del bmm_50
        del sum_26
        buf354 = reinterpret_tensor(buf347, (16, 64, 128), (8192, 128, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_682, buf353, out=buf354)
        del permute_682
        buf360 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_29, buf354, buf360, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_29
        buf361 = reinterpret_tensor(buf354, (128, 1024), (1024, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (128, 1024), (1024, 1), 0), permute_692, out=buf361)
        del permute_692
        buf355 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf353, permute_683, out=buf355)
        del permute_683
        buf364 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf355, buf364, 131072, grid=grid(131072), stream=stream0)
        buf365 = reinterpret_tensor(buf355, (128, 1024), (1024, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf364, permute_696, out=buf365)
        del permute_696
        buf372 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf365, primals_363, mul_117, buf346, div_55, buf372, 128, 1024, grid=grid(128), stream=stream0)
        del div_55
        del primals_363
        buf373 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (128, 1024), (1024, 1), 0), permute_700, out=buf373)
        del permute_700
        buf376 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_705, reinterpret_tensor(buf373, (16, 128, 64), (64, 1024, 1), 0), out=buf376)
        del permute_705
        buf382 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_28, buf376, buf382, 131072, grid=grid(131072), stream=stream0)
        del tangents_28
        buf383 = reinterpret_tensor(buf376, (128, 1024), (1024, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (128, 1024), (1024, 1), 0), permute_712, out=buf383)
        del permute_712
        buf377 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf373, (16, 128, 64), (64, 1024, 1), 0), permute_706, out=buf377)
        del permute_706
        buf379 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf377, alias_83, buf379, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_83
        buf380 = reinterpret_tensor(buf373, (16, 64, 128), (8192, 128, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_707, reinterpret_tensor(buf379, (16, 128, 128), (16384, 128, 1), 0), out=buf380)
        del permute_707
        buf386 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_27, buf380, buf386, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_27
        buf387 = reinterpret_tensor(buf380, (128, 1024), (1024, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (128, 1024), (1024, 1), 0), permute_717, out=buf387)
        del permute_717
        buf381 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf379, (16, 128, 128), (16384, 128, 1), 0), permute_708, out=buf381)
        del permute_708
        buf390 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf381, buf390, 131072, grid=grid(131072), stream=stream0)
        buf391 = reinterpret_tensor(buf381, (128, 1024), (1024, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf390, permute_721, out=buf391)
        del permute_721
        buf398 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf383, buf387, buf391, primals_353, mul_114, buf372, div_56, buf398, 128, 1024, grid=grid(128), stream=stream0)
        del div_56
        del primals_353
        buf399 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 1024), (1024, 1), 0), permute_725, out=buf399)
        del permute_725
        buf402 = reinterpret_tensor(buf399, (1, 128, 4096), (524288, 4096, 1), 0); del buf399  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf402, le_6, 524288, grid=grid(524288), stream=stream0)
        del le_6
        buf403 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (128, 4096), (4096, 1), 0), permute_729, out=buf403)
        del permute_729
        buf410 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf403, primals_347, mul_112, buf398, div_57, buf410, 128, 1024, grid=grid(128), stream=stream0)
        del div_57
        del primals_347
        buf411 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (128, 1024), (1024, 1), 0), permute_733, out=buf411)
        del permute_733
        buf414 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_738, reinterpret_tensor(buf411, (16, 128, 64), (64, 1024, 1), 0), out=buf414)
        del permute_738
        buf420 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_26, buf414, buf420, 131072, grid=grid(131072), stream=stream0)
        del tangents_26
        buf421 = reinterpret_tensor(buf414, (128, 1024), (1024, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (128, 1024), (1024, 1), 0), permute_745, out=buf421)
        del permute_745
        buf415 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf411, (16, 128, 64), (64, 1024, 1), 0), permute_739, out=buf415)
        del permute_739
        buf417 = buf377; del buf377  # reuse
        # Source Nodes: [attn_weights_59], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf415, bmm_46, amax_23, sum_24, buf417, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_23
        del bmm_46
        del sum_24
        buf418 = reinterpret_tensor(buf411, (16, 64, 128), (8192, 128, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_740, buf417, out=buf418)
        del permute_740
        buf424 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_25, buf418, buf424, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_25
        buf425 = reinterpret_tensor(buf418, (128, 1024), (1024, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (128, 1024), (1024, 1), 0), permute_750, out=buf425)
        del permute_750
        buf419 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf417, permute_741, out=buf419)
        del permute_741
        buf428 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf419, buf428, 131072, grid=grid(131072), stream=stream0)
        buf429 = reinterpret_tensor(buf419, (128, 1024), (1024, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf428, permute_754, out=buf429)
        del permute_754
        buf436 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf429, primals_337, mul_109, buf410, div_58, buf436, 128, 1024, grid=grid(128), stream=stream0)
        del div_58
        del primals_337
        buf437 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf436, (128, 1024), (1024, 1), 0), permute_758, out=buf437)
        del permute_758
        buf440 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_763, reinterpret_tensor(buf437, (16, 128, 64), (64, 1024, 1), 0), out=buf440)
        del permute_763
        buf446 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_24, buf440, buf446, 131072, grid=grid(131072), stream=stream0)
        del tangents_24
        buf447 = reinterpret_tensor(buf440, (128, 1024), (1024, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 1024), (1024, 1), 0), permute_770, out=buf447)
        del permute_770
        buf441 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf437, (16, 128, 64), (64, 1024, 1), 0), permute_764, out=buf441)
        del permute_764
        buf443 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf441, alias_86, buf443, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_86
        buf444 = reinterpret_tensor(buf437, (16, 64, 128), (8192, 128, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_765, reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0), out=buf444)
        del permute_765
        buf450 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_23, buf444, buf450, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_23
        buf451 = reinterpret_tensor(buf444, (128, 1024), (1024, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (128, 1024), (1024, 1), 0), permute_775, out=buf451)
        del permute_775
        buf445 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0), permute_766, out=buf445)
        del permute_766
        buf454 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf445, buf454, 131072, grid=grid(131072), stream=stream0)
        buf455 = reinterpret_tensor(buf445, (128, 1024), (1024, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf454, permute_779, out=buf455)
        del permute_779
        buf462 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf447, buf451, buf455, primals_327, mul_106, buf436, div_59, buf462, 128, 1024, grid=grid(128), stream=stream0)
        del div_59
        del primals_327
        buf463 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (128, 1024), (1024, 1), 0), permute_783, out=buf463)
        del permute_783
        buf466 = reinterpret_tensor(buf463, (1, 128, 4096), (524288, 4096, 1), 0); del buf463  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf466, le_7, 524288, grid=grid(524288), stream=stream0)
        del le_7
        buf467 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (128, 4096), (4096, 1), 0), permute_787, out=buf467)
        del permute_787
        buf474 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf467, primals_321, mul_104, buf462, div_60, buf474, 128, 1024, grid=grid(128), stream=stream0)
        del div_60
        del primals_321
        buf475 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 1024), (1024, 1), 0), permute_791, out=buf475)
        del permute_791
        buf478 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_796, reinterpret_tensor(buf475, (16, 128, 64), (64, 1024, 1), 0), out=buf478)
        del permute_796
        buf484 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_22, buf478, buf484, 131072, grid=grid(131072), stream=stream0)
        del tangents_22
        buf485 = reinterpret_tensor(buf478, (128, 1024), (1024, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (128, 1024), (1024, 1), 0), permute_803, out=buf485)
        del permute_803
        buf479 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf475, (16, 128, 64), (64, 1024, 1), 0), permute_797, out=buf479)
        del permute_797
        buf481 = buf441; del buf441  # reuse
        # Source Nodes: [attn_weights_53], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf479, bmm_42, amax_21, sum_22, buf481, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_21
        del bmm_42
        del sum_22
        buf482 = reinterpret_tensor(buf475, (16, 64, 128), (8192, 128, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_798, buf481, out=buf482)
        del permute_798
        buf488 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_21, buf482, buf488, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_21
        buf489 = reinterpret_tensor(buf482, (128, 1024), (1024, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (128, 1024), (1024, 1), 0), permute_808, out=buf489)
        del permute_808
        buf483 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf481, permute_799, out=buf483)
        del permute_799
        buf493 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf483, buf493, 131072, grid=grid(131072), stream=stream0)
        buf494 = reinterpret_tensor(buf483, (128, 1024), (1024, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf493, permute_812, out=buf494)
        del permute_812
        buf501 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf494, primals_311, mul_101, buf474, div_61, buf501, 128, 1024, grid=grid(128), stream=stream0)
        del div_61
        del primals_311
        buf502 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (128, 1024), (1024, 1), 0), permute_816, out=buf502)
        del permute_816
        buf505 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_821, reinterpret_tensor(buf502, (16, 128, 64), (64, 1024, 1), 0), out=buf505)
        del permute_821
        buf511 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_20, buf505, buf511, 131072, grid=grid(131072), stream=stream0)
        del tangents_20
        buf512 = reinterpret_tensor(buf505, (128, 1024), (1024, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (128, 1024), (1024, 1), 0), permute_828, out=buf512)
        del permute_828
        buf506 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (16, 128, 64), (64, 1024, 1), 0), permute_822, out=buf506)
        del permute_822
        buf508 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf506, alias_89, buf508, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_89
        buf509 = reinterpret_tensor(buf502, (16, 64, 128), (8192, 128, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_823, reinterpret_tensor(buf508, (16, 128, 128), (16384, 128, 1), 0), out=buf509)
        del permute_823
        buf515 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_19, buf509, buf515, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_19
        buf516 = reinterpret_tensor(buf509, (128, 1024), (1024, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (128, 1024), (1024, 1), 0), permute_833, out=buf516)
        del permute_833
        buf510 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf508, (16, 128, 128), (16384, 128, 1), 0), permute_824, out=buf510)
        del permute_824
        buf519 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf510, buf519, 131072, grid=grid(131072), stream=stream0)
        buf520 = reinterpret_tensor(buf510, (128, 1024), (1024, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf519, permute_837, out=buf520)
        del permute_837
        buf527 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf512, buf516, buf520, primals_301, mul_98, buf501, div_62, buf527, 128, 1024, grid=grid(128), stream=stream0)
        del div_62
        del primals_301
        buf528 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (128, 1024), (1024, 1), 0), permute_841, out=buf528)
        del permute_841
        buf531 = reinterpret_tensor(buf528, (1, 128, 4096), (524288, 4096, 1), 0); del buf528  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf531, le_8, 524288, grid=grid(524288), stream=stream0)
        del le_8
        buf532 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (128, 4096), (4096, 1), 0), permute_845, out=buf532)
        del permute_845
        buf539 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf532, primals_295, mul_96, buf527, div_63, buf539, 128, 1024, grid=grid(128), stream=stream0)
        del div_63
        del primals_295
        buf540 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf539, (128, 1024), (1024, 1), 0), permute_849, out=buf540)
        del permute_849
        buf543 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_854, reinterpret_tensor(buf540, (16, 128, 64), (64, 1024, 1), 0), out=buf543)
        del permute_854
        buf549 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_18, buf543, buf549, 131072, grid=grid(131072), stream=stream0)
        del tangents_18
        buf550 = reinterpret_tensor(buf543, (128, 1024), (1024, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (128, 1024), (1024, 1), 0), permute_861, out=buf550)
        del permute_861
        buf544 = buf508; del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf540, (16, 128, 64), (64, 1024, 1), 0), permute_855, out=buf544)
        del permute_855
        buf546 = buf506; del buf506  # reuse
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf544, bmm_38, amax_19, sum_20, buf546, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_19
        del bmm_38
        del sum_20
        buf547 = reinterpret_tensor(buf540, (16, 64, 128), (8192, 128, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_856, buf546, out=buf547)
        del permute_856
        buf553 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_17, buf547, buf553, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_17
        buf554 = reinterpret_tensor(buf547, (128, 1024), (1024, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0), permute_866, out=buf554)
        del permute_866
        buf548 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf546, permute_857, out=buf548)
        del permute_857
        buf557 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf548, buf557, 131072, grid=grid(131072), stream=stream0)
        buf558 = reinterpret_tensor(buf548, (128, 1024), (1024, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf557, permute_870, out=buf558)
        del permute_870
        buf565 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf558, primals_285, mul_93, buf539, div_64, buf565, 128, 1024, grid=grid(128), stream=stream0)
        del div_64
        del primals_285
        buf566 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (128, 1024), (1024, 1), 0), permute_874, out=buf566)
        del permute_874
        buf569 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_879, reinterpret_tensor(buf566, (16, 128, 64), (64, 1024, 1), 0), out=buf569)
        del permute_879
        buf575 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_16, buf569, buf575, 131072, grid=grid(131072), stream=stream0)
        del tangents_16
        buf576 = reinterpret_tensor(buf569, (128, 1024), (1024, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (128, 1024), (1024, 1), 0), permute_886, out=buf576)
        del permute_886
        buf570 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf566, (16, 128, 64), (64, 1024, 1), 0), permute_880, out=buf570)
        del permute_880
        buf572 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf570, alias_92, buf572, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_92
        buf573 = reinterpret_tensor(buf566, (16, 64, 128), (8192, 128, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_881, reinterpret_tensor(buf572, (16, 128, 128), (16384, 128, 1), 0), out=buf573)
        del permute_881
        buf579 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_15, buf573, buf579, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_15
        buf580 = reinterpret_tensor(buf573, (128, 1024), (1024, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (128, 1024), (1024, 1), 0), permute_891, out=buf580)
        del permute_891
        buf574 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf572, (16, 128, 128), (16384, 128, 1), 0), permute_882, out=buf574)
        del permute_882
        buf583 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf574, buf583, 131072, grid=grid(131072), stream=stream0)
        buf584 = reinterpret_tensor(buf574, (128, 1024), (1024, 1), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf583, permute_895, out=buf584)
        del permute_895
        buf591 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf576, buf580, buf584, primals_275, mul_90, buf565, div_65, buf591, 128, 1024, grid=grid(128), stream=stream0)
        del div_65
        del primals_275
        buf592 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (128, 1024), (1024, 1), 0), permute_899, out=buf592)
        del permute_899
        buf595 = reinterpret_tensor(buf592, (1, 128, 4096), (524288, 4096, 1), 0); del buf592  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf595, le_9, 524288, grid=grid(524288), stream=stream0)
        del le_9
        buf596 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (128, 4096), (4096, 1), 0), permute_903, out=buf596)
        del permute_903
        buf603 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf596, primals_269, mul_88, buf591, div_66, buf603, 128, 1024, grid=grid(128), stream=stream0)
        del div_66
        del primals_269
        buf604 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (128, 1024), (1024, 1), 0), permute_907, out=buf604)
        del permute_907
        buf607 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_912, reinterpret_tensor(buf604, (16, 128, 64), (64, 1024, 1), 0), out=buf607)
        del permute_912
        buf613 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_14, buf607, buf613, 131072, grid=grid(131072), stream=stream0)
        del tangents_14
        buf614 = reinterpret_tensor(buf607, (128, 1024), (1024, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (128, 1024), (1024, 1), 0), permute_919, out=buf614)
        del permute_919
        buf608 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf604, (16, 128, 64), (64, 1024, 1), 0), permute_913, out=buf608)
        del permute_913
        buf610 = buf570; del buf570  # reuse
        # Source Nodes: [attn_weights_41], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf608, bmm_34, amax_17, sum_18, buf610, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_17
        del bmm_34
        del sum_18
        buf611 = reinterpret_tensor(buf604, (16, 64, 128), (8192, 128, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_914, buf610, out=buf611)
        del permute_914
        buf617 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_13, buf611, buf617, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_13
        buf618 = reinterpret_tensor(buf611, (128, 1024), (1024, 1), 0); del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (128, 1024), (1024, 1), 0), permute_924, out=buf618)
        del permute_924
        buf612 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf610, permute_915, out=buf612)
        del permute_915
        buf621 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf612, buf621, 131072, grid=grid(131072), stream=stream0)
        buf622 = reinterpret_tensor(buf612, (128, 1024), (1024, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf621, permute_928, out=buf622)
        del permute_928
        buf629 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf622, primals_259, mul_85, buf603, div_67, buf629, 128, 1024, grid=grid(128), stream=stream0)
        del div_67
        del primals_259
        buf630 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (128, 1024), (1024, 1), 0), permute_932, out=buf630)
        del permute_932
        buf633 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_937, reinterpret_tensor(buf630, (16, 128, 64), (64, 1024, 1), 0), out=buf633)
        del permute_937
        buf639 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_12, buf633, buf639, 131072, grid=grid(131072), stream=stream0)
        del tangents_12
        buf640 = reinterpret_tensor(buf633, (128, 1024), (1024, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf639, (128, 1024), (1024, 1), 0), permute_944, out=buf640)
        del permute_944
        buf634 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf630, (16, 128, 64), (64, 1024, 1), 0), permute_938, out=buf634)
        del permute_938
        buf636 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf634, alias_95, buf636, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_95
        buf637 = reinterpret_tensor(buf630, (16, 64, 128), (8192, 128, 1), 0); del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_939, reinterpret_tensor(buf636, (16, 128, 128), (16384, 128, 1), 0), out=buf637)
        del permute_939
        buf643 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_11, buf637, buf643, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_11
        buf644 = reinterpret_tensor(buf637, (128, 1024), (1024, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf643, (128, 1024), (1024, 1), 0), permute_949, out=buf644)
        del permute_949
        buf638 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf636, (16, 128, 128), (16384, 128, 1), 0), permute_940, out=buf638)
        del permute_940
        buf647 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf638, buf647, 131072, grid=grid(131072), stream=stream0)
        buf648 = reinterpret_tensor(buf638, (128, 1024), (1024, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf647, permute_953, out=buf648)
        del permute_953
        buf655 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf640, buf644, buf648, primals_249, mul_82, buf629, div_68, buf655, 128, 1024, grid=grid(128), stream=stream0)
        del div_68
        del primals_249
        buf656 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (128, 1024), (1024, 1), 0), permute_957, out=buf656)
        del permute_957
        buf659 = reinterpret_tensor(buf656, (1, 128, 4096), (524288, 4096, 1), 0); del buf656  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf659, le_10, 524288, grid=grid(524288), stream=stream0)
        del le_10
        buf660 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (128, 4096), (4096, 1), 0), permute_961, out=buf660)
        del permute_961
        buf667 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf660, primals_243, mul_80, buf655, div_69, buf667, 128, 1024, grid=grid(128), stream=stream0)
        del div_69
        del primals_243
        buf668 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (128, 1024), (1024, 1), 0), permute_965, out=buf668)
        del permute_965
        buf671 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_970, reinterpret_tensor(buf668, (16, 128, 64), (64, 1024, 1), 0), out=buf671)
        del permute_970
        buf677 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_10, buf671, buf677, 131072, grid=grid(131072), stream=stream0)
        del tangents_10
        buf678 = reinterpret_tensor(buf671, (128, 1024), (1024, 1), 0); del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (128, 1024), (1024, 1), 0), permute_977, out=buf678)
        del permute_977
        buf672 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf668, (16, 128, 64), (64, 1024, 1), 0), permute_971, out=buf672)
        del permute_971
        buf674 = buf634; del buf634  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf672, bmm_30, amax_15, sum_16, buf674, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_15
        del bmm_30
        del sum_16
        buf675 = reinterpret_tensor(buf668, (16, 64, 128), (8192, 128, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_972, buf674, out=buf675)
        del permute_972
        buf681 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_9, buf675, buf681, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_9
        buf682 = reinterpret_tensor(buf675, (128, 1024), (1024, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf681, (128, 1024), (1024, 1), 0), permute_982, out=buf682)
        del permute_982
        buf676 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf674, permute_973, out=buf676)
        del permute_973
        buf685 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf676, buf685, 131072, grid=grid(131072), stream=stream0)
        buf686 = reinterpret_tensor(buf676, (128, 1024), (1024, 1), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf685, permute_986, out=buf686)
        del permute_986
        buf693 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf686, primals_233, mul_77, buf667, div_70, buf693, 128, 1024, grid=grid(128), stream=stream0)
        del div_70
        del primals_233
        buf694 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (128, 1024), (1024, 1), 0), permute_990, out=buf694)
        del permute_990
        buf697 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_995, reinterpret_tensor(buf694, (16, 128, 64), (64, 1024, 1), 0), out=buf697)
        del permute_995
        buf703 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_8, buf697, buf703, 131072, grid=grid(131072), stream=stream0)
        del tangents_8
        buf704 = reinterpret_tensor(buf697, (128, 1024), (1024, 1), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf703, (128, 1024), (1024, 1), 0), permute_1002, out=buf704)
        del permute_1002
        buf698 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf694, (16, 128, 64), (64, 1024, 1), 0), permute_996, out=buf698)
        del permute_996
        buf700 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf698, alias_98, buf700, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_98
        buf701 = reinterpret_tensor(buf694, (16, 64, 128), (8192, 128, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_997, reinterpret_tensor(buf700, (16, 128, 128), (16384, 128, 1), 0), out=buf701)
        del permute_997
        buf707 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_7, buf701, buf707, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_7
        buf708 = reinterpret_tensor(buf701, (128, 1024), (1024, 1), 0); del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (128, 1024), (1024, 1), 0), permute_1007, out=buf708)
        del permute_1007
        buf702 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf700, (16, 128, 128), (16384, 128, 1), 0), permute_998, out=buf702)
        del permute_998
        buf711 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf702, buf711, 131072, grid=grid(131072), stream=stream0)
        buf712 = reinterpret_tensor(buf702, (128, 1024), (1024, 1), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf711, permute_1011, out=buf712)
        del permute_1011
        buf719 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf704, buf708, buf712, primals_223, mul_74, buf693, div_71, buf719, 128, 1024, grid=grid(128), stream=stream0)
        del div_71
        del primals_223
        buf720 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (128, 1024), (1024, 1), 0), permute_1015, out=buf720)
        del permute_1015
        buf723 = reinterpret_tensor(buf720, (1, 128, 4096), (524288, 4096, 1), 0); del buf720  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf723, le_11, 524288, grid=grid(524288), stream=stream0)
        del le_11
        buf724 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (128, 4096), (4096, 1), 0), permute_1019, out=buf724)
        del permute_1019
        buf731 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf724, primals_217, mul_72, buf719, div_72, buf731, 128, 1024, grid=grid(128), stream=stream0)
        del div_72
        del primals_217
        buf732 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (128, 1024), (1024, 1), 0), permute_1023, out=buf732)
        del permute_1023
        buf735 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1028, reinterpret_tensor(buf732, (16, 128, 64), (64, 1024, 1), 0), out=buf735)
        del permute_1028
        buf741 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_6, buf735, buf741, 131072, grid=grid(131072), stream=stream0)
        del tangents_6
        buf742 = reinterpret_tensor(buf735, (128, 1024), (1024, 1), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (128, 1024), (1024, 1), 0), permute_1035, out=buf742)
        del permute_1035
        buf736 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf732, (16, 128, 64), (64, 1024, 1), 0), permute_1029, out=buf736)
        del permute_1029
        buf738 = buf698; del buf698  # reuse
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf736, bmm_26, amax_13, sum_14, buf738, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_13
        del bmm_26
        del sum_14
        buf739 = reinterpret_tensor(buf732, (16, 64, 128), (8192, 128, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1030, buf738, out=buf739)
        del permute_1030
        buf745 = empty((1, 128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_5, buf739, buf745, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_5
        buf746 = reinterpret_tensor(buf739, (128, 1024), (1024, 1), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf745, (128, 1024), (1024, 1), 0), permute_1040, out=buf746)
        del permute_1040
        buf235 = reinterpret_tensor(buf100, (1, 128, 1024), (131072, 1024, 1), 0); del buf100  # reuse
        buf492 = buf235; del buf235  # reuse
        buf749 = buf492; del buf492  # reuse
        buf791 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_21.run(buf749, tangents_51, buf36, buf40, buf104, buf164, buf168, buf228, buf232, buf293, buf297, buf357, buf361, buf421, buf425, buf485, buf489, buf550, buf554, buf614, buf618, buf678, buf682, buf742, buf746, primals_194, mul_62, div_75, buf791, 128, 1024, grid=grid(128), stream=stream0)
        del buf104
        del buf164
        del buf168
        del buf228
        del buf232
        del buf293
        del buf297
        del buf357
        del buf36
        del buf361
        del buf40
        del buf421
        del buf425
        del buf485
        del buf489
        del buf550
        del buf554
        del buf614
        del buf618
        del buf678
        del buf682
        del buf742
        del buf746
        del div_75
        del primals_194
        del tangents_51
        buf238 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (1024, 128), (1, 1024), 0), view_569, out=buf238)
        del view_569
        buf239 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf236, buf239, 1024, 128, grid=grid(1024), stream=stream0)
        del buf236
        buf242 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf243 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf237, mul_133, buf242, buf243, 1024, 128, grid=grid(1024), stream=stream0)
        del buf237
        del mul_133
        buf246 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (1024, 128), (1, 1024), 0), view_567, out=buf246)
        del view_567
        buf247 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf244, buf247, 1024, 128, grid=grid(1024), stream=stream0)
        del buf244
        buf256 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (1024, 128), (1, 1024), 0), view_551, out=buf256)
        buf257 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf254, buf257, 1024, 128, grid=grid(1024), stream=stream0)
        del buf254
        buf260 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (1024, 128), (1, 1024), 0), view_551, out=buf260)
        buf261 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf258, buf261, 1024, 128, grid=grid(1024), stream=stream0)
        del buf258
        buf264 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 128), (1, 1024), 0), view_551, out=buf264)
        del view_551
        buf265 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf262, buf265, 1024, 128, grid=grid(1024), stream=stream0)
        del buf262
        buf268 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf269 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf255, buf259, buf263, mul_130, buf268, buf269, 1024, 128, grid=grid(1024), stream=stream0)
        del buf255
        del buf259
        del buf263
        del mul_130
        buf272 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (1024, 128), (1, 1024), 0), view_549, out=buf272)
        del view_549
        buf273 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf270, buf273, 1024, 128, grid=grid(1024), stream=stream0)
        del buf270
        buf276 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (4096, 128), (1, 4096), 0), view_547, out=buf276)
        del view_547
        buf277 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf274, buf277, 4096, 128, grid=grid(4096), stream=stream0)
        del buf274
        buf280 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf281 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf275, mul_128, buf280, buf281, 1024, 128, grid=grid(1024), stream=stream0)
        del buf275
        del mul_128
        buf284 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 128), (1, 1024), 0), view_545, out=buf284)
        del view_545
        buf285 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf282, buf285, 1024, 128, grid=grid(1024), stream=stream0)
        del buf282
        buf294 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (1024, 128), (1, 1024), 0), view_267, out=buf294)
        buf295 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf292, buf295, 1024, 128, grid=grid(1024), stream=stream0)
        del buf292
        buf298 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (1024, 128), (1, 1024), 0), view_267, out=buf298)
        buf299 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf296, buf299, 1024, 128, grid=grid(1024), stream=stream0)
        del buf296
        buf302 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (1024, 128), (1, 1024), 0), view_531, out=buf302)
        del view_531
        buf303 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf300, buf303, 1024, 128, grid=grid(1024), stream=stream0)
        del buf300
        buf306 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf307 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf301, mul_125, buf306, buf307, 1024, 128, grid=grid(1024), stream=stream0)
        del buf301
        del mul_125
        buf310 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1024, 128), (1, 1024), 0), view_529, out=buf310)
        del view_529
        buf311 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf308, buf311, 1024, 128, grid=grid(1024), stream=stream0)
        del buf308
        buf320 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf318, (1024, 128), (1, 1024), 0), view_513, out=buf320)
        buf321 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf318, buf321, 1024, 128, grid=grid(1024), stream=stream0)
        del buf318
        buf324 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (1024, 128), (1, 1024), 0), view_513, out=buf324)
        buf325 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf322, buf325, 1024, 128, grid=grid(1024), stream=stream0)
        del buf322
        buf328 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (1024, 128), (1, 1024), 0), view_513, out=buf328)
        del view_513
        buf329 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf326, buf329, 1024, 128, grid=grid(1024), stream=stream0)
        del buf326
        buf332 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf333 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf319, buf323, buf327, mul_122, buf332, buf333, 1024, 128, grid=grid(1024), stream=stream0)
        del buf319
        del buf323
        del buf327
        del mul_122
        buf336 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (1024, 128), (1, 1024), 0), view_511, out=buf336)
        del view_511
        buf337 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf334, buf337, 1024, 128, grid=grid(1024), stream=stream0)
        del buf334
        buf340 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (4096, 128), (1, 4096), 0), view_509, out=buf340)
        del view_509
        buf341 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf338, buf341, 4096, 128, grid=grid(4096), stream=stream0)
        del buf338
        buf344 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf345 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf339, mul_120, buf344, buf345, 1024, 128, grid=grid(1024), stream=stream0)
        del buf339
        del mul_120
        buf348 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (1024, 128), (1, 1024), 0), view_507, out=buf348)
        del view_507
        buf349 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf346, buf349, 1024, 128, grid=grid(1024), stream=stream0)
        del buf346
        buf358 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (1024, 128), (1, 1024), 0), view_267, out=buf358)
        buf359 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf356, buf359, 1024, 128, grid=grid(1024), stream=stream0)
        del buf356
        buf362 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (1024, 128), (1, 1024), 0), view_267, out=buf362)
        buf363 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf360, buf363, 1024, 128, grid=grid(1024), stream=stream0)
        del buf360
        buf366 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (1024, 128), (1, 1024), 0), view_493, out=buf366)
        del view_493
        buf367 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf364, buf367, 1024, 128, grid=grid(1024), stream=stream0)
        del buf364
        buf370 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf371 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf365, mul_117, buf370, buf371, 1024, 128, grid=grid(1024), stream=stream0)
        del buf365
        del mul_117
        buf374 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1024, 128), (1, 1024), 0), view_491, out=buf374)
        del view_491
        buf375 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf372, buf375, 1024, 128, grid=grid(1024), stream=stream0)
        del buf372
        buf384 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (1024, 128), (1, 1024), 0), view_475, out=buf384)
        buf385 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf382, buf385, 1024, 128, grid=grid(1024), stream=stream0)
        del buf382
        buf388 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (1024, 128), (1, 1024), 0), view_475, out=buf388)
        buf389 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf386, buf389, 1024, 128, grid=grid(1024), stream=stream0)
        del buf386
        buf392 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1024, 128), (1, 1024), 0), view_475, out=buf392)
        del view_475
        buf393 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf390, buf393, 1024, 128, grid=grid(1024), stream=stream0)
        del buf390
        buf396 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf397 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf383, buf387, buf391, mul_114, buf396, buf397, 1024, 128, grid=grid(1024), stream=stream0)
        del buf383
        del buf387
        del buf391
        del mul_114
        buf400 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (1024, 128), (1, 1024), 0), view_473, out=buf400)
        del view_473
        buf401 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf398, buf401, 1024, 128, grid=grid(1024), stream=stream0)
        del buf398
        buf404 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (4096, 128), (1, 4096), 0), view_471, out=buf404)
        del view_471
        buf405 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf402, buf405, 4096, 128, grid=grid(4096), stream=stream0)
        del buf402
        buf408 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf409 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf403, mul_112, buf408, buf409, 1024, 128, grid=grid(1024), stream=stream0)
        del buf403
        del mul_112
        buf412 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (1024, 128), (1, 1024), 0), view_469, out=buf412)
        del view_469
        buf413 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf410, buf413, 1024, 128, grid=grid(1024), stream=stream0)
        del buf410
        buf422 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (1024, 128), (1, 1024), 0), view_267, out=buf422)
        buf423 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf420, buf423, 1024, 128, grid=grid(1024), stream=stream0)
        del buf420
        buf426 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (1024, 128), (1, 1024), 0), view_267, out=buf426)
        buf427 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf424, buf427, 1024, 128, grid=grid(1024), stream=stream0)
        del buf424
        buf430 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1024, 128), (1, 1024), 0), view_455, out=buf430)
        del view_455
        buf431 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf428, buf431, 1024, 128, grid=grid(1024), stream=stream0)
        del buf428
        buf434 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf435 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf429, mul_109, buf434, buf435, 1024, 128, grid=grid(1024), stream=stream0)
        del buf429
        del mul_109
        buf438 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf436, (1024, 128), (1, 1024), 0), view_453, out=buf438)
        del view_453
        buf439 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf436, buf439, 1024, 128, grid=grid(1024), stream=stream0)
        del buf436
        buf448 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1024, 128), (1, 1024), 0), view_437, out=buf448)
        buf449 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf446, buf449, 1024, 128, grid=grid(1024), stream=stream0)
        del buf446
        buf452 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (1024, 128), (1, 1024), 0), view_437, out=buf452)
        buf453 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf450, buf453, 1024, 128, grid=grid(1024), stream=stream0)
        del buf450
        buf456 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (1024, 128), (1, 1024), 0), view_437, out=buf456)
        del view_437
        buf457 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf454, buf457, 1024, 128, grid=grid(1024), stream=stream0)
        del buf454
        buf460 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf461 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf447, buf451, buf455, mul_106, buf460, buf461, 1024, 128, grid=grid(1024), stream=stream0)
        del buf447
        del buf451
        del buf455
        del mul_106
        buf464 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (1024, 128), (1, 1024), 0), view_435, out=buf464)
        del view_435
        buf465 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf462, buf465, 1024, 128, grid=grid(1024), stream=stream0)
        del buf462
        buf468 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (4096, 128), (1, 4096), 0), view_433, out=buf468)
        del view_433
        buf469 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf466, buf469, 4096, 128, grid=grid(4096), stream=stream0)
        del buf466
        buf472 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf473 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf467, mul_104, buf472, buf473, 1024, 128, grid=grid(1024), stream=stream0)
        del buf467
        del mul_104
        buf476 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (1024, 128), (1, 1024), 0), view_431, out=buf476)
        del view_431
        buf477 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf474, buf477, 1024, 128, grid=grid(1024), stream=stream0)
        del buf474
        buf486 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (1024, 128), (1, 1024), 0), view_267, out=buf486)
        buf487 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf484, buf487, 1024, 128, grid=grid(1024), stream=stream0)
        del buf484
        buf490 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (1024, 128), (1, 1024), 0), view_267, out=buf490)
        buf491 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf488, buf491, 1024, 128, grid=grid(1024), stream=stream0)
        del buf488
        buf495 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (1024, 128), (1, 1024), 0), view_417, out=buf495)
        del view_417
        buf496 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf493, buf496, 1024, 128, grid=grid(1024), stream=stream0)
        del buf493
        buf499 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf500 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf494, mul_101, buf499, buf500, 1024, 128, grid=grid(1024), stream=stream0)
        del buf494
        del mul_101
        buf503 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1024, 128), (1, 1024), 0), view_415, out=buf503)
        del view_415
        buf504 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf501, buf504, 1024, 128, grid=grid(1024), stream=stream0)
        del buf501
        buf513 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (1024, 128), (1, 1024), 0), view_399, out=buf513)
        buf514 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf511, buf514, 1024, 128, grid=grid(1024), stream=stream0)
        del buf511
        buf517 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (1024, 128), (1, 1024), 0), view_399, out=buf517)
        buf518 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf515, buf518, 1024, 128, grid=grid(1024), stream=stream0)
        del buf515
        buf521 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf519, (1024, 128), (1, 1024), 0), view_399, out=buf521)
        del view_399
        buf522 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf519, buf522, 1024, 128, grid=grid(1024), stream=stream0)
        del buf519
        buf525 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf526 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf512, buf516, buf520, mul_98, buf525, buf526, 1024, 128, grid=grid(1024), stream=stream0)
        del buf512
        del buf516
        del buf520
        del mul_98
        buf529 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (1024, 128), (1, 1024), 0), view_397, out=buf529)
        del view_397
        buf530 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf527, buf530, 1024, 128, grid=grid(1024), stream=stream0)
        del buf527
        buf533 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (4096, 128), (1, 4096), 0), view_395, out=buf533)
        del view_395
        buf534 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf531, buf534, 4096, 128, grid=grid(4096), stream=stream0)
        del buf531
        buf537 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf538 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf532, mul_96, buf537, buf538, 1024, 128, grid=grid(1024), stream=stream0)
        del buf532
        del mul_96
        buf541 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf539, (1024, 128), (1, 1024), 0), view_393, out=buf541)
        del view_393
        buf542 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf539, buf542, 1024, 128, grid=grid(1024), stream=stream0)
        del buf539
        buf551 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (1024, 128), (1, 1024), 0), view_267, out=buf551)
        buf552 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf549, buf552, 1024, 128, grid=grid(1024), stream=stream0)
        del buf549
        buf555 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (1024, 128), (1, 1024), 0), view_267, out=buf555)
        buf556 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf553, buf556, 1024, 128, grid=grid(1024), stream=stream0)
        del buf553
        buf559 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1024, 128), (1, 1024), 0), view_379, out=buf559)
        del view_379
        buf560 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf557, buf560, 1024, 128, grid=grid(1024), stream=stream0)
        del buf557
        buf563 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf564 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf558, mul_93, buf563, buf564, 1024, 128, grid=grid(1024), stream=stream0)
        del buf558
        del mul_93
        buf567 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1024, 128), (1, 1024), 0), view_377, out=buf567)
        del view_377
        buf568 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf565, buf568, 1024, 128, grid=grid(1024), stream=stream0)
        del buf565
        buf577 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (1024, 128), (1, 1024), 0), view_361, out=buf577)
        buf578 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf575, buf578, 1024, 128, grid=grid(1024), stream=stream0)
        del buf575
        buf581 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (1024, 128), (1, 1024), 0), view_361, out=buf581)
        buf582 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf579, buf582, 1024, 128, grid=grid(1024), stream=stream0)
        del buf579
        buf585 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (1024, 128), (1, 1024), 0), view_361, out=buf585)
        del view_361
        buf586 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf583, buf586, 1024, 128, grid=grid(1024), stream=stream0)
        del buf583
        buf589 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf590 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf576, buf580, buf584, mul_90, buf589, buf590, 1024, 128, grid=grid(1024), stream=stream0)
        del buf576
        del buf580
        del buf584
        del mul_90
        buf593 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1024, 128), (1, 1024), 0), view_359, out=buf593)
        del view_359
        buf594 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf591, buf594, 1024, 128, grid=grid(1024), stream=stream0)
        del buf591
        buf597 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (4096, 128), (1, 4096), 0), view_357, out=buf597)
        del view_357
        buf598 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf595, buf598, 4096, 128, grid=grid(4096), stream=stream0)
        del buf595
        buf601 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf602 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf596, mul_88, buf601, buf602, 1024, 128, grid=grid(1024), stream=stream0)
        del buf596
        del mul_88
        buf605 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (1024, 128), (1, 1024), 0), view_355, out=buf605)
        del view_355
        buf606 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf603, buf606, 1024, 128, grid=grid(1024), stream=stream0)
        del buf603
        buf615 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (1024, 128), (1, 1024), 0), view_267, out=buf615)
        buf616 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf613, buf616, 1024, 128, grid=grid(1024), stream=stream0)
        del buf613
        buf619 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (1024, 128), (1, 1024), 0), view_267, out=buf619)
        buf620 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf617, buf620, 1024, 128, grid=grid(1024), stream=stream0)
        del buf617
        buf623 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (1024, 128), (1, 1024), 0), view_341, out=buf623)
        del view_341
        buf624 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf621, buf624, 1024, 128, grid=grid(1024), stream=stream0)
        del buf621
        buf627 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf628 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf622, mul_85, buf627, buf628, 1024, 128, grid=grid(1024), stream=stream0)
        del buf622
        del mul_85
        buf631 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (1024, 128), (1, 1024), 0), view_339, out=buf631)
        del view_339
        buf632 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf629, buf632, 1024, 128, grid=grid(1024), stream=stream0)
        del buf629
        buf641 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf639, (1024, 128), (1, 1024), 0), view_323, out=buf641)
        buf642 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf639, buf642, 1024, 128, grid=grid(1024), stream=stream0)
        del buf639
        buf645 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf643, (1024, 128), (1, 1024), 0), view_323, out=buf645)
        buf646 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf643, buf646, 1024, 128, grid=grid(1024), stream=stream0)
        del buf643
        buf649 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf647, (1024, 128), (1, 1024), 0), view_323, out=buf649)
        del view_323
        buf650 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf647, buf650, 1024, 128, grid=grid(1024), stream=stream0)
        del buf647
        buf653 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf654 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf640, buf644, buf648, mul_82, buf653, buf654, 1024, 128, grid=grid(1024), stream=stream0)
        del buf640
        del buf644
        del buf648
        del mul_82
        buf657 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (1024, 128), (1, 1024), 0), view_321, out=buf657)
        del view_321
        buf658 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf655, buf658, 1024, 128, grid=grid(1024), stream=stream0)
        del buf655
        buf661 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (4096, 128), (1, 4096), 0), view_319, out=buf661)
        del view_319
        buf662 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf659, buf662, 4096, 128, grid=grid(4096), stream=stream0)
        del buf659
        buf665 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf666 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf660, mul_80, buf665, buf666, 1024, 128, grid=grid(1024), stream=stream0)
        del buf660
        del mul_80
        buf669 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (1024, 128), (1, 1024), 0), view_317, out=buf669)
        del view_317
        buf670 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf667, buf670, 1024, 128, grid=grid(1024), stream=stream0)
        del buf667
        buf679 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (1024, 128), (1, 1024), 0), view_267, out=buf679)
        buf680 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf677, buf680, 1024, 128, grid=grid(1024), stream=stream0)
        del buf677
        buf683 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf681, (1024, 128), (1, 1024), 0), view_267, out=buf683)
        buf684 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf681, buf684, 1024, 128, grid=grid(1024), stream=stream0)
        del buf681
        buf687 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (1024, 128), (1, 1024), 0), view_303, out=buf687)
        del view_303
        buf688 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf685, buf688, 1024, 128, grid=grid(1024), stream=stream0)
        del buf685
        buf691 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf692 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf686, mul_77, buf691, buf692, 1024, 128, grid=grid(1024), stream=stream0)
        del buf686
        del mul_77
        buf695 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1024, 128), (1, 1024), 0), view_301, out=buf695)
        del view_301
        buf696 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf693, buf696, 1024, 128, grid=grid(1024), stream=stream0)
        del buf693
        buf705 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf703, (1024, 128), (1, 1024), 0), view_285, out=buf705)
        buf706 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf703, buf706, 1024, 128, grid=grid(1024), stream=stream0)
        del buf703
        buf709 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1024, 128), (1, 1024), 0), view_285, out=buf709)
        buf710 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf707, buf710, 1024, 128, grid=grid(1024), stream=stream0)
        del buf707
        buf713 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (1024, 128), (1, 1024), 0), view_285, out=buf713)
        del view_285
        buf714 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf711, buf714, 1024, 128, grid=grid(1024), stream=stream0)
        del buf711
        buf717 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf718 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf704, buf708, buf712, mul_74, buf717, buf718, 1024, 128, grid=grid(1024), stream=stream0)
        del buf704
        del buf708
        del buf712
        del mul_74
        buf721 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (1024, 128), (1, 1024), 0), view_283, out=buf721)
        del view_283
        buf722 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf719, buf722, 1024, 128, grid=grid(1024), stream=stream0)
        buf725 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (4096, 128), (1, 4096), 0), view_281, out=buf725)
        del view_281
        buf726 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf723, buf726, 4096, 128, grid=grid(4096), stream=stream0)
        buf729 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf730 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf724, mul_72, buf729, buf730, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_72
        buf733 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (1024, 128), (1, 1024), 0), view_279, out=buf733)
        del view_279
        buf734 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf731, buf734, 1024, 128, grid=grid(1024), stream=stream0)
        buf740 = reinterpret_tensor(buf724, (16, 128, 64), (8192, 64, 1), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf738, permute_1031, out=buf740)
        del permute_1031
        buf743 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (1024, 128), (1, 1024), 0), view_267, out=buf743)
        buf744 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf741, buf744, 1024, 128, grid=grid(1024), stream=stream0)
        buf747 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf745, (1024, 128), (1, 1024), 0), view_267, out=buf747)
        del view_267
        buf748 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf745, buf748, 1024, 128, grid=grid(1024), stream=stream0)
        buf750 = reinterpret_tensor(buf745, (128, 1024), (1024, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf740, buf750, 131072, grid=grid(131072), stream=stream0)
        buf751 = reinterpret_tensor(buf740, (128, 1024), (1024, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf750, permute_1044, out=buf751)
        del permute_1044
        buf752 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf750, (1024, 128), (1, 1024), 0), view_265, out=buf752)
        del view_265
        buf753 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf750, buf753, 1024, 128, grid=grid(1024), stream=stream0)
        buf758 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf758, buf751, primals_207, mul_69, div_73, 128, 1024, grid=grid(128), stream=stream0)
        del div_73
        del primals_207
        buf756 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf757 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf751, mul_69, buf756, buf757, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_69
        buf759 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf758, (128, 1024), (1024, 1), 0), permute_1048, out=buf759)
        del permute_1048
        buf760 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf758, (1024, 128), (1, 1024), 0), view_263, out=buf760)
        del view_263
        buf761 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf758, buf761, 1024, 128, grid=grid(1024), stream=stream0)
        buf762 = reinterpret_tensor(buf750, (16, 128, 64), (8192, 64, 1), 0); del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1053, reinterpret_tensor(buf759, (16, 128, 64), (64, 1024, 1), 0), out=buf762)
        del permute_1053
        buf763 = buf738; del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf759, (16, 128, 64), (64, 1024, 1), 0), permute_1054, out=buf763)
        del permute_1054
        buf765 = buf736; del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_16.run(buf763, alias_101, buf765, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_101
        buf766 = reinterpret_tensor(buf759, (16, 64, 128), (8192, 128, 1), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf765, (16, 128, 128), (16384, 128, 1), 0), out=buf766)
        del permute_1055
        buf767 = reinterpret_tensor(buf741, (16, 128, 64), (8192, 64, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf765, (16, 128, 128), (16384, 128, 1), 0), permute_1056, out=buf767)
        del permute_1056
        buf768 = reinterpret_tensor(buf719, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_4, buf762, buf768, 131072, grid=grid(131072), stream=stream0)
        del tangents_4
        buf769 = reinterpret_tensor(buf762, (128, 1024), (1024, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (128, 1024), (1024, 1), 0), permute_1060, out=buf769)
        del permute_1060
        buf770 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (1024, 128), (1, 1024), 0), view_247, out=buf770)
        buf771 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf768, buf771, 1024, 128, grid=grid(1024), stream=stream0)
        buf772 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_3, buf766, buf772, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del tangents_3
        buf773 = reinterpret_tensor(buf766, (128, 1024), (1024, 1), 0); del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 1024), (1024, 1), 0), permute_1065, out=buf773)
        del permute_1065
        buf774 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (1024, 128), (1, 1024), 0), view_247, out=buf774)
        buf775 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf772, buf775, 1024, 128, grid=grid(1024), stream=stream0)
        buf776 = reinterpret_tensor(buf772, (128, 1024), (1024, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf767, buf776, 131072, grid=grid(131072), stream=stream0)
        buf777 = reinterpret_tensor(buf767, (128, 1024), (1024, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf776, permute_1069, out=buf777)
        del permute_1069
        buf778 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (1024, 128), (1, 1024), 0), view_247, out=buf778)
        del view_247
        buf779 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf776, buf779, 1024, 128, grid=grid(1024), stream=stream0)
        del buf776
        buf784 = buf758; del buf758  # reuse
        buf786 = buf784; del buf784  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_22.run(buf786, buf769, buf773, buf777, primals_197, mul_66, div_74, view_243, 128, 1024, grid=grid(128), stream=stream0)
        del div_74
        del primals_197
        buf782 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf783 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf769, buf773, buf777, mul_66, buf782, buf783, 1024, 128, grid=grid(1024), stream=stream0)
        del buf769
        del mul_66
        buf785 = empty((128112, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill, aten.mul]
        triton_poi_fused_embedding_dense_backward_masked_fill_mul_23.run(buf785, 131186688, grid=grid(131186688), stream=stream0)
        aten.index_put_(buf785, [view_243], buf786, True)
        del view_243
        buf792 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf793 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf749, mul_62, buf792, buf793, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_62
        buf794 = reinterpret_tensor(buf723, (128, 4096), (4096, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf791, (128, 1024), (1024, 1), 0), permute_1073, out=buf794)
        del permute_1073
        buf795 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf791, (1024, 128), (1, 1024), 0), view_241, out=buf795)
        del view_241
        buf796 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf791, buf796, 1024, 128, grid=grid(1024), stream=stream0)
        buf797 = reinterpret_tensor(buf794, (1, 128, 4096), (524288, 4096, 1), 0); del buf794  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf797, le_12, 524288, grid=grid(524288), stream=stream0)
        del le_12
        buf798 = reinterpret_tensor(buf749, (128, 1024), (1024, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf797, (128, 4096), (4096, 1), 0), permute_1077, out=buf798)
        del permute_1077
        buf799 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf797, (4096, 128), (1, 4096), 0), view_239, out=buf799)
        del view_239
        buf800 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf797, buf800, 4096, 128, grid=grid(4096), stream=stream0)
        buf805 = buf791; del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf805, buf798, primals_188, mul_60, div_76, 128, 1024, grid=grid(128), stream=stream0)
        del div_76
        del primals_188
        buf803 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf804 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf798, mul_60, buf803, buf804, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_60
        buf806 = buf798; del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf805, (128, 1024), (1024, 1), 0), permute_1081, out=buf806)
        del permute_1081
        buf807 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf805, (1024, 128), (1, 1024), 0), view_237, out=buf807)
        del view_237
        buf808 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf805, buf808, 1024, 128, grid=grid(1024), stream=stream0)
        buf809 = reinterpret_tensor(buf786, (16, 128, 64), (8192, 64, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1086, reinterpret_tensor(buf806, (16, 128, 64), (64, 1024, 1), 0), out=buf809)
        del permute_1086
        buf810 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf806, (16, 128, 64), (64, 1024, 1), 0), permute_1087, out=buf810)
        del permute_1087
        buf812 = buf763; del buf763  # reuse
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf810, bmm_22, amax_11, sum_12, buf812, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_11
        del bmm_22
        del sum_12
        buf813 = reinterpret_tensor(buf806, (16, 64, 128), (8192, 128, 1), 0); del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1088, buf812, out=buf813)
        del permute_1088
        buf814 = reinterpret_tensor(buf777, (16, 128, 64), (8192, 64, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf812, permute_1089, out=buf814)
        del permute_1089
        buf815 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf809, buf815, 131072, grid=grid(131072), stream=stream0)
        buf816 = reinterpret_tensor(buf809, (128, 1024), (1024, 1), 0); del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf815, permute_1093, out=buf816)
        del permute_1093
        buf817 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf815, (1024, 128), (1, 1024), 0), view_223, out=buf817)
        buf818 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf815, buf818, 1024, 128, grid=grid(1024), stream=stream0)
        buf819 = buf815; del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf813, (128, 1024), (1, 128), 0), permute_1098, out=buf819)
        del permute_1098
        buf820 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf813, (1024, 128), (128, 1), 0), view_223, out=buf820)
        buf821 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf813, buf821, 1024, 128, grid=grid(1024), stream=stream0)
        buf822 = reinterpret_tensor(buf813, (128, 1024), (1024, 1), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf814, buf822, 131072, grid=grid(131072), stream=stream0)
        buf823 = reinterpret_tensor(buf814, (128, 1024), (1024, 1), 0); del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf822, permute_1102, out=buf823)
        del permute_1102
        buf824 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf822, (1024, 128), (1, 1024), 0), view_223, out=buf824)
        del view_223
        buf825 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf822, buf825, 1024, 128, grid=grid(1024), stream=stream0)
        buf830 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf830, buf816, buf819, buf823, primals_178, mul_57, div_77, 128, 1024, grid=grid(128), stream=stream0)
        del div_77
        del primals_178
        buf828 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf829 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf816, buf819, buf823, mul_57, buf828, buf829, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_57
        buf831 = reinterpret_tensor(buf797, (128, 4096), (4096, 1), 0); del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (128, 1024), (1024, 1), 0), permute_1106, out=buf831)
        del permute_1106
        buf832 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (1024, 128), (1, 1024), 0), view_221, out=buf832)
        del view_221
        buf833 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf830, buf833, 1024, 128, grid=grid(1024), stream=stream0)
        buf834 = reinterpret_tensor(buf831, (1, 128, 4096), (524288, 4096, 1), 0); del buf831  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf834, le_13, 524288, grid=grid(524288), stream=stream0)
        del le_13
        buf835 = buf823; del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf834, (128, 4096), (4096, 1), 0), permute_1110, out=buf835)
        del permute_1110
        buf836 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf834, (4096, 128), (1, 4096), 0), view_219, out=buf836)
        del view_219
        buf837 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf834, buf837, 4096, 128, grid=grid(4096), stream=stream0)
        buf842 = buf830; del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf842, buf835, primals_172, mul_55, div_78, 128, 1024, grid=grid(128), stream=stream0)
        del div_78
        del primals_172
        buf840 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf841 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf835, mul_55, buf840, buf841, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_55
        buf843 = buf835; del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (128, 1024), (1024, 1), 0), permute_1114, out=buf843)
        del permute_1114
        buf844 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (1024, 128), (1, 1024), 0), view_217, out=buf844)
        del view_217
        buf845 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf842, buf845, 1024, 128, grid=grid(1024), stream=stream0)
        buf846 = reinterpret_tensor(buf819, (16, 128, 64), (8192, 64, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1119, reinterpret_tensor(buf843, (16, 128, 64), (64, 1024, 1), 0), out=buf846)
        del permute_1119
        buf847 = buf812; del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf843, (16, 128, 64), (64, 1024, 1), 0), permute_1120, out=buf847)
        del permute_1120
        buf849 = buf810; del buf810  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf847, bmm_20, amax_10, sum_11, buf849, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_10
        del bmm_20
        del sum_11
        buf850 = reinterpret_tensor(buf843, (16, 64, 128), (8192, 128, 1), 0); del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1121, buf849, out=buf850)
        del permute_1121
        buf851 = reinterpret_tensor(buf816, (16, 128, 64), (8192, 64, 1), 0); del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf849, permute_1122, out=buf851)
        del permute_1122
        buf852 = buf822; del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf846, buf852, 131072, grid=grid(131072), stream=stream0)
        buf853 = reinterpret_tensor(buf846, (128, 1024), (1024, 1), 0); del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf852, permute_1126, out=buf853)
        del permute_1126
        buf854 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf852, (1024, 128), (1, 1024), 0), view_203, out=buf854)
        buf855 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf852, buf855, 1024, 128, grid=grid(1024), stream=stream0)
        buf856 = buf852; del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (128, 1024), (1, 128), 0), permute_1131, out=buf856)
        del permute_1131
        buf857 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (1024, 128), (128, 1), 0), view_203, out=buf857)
        buf858 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf850, buf858, 1024, 128, grid=grid(1024), stream=stream0)
        buf859 = reinterpret_tensor(buf850, (128, 1024), (1024, 1), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf851, buf859, 131072, grid=grid(131072), stream=stream0)
        buf860 = reinterpret_tensor(buf851, (128, 1024), (1024, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf859, permute_1135, out=buf860)
        del permute_1135
        buf861 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (1024, 128), (1, 1024), 0), view_203, out=buf861)
        del view_203
        buf862 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf859, buf862, 1024, 128, grid=grid(1024), stream=stream0)
        buf867 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf867, buf853, buf856, buf860, primals_162, mul_52, div_79, 128, 1024, grid=grid(128), stream=stream0)
        del div_79
        del primals_162
        buf865 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf866 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf853, buf856, buf860, mul_52, buf865, buf866, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_52
        buf868 = reinterpret_tensor(buf834, (128, 4096), (4096, 1), 0); del buf834  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (128, 1024), (1024, 1), 0), permute_1139, out=buf868)
        del permute_1139
        buf869 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (1024, 128), (1, 1024), 0), view_201, out=buf869)
        del view_201
        buf870 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf867, buf870, 1024, 128, grid=grid(1024), stream=stream0)
        buf871 = reinterpret_tensor(buf868, (1, 128, 4096), (524288, 4096, 1), 0); del buf868  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf871, le_14, 524288, grid=grid(524288), stream=stream0)
        del le_14
        buf872 = buf860; del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf871, (128, 4096), (4096, 1), 0), permute_1143, out=buf872)
        del permute_1143
        buf873 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf871, (4096, 128), (1, 4096), 0), view_199, out=buf873)
        del view_199
        buf874 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf871, buf874, 4096, 128, grid=grid(4096), stream=stream0)
        buf879 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf879, buf872, primals_156, mul_50, div_80, 128, 1024, grid=grid(128), stream=stream0)
        del div_80
        del primals_156
        buf877 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf878 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf872, mul_50, buf877, buf878, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_50
        buf880 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (128, 1024), (1024, 1), 0), permute_1147, out=buf880)
        del permute_1147
        buf881 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (1024, 128), (1, 1024), 0), view_197, out=buf881)
        del view_197
        buf882 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf879, buf882, 1024, 128, grid=grid(1024), stream=stream0)
        buf883 = reinterpret_tensor(buf856, (16, 128, 64), (8192, 64, 1), 0); del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1152, reinterpret_tensor(buf880, (16, 128, 64), (64, 1024, 1), 0), out=buf883)
        del permute_1152
        buf884 = buf849; del buf849  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf880, (16, 128, 64), (64, 1024, 1), 0), permute_1153, out=buf884)
        del permute_1153
        buf886 = buf847; del buf847  # reuse
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf884, bmm_18, amax_9, sum_10, buf886, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_9
        del bmm_18
        del sum_10
        buf887 = reinterpret_tensor(buf880, (16, 64, 128), (8192, 128, 1), 0); del buf880  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1154, buf886, out=buf887)
        del permute_1154
        buf888 = reinterpret_tensor(buf853, (16, 128, 64), (8192, 64, 1), 0); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf886, permute_1155, out=buf888)
        del permute_1155
        buf889 = buf859; del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf883, buf889, 131072, grid=grid(131072), stream=stream0)
        buf890 = reinterpret_tensor(buf883, (128, 1024), (1024, 1), 0); del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf889, permute_1159, out=buf890)
        del permute_1159
        buf891 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf889, (1024, 128), (1, 1024), 0), view_183, out=buf891)
        buf892 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf889, buf892, 1024, 128, grid=grid(1024), stream=stream0)
        buf893 = buf889; del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf887, (128, 1024), (1, 128), 0), permute_1164, out=buf893)
        del permute_1164
        buf894 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf887, (1024, 128), (128, 1), 0), view_183, out=buf894)
        buf895 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf887, buf895, 1024, 128, grid=grid(1024), stream=stream0)
        buf896 = reinterpret_tensor(buf887, (128, 1024), (1024, 1), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf888, buf896, 131072, grid=grid(131072), stream=stream0)
        buf897 = reinterpret_tensor(buf888, (128, 1024), (1024, 1), 0); del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf896, permute_1168, out=buf897)
        del permute_1168
        buf898 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (1024, 128), (1, 1024), 0), view_183, out=buf898)
        del view_183
        buf899 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf896, buf899, 1024, 128, grid=grid(1024), stream=stream0)
        buf904 = buf879; del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf904, buf890, buf893, buf897, primals_146, mul_47, div_81, 128, 1024, grid=grid(128), stream=stream0)
        del div_81
        del primals_146
        buf902 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf903 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf890, buf893, buf897, mul_47, buf902, buf903, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_47
        buf905 = reinterpret_tensor(buf871, (128, 4096), (4096, 1), 0); del buf871  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (128, 1024), (1024, 1), 0), permute_1172, out=buf905)
        del permute_1172
        buf906 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (1024, 128), (1, 1024), 0), view_181, out=buf906)
        del view_181
        buf907 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf904, buf907, 1024, 128, grid=grid(1024), stream=stream0)
        buf908 = reinterpret_tensor(buf905, (1, 128, 4096), (524288, 4096, 1), 0); del buf905  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf908, le_15, 524288, grid=grid(524288), stream=stream0)
        del le_15
        buf909 = buf897; del buf897  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (128, 4096), (4096, 1), 0), permute_1176, out=buf909)
        del permute_1176
        buf910 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (4096, 128), (1, 4096), 0), view_179, out=buf910)
        del view_179
        buf911 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf908, buf911, 4096, 128, grid=grid(4096), stream=stream0)
        buf916 = buf904; del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf916, buf909, primals_140, mul_45, div_82, 128, 1024, grid=grid(128), stream=stream0)
        del div_82
        del primals_140
        buf914 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf915 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf909, mul_45, buf914, buf915, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_45
        buf917 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf916, (128, 1024), (1024, 1), 0), permute_1180, out=buf917)
        del permute_1180
        buf918 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf916, (1024, 128), (1, 1024), 0), view_177, out=buf918)
        del view_177
        buf919 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf916, buf919, 1024, 128, grid=grid(1024), stream=stream0)
        buf920 = reinterpret_tensor(buf893, (16, 128, 64), (8192, 64, 1), 0); del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1185, reinterpret_tensor(buf917, (16, 128, 64), (64, 1024, 1), 0), out=buf920)
        del permute_1185
        buf921 = buf886; del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf917, (16, 128, 64), (64, 1024, 1), 0), permute_1186, out=buf921)
        del permute_1186
        buf923 = buf884; del buf884  # reuse
        # Source Nodes: [attn_weights_17], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf921, bmm_16, amax_8, sum_9, buf923, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_8
        del bmm_16
        del sum_9
        buf924 = reinterpret_tensor(buf917, (16, 64, 128), (8192, 128, 1), 0); del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1187, buf923, out=buf924)
        del permute_1187
        buf925 = reinterpret_tensor(buf890, (16, 128, 64), (8192, 64, 1), 0); del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf923, permute_1188, out=buf925)
        del permute_1188
        buf926 = buf896; del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf920, buf926, 131072, grid=grid(131072), stream=stream0)
        buf927 = reinterpret_tensor(buf920, (128, 1024), (1024, 1), 0); del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf926, permute_1192, out=buf927)
        del permute_1192
        buf928 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (1024, 128), (1, 1024), 0), view_163, out=buf928)
        buf929 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf926, buf929, 1024, 128, grid=grid(1024), stream=stream0)
        buf930 = buf926; del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf924, (128, 1024), (1, 128), 0), permute_1197, out=buf930)
        del permute_1197
        buf931 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf924, (1024, 128), (128, 1), 0), view_163, out=buf931)
        buf932 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf924, buf932, 1024, 128, grid=grid(1024), stream=stream0)
        buf933 = reinterpret_tensor(buf924, (128, 1024), (1024, 1), 0); del buf924  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf925, buf933, 131072, grid=grid(131072), stream=stream0)
        buf934 = reinterpret_tensor(buf925, (128, 1024), (1024, 1), 0); del buf925  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf933, permute_1201, out=buf934)
        del permute_1201
        buf935 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (1024, 128), (1, 1024), 0), view_163, out=buf935)
        del view_163
        buf936 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf933, buf936, 1024, 128, grid=grid(1024), stream=stream0)
        buf941 = buf916; del buf916  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf941, buf927, buf930, buf934, primals_130, mul_42, div_83, 128, 1024, grid=grid(128), stream=stream0)
        del div_83
        del primals_130
        buf939 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf940 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf927, buf930, buf934, mul_42, buf939, buf940, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_42
        buf942 = reinterpret_tensor(buf908, (128, 4096), (4096, 1), 0); del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf941, (128, 1024), (1024, 1), 0), permute_1205, out=buf942)
        del permute_1205
        buf943 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf941, (1024, 128), (1, 1024), 0), view_161, out=buf943)
        del view_161
        buf944 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf941, buf944, 1024, 128, grid=grid(1024), stream=stream0)
        buf945 = reinterpret_tensor(buf942, (1, 128, 4096), (524288, 4096, 1), 0); del buf942  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf945, le_16, 524288, grid=grid(524288), stream=stream0)
        del le_16
        buf946 = buf934; del buf934  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (128, 4096), (4096, 1), 0), permute_1209, out=buf946)
        del permute_1209
        buf947 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (4096, 128), (1, 4096), 0), view_159, out=buf947)
        del view_159
        buf948 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf945, buf948, 4096, 128, grid=grid(4096), stream=stream0)
        buf953 = buf941; del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf953, buf946, primals_124, mul_40, div_84, 128, 1024, grid=grid(128), stream=stream0)
        del div_84
        del primals_124
        buf951 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf952 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf946, mul_40, buf951, buf952, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_40
        buf954 = buf946; del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf953, (128, 1024), (1024, 1), 0), permute_1213, out=buf954)
        del permute_1213
        buf955 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf953, (1024, 128), (1, 1024), 0), view_157, out=buf955)
        del view_157
        buf956 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf953, buf956, 1024, 128, grid=grid(1024), stream=stream0)
        buf957 = reinterpret_tensor(buf930, (16, 128, 64), (8192, 64, 1), 0); del buf930  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1218, reinterpret_tensor(buf954, (16, 128, 64), (64, 1024, 1), 0), out=buf957)
        del permute_1218
        buf958 = buf923; del buf923  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf954, (16, 128, 64), (64, 1024, 1), 0), permute_1219, out=buf958)
        del permute_1219
        buf960 = buf921; del buf921  # reuse
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf958, bmm_14, amax_7, sum_8, buf960, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_7
        del bmm_14
        del sum_8
        buf961 = reinterpret_tensor(buf954, (16, 64, 128), (8192, 128, 1), 0); del buf954  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1220, buf960, out=buf961)
        del permute_1220
        buf962 = reinterpret_tensor(buf927, (16, 128, 64), (8192, 64, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf960, permute_1221, out=buf962)
        del permute_1221
        buf963 = buf933; del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf957, buf963, 131072, grid=grid(131072), stream=stream0)
        buf964 = reinterpret_tensor(buf957, (128, 1024), (1024, 1), 0); del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf963, permute_1225, out=buf964)
        del permute_1225
        buf965 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf963, (1024, 128), (1, 1024), 0), view_143, out=buf965)
        buf966 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf963, buf966, 1024, 128, grid=grid(1024), stream=stream0)
        buf967 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf961, (128, 1024), (1, 128), 0), permute_1230, out=buf967)
        del permute_1230
        buf968 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf961, (1024, 128), (128, 1), 0), view_143, out=buf968)
        buf969 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf961, buf969, 1024, 128, grid=grid(1024), stream=stream0)
        buf970 = reinterpret_tensor(buf961, (128, 1024), (1024, 1), 0); del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf962, buf970, 131072, grid=grid(131072), stream=stream0)
        buf971 = reinterpret_tensor(buf962, (128, 1024), (1024, 1), 0); del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf970, permute_1234, out=buf971)
        del permute_1234
        buf972 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf970, (1024, 128), (1, 1024), 0), view_143, out=buf972)
        del view_143
        buf973 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf970, buf973, 1024, 128, grid=grid(1024), stream=stream0)
        buf978 = buf953; del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf978, buf964, buf967, buf971, primals_114, mul_37, div_85, 128, 1024, grid=grid(128), stream=stream0)
        del div_85
        del primals_114
        buf976 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf977 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf964, buf967, buf971, mul_37, buf976, buf977, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_37
        buf979 = reinterpret_tensor(buf945, (128, 4096), (4096, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (128, 1024), (1024, 1), 0), permute_1238, out=buf979)
        del permute_1238
        buf980 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (1024, 128), (1, 1024), 0), view_141, out=buf980)
        del view_141
        buf981 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf978, buf981, 1024, 128, grid=grid(1024), stream=stream0)
        buf982 = reinterpret_tensor(buf979, (1, 128, 4096), (524288, 4096, 1), 0); del buf979  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf982, le_17, 524288, grid=grid(524288), stream=stream0)
        del le_17
        buf983 = buf971; del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf982, (128, 4096), (4096, 1), 0), permute_1242, out=buf983)
        del permute_1242
        buf984 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf982, (4096, 128), (1, 4096), 0), view_139, out=buf984)
        del view_139
        buf985 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf982, buf985, 4096, 128, grid=grid(4096), stream=stream0)
        buf990 = buf978; del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf990, buf983, primals_108, mul_35, div_86, 128, 1024, grid=grid(128), stream=stream0)
        del div_86
        del primals_108
        buf988 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf989 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf983, mul_35, buf988, buf989, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_35
        buf991 = buf983; del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf990, (128, 1024), (1024, 1), 0), permute_1246, out=buf991)
        del permute_1246
        buf992 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf990, (1024, 128), (1, 1024), 0), view_137, out=buf992)
        del view_137
        buf993 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf990, buf993, 1024, 128, grid=grid(1024), stream=stream0)
        buf994 = reinterpret_tensor(buf967, (16, 128, 64), (8192, 64, 1), 0); del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1251, reinterpret_tensor(buf991, (16, 128, 64), (64, 1024, 1), 0), out=buf994)
        del permute_1251
        buf995 = buf960; del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf991, (16, 128, 64), (64, 1024, 1), 0), permute_1252, out=buf995)
        del permute_1252
        buf997 = buf958; del buf958  # reuse
        # Source Nodes: [attn_weights_13], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf995, bmm_12, amax_6, sum_7, buf997, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_6
        del bmm_12
        del sum_7
        buf998 = reinterpret_tensor(buf991, (16, 64, 128), (8192, 128, 1), 0); del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1253, buf997, out=buf998)
        del permute_1253
        buf999 = reinterpret_tensor(buf964, (16, 128, 64), (8192, 64, 1), 0); del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf997, permute_1254, out=buf999)
        del permute_1254
        buf1000 = buf970; del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf994, buf1000, 131072, grid=grid(131072), stream=stream0)
        buf1001 = reinterpret_tensor(buf994, (128, 1024), (1024, 1), 0); del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1000, permute_1258, out=buf1001)
        del permute_1258
        buf1002 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1000, (1024, 128), (1, 1024), 0), view_123, out=buf1002)
        buf1003 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1000, buf1003, 1024, 128, grid=grid(1024), stream=stream0)
        buf1004 = buf1000; del buf1000  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf998, (128, 1024), (1, 128), 0), permute_1263, out=buf1004)
        del permute_1263
        buf1005 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf998, (1024, 128), (128, 1), 0), view_123, out=buf1005)
        buf1006 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf998, buf1006, 1024, 128, grid=grid(1024), stream=stream0)
        buf1007 = reinterpret_tensor(buf998, (128, 1024), (1024, 1), 0); del buf998  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf999, buf1007, 131072, grid=grid(131072), stream=stream0)
        buf1008 = reinterpret_tensor(buf999, (128, 1024), (1024, 1), 0); del buf999  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1007, permute_1267, out=buf1008)
        del permute_1267
        buf1009 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1007, (1024, 128), (1, 1024), 0), view_123, out=buf1009)
        del view_123
        buf1010 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1007, buf1010, 1024, 128, grid=grid(1024), stream=stream0)
        buf1015 = buf990; del buf990  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1015, buf1001, buf1004, buf1008, primals_98, mul_32, div_87, 128, 1024, grid=grid(128), stream=stream0)
        del div_87
        del primals_98
        buf1013 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1014 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1001, buf1004, buf1008, mul_32, buf1013, buf1014, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_32
        buf1016 = reinterpret_tensor(buf982, (128, 4096), (4096, 1), 0); del buf982  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1015, (128, 1024), (1024, 1), 0), permute_1271, out=buf1016)
        del permute_1271
        buf1017 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1015, (1024, 128), (1, 1024), 0), view_121, out=buf1017)
        del view_121
        buf1018 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1015, buf1018, 1024, 128, grid=grid(1024), stream=stream0)
        buf1019 = reinterpret_tensor(buf1016, (1, 128, 4096), (524288, 4096, 1), 0); del buf1016  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1019, le_18, 524288, grid=grid(524288), stream=stream0)
        del le_18
        buf1020 = buf1008; del buf1008  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1019, (128, 4096), (4096, 1), 0), permute_1275, out=buf1020)
        del permute_1275
        buf1021 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1019, (4096, 128), (1, 4096), 0), view_119, out=buf1021)
        del view_119
        buf1022 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1019, buf1022, 4096, 128, grid=grid(4096), stream=stream0)
        buf1027 = buf1015; del buf1015  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1027, buf1020, primals_92, mul_30, div_88, 128, 1024, grid=grid(128), stream=stream0)
        del div_88
        del primals_92
        buf1025 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1026 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1020, mul_30, buf1025, buf1026, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_30
        buf1028 = buf1020; del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (128, 1024), (1024, 1), 0), permute_1279, out=buf1028)
        del permute_1279
        buf1029 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (1024, 128), (1, 1024), 0), view_117, out=buf1029)
        del view_117
        buf1030 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1027, buf1030, 1024, 128, grid=grid(1024), stream=stream0)
        buf1031 = reinterpret_tensor(buf1004, (16, 128, 64), (8192, 64, 1), 0); del buf1004  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1284, reinterpret_tensor(buf1028, (16, 128, 64), (64, 1024, 1), 0), out=buf1031)
        del permute_1284
        buf1032 = buf997; del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1028, (16, 128, 64), (64, 1024, 1), 0), permute_1285, out=buf1032)
        del permute_1285
        buf1034 = buf995; del buf995  # reuse
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1032, bmm_10, amax_5, sum_6, buf1034, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_5
        del bmm_10
        del sum_6
        buf1035 = reinterpret_tensor(buf1028, (16, 64, 128), (8192, 128, 1), 0); del buf1028  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1286, buf1034, out=buf1035)
        del permute_1286
        buf1036 = reinterpret_tensor(buf1001, (16, 128, 64), (8192, 64, 1), 0); del buf1001  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1034, permute_1287, out=buf1036)
        del permute_1287
        buf1037 = buf1007; del buf1007  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1031, buf1037, 131072, grid=grid(131072), stream=stream0)
        buf1038 = reinterpret_tensor(buf1031, (128, 1024), (1024, 1), 0); del buf1031  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1037, permute_1291, out=buf1038)
        del permute_1291
        buf1039 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (1024, 128), (1, 1024), 0), view_103, out=buf1039)
        buf1040 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1037, buf1040, 1024, 128, grid=grid(1024), stream=stream0)
        buf1041 = buf1037; del buf1037  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (128, 1024), (1, 128), 0), permute_1296, out=buf1041)
        del permute_1296
        buf1042 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (1024, 128), (128, 1), 0), view_103, out=buf1042)
        buf1043 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1035, buf1043, 1024, 128, grid=grid(1024), stream=stream0)
        buf1044 = reinterpret_tensor(buf1035, (128, 1024), (1024, 1), 0); del buf1035  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1036, buf1044, 131072, grid=grid(131072), stream=stream0)
        buf1045 = reinterpret_tensor(buf1036, (128, 1024), (1024, 1), 0); del buf1036  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1044, permute_1300, out=buf1045)
        del permute_1300
        buf1046 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1044, (1024, 128), (1, 1024), 0), view_103, out=buf1046)
        del view_103
        buf1047 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1044, buf1047, 1024, 128, grid=grid(1024), stream=stream0)
        buf1052 = buf1027; del buf1027  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1052, buf1038, buf1041, buf1045, primals_82, mul_27, div_89, 128, 1024, grid=grid(128), stream=stream0)
        del div_89
        del primals_82
        buf1050 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1051 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1038, buf1041, buf1045, mul_27, buf1050, buf1051, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_27
        buf1053 = reinterpret_tensor(buf1019, (128, 4096), (4096, 1), 0); del buf1019  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1052, (128, 1024), (1024, 1), 0), permute_1304, out=buf1053)
        del permute_1304
        buf1054 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1052, (1024, 128), (1, 1024), 0), view_101, out=buf1054)
        del view_101
        buf1055 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1052, buf1055, 1024, 128, grid=grid(1024), stream=stream0)
        buf1056 = reinterpret_tensor(buf1053, (1, 128, 4096), (524288, 4096, 1), 0); del buf1053  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1056, le_19, 524288, grid=grid(524288), stream=stream0)
        del le_19
        buf1057 = buf1045; del buf1045  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1056, (128, 4096), (4096, 1), 0), permute_1308, out=buf1057)
        del permute_1308
        buf1058 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1056, (4096, 128), (1, 4096), 0), view_99, out=buf1058)
        del view_99
        buf1059 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1056, buf1059, 4096, 128, grid=grid(4096), stream=stream0)
        buf1064 = buf1052; del buf1052  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1064, buf1057, primals_76, mul_25, div_90, 128, 1024, grid=grid(128), stream=stream0)
        del div_90
        del primals_76
        buf1062 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1063 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1057, mul_25, buf1062, buf1063, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_25
        buf1065 = buf1057; del buf1057  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1064, (128, 1024), (1024, 1), 0), permute_1312, out=buf1065)
        del permute_1312
        buf1066 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1064, (1024, 128), (1, 1024), 0), view_97, out=buf1066)
        del view_97
        buf1067 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1064, buf1067, 1024, 128, grid=grid(1024), stream=stream0)
        buf1068 = reinterpret_tensor(buf1041, (16, 128, 64), (8192, 64, 1), 0); del buf1041  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1317, reinterpret_tensor(buf1065, (16, 128, 64), (64, 1024, 1), 0), out=buf1068)
        del permute_1317
        buf1069 = buf1034; del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1065, (16, 128, 64), (64, 1024, 1), 0), permute_1318, out=buf1069)
        del permute_1318
        buf1071 = buf1032; del buf1032  # reuse
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1069, bmm_8, amax_4, sum_5, buf1071, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_4
        del bmm_8
        del sum_5
        buf1072 = reinterpret_tensor(buf1065, (16, 64, 128), (8192, 128, 1), 0); del buf1065  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1319, buf1071, out=buf1072)
        del permute_1319
        buf1073 = reinterpret_tensor(buf1038, (16, 128, 64), (8192, 64, 1), 0); del buf1038  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1071, permute_1320, out=buf1073)
        del permute_1320
        buf1074 = buf1044; del buf1044  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1068, buf1074, 131072, grid=grid(131072), stream=stream0)
        buf1075 = reinterpret_tensor(buf1068, (128, 1024), (1024, 1), 0); del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1074, permute_1324, out=buf1075)
        del permute_1324
        buf1076 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1074, (1024, 128), (1, 1024), 0), view_83, out=buf1076)
        buf1077 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1074, buf1077, 1024, 128, grid=grid(1024), stream=stream0)
        buf1078 = buf1074; del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1072, (128, 1024), (1, 128), 0), permute_1329, out=buf1078)
        del permute_1329
        buf1079 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1072, (1024, 128), (128, 1), 0), view_83, out=buf1079)
        buf1080 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1072, buf1080, 1024, 128, grid=grid(1024), stream=stream0)
        buf1081 = reinterpret_tensor(buf1072, (128, 1024), (1024, 1), 0); del buf1072  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1073, buf1081, 131072, grid=grid(131072), stream=stream0)
        buf1082 = reinterpret_tensor(buf1073, (128, 1024), (1024, 1), 0); del buf1073  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1081, permute_1333, out=buf1082)
        del permute_1333
        buf1083 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1081, (1024, 128), (1, 1024), 0), view_83, out=buf1083)
        del view_83
        buf1084 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1081, buf1084, 1024, 128, grid=grid(1024), stream=stream0)
        buf1089 = buf1064; del buf1064  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1089, buf1075, buf1078, buf1082, primals_66, mul_22, div_91, 128, 1024, grid=grid(128), stream=stream0)
        del div_91
        del primals_66
        buf1087 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1088 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1075, buf1078, buf1082, mul_22, buf1087, buf1088, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_22
        buf1090 = reinterpret_tensor(buf1056, (128, 4096), (4096, 1), 0); del buf1056  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1089, (128, 1024), (1024, 1), 0), permute_1337, out=buf1090)
        del permute_1337
        buf1091 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1089, (1024, 128), (1, 1024), 0), view_81, out=buf1091)
        del view_81
        buf1092 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1089, buf1092, 1024, 128, grid=grid(1024), stream=stream0)
        buf1093 = reinterpret_tensor(buf1090, (1, 128, 4096), (524288, 4096, 1), 0); del buf1090  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1093, le_20, 524288, grid=grid(524288), stream=stream0)
        del le_20
        buf1094 = buf1082; del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1093, (128, 4096), (4096, 1), 0), permute_1341, out=buf1094)
        del permute_1341
        buf1095 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1093, (4096, 128), (1, 4096), 0), view_79, out=buf1095)
        del view_79
        buf1096 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1093, buf1096, 4096, 128, grid=grid(4096), stream=stream0)
        buf1101 = buf1089; del buf1089  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1101, buf1094, primals_60, mul_20, div_92, 128, 1024, grid=grid(128), stream=stream0)
        del div_92
        del primals_60
        buf1099 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1100 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1094, mul_20, buf1099, buf1100, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_20
        buf1102 = buf1094; del buf1094  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1101, (128, 1024), (1024, 1), 0), permute_1345, out=buf1102)
        del permute_1345
        buf1103 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1101, (1024, 128), (1, 1024), 0), view_77, out=buf1103)
        del view_77
        buf1104 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1101, buf1104, 1024, 128, grid=grid(1024), stream=stream0)
        buf1105 = reinterpret_tensor(buf1078, (16, 128, 64), (8192, 64, 1), 0); del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1350, reinterpret_tensor(buf1102, (16, 128, 64), (64, 1024, 1), 0), out=buf1105)
        del permute_1350
        buf1106 = buf1071; del buf1071  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1102, (16, 128, 64), (64, 1024, 1), 0), permute_1351, out=buf1106)
        del permute_1351
        buf1108 = buf1069; del buf1069  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1106, bmm_6, amax_3, sum_4, buf1108, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_3
        del bmm_6
        del sum_4
        buf1109 = reinterpret_tensor(buf1102, (16, 64, 128), (8192, 128, 1), 0); del buf1102  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1352, buf1108, out=buf1109)
        del permute_1352
        buf1110 = reinterpret_tensor(buf1075, (16, 128, 64), (8192, 64, 1), 0); del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1108, permute_1353, out=buf1110)
        del permute_1353
        buf1111 = buf1081; del buf1081  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1105, buf1111, 131072, grid=grid(131072), stream=stream0)
        buf1112 = reinterpret_tensor(buf1105, (128, 1024), (1024, 1), 0); del buf1105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1111, permute_1357, out=buf1112)
        del permute_1357
        buf1113 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1111, (1024, 128), (1, 1024), 0), view_63, out=buf1113)
        buf1114 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1111, buf1114, 1024, 128, grid=grid(1024), stream=stream0)
        buf1115 = buf1111; del buf1111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1109, (128, 1024), (1, 128), 0), permute_1362, out=buf1115)
        del permute_1362
        buf1116 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1109, (1024, 128), (128, 1), 0), view_63, out=buf1116)
        buf1117 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1109, buf1117, 1024, 128, grid=grid(1024), stream=stream0)
        buf1118 = reinterpret_tensor(buf1109, (128, 1024), (1024, 1), 0); del buf1109  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1110, buf1118, 131072, grid=grid(131072), stream=stream0)
        buf1119 = reinterpret_tensor(buf1110, (128, 1024), (1024, 1), 0); del buf1110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1118, permute_1366, out=buf1119)
        del permute_1366
        buf1120 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1118, (1024, 128), (1, 1024), 0), view_63, out=buf1120)
        del view_63
        buf1121 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1118, buf1121, 1024, 128, grid=grid(1024), stream=stream0)
        buf1126 = buf1101; del buf1101  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1126, buf1112, buf1115, buf1119, primals_50, mul_17, div_93, 128, 1024, grid=grid(128), stream=stream0)
        del div_93
        del primals_50
        buf1124 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1125 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1112, buf1115, buf1119, mul_17, buf1124, buf1125, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_17
        buf1127 = reinterpret_tensor(buf1093, (128, 4096), (4096, 1), 0); del buf1093  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (128, 1024), (1024, 1), 0), permute_1370, out=buf1127)
        del permute_1370
        buf1128 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (1024, 128), (1, 1024), 0), view_61, out=buf1128)
        del view_61
        buf1129 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1126, buf1129, 1024, 128, grid=grid(1024), stream=stream0)
        buf1130 = reinterpret_tensor(buf1127, (1, 128, 4096), (524288, 4096, 1), 0); del buf1127  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1130, le_21, 524288, grid=grid(524288), stream=stream0)
        del le_21
        buf1131 = buf1119; del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1130, (128, 4096), (4096, 1), 0), permute_1374, out=buf1131)
        del permute_1374
        buf1132 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1130, (4096, 128), (1, 4096), 0), view_59, out=buf1132)
        del view_59
        buf1133 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1130, buf1133, 4096, 128, grid=grid(4096), stream=stream0)
        buf1138 = buf1126; del buf1126  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1138, buf1131, primals_44, mul_15, div_94, 128, 1024, grid=grid(128), stream=stream0)
        del div_94
        del primals_44
        buf1136 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1137 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1131, mul_15, buf1136, buf1137, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_15
        buf1139 = buf1131; del buf1131  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1138, (128, 1024), (1024, 1), 0), permute_1378, out=buf1139)
        del permute_1378
        buf1140 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1138, (1024, 128), (1, 1024), 0), view_57, out=buf1140)
        del view_57
        buf1141 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1138, buf1141, 1024, 128, grid=grid(1024), stream=stream0)
        buf1142 = reinterpret_tensor(buf1115, (16, 128, 64), (8192, 64, 1), 0); del buf1115  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1383, reinterpret_tensor(buf1139, (16, 128, 64), (64, 1024, 1), 0), out=buf1142)
        del permute_1383
        buf1143 = buf1108; del buf1108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1139, (16, 128, 64), (64, 1024, 1), 0), permute_1384, out=buf1143)
        del permute_1384
        buf1145 = buf1106; del buf1106  # reuse
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1143, bmm_4, amax_2, sum_3, buf1145, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_2
        del bmm_4
        del sum_3
        buf1146 = reinterpret_tensor(buf1139, (16, 64, 128), (8192, 128, 1), 0); del buf1139  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1385, buf1145, out=buf1146)
        del permute_1385
        buf1147 = reinterpret_tensor(buf1112, (16, 128, 64), (8192, 64, 1), 0); del buf1112  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1145, permute_1386, out=buf1147)
        del permute_1386
        buf1148 = buf1118; del buf1118  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1142, buf1148, 131072, grid=grid(131072), stream=stream0)
        buf1149 = reinterpret_tensor(buf1142, (128, 1024), (1024, 1), 0); del buf1142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1148, permute_1390, out=buf1149)
        del permute_1390
        buf1150 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1148, (1024, 128), (1, 1024), 0), view_43, out=buf1150)
        buf1151 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1148, buf1151, 1024, 128, grid=grid(1024), stream=stream0)
        buf1152 = buf1148; del buf1148  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1146, (128, 1024), (1, 128), 0), permute_1395, out=buf1152)
        del permute_1395
        buf1153 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1146, (1024, 128), (128, 1), 0), view_43, out=buf1153)
        buf1154 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1146, buf1154, 1024, 128, grid=grid(1024), stream=stream0)
        buf1155 = reinterpret_tensor(buf1146, (128, 1024), (1024, 1), 0); del buf1146  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1147, buf1155, 131072, grid=grid(131072), stream=stream0)
        buf1156 = reinterpret_tensor(buf1147, (128, 1024), (1024, 1), 0); del buf1147  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1155, permute_1399, out=buf1156)
        del permute_1399
        buf1157 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1155, (1024, 128), (1, 1024), 0), view_43, out=buf1157)
        del view_43
        buf1158 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1155, buf1158, 1024, 128, grid=grid(1024), stream=stream0)
        buf1163 = buf1138; del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1163, buf1149, buf1152, buf1156, primals_34, mul_12, div_95, 128, 1024, grid=grid(128), stream=stream0)
        del div_95
        del primals_34
        buf1161 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1162 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1149, buf1152, buf1156, mul_12, buf1161, buf1162, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_12
        buf1164 = reinterpret_tensor(buf1130, (128, 4096), (4096, 1), 0); del buf1130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1163, (128, 1024), (1024, 1), 0), permute_1403, out=buf1164)
        del permute_1403
        buf1165 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1163, (1024, 128), (1, 1024), 0), view_41, out=buf1165)
        del view_41
        buf1166 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1163, buf1166, 1024, 128, grid=grid(1024), stream=stream0)
        buf1167 = reinterpret_tensor(buf1164, (1, 128, 4096), (524288, 4096, 1), 0); del buf1164  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1167, le_22, 524288, grid=grid(524288), stream=stream0)
        del le_22
        buf1168 = buf1156; del buf1156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1167, (128, 4096), (4096, 1), 0), permute_1407, out=buf1168)
        del permute_1407
        buf1169 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1167, (4096, 128), (1, 4096), 0), view_39, out=buf1169)
        del view_39
        buf1170 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1167, buf1170, 4096, 128, grid=grid(4096), stream=stream0)
        buf1175 = buf1163; del buf1163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1175, buf1168, primals_28, mul_10, div_96, 128, 1024, grid=grid(128), stream=stream0)
        del div_96
        del primals_28
        buf1173 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1174 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1168, mul_10, buf1173, buf1174, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_10
        buf1176 = buf1168; del buf1168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1175, (128, 1024), (1024, 1), 0), permute_1411, out=buf1176)
        del permute_1411
        buf1177 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1175, (1024, 128), (1, 1024), 0), view_37, out=buf1177)
        del view_37
        buf1178 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1175, buf1178, 1024, 128, grid=grid(1024), stream=stream0)
        buf1179 = reinterpret_tensor(buf1152, (16, 128, 64), (8192, 64, 1), 0); del buf1152  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1416, reinterpret_tensor(buf1176, (16, 128, 64), (64, 1024, 1), 0), out=buf1179)
        del permute_1416
        buf1180 = buf1145; del buf1145  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1176, (16, 128, 64), (64, 1024, 1), 0), permute_1417, out=buf1180)
        del permute_1417
        buf1182 = buf1143; del buf1143  # reuse
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1180, bmm_2, amax_1, sum_2, buf1182, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_1
        del bmm_2
        del sum_2
        buf1183 = reinterpret_tensor(buf1176, (16, 64, 128), (8192, 128, 1), 0); del buf1176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1418, buf1182, out=buf1183)
        del permute_1418
        buf1184 = reinterpret_tensor(buf1149, (16, 128, 64), (8192, 64, 1), 0); del buf1149  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1182, permute_1419, out=buf1184)
        del permute_1419
        buf1185 = buf1155; del buf1155  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1179, buf1185, 131072, grid=grid(131072), stream=stream0)
        buf1186 = reinterpret_tensor(buf1179, (128, 1024), (1024, 1), 0); del buf1179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1185, permute_1423, out=buf1186)
        del permute_1423
        buf1187 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1185, (1024, 128), (1, 1024), 0), view_23, out=buf1187)
        buf1188 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1185, buf1188, 1024, 128, grid=grid(1024), stream=stream0)
        buf1189 = buf1185; del buf1185  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1183, (128, 1024), (1, 128), 0), permute_1428, out=buf1189)
        del permute_1428
        buf1190 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1183, (1024, 128), (128, 1), 0), view_23, out=buf1190)
        buf1191 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1183, buf1191, 1024, 128, grid=grid(1024), stream=stream0)
        buf1192 = reinterpret_tensor(buf1183, (128, 1024), (1024, 1), 0); del buf1183  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1184, buf1192, 131072, grid=grid(131072), stream=stream0)
        buf1193 = reinterpret_tensor(buf1184, (128, 1024), (1024, 1), 0); del buf1184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1192, permute_1432, out=buf1193)
        del permute_1432
        buf1194 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1192, (1024, 128), (1, 1024), 0), view_23, out=buf1194)
        del view_23
        buf1195 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1192, buf1195, 1024, 128, grid=grid(1024), stream=stream0)
        buf1200 = buf1175; del buf1175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf1200, buf1186, buf1189, buf1193, primals_18, mul_7, div_97, 128, 1024, grid=grid(128), stream=stream0)
        del div_97
        del primals_18
        buf1198 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1199 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1186, buf1189, buf1193, mul_7, buf1198, buf1199, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_7
        buf1201 = reinterpret_tensor(buf1167, (128, 4096), (4096, 1), 0); del buf1167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1200, (128, 1024), (1024, 1), 0), permute_1436, out=buf1201)
        del permute_1436
        buf1202 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1200, (1024, 128), (1, 1024), 0), view_21, out=buf1202)
        del view_21
        buf1203 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1200, buf1203, 1024, 128, grid=grid(1024), stream=stream0)
        buf1204 = reinterpret_tensor(buf1201, (1, 128, 4096), (524288, 4096, 1), 0); del buf1201  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.masked_fill, aten.threshold_backward]
        triton_poi_fused_masked_fill_threshold_backward_8.run(buf1204, le_23, 524288, grid=grid(524288), stream=stream0)
        del le_23
        buf1205 = buf1193; del buf1193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1204, (128, 4096), (4096, 1), 0), permute_1440, out=buf1205)
        del permute_1440
        buf1206 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1204, (4096, 128), (1, 4096), 0), view_19, out=buf1206)
        del view_19
        buf1207 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1204, buf1207, 4096, 128, grid=grid(4096), stream=stream0)
        del buf1204
        buf1212 = buf1200; del buf1200  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf1212, buf1205, primals_12, mul_5, div_98, 128, 1024, grid=grid(128), stream=stream0)
        del div_98
        del primals_12
        buf1210 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1211 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf1205, mul_5, buf1210, buf1211, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_5
        buf1213 = buf1205; del buf1205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1212, (128, 1024), (1024, 1), 0), permute_1444, out=buf1213)
        del permute_1444
        buf1214 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1212, (1024, 128), (1, 1024), 0), view_17, out=buf1214)
        del view_17
        buf1215 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf1212, buf1215, 1024, 128, grid=grid(1024), stream=stream0)
        buf1216 = reinterpret_tensor(buf1189, (16, 128, 64), (8192, 64, 1), 0); del buf1189  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1449, reinterpret_tensor(buf1213, (16, 128, 64), (64, 1024, 1), 0), out=buf1216)
        del permute_1449
        buf1217 = buf1182; del buf1182  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1213, (16, 128, 64), (64, 1024, 1), 0), permute_1450, out=buf1217)
        del permute_1450
        buf1219 = buf1180; del buf1180  # reuse
        # Source Nodes: [attn_weights_1], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_11.run(buf1217, bmm, amax, sum_1, buf1219, 2048, 128, grid=grid(2048), stream=stream0)
        del amax
        del bmm
        del buf1217
        del sum_1
        buf1220 = reinterpret_tensor(buf1213, (16, 64, 128), (8192, 128, 1), 0); del buf1213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1451, buf1219, out=buf1220)
        del permute_1451
        buf1221 = reinterpret_tensor(buf1186, (16, 128, 64), (8192, 64, 1), 0); del buf1186  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1219, permute_1452, out=buf1221)
        del buf1219
        del permute_1452
        buf1222 = buf1192; del buf1192  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf1216, buf1222, 131072, grid=grid(131072), stream=stream0)
        buf1223 = reinterpret_tensor(buf1216, (128, 1024), (1024, 1), 0); del buf1216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1222, permute_1456, out=buf1223)
        del permute_1456
        buf1224 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1222, (1024, 128), (1, 1024), 0), view_3, out=buf1224)
        buf1225 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1222, buf1225, 1024, 128, grid=grid(1024), stream=stream0)
        buf1226 = buf1222; del buf1222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1220, (128, 1024), (1, 128), 0), permute_1461, out=buf1226)
        del permute_1461
        buf1227 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1220, (1024, 128), (128, 1), 0), view_3, out=buf1227)
        buf1228 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf1220, buf1228, 1024, 128, grid=grid(1024), stream=stream0)
        buf1229 = reinterpret_tensor(buf1220, (128, 1024), (1024, 1), 0); del buf1220  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf1221, buf1229, 131072, grid=grid(131072), stream=stream0)
        buf1230 = reinterpret_tensor(buf1221, (128, 1024), (1024, 1), 0); del buf1221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1229, permute_1465, out=buf1230)
        del permute_1465
        buf1231 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1229, (1024, 128), (1, 1024), 0), view_3, out=buf1231)
        del view_3
        buf1232 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf1229, buf1232, 1024, 128, grid=grid(1024), stream=stream0)
        del buf1229
        buf1237 = buf1212; del buf1212  # reuse
        buf1239 = buf1237; del buf1237  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_22.run(buf1239, buf1223, buf1226, buf1230, primals_2, mul_2, div_99, view, 128, 1024, grid=grid(128), stream=stream0)
        del div_99
        del primals_2
        buf1235 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1236 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf1223, buf1226, buf1230, mul_2, buf1235, buf1236, 1024, 128, grid=grid(1024), stream=stream0)
        del buf1223
        del buf1226
        del buf1230
        del mul_2
        buf1238 = empty((128112, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_masked_fill_mul_23.run(buf1238, 131186688, grid=grid(131186688), stream=stream0)
        aten.index_put_(buf1238, [view], buf1239, True)
        del buf1239
        del view
        return (buf1238, buf1235, buf1236, reinterpret_tensor(buf1231, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1232, (1024, ), (1, ), 0), reinterpret_tensor(buf1227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1228, (1024, ), (1, ), 0), reinterpret_tensor(buf1224, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1225, (1024, ), (1, ), 0), reinterpret_tensor(buf1214, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1215, (1024, ), (1, ), 0), buf1210, buf1211, reinterpret_tensor(buf1206, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1207, (4096, ), (1, ), 0), reinterpret_tensor(buf1202, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1203, (1024, ), (1, ), 0), buf1198, buf1199, reinterpret_tensor(buf1194, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1195, (1024, ), (1, ), 0), reinterpret_tensor(buf1190, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1191, (1024, ), (1, ), 0), reinterpret_tensor(buf1187, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1188, (1024, ), (1, ), 0), reinterpret_tensor(buf1177, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1178, (1024, ), (1, ), 0), buf1173, buf1174, reinterpret_tensor(buf1169, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1170, (4096, ), (1, ), 0), reinterpret_tensor(buf1165, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1166, (1024, ), (1, ), 0), buf1161, buf1162, reinterpret_tensor(buf1157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1158, (1024, ), (1, ), 0), reinterpret_tensor(buf1153, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1154, (1024, ), (1, ), 0), reinterpret_tensor(buf1150, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1151, (1024, ), (1, ), 0), reinterpret_tensor(buf1140, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1141, (1024, ), (1, ), 0), buf1136, buf1137, reinterpret_tensor(buf1132, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1133, (4096, ), (1, ), 0), reinterpret_tensor(buf1128, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1129, (1024, ), (1, ), 0), buf1124, buf1125, reinterpret_tensor(buf1120, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1121, (1024, ), (1, ), 0), reinterpret_tensor(buf1116, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1117, (1024, ), (1, ), 0), reinterpret_tensor(buf1113, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1114, (1024, ), (1, ), 0), reinterpret_tensor(buf1103, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1104, (1024, ), (1, ), 0), buf1099, buf1100, reinterpret_tensor(buf1095, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1096, (4096, ), (1, ), 0), reinterpret_tensor(buf1091, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1092, (1024, ), (1, ), 0), buf1087, buf1088, reinterpret_tensor(buf1083, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1084, (1024, ), (1, ), 0), reinterpret_tensor(buf1079, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1080, (1024, ), (1, ), 0), reinterpret_tensor(buf1076, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1077, (1024, ), (1, ), 0), reinterpret_tensor(buf1066, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1067, (1024, ), (1, ), 0), buf1062, buf1063, reinterpret_tensor(buf1058, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1059, (4096, ), (1, ), 0), reinterpret_tensor(buf1054, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1055, (1024, ), (1, ), 0), buf1050, buf1051, reinterpret_tensor(buf1046, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1047, (1024, ), (1, ), 0), reinterpret_tensor(buf1042, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1043, (1024, ), (1, ), 0), reinterpret_tensor(buf1039, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1040, (1024, ), (1, ), 0), reinterpret_tensor(buf1029, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1030, (1024, ), (1, ), 0), buf1025, buf1026, reinterpret_tensor(buf1021, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1022, (4096, ), (1, ), 0), reinterpret_tensor(buf1017, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1018, (1024, ), (1, ), 0), buf1013, buf1014, reinterpret_tensor(buf1009, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1010, (1024, ), (1, ), 0), reinterpret_tensor(buf1005, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1006, (1024, ), (1, ), 0), reinterpret_tensor(buf1002, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1003, (1024, ), (1, ), 0), reinterpret_tensor(buf992, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf993, (1024, ), (1, ), 0), buf988, buf989, reinterpret_tensor(buf984, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf985, (4096, ), (1, ), 0), reinterpret_tensor(buf980, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf981, (1024, ), (1, ), 0), buf976, buf977, reinterpret_tensor(buf972, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf973, (1024, ), (1, ), 0), reinterpret_tensor(buf968, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf969, (1024, ), (1, ), 0), reinterpret_tensor(buf965, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf966, (1024, ), (1, ), 0), reinterpret_tensor(buf955, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf956, (1024, ), (1, ), 0), buf951, buf952, reinterpret_tensor(buf947, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf948, (4096, ), (1, ), 0), reinterpret_tensor(buf943, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf944, (1024, ), (1, ), 0), buf939, buf940, reinterpret_tensor(buf935, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf936, (1024, ), (1, ), 0), reinterpret_tensor(buf931, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf932, (1024, ), (1, ), 0), reinterpret_tensor(buf928, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf929, (1024, ), (1, ), 0), reinterpret_tensor(buf918, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf919, (1024, ), (1, ), 0), buf914, buf915, reinterpret_tensor(buf910, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf911, (4096, ), (1, ), 0), reinterpret_tensor(buf906, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf907, (1024, ), (1, ), 0), buf902, buf903, reinterpret_tensor(buf898, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf899, (1024, ), (1, ), 0), reinterpret_tensor(buf894, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf895, (1024, ), (1, ), 0), reinterpret_tensor(buf891, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf892, (1024, ), (1, ), 0), reinterpret_tensor(buf881, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf882, (1024, ), (1, ), 0), buf877, buf878, reinterpret_tensor(buf873, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf874, (4096, ), (1, ), 0), reinterpret_tensor(buf869, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf870, (1024, ), (1, ), 0), buf865, buf866, reinterpret_tensor(buf861, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf862, (1024, ), (1, ), 0), reinterpret_tensor(buf857, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf858, (1024, ), (1, ), 0), reinterpret_tensor(buf854, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf855, (1024, ), (1, ), 0), reinterpret_tensor(buf844, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf845, (1024, ), (1, ), 0), buf840, buf841, reinterpret_tensor(buf836, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf837, (4096, ), (1, ), 0), reinterpret_tensor(buf832, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf833, (1024, ), (1, ), 0), buf828, buf829, reinterpret_tensor(buf824, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf825, (1024, ), (1, ), 0), reinterpret_tensor(buf820, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf821, (1024, ), (1, ), 0), reinterpret_tensor(buf817, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf818, (1024, ), (1, ), 0), reinterpret_tensor(buf807, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf808, (1024, ), (1, ), 0), buf803, buf804, reinterpret_tensor(buf799, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf800, (4096, ), (1, ), 0), reinterpret_tensor(buf795, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf796, (1024, ), (1, ), 0), buf792, buf793, buf785, buf782, buf783, reinterpret_tensor(buf778, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf779, (1024, ), (1, ), 0), reinterpret_tensor(buf774, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf775, (1024, ), (1, ), 0), reinterpret_tensor(buf770, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf771, (1024, ), (1, ), 0), reinterpret_tensor(buf760, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf761, (1024, ), (1, ), 0), buf756, buf757, reinterpret_tensor(buf752, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf753, (1024, ), (1, ), 0), reinterpret_tensor(buf747, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf748, (1024, ), (1, ), 0), reinterpret_tensor(buf743, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf744, (1024, ), (1, ), 0), reinterpret_tensor(buf733, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf734, (1024, ), (1, ), 0), buf729, buf730, reinterpret_tensor(buf725, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf726, (4096, ), (1, ), 0), reinterpret_tensor(buf721, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf722, (1024, ), (1, ), 0), buf717, buf718, reinterpret_tensor(buf713, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf714, (1024, ), (1, ), 0), reinterpret_tensor(buf709, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf710, (1024, ), (1, ), 0), reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf706, (1024, ), (1, ), 0), reinterpret_tensor(buf695, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf696, (1024, ), (1, ), 0), buf691, buf692, reinterpret_tensor(buf687, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf688, (1024, ), (1, ), 0), reinterpret_tensor(buf683, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf684, (1024, ), (1, ), 0), reinterpret_tensor(buf679, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf680, (1024, ), (1, ), 0), reinterpret_tensor(buf669, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf670, (1024, ), (1, ), 0), buf665, buf666, reinterpret_tensor(buf661, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf662, (4096, ), (1, ), 0), reinterpret_tensor(buf657, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf658, (1024, ), (1, ), 0), buf653, buf654, reinterpret_tensor(buf649, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf650, (1024, ), (1, ), 0), reinterpret_tensor(buf645, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf646, (1024, ), (1, ), 0), reinterpret_tensor(buf641, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf642, (1024, ), (1, ), 0), reinterpret_tensor(buf631, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf632, (1024, ), (1, ), 0), buf627, buf628, reinterpret_tensor(buf623, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf624, (1024, ), (1, ), 0), reinterpret_tensor(buf619, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf620, (1024, ), (1, ), 0), reinterpret_tensor(buf615, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf616, (1024, ), (1, ), 0), reinterpret_tensor(buf605, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf606, (1024, ), (1, ), 0), buf601, buf602, reinterpret_tensor(buf597, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf598, (4096, ), (1, ), 0), reinterpret_tensor(buf593, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf594, (1024, ), (1, ), 0), buf589, buf590, reinterpret_tensor(buf585, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf586, (1024, ), (1, ), 0), reinterpret_tensor(buf581, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf582, (1024, ), (1, ), 0), reinterpret_tensor(buf577, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf578, (1024, ), (1, ), 0), reinterpret_tensor(buf567, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf568, (1024, ), (1, ), 0), buf563, buf564, reinterpret_tensor(buf559, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf560, (1024, ), (1, ), 0), reinterpret_tensor(buf555, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf556, (1024, ), (1, ), 0), reinterpret_tensor(buf551, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf552, (1024, ), (1, ), 0), reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf542, (1024, ), (1, ), 0), buf537, buf538, reinterpret_tensor(buf533, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf534, (4096, ), (1, ), 0), reinterpret_tensor(buf529, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf530, (1024, ), (1, ), 0), buf525, buf526, reinterpret_tensor(buf521, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf522, (1024, ), (1, ), 0), reinterpret_tensor(buf517, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf518, (1024, ), (1, ), 0), reinterpret_tensor(buf513, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf514, (1024, ), (1, ), 0), reinterpret_tensor(buf503, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf504, (1024, ), (1, ), 0), buf499, buf500, reinterpret_tensor(buf495, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf496, (1024, ), (1, ), 0), reinterpret_tensor(buf490, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf491, (1024, ), (1, ), 0), reinterpret_tensor(buf486, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf487, (1024, ), (1, ), 0), reinterpret_tensor(buf476, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf477, (1024, ), (1, ), 0), buf472, buf473, reinterpret_tensor(buf468, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf469, (4096, ), (1, ), 0), reinterpret_tensor(buf464, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf465, (1024, ), (1, ), 0), buf460, buf461, reinterpret_tensor(buf456, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf457, (1024, ), (1, ), 0), reinterpret_tensor(buf452, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf453, (1024, ), (1, ), 0), reinterpret_tensor(buf448, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf449, (1024, ), (1, ), 0), reinterpret_tensor(buf438, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf439, (1024, ), (1, ), 0), buf434, buf435, reinterpret_tensor(buf430, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf431, (1024, ), (1, ), 0), reinterpret_tensor(buf426, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf427, (1024, ), (1, ), 0), reinterpret_tensor(buf422, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf423, (1024, ), (1, ), 0), reinterpret_tensor(buf412, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf413, (1024, ), (1, ), 0), buf408, buf409, reinterpret_tensor(buf404, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf405, (4096, ), (1, ), 0), reinterpret_tensor(buf400, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf401, (1024, ), (1, ), 0), buf396, buf397, reinterpret_tensor(buf392, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf393, (1024, ), (1, ), 0), reinterpret_tensor(buf388, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf389, (1024, ), (1, ), 0), reinterpret_tensor(buf384, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf385, (1024, ), (1, ), 0), reinterpret_tensor(buf374, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf375, (1024, ), (1, ), 0), buf370, buf371, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf367, (1024, ), (1, ), 0), reinterpret_tensor(buf362, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf363, (1024, ), (1, ), 0), reinterpret_tensor(buf358, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf359, (1024, ), (1, ), 0), reinterpret_tensor(buf348, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf349, (1024, ), (1, ), 0), buf344, buf345, reinterpret_tensor(buf340, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf341, (4096, ), (1, ), 0), reinterpret_tensor(buf336, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf337, (1024, ), (1, ), 0), buf332, buf333, reinterpret_tensor(buf328, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf329, (1024, ), (1, ), 0), reinterpret_tensor(buf324, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf325, (1024, ), (1, ), 0), reinterpret_tensor(buf320, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf321, (1024, ), (1, ), 0), reinterpret_tensor(buf310, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf311, (1024, ), (1, ), 0), buf306, buf307, reinterpret_tensor(buf302, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf303, (1024, ), (1, ), 0), reinterpret_tensor(buf298, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf299, (1024, ), (1, ), 0), reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf295, (1024, ), (1, ), 0), reinterpret_tensor(buf284, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf285, (1024, ), (1, ), 0), buf280, buf281, reinterpret_tensor(buf276, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf277, (4096, ), (1, ), 0), reinterpret_tensor(buf272, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf273, (1024, ), (1, ), 0), buf268, buf269, reinterpret_tensor(buf264, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf265, (1024, ), (1, ), 0), reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf261, (1024, ), (1, ), 0), reinterpret_tensor(buf256, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf257, (1024, ), (1, ), 0), reinterpret_tensor(buf246, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf247, (1024, ), (1, ), 0), buf242, buf243, reinterpret_tensor(buf238, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf239, (1024, ), (1, ), 0), reinterpret_tensor(buf233, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf234, (1024, ), (1, ), 0), reinterpret_tensor(buf229, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf230, (1024, ), (1, ), 0), reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf220, (1024, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf211, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf212, (4096, ), (1, ), 0), reinterpret_tensor(buf207, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf208, (1024, ), (1, ), 0), buf203, buf204, reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf200, (1024, ), (1, ), 0), reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf196, (1024, ), (1, ), 0), reinterpret_tensor(buf191, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf192, (1024, ), (1, ), 0), reinterpret_tensor(buf181, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf182, (1024, ), (1, ), 0), buf177, buf178, reinterpret_tensor(buf173, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf174, (1024, ), (1, ), 0), reinterpret_tensor(buf169, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf170, (1024, ), (1, ), 0), reinterpret_tensor(buf165, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf166, (1024, ), (1, ), 0), reinterpret_tensor(buf155, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf156, (1024, ), (1, ), 0), buf151, buf152, reinterpret_tensor(buf147, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf148, (4096, ), (1, ), 0), reinterpret_tensor(buf143, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf144, (1024, ), (1, ), 0), buf139, buf140, reinterpret_tensor(buf135, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf136, (1024, ), (1, ), 0), reinterpret_tensor(buf131, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf132, (1024, ), (1, ), 0), reinterpret_tensor(buf127, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf128, (1024, ), (1, ), 0), reinterpret_tensor(buf117, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf118, (1024, ), (1, ), 0), buf113, buf114, reinterpret_tensor(buf109, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf110, (1024, ), (1, ), 0), reinterpret_tensor(buf105, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf106, (1024, ), (1, ), 0), reinterpret_tensor(buf101, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf102, (1024, ), (1, ), 0), reinterpret_tensor(buf91, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf92, (1024, ), (1, ), 0), buf87, buf88, reinterpret_tensor(buf83, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf84, (4096, ), (1, ), 0), reinterpret_tensor(buf79, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf80, (1024, ), (1, ), 0), buf75, buf76, reinterpret_tensor(buf71, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (1024, ), (1, ), 0), reinterpret_tensor(buf67, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf68, (1024, ), (1, ), 0), reinterpret_tensor(buf63, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf64, (1024, ), (1, ), 0), reinterpret_tensor(buf53, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf54, (1024, ), (1, ), 0), buf49, buf50, reinterpret_tensor(buf45, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf46, (1024, ), (1, ), 0), reinterpret_tensor(buf41, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf42, (1024, ), (1, ), 0), reinterpret_tensor(buf37, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf38, (1024, ), (1, ), 0), reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf28, (1024, ), (1, ), 0), buf23, buf24, reinterpret_tensor(buf19, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf20, (4096, ), (1, ), 0), reinterpret_tensor(buf15, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf16, (1024, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf7, (128112, 1024), (1024, 1), 0), None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    mul_2 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_4 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_2 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_6 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_3 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_4 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_8 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_4 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_5 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_10 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_5 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_12 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_6 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_7 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_14 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_7 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_8 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_159 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_16 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_8 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_9 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_181 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_47 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_18 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_9 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_10 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_197 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_199 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_201 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_20 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_10 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_11 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_217 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_221 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_223 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_22 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_11 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_12 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_237 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_239 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_241 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_62 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_243 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    mul_66 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_247 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_263 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_265 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_267 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_26 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_13 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_14 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_279 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_281 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_283 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_285 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_301 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_303 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_30 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_15 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_16 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_317 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_319 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_321 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_323 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_339 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_341 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_34 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_17 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_18 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_355 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_357 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_359 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_361 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_377 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_379 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_38 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_19 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_20 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_393 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_395 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_397 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_98 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_399 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_415 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_101 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_417 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_42 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_21 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_22 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_431 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_433 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_435 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_106 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_437 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_453 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_109 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_455 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_46 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_23 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_24 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_469 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_471 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_473 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_114 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_475 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_491 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_493 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_50 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_25 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_26 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_507 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_509 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_511 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_122 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_513 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_529 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_531 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_54 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_27 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_28 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_545 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_547 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_549 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_130 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_551 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_567 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_569 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_58 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_29 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_30 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_583 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_585 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_587 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_589 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_605 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_607 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_62 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_31 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_32 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_621 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_623 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_625 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_146 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_627 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_643 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_149 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_645 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_66 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_33 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_34 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_659 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_661 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_663 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_665 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_681 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_683 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_70 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_35 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_36 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_697 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_160 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_699 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_701 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_162 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_703 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    sub_99 = rand_strided((128, 128112), (128112, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_381 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_391 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_392 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_397 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_68 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_439 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_449 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_473 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_474 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_71 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_489 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_497 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_507 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_531 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_532 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_74 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_533 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_538 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_555 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_559 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_564 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_565 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_566 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_576 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_580 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_584 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_589 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_77 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_592 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_605 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_609 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_613 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_617 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_622 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_634 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_638 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_647 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_648 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_80 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_649 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_650 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_659 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_663 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_671 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_680 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_681 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_682 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_683 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_692 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_696 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_705 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_706 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_83 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_707 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_708 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_712 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_729 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_733 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_738 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_739 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_741 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_745 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_754 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_758 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_763 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_764 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_86 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_770 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_775 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_779 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_783 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_787 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_791 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_796 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_797 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_799 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_803 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_808 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_812 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_816 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_821 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_89 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_824 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_828 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_833 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_837 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_841 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_845 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_849 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_854 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_855 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_856 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_857 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_861 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_866 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_870 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_874 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_879 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_880 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_92 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_881 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_882 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_886 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_891 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_895 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_65 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_899 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_903 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_907 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_912 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_913 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_914 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_915 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_919 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_924 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_928 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_932 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_937 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_938 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_95 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_939 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_940 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_944 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_949 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_953 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_68 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_957 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_961 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_965 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_970 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_971 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_972 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_973 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_977 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_982 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_986 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_990 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_995 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_996 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_98 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_997 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_998 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1002 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1007 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1011 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1015 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1019 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1023 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1028 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1029 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1030 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1031 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1035 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1040 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1044 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1048 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1053 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1054 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_101 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_1055 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1056 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1060 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1065 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1069 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1073 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1077 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1081 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1086 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1087 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1088 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1089 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1093 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1098 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1102 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1106 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1110 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1114 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1119 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1120 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1121 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1122 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1126 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1131 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1135 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1139 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1143 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1147 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1152 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1153 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1154 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1155 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1159 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1164 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1168 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1172 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1176 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1180 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1185 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1186 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1187 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1188 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1192 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1197 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1201 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1205 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1209 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1213 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1218 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1219 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1220 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1221 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1225 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1230 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1234 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1238 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1242 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1246 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1251 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1252 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1253 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1254 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1258 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1263 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1267 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1271 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1275 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1279 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1284 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1285 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1286 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1287 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1291 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1296 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1300 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_89 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1304 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1308 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1312 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1317 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1318 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1319 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1320 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1324 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1329 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1333 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1337 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_20 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1341 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_92 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1345 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1350 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1351 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1352 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1353 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1357 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1362 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1366 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1370 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1374 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1378 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1383 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1384 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1385 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1386 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1390 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1395 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1399 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_95 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1403 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1407 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_96 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1411 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1416 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1417 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1418 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1419 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1423 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1428 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1432 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_97 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1436 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((1, 128, 4096), (524288, 4096, 1), device='cuda:0', dtype=torch.bool)
    permute_1440 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_98 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1444 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1449 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1450 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1451 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1452 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_1456 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1461 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1465 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_99 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 128112), (16398336, 128112, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_37 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_40 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_43 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_46 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_49 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_50 = rand_strided((1, 16, 128, 64), (131072, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_51 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_12, primals_18, primals_28, primals_34, primals_44, primals_50, primals_60, primals_66, primals_76, primals_82, primals_92, primals_98, primals_108, primals_114, primals_124, primals_130, primals_140, primals_146, primals_156, primals_162, primals_172, primals_178, primals_188, primals_194, primals_197, primals_207, primals_217, primals_223, primals_233, primals_243, primals_249, primals_259, primals_269, primals_275, primals_285, primals_295, primals_301, primals_311, primals_321, primals_327, primals_337, primals_347, primals_353, primals_363, primals_373, primals_379, primals_389, primals_399, primals_405, primals_415, primals_425, primals_431, primals_441, primals_451, primals_457, primals_467, primals_477, primals_483, primals_493, primals_503, primals_509, primals_514, view, mul_2, view_3, bmm, amax, sum_1, view_17, mul_5, view_19, view_21, mul_7, view_23, bmm_2, amax_1, sum_2, view_37, mul_10, view_39, view_41, mul_12, view_43, bmm_4, amax_2, sum_3, view_57, mul_15, view_59, view_61, mul_17, view_63, bmm_6, amax_3, sum_4, view_77, mul_20, view_79, view_81, mul_22, view_83, bmm_8, amax_4, sum_5, view_97, mul_25, view_99, view_101, mul_27, view_103, bmm_10, amax_5, sum_6, view_117, mul_30, view_119, view_121, mul_32, view_123, bmm_12, amax_6, sum_7, view_137, mul_35, view_139, view_141, mul_37, view_143, bmm_14, amax_7, sum_8, view_157, mul_40, view_159, view_161, mul_42, view_163, bmm_16, amax_8, sum_9, view_177, mul_45, view_179, view_181, mul_47, view_183, bmm_18, amax_9, sum_10, view_197, mul_50, view_199, view_201, mul_52, view_203, bmm_20, amax_10, sum_11, view_217, mul_55, view_219, view_221, mul_57, view_223, bmm_22, amax_11, sum_12, view_237, mul_60, view_239, view_241, mul_62, view_243, mul_66, view_247, view_263, mul_69, view_265, view_267, bmm_26, amax_13, sum_14, view_279, mul_72, view_281, view_283, mul_74, view_285, view_301, mul_77, view_303, bmm_30, amax_15, sum_16, view_317, mul_80, view_319, view_321, mul_82, view_323, view_339, mul_85, view_341, bmm_34, amax_17, sum_18, view_355, mul_88, view_357, view_359, mul_90, view_361, view_377, mul_93, view_379, bmm_38, amax_19, sum_20, view_393, mul_96, view_395, view_397, mul_98, view_399, view_415, mul_101, view_417, bmm_42, amax_21, sum_22, view_431, mul_104, view_433, view_435, mul_106, view_437, view_453, mul_109, view_455, bmm_46, amax_23, sum_24, view_469, mul_112, view_471, view_473, mul_114, view_475, view_491, mul_117, view_493, bmm_50, amax_25, sum_26, view_507, mul_120, view_509, view_511, mul_122, view_513, view_529, mul_125, view_531, bmm_54, amax_27, sum_28, view_545, mul_128, view_547, view_549, mul_130, view_551, view_567, mul_133, view_569, bmm_58, amax_29, sum_30, view_583, mul_136, view_585, view_587, mul_138, view_589, view_605, mul_141, view_607, bmm_62, amax_31, sum_32, view_621, mul_144, view_623, view_625, mul_146, view_627, view_643, mul_149, view_645, bmm_66, amax_33, sum_34, view_659, mul_152, view_661, view_663, mul_154, view_665, view_681, mul_157, view_683, bmm_70, amax_35, sum_36, view_697, mul_160, view_699, view_701, mul_162, view_703, sub_99, convert_element_type_6, permute_375, div_38, permute_377, le, permute_381, div_39, permute_385, permute_390, permute_391, permute_392, permute_393, permute_397, permute_402, permute_406, div_40, permute_410, permute_415, permute_416, alias_68, permute_417, permute_418, permute_422, permute_427, permute_431, div_41, permute_435, le_1, permute_439, div_42, permute_443, permute_448, permute_449, permute_450, permute_451, permute_455, permute_460, permute_464, div_43, permute_468, permute_473, permute_474, alias_71, permute_475, permute_476, permute_480, permute_485, permute_489, div_44, permute_493, le_2, permute_497, div_45, permute_501, permute_506, permute_507, permute_508, permute_509, permute_513, permute_518, permute_522, div_46, permute_526, permute_531, permute_532, alias_74, permute_533, permute_534, permute_538, permute_543, permute_547, div_47, permute_551, le_3, permute_555, div_48, permute_559, permute_564, permute_565, permute_566, permute_567, permute_571, permute_576, permute_580, div_49, permute_584, permute_589, permute_590, alias_77, permute_591, permute_592, permute_596, permute_601, permute_605, div_50, permute_609, le_4, permute_613, div_51, permute_617, permute_622, permute_623, permute_624, permute_625, permute_629, permute_634, permute_638, div_52, permute_642, permute_647, permute_648, alias_80, permute_649, permute_650, permute_654, permute_659, permute_663, div_53, permute_667, le_5, permute_671, div_54, permute_675, permute_680, permute_681, permute_682, permute_683, permute_687, permute_692, permute_696, div_55, permute_700, permute_705, permute_706, alias_83, permute_707, permute_708, permute_712, permute_717, permute_721, div_56, permute_725, le_6, permute_729, div_57, permute_733, permute_738, permute_739, permute_740, permute_741, permute_745, permute_750, permute_754, div_58, permute_758, permute_763, permute_764, alias_86, permute_765, permute_766, permute_770, permute_775, permute_779, div_59, permute_783, le_7, permute_787, div_60, permute_791, permute_796, permute_797, permute_798, permute_799, permute_803, permute_808, permute_812, div_61, permute_816, permute_821, permute_822, alias_89, permute_823, permute_824, permute_828, permute_833, permute_837, div_62, permute_841, le_8, permute_845, div_63, permute_849, permute_854, permute_855, permute_856, permute_857, permute_861, permute_866, permute_870, div_64, permute_874, permute_879, permute_880, alias_92, permute_881, permute_882, permute_886, permute_891, permute_895, div_65, permute_899, le_9, permute_903, div_66, permute_907, permute_912, permute_913, permute_914, permute_915, permute_919, permute_924, permute_928, div_67, permute_932, permute_937, permute_938, alias_95, permute_939, permute_940, permute_944, permute_949, permute_953, div_68, permute_957, le_10, permute_961, div_69, permute_965, permute_970, permute_971, permute_972, permute_973, permute_977, permute_982, permute_986, div_70, permute_990, permute_995, permute_996, alias_98, permute_997, permute_998, permute_1002, permute_1007, permute_1011, div_71, permute_1015, le_11, permute_1019, div_72, permute_1023, permute_1028, permute_1029, permute_1030, permute_1031, permute_1035, permute_1040, permute_1044, div_73, permute_1048, permute_1053, permute_1054, alias_101, permute_1055, permute_1056, permute_1060, permute_1065, permute_1069, div_74, div_75, permute_1073, le_12, permute_1077, div_76, permute_1081, permute_1086, permute_1087, permute_1088, permute_1089, permute_1093, permute_1098, permute_1102, div_77, permute_1106, le_13, permute_1110, div_78, permute_1114, permute_1119, permute_1120, permute_1121, permute_1122, permute_1126, permute_1131, permute_1135, div_79, permute_1139, le_14, permute_1143, div_80, permute_1147, permute_1152, permute_1153, permute_1154, permute_1155, permute_1159, permute_1164, permute_1168, div_81, permute_1172, le_15, permute_1176, div_82, permute_1180, permute_1185, permute_1186, permute_1187, permute_1188, permute_1192, permute_1197, permute_1201, div_83, permute_1205, le_16, permute_1209, div_84, permute_1213, permute_1218, permute_1219, permute_1220, permute_1221, permute_1225, permute_1230, permute_1234, div_85, permute_1238, le_17, permute_1242, div_86, permute_1246, permute_1251, permute_1252, permute_1253, permute_1254, permute_1258, permute_1263, permute_1267, div_87, permute_1271, le_18, permute_1275, div_88, permute_1279, permute_1284, permute_1285, permute_1286, permute_1287, permute_1291, permute_1296, permute_1300, div_89, permute_1304, le_19, permute_1308, div_90, permute_1312, permute_1317, permute_1318, permute_1319, permute_1320, permute_1324, permute_1329, permute_1333, div_91, permute_1337, le_20, permute_1341, div_92, permute_1345, permute_1350, permute_1351, permute_1352, permute_1353, permute_1357, permute_1362, permute_1366, div_93, permute_1370, le_21, permute_1374, div_94, permute_1378, permute_1383, permute_1384, permute_1385, permute_1386, permute_1390, permute_1395, permute_1399, div_95, permute_1403, le_22, permute_1407, div_96, permute_1411, permute_1416, permute_1417, permute_1418, permute_1419, permute_1423, permute_1428, permute_1432, div_97, permute_1436, le_23, permute_1440, div_98, permute_1444, permute_1449, permute_1450, permute_1451, permute_1452, permute_1456, permute_1461, permute_1465, div_99, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('M2M100ForConditionalGeneration', benchmark_compiled_module)
