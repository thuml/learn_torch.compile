
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


# kernel path: /tmp/torchinductor_youkaichao/qk/cqk6fvdvspjgvfgzvdzwndzxatevfmz4gf4xr4erp2mtnqxudmqd.py
# Source Nodes: [], Original ATen: [aten.nll_loss_backward]

triton_poi_fused_nll_loss_backward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14847616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/4s/c4s77v7baprcncxanw6amxrzdpwfjmdcfhvb5f5fgf3ctguqmewz.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_3
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 29056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (29056*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp3 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp1, tmp6, tmp7)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3mu2l4zldodp7laoccjkx7njrpmybgzdhtp4rt5mmeve5zytcl.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14876672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 29056)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x1), tmp3, eviction_policy='evict_last').to(tl.int1)
    tmp10 = tmp7 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tl.load(in_ptr5 + (x2), tmp3, other=0.0)
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.load(in_ptr6 + (x1), tmp3, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp3, tmp18, tmp19)
    tmp21 = tl.where(tmp3, tmp20, tmp11)
    tmp22 = tmp0 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3rpgl5i6va4g6omnebvjldvt3tkq7vu6irpkj2kgafslotho22.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29056
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (29056*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllqsswsi3ft4lupk7u7nq5o2lbt6zvqoa3nfb7pczvm36vdb2td.py
# Source Nodes: [hidden_states_170], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# hidden_states_170 => add_196, erf_24, mul_172
triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp20 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp21 = 0.7071067811865476
    tmp22 = tmp20 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp20
    tmp29 = -0.5
    tmp30 = tmp28 * tmp29
    tmp31 = tl.exp(tmp30)
    tmp32 = 0.3989422804014327
    tmp33 = tmp31 * tmp32
    tmp34 = tmp20 * tmp33
    tmp35 = tmp27 + tmp34
    tmp36 = tmp19 * tmp35
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pjvfottjtsg4t5x2evrmbv6qwiegya5a4yocdnnfkezjjqc6pb.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnaeswmp3uqzn7xwlycneyhxluabiwieeryuwpvnsedubwgyz2e.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dv4lgev253toikcyfs5sbbcoc7oyvqtu5srkfrcxgwi55qqkm5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2h5c6ukgv5gyqcyrogxflxeirgxmm6qja5nzsnjt6vyrn4i7fgl.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp20 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
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
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c267ylaj45yt2l6gkypo72a7upnqgtqmm7chali6emmqfdetg4x5.py
# Source Nodes: [intermediate_output_23], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_23 => add_192, erf_23, mul_167
triton_poi_fused_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdfujenr2tw3oyqqqzv6eznn7feymlij5uehnnwictodlohejo4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67oc26slmbqjpazd4dqkmiyqhtaowvec6vumzv6jx3swum6zha7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2efhjuh4iefn3hg5l5obyljidxhc77y7avr4nocrb2rzqchoir.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp22 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
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
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciacbwkx4xm3p3y7oategi5xoa4tdfxvjbfk6omyhhisquhc5w2m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp26 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
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
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4vynu3ogqqg5fvfjbz5bxrheuwzzwktnltssi5y6wzpis45vk5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
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
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpddjq2fspsdhzozgh7vcyby5zwhjo22yqdxke5jk2qflrcgxvuu.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_3
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*i1', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp29 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp27 = tl.full([1], -1, tl.int64)
    tmp28 = tmp26 == tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tmp33 = tmp25 * tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp28, tmp34, tmp33)
    tmp36 = tl.full([1], False, tl.int1)
    tmp37 = tl.where(tmp36, tmp34, tmp33)
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 == tmp39
    tmp41 = tl.where(tmp40, tmp34, tmp33)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuc2jboh5cbe4q6o6fv6jblh3vososvcel7755rh56aiydenesi.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgx3pdu5vyhmht35qieahknnyuvzktyqlbb3utg7bblq4mjincdi.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hmhxi7hzairwxghuxiwgje2qo7pxkquneokg3ahturx2mtz6kg.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29753344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_392, primals_398, full_default, slice_3, getitem_1, mul_1, view, clone_default_69, clone_default_70, clone_default_71, getitem_408, getitem_409, getitem_410, alias_default_47, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_66, clone_default_67, clone_default_68, getitem_401, getitem_402, getitem_403, alias_default_45, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_63, clone_default_64, clone_default_65, getitem_394, getitem_395, getitem_396, alias_default_43, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_60, clone_default_61, clone_default_62, getitem_387, getitem_388, getitem_389, alias_default_41, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_57, clone_default_58, clone_default_59, getitem_380, getitem_381, getitem_382, alias_default_39, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_54, clone_default_55, clone_default_56, getitem_373, getitem_374, getitem_375, alias_default_37, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_51, clone_default_52, clone_default_53, getitem_366, getitem_367, getitem_368, alias_default_35, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_48, clone_default_49, clone_default_50, getitem_359, getitem_360, getitem_361, alias_default_33, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_45, clone_default_46, clone_default_47, getitem_352, getitem_353, getitem_354, alias_default_31, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_42, clone_default_43, clone_default_44, getitem_345, getitem_346, getitem_347, alias_default_29, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_39, clone_default_40, clone_default_41, getitem_338, getitem_339, getitem_340, alias_default_27, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default_36, clone_default_37, clone_default_38, getitem_331, getitem_332, getitem_333, alias_default_25, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, clone_default_33, clone_default_34, clone_default_35, getitem_324, getitem_325, getitem_326, alias_default_23, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, clone_default_30, clone_default_31, clone_default_32, getitem_317, getitem_318, getitem_319, alias_default_21, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, clone_default_27, clone_default_28, clone_default_29, getitem_310, getitem_311, getitem_312, alias_default_19, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, clone_default_24, clone_default_25, clone_default_26, getitem_303, getitem_304, getitem_305, alias_default_17, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, clone_default_21, clone_default_22, clone_default_23, getitem_296, getitem_297, getitem_298, alias_default_15, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, clone_default_18, clone_default_19, clone_default_20, getitem_289, getitem_290, getitem_291, alias_default_13, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, clone_default_15, clone_default_16, clone_default_17, getitem_282, getitem_283, getitem_284, alias_default_11, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, clone_default_12, clone_default_13, clone_default_14, getitem_275, getitem_276, getitem_277, alias_default_9, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, clone_default_9, clone_default_10, clone_default_11, getitem_268, getitem_269, getitem_270, alias_default_7, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, clone_default_6, clone_default_7, clone_default_8, getitem_261, getitem_262, getitem_263, alias_default_5, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, clone_default_3, clone_default_4, clone_default_5, getitem_254, getitem_255, getitem_256, alias_default_3, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, clone_default, clone_default_1, clone_default_2, getitem_247, getitem_248, getitem_249, alias_default_1, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, addmm_144, mul_174, view_530, sub_76, convert_element_type, ne_3, where_2, permute_266, div_50, permute_270, div_51, permute_274, permute_278, div_52, permute_282, permute_294, permute_299, permute_303, div_54, permute_307, permute_311, div_55, permute_315, permute_327, permute_332, permute_336, div_57, permute_340, permute_344, div_58, permute_348, permute_360, permute_365, permute_369, div_60, permute_373, permute_377, div_61, permute_381, permute_393, permute_398, permute_402, div_63, permute_406, permute_410, div_64, permute_414, permute_426, permute_431, permute_435, div_66, permute_439, permute_443, div_67, permute_447, permute_459, permute_464, permute_468, div_69, permute_472, permute_476, div_70, permute_480, permute_492, permute_497, permute_501, div_72, permute_505, permute_509, div_73, permute_513, permute_525, permute_530, permute_534, div_75, permute_538, permute_542, div_76, permute_546, permute_558, permute_563, permute_567, div_78, permute_571, permute_575, div_79, permute_579, permute_591, permute_596, permute_600, div_81, permute_604, permute_608, div_82, permute_612, permute_624, permute_629, permute_633, div_84, permute_637, permute_641, div_85, permute_645, permute_657, permute_662, permute_666, div_87, permute_670, permute_674, div_88, permute_678, permute_690, permute_695, permute_699, div_90, permute_703, permute_707, div_91, permute_711, permute_723, permute_728, permute_732, div_93, permute_736, permute_740, div_94, permute_744, permute_756, permute_761, permute_765, div_96, permute_769, permute_773, div_97, permute_777, permute_789, permute_794, permute_798, div_99, permute_802, permute_806, div_100, permute_810, permute_822, permute_827, permute_831, div_102, permute_835, permute_839, div_103, permute_843, permute_855, permute_860, permute_864, div_105, permute_868, permute_872, div_106, permute_876, permute_888, permute_893, permute_897, div_108, permute_901, permute_905, div_109, permute_909, permute_921, permute_926, permute_930, div_111, permute_934, permute_938, div_112, permute_942, permute_954, permute_959, permute_963, div_114, permute_967, permute_971, div_115, permute_975, permute_987, permute_992, permute_996, div_117, permute_1000, permute_1004, div_118, permute_1008, permute_1020, permute_1025, permute_1029, div_120, permute_1033, permute_1037, div_121, permute_1041, permute_1053, permute_1058, permute_1062, div_123, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_94, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_174, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, ), (1, ))
    assert_size_stride(primals_190, (1024, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_392, (1024, ), (1, ))
    assert_size_stride(primals_398, (1, 512), (512, 1))
    assert_size_stride(full_default, (1, 512), (512, 1))
    assert_size_stride(slice_3, (1, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_69, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_70, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_71, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_408, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_409, (), ())
    assert_size_stride(getitem_410, (), ())
    assert_size_stride(alias_default_47, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_16, (512, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_3, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_18, (512, 1024), (1024, 1))
    assert_size_stride(addmm_4, (512, 4096), (4096, 1))
    assert_size_stride(view_20, (512, 4096), (4096, 1))
    assert_size_stride(getitem_11, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_8, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_22, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_66, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_67, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_68, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_401, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_402, (), ())
    assert_size_stride(getitem_403, (), ())
    assert_size_stride(alias_default_45, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_38, (512, 1024), (1024, 1))
    assert_size_stride(getitem_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_10, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_40, (512, 1024), (1024, 1))
    assert_size_stride(addmm_10, (512, 4096), (4096, 1))
    assert_size_stride(view_42, (512, 4096), (4096, 1))
    assert_size_stride(getitem_21, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_15, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_44, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_63, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_64, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_65, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_394, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_395, (), ())
    assert_size_stride(getitem_396, (), ())
    assert_size_stride(alias_default_43, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_60, (512, 1024), (1024, 1))
    assert_size_stride(getitem_27, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_62, (512, 1024), (1024, 1))
    assert_size_stride(addmm_16, (512, 4096), (4096, 1))
    assert_size_stride(view_64, (512, 4096), (4096, 1))
    assert_size_stride(getitem_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_22, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_66, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_60, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_61, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_62, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_387, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_388, (), ())
    assert_size_stride(getitem_389, (), ())
    assert_size_stride(alias_default_41, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_82, (512, 1024), (1024, 1))
    assert_size_stride(getitem_37, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_24, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_84, (512, 1024), (1024, 1))
    assert_size_stride(addmm_22, (512, 4096), (4096, 1))
    assert_size_stride(view_86, (512, 4096), (4096, 1))
    assert_size_stride(getitem_41, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_29, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_88, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_57, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_58, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_59, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_380, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_381, (), ())
    assert_size_stride(getitem_382, (), ())
    assert_size_stride(alias_default_39, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_104, (512, 1024), (1024, 1))
    assert_size_stride(getitem_47, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_106, (512, 1024), (1024, 1))
    assert_size_stride(addmm_28, (512, 4096), (4096, 1))
    assert_size_stride(view_108, (512, 4096), (4096, 1))
    assert_size_stride(getitem_51, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_36, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_54, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_55, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_56, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_373, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_374, (), ())
    assert_size_stride(getitem_375, (), ())
    assert_size_stride(alias_default_37, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_126, (512, 1024), (1024, 1))
    assert_size_stride(getitem_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_38, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_128, (512, 1024), (1024, 1))
    assert_size_stride(addmm_34, (512, 4096), (4096, 1))
    assert_size_stride(view_130, (512, 4096), (4096, 1))
    assert_size_stride(getitem_61, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_43, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_132, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_51, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_52, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_53, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_366, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_367, (), ())
    assert_size_stride(getitem_368, (), ())
    assert_size_stride(alias_default_35, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_148, (512, 1024), (1024, 1))
    assert_size_stride(getitem_67, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_45, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_150, (512, 1024), (1024, 1))
    assert_size_stride(addmm_40, (512, 4096), (4096, 1))
    assert_size_stride(view_152, (512, 4096), (4096, 1))
    assert_size_stride(getitem_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_50, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_154, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_48, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_49, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_50, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_359, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_360, (), ())
    assert_size_stride(getitem_361, (), ())
    assert_size_stride(alias_default_33, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_170, (512, 1024), (1024, 1))
    assert_size_stride(getitem_77, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_52, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_172, (512, 1024), (1024, 1))
    assert_size_stride(addmm_46, (512, 4096), (4096, 1))
    assert_size_stride(view_174, (512, 4096), (4096, 1))
    assert_size_stride(getitem_81, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_176, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_45, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_46, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_47, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_352, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_353, (), ())
    assert_size_stride(getitem_354, (), ())
    assert_size_stride(alias_default_31, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_192, (512, 1024), (1024, 1))
    assert_size_stride(getitem_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_59, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_194, (512, 1024), (1024, 1))
    assert_size_stride(addmm_52, (512, 4096), (4096, 1))
    assert_size_stride(view_196, (512, 4096), (4096, 1))
    assert_size_stride(getitem_91, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_64, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_198, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_42, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_43, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_44, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_345, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_346, (), ())
    assert_size_stride(getitem_347, (), ())
    assert_size_stride(alias_default_29, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_214, (512, 1024), (1024, 1))
    assert_size_stride(getitem_97, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_66, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_216, (512, 1024), (1024, 1))
    assert_size_stride(addmm_58, (512, 4096), (4096, 1))
    assert_size_stride(view_218, (512, 4096), (4096, 1))
    assert_size_stride(getitem_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_220, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_39, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_40, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_41, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_338, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_339, (), ())
    assert_size_stride(getitem_340, (), ())
    assert_size_stride(alias_default_27, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_236, (512, 1024), (1024, 1))
    assert_size_stride(getitem_107, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_73, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_238, (512, 1024), (1024, 1))
    assert_size_stride(addmm_64, (512, 4096), (4096, 1))
    assert_size_stride(view_240, (512, 4096), (4096, 1))
    assert_size_stride(getitem_111, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_78, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_242, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_36, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_37, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_38, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_331, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_332, (), ())
    assert_size_stride(getitem_333, (), ())
    assert_size_stride(alias_default_25, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_258, (512, 1024), (1024, 1))
    assert_size_stride(getitem_117, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_80, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_260, (512, 1024), (1024, 1))
    assert_size_stride(addmm_70, (512, 4096), (4096, 1))
    assert_size_stride(view_262, (512, 4096), (4096, 1))
    assert_size_stride(getitem_121, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_85, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_264, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_33, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_34, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_35, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_324, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_325, (), ())
    assert_size_stride(getitem_326, (), ())
    assert_size_stride(alias_default_23, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_280, (512, 1024), (1024, 1))
    assert_size_stride(getitem_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_282, (512, 1024), (1024, 1))
    assert_size_stride(addmm_76, (512, 4096), (4096, 1))
    assert_size_stride(view_284, (512, 4096), (4096, 1))
    assert_size_stride(getitem_131, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_92, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_286, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_30, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_31, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_32, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_317, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_318, (), ())
    assert_size_stride(getitem_319, (), ())
    assert_size_stride(alias_default_21, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_302, (512, 1024), (1024, 1))
    assert_size_stride(getitem_137, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_94, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_304, (512, 1024), (1024, 1))
    assert_size_stride(addmm_82, (512, 4096), (4096, 1))
    assert_size_stride(view_306, (512, 4096), (4096, 1))
    assert_size_stride(getitem_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_99, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_308, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_27, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_28, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_29, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_310, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_311, (), ())
    assert_size_stride(getitem_312, (), ())
    assert_size_stride(alias_default_19, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_324, (512, 1024), (1024, 1))
    assert_size_stride(getitem_147, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_326, (512, 1024), (1024, 1))
    assert_size_stride(addmm_88, (512, 4096), (4096, 1))
    assert_size_stride(view_328, (512, 4096), (4096, 1))
    assert_size_stride(getitem_151, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_106, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_330, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_24, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_25, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_26, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_303, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_304, (), ())
    assert_size_stride(getitem_305, (), ())
    assert_size_stride(alias_default_17, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_346, (512, 1024), (1024, 1))
    assert_size_stride(getitem_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_108, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_348, (512, 1024), (1024, 1))
    assert_size_stride(addmm_94, (512, 4096), (4096, 1))
    assert_size_stride(view_350, (512, 4096), (4096, 1))
    assert_size_stride(getitem_161, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_113, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_352, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_21, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_22, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_23, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_296, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_297, (), ())
    assert_size_stride(getitem_298, (), ())
    assert_size_stride(alias_default_15, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_368, (512, 1024), (1024, 1))
    assert_size_stride(getitem_167, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_115, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_370, (512, 1024), (1024, 1))
    assert_size_stride(addmm_100, (512, 4096), (4096, 1))
    assert_size_stride(view_372, (512, 4096), (4096, 1))
    assert_size_stride(getitem_171, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_120, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_374, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_18, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_19, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_20, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_289, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_290, (), ())
    assert_size_stride(getitem_291, (), ())
    assert_size_stride(alias_default_13, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_390, (512, 1024), (1024, 1))
    assert_size_stride(getitem_177, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_122, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_392, (512, 1024), (1024, 1))
    assert_size_stride(addmm_106, (512, 4096), (4096, 1))
    assert_size_stride(view_394, (512, 4096), (4096, 1))
    assert_size_stride(getitem_181, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_396, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_15, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_16, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_17, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_282, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_283, (), ())
    assert_size_stride(getitem_284, (), ())
    assert_size_stride(alias_default_11, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_412, (512, 1024), (1024, 1))
    assert_size_stride(getitem_187, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_129, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_414, (512, 1024), (1024, 1))
    assert_size_stride(addmm_112, (512, 4096), (4096, 1))
    assert_size_stride(view_416, (512, 4096), (4096, 1))
    assert_size_stride(getitem_191, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_134, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_418, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_12, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_13, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_14, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_275, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_276, (), ())
    assert_size_stride(getitem_277, (), ())
    assert_size_stride(alias_default_9, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_434, (512, 1024), (1024, 1))
    assert_size_stride(getitem_197, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_136, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_436, (512, 1024), (1024, 1))
    assert_size_stride(addmm_118, (512, 4096), (4096, 1))
    assert_size_stride(view_438, (512, 4096), (4096, 1))
    assert_size_stride(getitem_201, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_440, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_9, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_10, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_11, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_268, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_269, (), ())
    assert_size_stride(getitem_270, (), ())
    assert_size_stride(alias_default_7, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_456, (512, 1024), (1024, 1))
    assert_size_stride(getitem_207, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_143, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_458, (512, 1024), (1024, 1))
    assert_size_stride(addmm_124, (512, 4096), (4096, 1))
    assert_size_stride(view_460, (512, 4096), (4096, 1))
    assert_size_stride(getitem_211, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_148, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_462, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_6, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_7, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_8, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_261, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_262, (), ())
    assert_size_stride(getitem_263, (), ())
    assert_size_stride(alias_default_5, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_478, (512, 1024), (1024, 1))
    assert_size_stride(getitem_217, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_150, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_480, (512, 1024), (1024, 1))
    assert_size_stride(addmm_130, (512, 4096), (4096, 1))
    assert_size_stride(view_482, (512, 4096), (4096, 1))
    assert_size_stride(getitem_221, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_155, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_484, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_3, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_4, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_5, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_254, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_255, (), ())
    assert_size_stride(getitem_256, (), ())
    assert_size_stride(alias_default_3, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_500, (512, 1024), (1024, 1))
    assert_size_stride(getitem_227, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_502, (512, 1024), (1024, 1))
    assert_size_stride(addmm_136, (512, 4096), (4096, 1))
    assert_size_stride(view_504, (512, 4096), (4096, 1))
    assert_size_stride(getitem_231, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_162, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_506, (512, 1024), (1024, 1))
    assert_size_stride(clone_default, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_1, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_2, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_247, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_248, (), ())
    assert_size_stride(getitem_249, (), ())
    assert_size_stride(alias_default_1, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_522, (512, 1024), (1024, 1))
    assert_size_stride(getitem_237, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_164, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_524, (512, 1024), (1024, 1))
    assert_size_stride(addmm_142, (512, 4096), (4096, 1))
    assert_size_stride(view_526, (512, 4096), (4096, 1))
    assert_size_stride(getitem_241, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_169, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_528, (512, 1024), (1024, 1))
    assert_size_stride(addmm_144, (512, 1024), (1024, 1))
    assert_size_stride(mul_174, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_530, (512, 1024), (1024, 1))
    assert_size_stride(sub_76, (511, 29056), (29056, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(ne_3, (511, 1), (1, 1))
    assert_size_stride(where_2, (511, 1), (1, 1))
    assert_size_stride(permute_266, (29056, 1024), (1024, 1))
    assert_size_stride(div_50, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_270, (1024, 1024), (1024, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (1024, 4096), (4096, 1))
    assert_size_stride(permute_278, (4096, 1024), (1024, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (1024, 1024), (1024, 1))
    assert_size_stride(permute_294, (1024, 1024), (1024, 1))
    assert_size_stride(permute_299, (1024, 1024), (1024, 1))
    assert_size_stride(permute_303, (1024, 1024), (1024, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (1024, 4096), (4096, 1))
    assert_size_stride(permute_311, (4096, 1024), (1024, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (1024, 1024), (1024, 1))
    assert_size_stride(permute_327, (1024, 1024), (1024, 1))
    assert_size_stride(permute_332, (1024, 1024), (1024, 1))
    assert_size_stride(permute_336, (1024, 1024), (1024, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (1024, 4096), (4096, 1))
    assert_size_stride(permute_344, (4096, 1024), (1024, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (1024, 1024), (1024, 1))
    assert_size_stride(permute_360, (1024, 1024), (1024, 1))
    assert_size_stride(permute_365, (1024, 1024), (1024, 1))
    assert_size_stride(permute_369, (1024, 1024), (1024, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (1024, 4096), (4096, 1))
    assert_size_stride(permute_377, (4096, 1024), (1024, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (1024, 1024), (1024, 1))
    assert_size_stride(permute_393, (1024, 1024), (1024, 1))
    assert_size_stride(permute_398, (1024, 1024), (1024, 1))
    assert_size_stride(permute_402, (1024, 1024), (1024, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (1024, 4096), (4096, 1))
    assert_size_stride(permute_410, (4096, 1024), (1024, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (1024, 1024), (1024, 1))
    assert_size_stride(permute_426, (1024, 1024), (1024, 1))
    assert_size_stride(permute_431, (1024, 1024), (1024, 1))
    assert_size_stride(permute_435, (1024, 1024), (1024, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (1024, 4096), (4096, 1))
    assert_size_stride(permute_443, (4096, 1024), (1024, 1))
    assert_size_stride(div_67, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (1024, 1024), (1024, 1))
    assert_size_stride(permute_459, (1024, 1024), (1024, 1))
    assert_size_stride(permute_464, (1024, 1024), (1024, 1))
    assert_size_stride(permute_468, (1024, 1024), (1024, 1))
    assert_size_stride(div_69, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (1024, 4096), (4096, 1))
    assert_size_stride(permute_476, (4096, 1024), (1024, 1))
    assert_size_stride(div_70, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (1024, 1024), (1024, 1))
    assert_size_stride(permute_492, (1024, 1024), (1024, 1))
    assert_size_stride(permute_497, (1024, 1024), (1024, 1))
    assert_size_stride(permute_501, (1024, 1024), (1024, 1))
    assert_size_stride(div_72, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (1024, 4096), (4096, 1))
    assert_size_stride(permute_509, (4096, 1024), (1024, 1))
    assert_size_stride(div_73, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (1024, 1024), (1024, 1))
    assert_size_stride(permute_525, (1024, 1024), (1024, 1))
    assert_size_stride(permute_530, (1024, 1024), (1024, 1))
    assert_size_stride(permute_534, (1024, 1024), (1024, 1))
    assert_size_stride(div_75, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_538, (1024, 4096), (4096, 1))
    assert_size_stride(permute_542, (4096, 1024), (1024, 1))
    assert_size_stride(div_76, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_546, (1024, 1024), (1024, 1))
    assert_size_stride(permute_558, (1024, 1024), (1024, 1))
    assert_size_stride(permute_563, (1024, 1024), (1024, 1))
    assert_size_stride(permute_567, (1024, 1024), (1024, 1))
    assert_size_stride(div_78, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_571, (1024, 4096), (4096, 1))
    assert_size_stride(permute_575, (4096, 1024), (1024, 1))
    assert_size_stride(div_79, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_579, (1024, 1024), (1024, 1))
    assert_size_stride(permute_591, (1024, 1024), (1024, 1))
    assert_size_stride(permute_596, (1024, 1024), (1024, 1))
    assert_size_stride(permute_600, (1024, 1024), (1024, 1))
    assert_size_stride(div_81, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_604, (1024, 4096), (4096, 1))
    assert_size_stride(permute_608, (4096, 1024), (1024, 1))
    assert_size_stride(div_82, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_612, (1024, 1024), (1024, 1))
    assert_size_stride(permute_624, (1024, 1024), (1024, 1))
    assert_size_stride(permute_629, (1024, 1024), (1024, 1))
    assert_size_stride(permute_633, (1024, 1024), (1024, 1))
    assert_size_stride(div_84, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_637, (1024, 4096), (4096, 1))
    assert_size_stride(permute_641, (4096, 1024), (1024, 1))
    assert_size_stride(div_85, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_645, (1024, 1024), (1024, 1))
    assert_size_stride(permute_657, (1024, 1024), (1024, 1))
    assert_size_stride(permute_662, (1024, 1024), (1024, 1))
    assert_size_stride(permute_666, (1024, 1024), (1024, 1))
    assert_size_stride(div_87, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_670, (1024, 4096), (4096, 1))
    assert_size_stride(permute_674, (4096, 1024), (1024, 1))
    assert_size_stride(div_88, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_678, (1024, 1024), (1024, 1))
    assert_size_stride(permute_690, (1024, 1024), (1024, 1))
    assert_size_stride(permute_695, (1024, 1024), (1024, 1))
    assert_size_stride(permute_699, (1024, 1024), (1024, 1))
    assert_size_stride(div_90, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_703, (1024, 4096), (4096, 1))
    assert_size_stride(permute_707, (4096, 1024), (1024, 1))
    assert_size_stride(div_91, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_711, (1024, 1024), (1024, 1))
    assert_size_stride(permute_723, (1024, 1024), (1024, 1))
    assert_size_stride(permute_728, (1024, 1024), (1024, 1))
    assert_size_stride(permute_732, (1024, 1024), (1024, 1))
    assert_size_stride(div_93, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_736, (1024, 4096), (4096, 1))
    assert_size_stride(permute_740, (4096, 1024), (1024, 1))
    assert_size_stride(div_94, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_744, (1024, 1024), (1024, 1))
    assert_size_stride(permute_756, (1024, 1024), (1024, 1))
    assert_size_stride(permute_761, (1024, 1024), (1024, 1))
    assert_size_stride(permute_765, (1024, 1024), (1024, 1))
    assert_size_stride(div_96, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_769, (1024, 4096), (4096, 1))
    assert_size_stride(permute_773, (4096, 1024), (1024, 1))
    assert_size_stride(div_97, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_777, (1024, 1024), (1024, 1))
    assert_size_stride(permute_789, (1024, 1024), (1024, 1))
    assert_size_stride(permute_794, (1024, 1024), (1024, 1))
    assert_size_stride(permute_798, (1024, 1024), (1024, 1))
    assert_size_stride(div_99, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_802, (1024, 4096), (4096, 1))
    assert_size_stride(permute_806, (4096, 1024), (1024, 1))
    assert_size_stride(div_100, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_810, (1024, 1024), (1024, 1))
    assert_size_stride(permute_822, (1024, 1024), (1024, 1))
    assert_size_stride(permute_827, (1024, 1024), (1024, 1))
    assert_size_stride(permute_831, (1024, 1024), (1024, 1))
    assert_size_stride(div_102, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_835, (1024, 4096), (4096, 1))
    assert_size_stride(permute_839, (4096, 1024), (1024, 1))
    assert_size_stride(div_103, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_843, (1024, 1024), (1024, 1))
    assert_size_stride(permute_855, (1024, 1024), (1024, 1))
    assert_size_stride(permute_860, (1024, 1024), (1024, 1))
    assert_size_stride(permute_864, (1024, 1024), (1024, 1))
    assert_size_stride(div_105, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_868, (1024, 4096), (4096, 1))
    assert_size_stride(permute_872, (4096, 1024), (1024, 1))
    assert_size_stride(div_106, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_876, (1024, 1024), (1024, 1))
    assert_size_stride(permute_888, (1024, 1024), (1024, 1))
    assert_size_stride(permute_893, (1024, 1024), (1024, 1))
    assert_size_stride(permute_897, (1024, 1024), (1024, 1))
    assert_size_stride(div_108, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_901, (1024, 4096), (4096, 1))
    assert_size_stride(permute_905, (4096, 1024), (1024, 1))
    assert_size_stride(div_109, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_909, (1024, 1024), (1024, 1))
    assert_size_stride(permute_921, (1024, 1024), (1024, 1))
    assert_size_stride(permute_926, (1024, 1024), (1024, 1))
    assert_size_stride(permute_930, (1024, 1024), (1024, 1))
    assert_size_stride(div_111, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_934, (1024, 4096), (4096, 1))
    assert_size_stride(permute_938, (4096, 1024), (1024, 1))
    assert_size_stride(div_112, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_942, (1024, 1024), (1024, 1))
    assert_size_stride(permute_954, (1024, 1024), (1024, 1))
    assert_size_stride(permute_959, (1024, 1024), (1024, 1))
    assert_size_stride(permute_963, (1024, 1024), (1024, 1))
    assert_size_stride(div_114, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_967, (1024, 4096), (4096, 1))
    assert_size_stride(permute_971, (4096, 1024), (1024, 1))
    assert_size_stride(div_115, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_975, (1024, 1024), (1024, 1))
    assert_size_stride(permute_987, (1024, 1024), (1024, 1))
    assert_size_stride(permute_992, (1024, 1024), (1024, 1))
    assert_size_stride(permute_996, (1024, 1024), (1024, 1))
    assert_size_stride(div_117, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1000, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1004, (4096, 1024), (1024, 1))
    assert_size_stride(div_118, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1008, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1020, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1025, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1029, (1024, 1024), (1024, 1))
    assert_size_stride(div_120, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1033, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1037, (4096, 1024), (1024, 1))
    assert_size_stride(div_121, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1041, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1053, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1058, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1062, (1024, 1024), (1024, 1))
    assert_size_stride(div_123, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 29056), (14876672, 29056, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((511, 29056), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 14847616, grid=grid(14847616), stream=stream0)
        aten.scatter_(buf0,1,where_2,-1.0)
        del where_2
        buf3 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_3, tangents_1, convert_element_type, buf3, 511, 29056, grid=grid(511), stream=stream0)
        buf4 = empty((1, 512, 29056), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_2.run(tangents_2, buf0, ne_3, tangents_1, convert_element_type, sub_76, buf3, buf4, 14876672, grid=grid(14876672), stream=stream0)
        del buf0
        del buf3
        del convert_element_type
        del ne_3
        del sub_76
        del tangents_1
        del tangents_2
        buf5 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 29056), (29056, 1), 0), permute_266, out=buf5)
        del permute_266
        buf6 = empty((29056, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (29056, 512), (1, 29056), 0), view_530, out=buf6)
        del view_530
        buf7 = empty((1, 29056), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf7, 29056, 512, grid=grid(29056), stream=stream0)
        del buf4
        buf12 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_170], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf5, primals_392, mul_174, div_50, addmm_144, buf12, 512, 1024, grid=grid(512), stream=stream0)
        del addmm_144
        del div_50
        del primals_392
        buf10 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf11 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf5, mul_174, buf10, buf11, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_174
        buf13 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 1024), (1024, 1), 0), permute_270, out=buf13)
        del permute_270
        buf14 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (1024, 512), (1, 1024), 0), view_528, out=buf14)
        del view_528
        buf15 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf12, buf15, 4096, 128, grid=grid(4096), stream=stream0)
        buf16 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf15, buf16, 1024, 4, grid=grid(1024), stream=stream0)
        buf19 = buf12; del buf12  # reuse
        buf22 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf13, primals_388, mul_169, div_51, getitem_241, buf19, buf22, 512, 1024, grid=grid(512), stream=stream0)
        del div_51
        del getitem_241
        del primals_388
        buf20 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf21 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf13, mul_169, buf20, buf21, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_169
        buf23 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 1024), (1024, 1), 0), permute_274, out=buf23)
        del permute_274
        buf24 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (1024, 512), (1, 1024), 0), view_526, out=buf24)
        del view_526
        buf25 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf22, buf25, 4096, 128, grid=grid(4096), stream=stream0)
        buf26 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf25, buf26, 1024, 4, grid=grid(1024), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (1, 512, 4096), (2097152, 4096, 1), 0); del buf23  # reuse
        # Source Nodes: [intermediate_output_23], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf27, addmm_142, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_142
        buf28 = reinterpret_tensor(buf22, (512, 1024), (1024, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 4096), (4096, 1), 0), permute_278, out=buf28)
        del permute_278
        buf29 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 512), (1, 4096), 0), view_524, out=buf29)
        del view_524
        buf30 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf27, buf30, 16384, 128, grid=grid(16384), stream=stream0)
        buf31 = reinterpret_tensor(buf25, (1, 4096), (4096, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf30, buf31, 4096, 4, grid=grid(4096), stream=stream0)
        buf36 = buf19; del buf19  # reuse
        buf37 = reinterpret_tensor(buf13, (1, 512, 1024), (524288, 1024, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf36, buf28, primals_382, mul_164, div_52, getitem_237, buf37, 512, 1024, grid=grid(512), stream=stream0)
        del div_52
        del getitem_237
        del primals_382
        buf34 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf35 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf28, mul_164, buf34, buf35, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_164
        buf38 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (512, 1024), (1024, 1), 0), permute_282, out=buf38)
        del permute_282
        buf39 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (1024, 512), (1, 1024), 0), view_522, out=buf39)
        del view_522
        buf40 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf37, buf40, 4096, 128, grid=grid(4096), stream=stream0)
        del buf37
        buf41 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf40, buf41, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf42 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf38, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_247, getitem_248, getitem_249, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_247
        del getitem_248
        del getitem_249
        buf43 = buf42[0]
        buf44 = buf42[1]
        buf45 = buf42[2]
        del buf42
        buf46 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 1024), (1024, 1), 0), permute_294, out=buf46)
        del permute_294
        buf47 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1024, 512), (1, 1024), 0), view_506, out=buf47)
        buf48 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf45, buf48, 4096, 128, grid=grid(4096), stream=stream0)
        buf49 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf48, buf49, 1024, 4, grid=grid(1024), stream=stream0)
        buf50 = reinterpret_tensor(buf45, (512, 1024), (1024, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0), permute_299, out=buf50)
        del permute_299
        buf51 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 512), (1, 1024), 0), view_506, out=buf51)
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf44, buf52, 4096, 128, grid=grid(4096), stream=stream0)
        buf53 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf52, buf53, 1024, 4, grid=grid(1024), stream=stream0)
        buf54 = reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (512, 1024), (1024, 1), 0), permute_303, out=buf54)
        del permute_303
        buf55 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (1024, 512), (1, 1024), 0), view_506, out=buf55)
        del view_506
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf43, buf56, 4096, 128, grid=grid(4096), stream=stream0)
        buf57 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf56, buf57, 1024, 4, grid=grid(1024), stream=stream0)
        buf62 = buf36; del buf36  # reuse
        buf63 = reinterpret_tensor(buf43, (1, 512, 1024), (524288, 1024, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf62, buf46, buf50, buf54, primals_372, mul_162, div_54, getitem_231, buf63, 512, 1024, grid=grid(512), stream=stream0)
        del div_54
        del getitem_231
        del primals_372
        buf60 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf61 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf46, buf50, buf54, mul_162, buf60, buf61, 1024, 512, grid=grid(1024), stream=stream0)
        del buf46
        del buf50
        del mul_162
        buf64 = reinterpret_tensor(buf27, (512, 4096), (4096, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (512, 1024), (1024, 1), 0), permute_307, out=buf64)
        del permute_307
        buf65 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 512), (1, 1024), 0), view_504, out=buf65)
        del view_504
        buf66 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf63, buf66, 4096, 128, grid=grid(4096), stream=stream0)
        buf67 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf66, buf67, 1024, 4, grid=grid(1024), stream=stream0)
        buf68 = reinterpret_tensor(buf64, (1, 512, 4096), (2097152, 4096, 1), 0); del buf64  # reuse
        # Source Nodes: [intermediate_output_22], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf68, addmm_136, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_136
        buf69 = reinterpret_tensor(buf63, (512, 1024), (1024, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 4096), (4096, 1), 0), permute_311, out=buf69)
        del permute_311
        buf70 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (4096, 512), (1, 4096), 0), view_502, out=buf70)
        del view_502
        buf71 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf68, buf71, 16384, 128, grid=grid(16384), stream=stream0)
        buf72 = reinterpret_tensor(buf66, (1, 4096), (4096, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf71, buf72, 4096, 4, grid=grid(4096), stream=stream0)
        buf77 = buf62; del buf62  # reuse
        buf78 = reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf77, buf69, primals_366, mul_157, div_55, getitem_227, buf78, 512, 1024, grid=grid(512), stream=stream0)
        del div_55
        del getitem_227
        del primals_366
        buf75 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf76 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf69, mul_157, buf75, buf76, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_157
        buf79 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 1024), (1024, 1), 0), permute_315, out=buf79)
        del permute_315
        buf80 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (1024, 512), (1, 1024), 0), view_500, out=buf80)
        del view_500
        buf81 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf78, buf81, 4096, 128, grid=grid(4096), stream=stream0)
        del buf78
        buf82 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf81, buf82, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf83 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf79, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_254, getitem_255, getitem_256, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_254
        del getitem_255
        del getitem_256
        buf84 = buf83[0]
        buf85 = buf83[1]
        buf86 = buf83[2]
        del buf83
        buf87 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (512, 1024), (1024, 1), 0), permute_327, out=buf87)
        del permute_327
        buf88 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1024, 512), (1, 1024), 0), view_484, out=buf88)
        buf89 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf86, buf89, 4096, 128, grid=grid(4096), stream=stream0)
        buf90 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf89, buf90, 1024, 4, grid=grid(1024), stream=stream0)
        buf91 = reinterpret_tensor(buf86, (512, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 1024), (1024, 1), 0), permute_332, out=buf91)
        del permute_332
        buf92 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1024, 512), (1, 1024), 0), view_484, out=buf92)
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf85, buf93, 4096, 128, grid=grid(4096), stream=stream0)
        buf94 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf93, buf94, 1024, 4, grid=grid(1024), stream=stream0)
        buf95 = reinterpret_tensor(buf85, (512, 1024), (1024, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (512, 1024), (1024, 1), 0), permute_336, out=buf95)
        del permute_336
        buf96 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (1024, 512), (1, 1024), 0), view_484, out=buf96)
        del view_484
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf84, buf97, 4096, 128, grid=grid(4096), stream=stream0)
        buf98 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf97, buf98, 1024, 4, grid=grid(1024), stream=stream0)
        buf103 = buf77; del buf77  # reuse
        buf104 = reinterpret_tensor(buf84, (1, 512, 1024), (524288, 1024, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf103, buf87, buf91, buf95, primals_356, mul_155, div_57, getitem_221, buf104, 512, 1024, grid=grid(512), stream=stream0)
        del div_57
        del getitem_221
        del primals_356
        buf101 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf102 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf87, buf91, buf95, mul_155, buf101, buf102, 1024, 512, grid=grid(1024), stream=stream0)
        del buf87
        del buf91
        del mul_155
        buf105 = reinterpret_tensor(buf68, (512, 4096), (4096, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 1024), (1024, 1), 0), permute_340, out=buf105)
        del permute_340
        buf106 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1024, 512), (1, 1024), 0), view_482, out=buf106)
        del view_482
        buf107 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf104, buf107, 4096, 128, grid=grid(4096), stream=stream0)
        buf108 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf107, buf108, 1024, 4, grid=grid(1024), stream=stream0)
        buf109 = reinterpret_tensor(buf105, (1, 512, 4096), (2097152, 4096, 1), 0); del buf105  # reuse
        # Source Nodes: [intermediate_output_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf109, addmm_130, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_130
        buf110 = reinterpret_tensor(buf104, (512, 1024), (1024, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 4096), (4096, 1), 0), permute_344, out=buf110)
        del permute_344
        buf111 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (4096, 512), (1, 4096), 0), view_480, out=buf111)
        del view_480
        buf112 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf109, buf112, 16384, 128, grid=grid(16384), stream=stream0)
        buf113 = reinterpret_tensor(buf107, (1, 4096), (4096, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf112, buf113, 4096, 4, grid=grid(4096), stream=stream0)
        buf118 = buf103; del buf103  # reuse
        buf119 = reinterpret_tensor(buf95, (1, 512, 1024), (524288, 1024, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf118, buf110, primals_350, mul_150, div_58, getitem_217, buf119, 512, 1024, grid=grid(512), stream=stream0)
        del div_58
        del getitem_217
        del primals_350
        buf116 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf117 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf110, mul_150, buf116, buf117, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_150
        buf120 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 1024), (1024, 1), 0), permute_348, out=buf120)
        del permute_348
        buf121 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (1024, 512), (1, 1024), 0), view_478, out=buf121)
        del view_478
        buf122 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf119, buf122, 4096, 128, grid=grid(4096), stream=stream0)
        del buf119
        buf123 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf122, buf123, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf124 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf120, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_261, getitem_262, getitem_263, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_261
        del getitem_262
        del getitem_263
        buf125 = buf124[0]
        buf126 = buf124[1]
        buf127 = buf124[2]
        del buf124
        buf128 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 1024), (1024, 1), 0), permute_360, out=buf128)
        del permute_360
        buf129 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (1024, 512), (1, 1024), 0), view_462, out=buf129)
        buf130 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf127, buf130, 4096, 128, grid=grid(4096), stream=stream0)
        buf131 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf130, buf131, 1024, 4, grid=grid(1024), stream=stream0)
        buf132 = reinterpret_tensor(buf127, (512, 1024), (1024, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (512, 1024), (1024, 1), 0), permute_365, out=buf132)
        del permute_365
        buf133 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (1024, 512), (1, 1024), 0), view_462, out=buf133)
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf126, buf134, 4096, 128, grid=grid(4096), stream=stream0)
        buf135 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf134, buf135, 1024, 4, grid=grid(1024), stream=stream0)
        buf136 = reinterpret_tensor(buf126, (512, 1024), (1024, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 1024), (1024, 1), 0), permute_369, out=buf136)
        del permute_369
        buf137 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1024, 512), (1, 1024), 0), view_462, out=buf137)
        del view_462
        buf138 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf125, buf138, 4096, 128, grid=grid(4096), stream=stream0)
        buf139 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf138, buf139, 1024, 4, grid=grid(1024), stream=stream0)
        buf144 = buf118; del buf118  # reuse
        buf145 = reinterpret_tensor(buf125, (1, 512, 1024), (524288, 1024, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf144, buf128, buf132, buf136, primals_340, mul_148, div_60, getitem_211, buf145, 512, 1024, grid=grid(512), stream=stream0)
        del div_60
        del getitem_211
        del primals_340
        buf142 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf143 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf128, buf132, buf136, mul_148, buf142, buf143, 1024, 512, grid=grid(1024), stream=stream0)
        del buf128
        del buf132
        del mul_148
        buf146 = reinterpret_tensor(buf109, (512, 4096), (4096, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (512, 1024), (1024, 1), 0), permute_373, out=buf146)
        del permute_373
        buf147 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (1024, 512), (1, 1024), 0), view_460, out=buf147)
        del view_460
        buf148 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf145, buf148, 4096, 128, grid=grid(4096), stream=stream0)
        buf149 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf148, buf149, 1024, 4, grid=grid(1024), stream=stream0)
        buf150 = reinterpret_tensor(buf146, (1, 512, 4096), (2097152, 4096, 1), 0); del buf146  # reuse
        # Source Nodes: [intermediate_output_20], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf150, addmm_124, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_124
        buf151 = reinterpret_tensor(buf145, (512, 1024), (1024, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (512, 4096), (4096, 1), 0), permute_377, out=buf151)
        del permute_377
        buf152 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (4096, 512), (1, 4096), 0), view_458, out=buf152)
        del view_458
        buf153 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf150, buf153, 16384, 128, grid=grid(16384), stream=stream0)
        buf154 = reinterpret_tensor(buf148, (1, 4096), (4096, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf153, buf154, 4096, 4, grid=grid(4096), stream=stream0)
        buf159 = buf144; del buf144  # reuse
        buf160 = reinterpret_tensor(buf136, (1, 512, 1024), (524288, 1024, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf159, buf151, primals_334, mul_143, div_61, getitem_207, buf160, 512, 1024, grid=grid(512), stream=stream0)
        del div_61
        del getitem_207
        del primals_334
        buf157 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf158 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf151, mul_143, buf157, buf158, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_143
        buf161 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 1024), (1024, 1), 0), permute_381, out=buf161)
        del permute_381
        buf162 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (1024, 512), (1, 1024), 0), view_456, out=buf162)
        del view_456
        buf163 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf160, buf163, 4096, 128, grid=grid(4096), stream=stream0)
        del buf160
        buf164 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf163, buf164, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf165 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf161, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_268, getitem_269, getitem_270, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_268
        del getitem_269
        del getitem_270
        buf166 = buf165[0]
        buf167 = buf165[1]
        buf168 = buf165[2]
        del buf165
        buf169 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 1024), (1024, 1), 0), permute_393, out=buf169)
        del permute_393
        buf170 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (1024, 512), (1, 1024), 0), view_440, out=buf170)
        buf171 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf168, buf171, 4096, 128, grid=grid(4096), stream=stream0)
        buf172 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf171, buf172, 1024, 4, grid=grid(1024), stream=stream0)
        buf173 = reinterpret_tensor(buf168, (512, 1024), (1024, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (512, 1024), (1024, 1), 0), permute_398, out=buf173)
        del permute_398
        buf174 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1024, 512), (1, 1024), 0), view_440, out=buf174)
        buf175 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf167, buf175, 4096, 128, grid=grid(4096), stream=stream0)
        buf176 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf175, buf176, 1024, 4, grid=grid(1024), stream=stream0)
        buf177 = reinterpret_tensor(buf167, (512, 1024), (1024, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 1024), (1024, 1), 0), permute_402, out=buf177)
        del permute_402
        buf178 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (1024, 512), (1, 1024), 0), view_440, out=buf178)
        del view_440
        buf179 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf166, buf179, 4096, 128, grid=grid(4096), stream=stream0)
        buf180 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf179, buf180, 1024, 4, grid=grid(1024), stream=stream0)
        buf185 = buf159; del buf159  # reuse
        buf186 = reinterpret_tensor(buf166, (1, 512, 1024), (524288, 1024, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf185, buf169, buf173, buf177, primals_324, mul_141, div_63, getitem_201, buf186, 512, 1024, grid=grid(512), stream=stream0)
        del div_63
        del getitem_201
        del primals_324
        buf183 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf184 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf169, buf173, buf177, mul_141, buf183, buf184, 1024, 512, grid=grid(1024), stream=stream0)
        del buf169
        del buf173
        del mul_141
        buf187 = reinterpret_tensor(buf150, (512, 4096), (4096, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 1024), (1024, 1), 0), permute_406, out=buf187)
        del permute_406
        buf188 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (1024, 512), (1, 1024), 0), view_438, out=buf188)
        del view_438
        buf189 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf186, buf189, 4096, 128, grid=grid(4096), stream=stream0)
        buf190 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf189, buf190, 1024, 4, grid=grid(1024), stream=stream0)
        buf191 = reinterpret_tensor(buf187, (1, 512, 4096), (2097152, 4096, 1), 0); del buf187  # reuse
        # Source Nodes: [intermediate_output_19], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf191, addmm_118, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_118
        buf192 = reinterpret_tensor(buf186, (512, 1024), (1024, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 4096), (4096, 1), 0), permute_410, out=buf192)
        del permute_410
        buf193 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (4096, 512), (1, 4096), 0), view_436, out=buf193)
        del view_436
        buf194 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf191, buf194, 16384, 128, grid=grid(16384), stream=stream0)
        buf195 = reinterpret_tensor(buf189, (1, 4096), (4096, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf194, buf195, 4096, 4, grid=grid(4096), stream=stream0)
        buf200 = buf185; del buf185  # reuse
        buf201 = reinterpret_tensor(buf177, (1, 512, 1024), (524288, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf200, buf192, primals_318, mul_136, div_64, getitem_197, buf201, 512, 1024, grid=grid(512), stream=stream0)
        del div_64
        del getitem_197
        del primals_318
        buf198 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf199 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf192, mul_136, buf198, buf199, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_136
        buf202 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (512, 1024), (1024, 1), 0), permute_414, out=buf202)
        del permute_414
        buf203 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (1024, 512), (1, 1024), 0), view_434, out=buf203)
        del view_434
        buf204 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf201, buf204, 4096, 128, grid=grid(4096), stream=stream0)
        del buf201
        buf205 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf204, buf205, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf206 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf202, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_275, getitem_276, getitem_277, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_275
        del getitem_276
        del getitem_277
        buf207 = buf206[0]
        buf208 = buf206[1]
        buf209 = buf206[2]
        del buf206
        buf210 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (512, 1024), (1024, 1), 0), permute_426, out=buf210)
        del permute_426
        buf211 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (1024, 512), (1, 1024), 0), view_418, out=buf211)
        buf212 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf209, buf212, 4096, 128, grid=grid(4096), stream=stream0)
        buf213 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf212, buf213, 1024, 4, grid=grid(1024), stream=stream0)
        buf214 = reinterpret_tensor(buf209, (512, 1024), (1024, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0), permute_431, out=buf214)
        del permute_431
        buf215 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1024, 512), (1, 1024), 0), view_418, out=buf215)
        buf216 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf208, buf216, 4096, 128, grid=grid(4096), stream=stream0)
        buf217 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf216, buf217, 1024, 4, grid=grid(1024), stream=stream0)
        buf218 = reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 1024), (1024, 1), 0), permute_435, out=buf218)
        del permute_435
        buf219 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (1, 1024), 0), view_418, out=buf219)
        del view_418
        buf220 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf207, buf220, 4096, 128, grid=grid(4096), stream=stream0)
        buf221 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf220, buf221, 1024, 4, grid=grid(1024), stream=stream0)
        buf226 = buf200; del buf200  # reuse
        buf227 = reinterpret_tensor(buf207, (1, 512, 1024), (524288, 1024, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf226, buf210, buf214, buf218, primals_308, mul_134, div_66, getitem_191, buf227, 512, 1024, grid=grid(512), stream=stream0)
        del div_66
        del getitem_191
        del primals_308
        buf224 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf225 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf210, buf214, buf218, mul_134, buf224, buf225, 1024, 512, grid=grid(1024), stream=stream0)
        del buf210
        del buf214
        del mul_134
        buf228 = reinterpret_tensor(buf191, (512, 4096), (4096, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (512, 1024), (1024, 1), 0), permute_439, out=buf228)
        del permute_439
        buf229 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1024, 512), (1, 1024), 0), view_416, out=buf229)
        del view_416
        buf230 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf227, buf230, 4096, 128, grid=grid(4096), stream=stream0)
        buf231 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf230, buf231, 1024, 4, grid=grid(1024), stream=stream0)
        buf232 = reinterpret_tensor(buf228, (1, 512, 4096), (2097152, 4096, 1), 0); del buf228  # reuse
        # Source Nodes: [intermediate_output_18], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf232, addmm_112, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_112
        buf233 = reinterpret_tensor(buf227, (512, 1024), (1024, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 4096), (4096, 1), 0), permute_443, out=buf233)
        del permute_443
        buf234 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 512), (1, 4096), 0), view_414, out=buf234)
        del view_414
        buf235 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf232, buf235, 16384, 128, grid=grid(16384), stream=stream0)
        buf236 = reinterpret_tensor(buf230, (1, 4096), (4096, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf235, buf236, 4096, 4, grid=grid(4096), stream=stream0)
        buf241 = buf226; del buf226  # reuse
        buf242 = reinterpret_tensor(buf218, (1, 512, 1024), (524288, 1024, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf241, buf233, primals_302, mul_129, div_67, getitem_187, buf242, 512, 1024, grid=grid(512), stream=stream0)
        del div_67
        del getitem_187
        del primals_302
        buf239 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf240 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf233, mul_129, buf239, buf240, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_129
        buf243 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 1024), (1024, 1), 0), permute_447, out=buf243)
        del permute_447
        buf244 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1024, 512), (1, 1024), 0), view_412, out=buf244)
        del view_412
        buf245 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf242, buf245, 4096, 128, grid=grid(4096), stream=stream0)
        del buf242
        buf246 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf245, buf246, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf247 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf243, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_282, getitem_283, getitem_284, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_282
        del getitem_283
        del getitem_284
        buf248 = buf247[0]
        buf249 = buf247[1]
        buf250 = buf247[2]
        del buf247
        buf251 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (512, 1024), (1024, 1), 0), permute_459, out=buf251)
        del permute_459
        buf252 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (1024, 512), (1, 1024), 0), view_396, out=buf252)
        buf253 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf250, buf253, 4096, 128, grid=grid(4096), stream=stream0)
        buf254 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf253, buf254, 1024, 4, grid=grid(1024), stream=stream0)
        buf255 = reinterpret_tensor(buf250, (512, 1024), (1024, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 1024), (1024, 1), 0), permute_464, out=buf255)
        del permute_464
        buf256 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1024, 512), (1, 1024), 0), view_396, out=buf256)
        buf257 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf249, buf257, 4096, 128, grid=grid(4096), stream=stream0)
        buf258 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf257, buf258, 1024, 4, grid=grid(1024), stream=stream0)
        buf259 = reinterpret_tensor(buf249, (512, 1024), (1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 1024), (1024, 1), 0), permute_468, out=buf259)
        del permute_468
        buf260 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1024, 512), (1, 1024), 0), view_396, out=buf260)
        del view_396
        buf261 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf248, buf261, 4096, 128, grid=grid(4096), stream=stream0)
        buf262 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf261, buf262, 1024, 4, grid=grid(1024), stream=stream0)
        buf267 = buf241; del buf241  # reuse
        buf268 = reinterpret_tensor(buf248, (1, 512, 1024), (524288, 1024, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf267, buf251, buf255, buf259, primals_292, mul_127, div_69, getitem_181, buf268, 512, 1024, grid=grid(512), stream=stream0)
        del div_69
        del getitem_181
        del primals_292
        buf265 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf266 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf251, buf255, buf259, mul_127, buf265, buf266, 1024, 512, grid=grid(1024), stream=stream0)
        del buf251
        del buf255
        del mul_127
        buf269 = reinterpret_tensor(buf232, (512, 4096), (4096, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 1024), (1024, 1), 0), permute_472, out=buf269)
        del permute_472
        buf270 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (1024, 512), (1, 1024), 0), view_394, out=buf270)
        del view_394
        buf271 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf268, buf271, 4096, 128, grid=grid(4096), stream=stream0)
        buf272 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf271, buf272, 1024, 4, grid=grid(1024), stream=stream0)
        buf273 = reinterpret_tensor(buf269, (1, 512, 4096), (2097152, 4096, 1), 0); del buf269  # reuse
        # Source Nodes: [intermediate_output_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf273, addmm_106, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_106
        buf274 = reinterpret_tensor(buf268, (512, 1024), (1024, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (512, 4096), (4096, 1), 0), permute_476, out=buf274)
        del permute_476
        buf275 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (4096, 512), (1, 4096), 0), view_392, out=buf275)
        del view_392
        buf276 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf273, buf276, 16384, 128, grid=grid(16384), stream=stream0)
        buf277 = reinterpret_tensor(buf271, (1, 4096), (4096, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf276, buf277, 4096, 4, grid=grid(4096), stream=stream0)
        buf282 = buf267; del buf267  # reuse
        buf283 = reinterpret_tensor(buf259, (1, 512, 1024), (524288, 1024, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf282, buf274, primals_286, mul_122, div_70, getitem_177, buf283, 512, 1024, grid=grid(512), stream=stream0)
        del div_70
        del getitem_177
        del primals_286
        buf280 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf281 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf274, mul_122, buf280, buf281, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_122
        buf284 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (512, 1024), (1024, 1), 0), permute_480, out=buf284)
        del permute_480
        buf285 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (1024, 512), (1, 1024), 0), view_390, out=buf285)
        del view_390
        buf286 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf283, buf286, 4096, 128, grid=grid(4096), stream=stream0)
        del buf283
        buf287 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf286, buf287, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf288 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf284, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_289, getitem_290, getitem_291, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_289
        del getitem_290
        del getitem_291
        buf289 = buf288[0]
        buf290 = buf288[1]
        buf291 = buf288[2]
        del buf288
        buf292 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 1024), (1024, 1), 0), permute_492, out=buf292)
        del permute_492
        buf293 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (1024, 512), (1, 1024), 0), view_374, out=buf293)
        buf294 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf291, buf294, 4096, 128, grid=grid(4096), stream=stream0)
        buf295 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf294, buf295, 1024, 4, grid=grid(1024), stream=stream0)
        buf296 = reinterpret_tensor(buf291, (512, 1024), (1024, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (512, 1024), (1024, 1), 0), permute_497, out=buf296)
        del permute_497
        buf297 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (1024, 512), (1, 1024), 0), view_374, out=buf297)
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf290, buf298, 4096, 128, grid=grid(4096), stream=stream0)
        buf299 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf298, buf299, 1024, 4, grid=grid(1024), stream=stream0)
        buf300 = reinterpret_tensor(buf290, (512, 1024), (1024, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (512, 1024), (1024, 1), 0), permute_501, out=buf300)
        del permute_501
        buf301 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1024, 512), (1, 1024), 0), view_374, out=buf301)
        del view_374
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf289, buf302, 4096, 128, grid=grid(4096), stream=stream0)
        buf303 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf302, buf303, 1024, 4, grid=grid(1024), stream=stream0)
        buf308 = buf282; del buf282  # reuse
        buf309 = reinterpret_tensor(buf289, (1, 512, 1024), (524288, 1024, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf308, buf292, buf296, buf300, primals_276, mul_120, div_72, getitem_171, buf309, 512, 1024, grid=grid(512), stream=stream0)
        del div_72
        del getitem_171
        del primals_276
        buf306 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf307 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf292, buf296, buf300, mul_120, buf306, buf307, 1024, 512, grid=grid(1024), stream=stream0)
        del buf292
        del buf296
        del mul_120
        buf310 = reinterpret_tensor(buf273, (512, 4096), (4096, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 1024), (1024, 1), 0), permute_505, out=buf310)
        del permute_505
        buf311 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (1024, 512), (1, 1024), 0), view_372, out=buf311)
        del view_372
        buf312 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf309, buf312, 4096, 128, grid=grid(4096), stream=stream0)
        buf313 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf312, buf313, 1024, 4, grid=grid(1024), stream=stream0)
        buf314 = reinterpret_tensor(buf310, (1, 512, 4096), (2097152, 4096, 1), 0); del buf310  # reuse
        # Source Nodes: [intermediate_output_16], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf314, addmm_100, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_100
        buf315 = reinterpret_tensor(buf309, (512, 1024), (1024, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (512, 4096), (4096, 1), 0), permute_509, out=buf315)
        del permute_509
        buf316 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (4096, 512), (1, 4096), 0), view_370, out=buf316)
        del view_370
        buf317 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf314, buf317, 16384, 128, grid=grid(16384), stream=stream0)
        buf318 = reinterpret_tensor(buf312, (1, 4096), (4096, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf317, buf318, 4096, 4, grid=grid(4096), stream=stream0)
        buf323 = buf308; del buf308  # reuse
        buf324 = reinterpret_tensor(buf300, (1, 512, 1024), (524288, 1024, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf323, buf315, primals_270, mul_115, div_73, getitem_167, buf324, 512, 1024, grid=grid(512), stream=stream0)
        del div_73
        del getitem_167
        del primals_270
        buf321 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf322 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf315, mul_115, buf321, buf322, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_115
        buf325 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (512, 1024), (1024, 1), 0), permute_513, out=buf325)
        del permute_513
        buf326 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (1024, 512), (1, 1024), 0), view_368, out=buf326)
        del view_368
        buf327 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf324, buf327, 4096, 128, grid=grid(4096), stream=stream0)
        del buf324
        buf328 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf327, buf328, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf329 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf325, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_296, getitem_297, getitem_298, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_296
        del getitem_297
        del getitem_298
        buf330 = buf329[0]
        buf331 = buf329[1]
        buf332 = buf329[2]
        del buf329
        buf333 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (512, 1024), (1024, 1), 0), permute_525, out=buf333)
        del permute_525
        buf334 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (1024, 512), (1, 1024), 0), view_352, out=buf334)
        buf335 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf332, buf335, 4096, 128, grid=grid(4096), stream=stream0)
        buf336 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf335, buf336, 1024, 4, grid=grid(1024), stream=stream0)
        buf337 = reinterpret_tensor(buf332, (512, 1024), (1024, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (512, 1024), (1024, 1), 0), permute_530, out=buf337)
        del permute_530
        buf338 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (1024, 512), (1, 1024), 0), view_352, out=buf338)
        buf339 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf331, buf339, 4096, 128, grid=grid(4096), stream=stream0)
        buf340 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf339, buf340, 1024, 4, grid=grid(1024), stream=stream0)
        buf341 = reinterpret_tensor(buf331, (512, 1024), (1024, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (512, 1024), (1024, 1), 0), permute_534, out=buf341)
        del permute_534
        buf342 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (1024, 512), (1, 1024), 0), view_352, out=buf342)
        del view_352
        buf343 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf330, buf343, 4096, 128, grid=grid(4096), stream=stream0)
        buf344 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf343, buf344, 1024, 4, grid=grid(1024), stream=stream0)
        buf349 = buf323; del buf323  # reuse
        buf350 = reinterpret_tensor(buf330, (1, 512, 1024), (524288, 1024, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf349, buf333, buf337, buf341, primals_260, mul_113, div_75, getitem_161, buf350, 512, 1024, grid=grid(512), stream=stream0)
        del div_75
        del getitem_161
        del primals_260
        buf347 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf348 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf333, buf337, buf341, mul_113, buf347, buf348, 1024, 512, grid=grid(1024), stream=stream0)
        del buf333
        del buf337
        del mul_113
        buf351 = reinterpret_tensor(buf314, (512, 4096), (4096, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (512, 1024), (1024, 1), 0), permute_538, out=buf351)
        del permute_538
        buf352 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (1024, 512), (1, 1024), 0), view_350, out=buf352)
        del view_350
        buf353 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf350, buf353, 4096, 128, grid=grid(4096), stream=stream0)
        buf354 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf353, buf354, 1024, 4, grid=grid(1024), stream=stream0)
        buf355 = reinterpret_tensor(buf351, (1, 512, 4096), (2097152, 4096, 1), 0); del buf351  # reuse
        # Source Nodes: [intermediate_output_15], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf355, addmm_94, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_94
        buf356 = reinterpret_tensor(buf350, (512, 1024), (1024, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (512, 4096), (4096, 1), 0), permute_542, out=buf356)
        del permute_542
        buf357 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (4096, 512), (1, 4096), 0), view_348, out=buf357)
        del view_348
        buf358 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf355, buf358, 16384, 128, grid=grid(16384), stream=stream0)
        buf359 = reinterpret_tensor(buf353, (1, 4096), (4096, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf358, buf359, 4096, 4, grid=grid(4096), stream=stream0)
        buf364 = buf349; del buf349  # reuse
        buf365 = reinterpret_tensor(buf341, (1, 512, 1024), (524288, 1024, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf364, buf356, primals_254, mul_108, div_76, getitem_157, buf365, 512, 1024, grid=grid(512), stream=stream0)
        del div_76
        del getitem_157
        del primals_254
        buf362 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf363 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf356, mul_108, buf362, buf363, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_108
        buf366 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (512, 1024), (1024, 1), 0), permute_546, out=buf366)
        del permute_546
        buf367 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (1024, 512), (1, 1024), 0), view_346, out=buf367)
        del view_346
        buf368 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf365, buf368, 4096, 128, grid=grid(4096), stream=stream0)
        del buf365
        buf369 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf368, buf369, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf370 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf366, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_303, getitem_304, getitem_305, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_303
        del getitem_304
        del getitem_305
        buf371 = buf370[0]
        buf372 = buf370[1]
        buf373 = buf370[2]
        del buf370
        buf374 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (512, 1024), (1024, 1), 0), permute_558, out=buf374)
        del permute_558
        buf375 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (1024, 512), (1, 1024), 0), view_330, out=buf375)
        buf376 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf373, buf376, 4096, 128, grid=grid(4096), stream=stream0)
        buf377 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf376, buf377, 1024, 4, grid=grid(1024), stream=stream0)
        buf378 = reinterpret_tensor(buf373, (512, 1024), (1024, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (512, 1024), (1024, 1), 0), permute_563, out=buf378)
        del permute_563
        buf379 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1024, 512), (1, 1024), 0), view_330, out=buf379)
        buf380 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf372, buf380, 4096, 128, grid=grid(4096), stream=stream0)
        buf381 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf380, buf381, 1024, 4, grid=grid(1024), stream=stream0)
        buf382 = reinterpret_tensor(buf372, (512, 1024), (1024, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (512, 1024), (1024, 1), 0), permute_567, out=buf382)
        del permute_567
        buf383 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (1024, 512), (1, 1024), 0), view_330, out=buf383)
        del view_330
        buf384 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf371, buf384, 4096, 128, grid=grid(4096), stream=stream0)
        buf385 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf384, buf385, 1024, 4, grid=grid(1024), stream=stream0)
        buf390 = buf364; del buf364  # reuse
        buf391 = reinterpret_tensor(buf371, (1, 512, 1024), (524288, 1024, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf390, buf374, buf378, buf382, primals_244, mul_106, div_78, getitem_151, buf391, 512, 1024, grid=grid(512), stream=stream0)
        del div_78
        del getitem_151
        del primals_244
        buf388 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf389 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf374, buf378, buf382, mul_106, buf388, buf389, 1024, 512, grid=grid(1024), stream=stream0)
        del buf374
        del buf378
        del mul_106
        buf392 = reinterpret_tensor(buf355, (512, 4096), (4096, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (512, 1024), (1024, 1), 0), permute_571, out=buf392)
        del permute_571
        buf393 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (1024, 512), (1, 1024), 0), view_328, out=buf393)
        del view_328
        buf394 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf391, buf394, 4096, 128, grid=grid(4096), stream=stream0)
        buf395 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf394, buf395, 1024, 4, grid=grid(1024), stream=stream0)
        buf396 = reinterpret_tensor(buf392, (1, 512, 4096), (2097152, 4096, 1), 0); del buf392  # reuse
        # Source Nodes: [intermediate_output_14], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf396, addmm_88, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_88
        buf397 = reinterpret_tensor(buf391, (512, 1024), (1024, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (512, 4096), (4096, 1), 0), permute_575, out=buf397)
        del permute_575
        buf398 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (4096, 512), (1, 4096), 0), view_326, out=buf398)
        del view_326
        buf399 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf396, buf399, 16384, 128, grid=grid(16384), stream=stream0)
        buf400 = reinterpret_tensor(buf394, (1, 4096), (4096, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf399, buf400, 4096, 4, grid=grid(4096), stream=stream0)
        buf405 = buf390; del buf390  # reuse
        buf406 = reinterpret_tensor(buf382, (1, 512, 1024), (524288, 1024, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf405, buf397, primals_238, mul_101, div_79, getitem_147, buf406, 512, 1024, grid=grid(512), stream=stream0)
        del div_79
        del getitem_147
        del primals_238
        buf403 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf404 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf397, mul_101, buf403, buf404, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_101
        buf407 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (512, 1024), (1024, 1), 0), permute_579, out=buf407)
        del permute_579
        buf408 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (1024, 512), (1, 1024), 0), view_324, out=buf408)
        del view_324
        buf409 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf406, buf409, 4096, 128, grid=grid(4096), stream=stream0)
        del buf406
        buf410 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf409, buf410, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf411 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf407, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_310, getitem_311, getitem_312, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_310
        del getitem_311
        del getitem_312
        buf412 = buf411[0]
        buf413 = buf411[1]
        buf414 = buf411[2]
        del buf411
        buf415 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (512, 1024), (1024, 1), 0), permute_591, out=buf415)
        del permute_591
        buf416 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (1024, 512), (1, 1024), 0), view_308, out=buf416)
        buf417 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf414, buf417, 4096, 128, grid=grid(4096), stream=stream0)
        buf418 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf417, buf418, 1024, 4, grid=grid(1024), stream=stream0)
        buf419 = reinterpret_tensor(buf414, (512, 1024), (1024, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (512, 1024), (1024, 1), 0), permute_596, out=buf419)
        del permute_596
        buf420 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (1024, 512), (1, 1024), 0), view_308, out=buf420)
        buf421 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf413, buf421, 4096, 128, grid=grid(4096), stream=stream0)
        buf422 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf421, buf422, 1024, 4, grid=grid(1024), stream=stream0)
        buf423 = reinterpret_tensor(buf413, (512, 1024), (1024, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), permute_600, out=buf423)
        del permute_600
        buf424 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (1024, 512), (1, 1024), 0), view_308, out=buf424)
        del view_308
        buf425 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf412, buf425, 4096, 128, grid=grid(4096), stream=stream0)
        buf426 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf425, buf426, 1024, 4, grid=grid(1024), stream=stream0)
        buf431 = buf405; del buf405  # reuse
        buf432 = reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf431, buf415, buf419, buf423, primals_228, mul_99, div_81, getitem_141, buf432, 512, 1024, grid=grid(512), stream=stream0)
        del div_81
        del getitem_141
        del primals_228
        buf429 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf430 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf415, buf419, buf423, mul_99, buf429, buf430, 1024, 512, grid=grid(1024), stream=stream0)
        del buf415
        del buf419
        del mul_99
        buf433 = reinterpret_tensor(buf396, (512, 4096), (4096, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (512, 1024), (1024, 1), 0), permute_604, out=buf433)
        del permute_604
        buf434 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (1024, 512), (1, 1024), 0), view_306, out=buf434)
        del view_306
        buf435 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf432, buf435, 4096, 128, grid=grid(4096), stream=stream0)
        buf436 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf435, buf436, 1024, 4, grid=grid(1024), stream=stream0)
        buf437 = reinterpret_tensor(buf433, (1, 512, 4096), (2097152, 4096, 1), 0); del buf433  # reuse
        # Source Nodes: [intermediate_output_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf437, addmm_82, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_82
        buf438 = reinterpret_tensor(buf432, (512, 1024), (1024, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (512, 4096), (4096, 1), 0), permute_608, out=buf438)
        del permute_608
        buf439 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (4096, 512), (1, 4096), 0), view_304, out=buf439)
        del view_304
        buf440 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf437, buf440, 16384, 128, grid=grid(16384), stream=stream0)
        buf441 = reinterpret_tensor(buf435, (1, 4096), (4096, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf440, buf441, 4096, 4, grid=grid(4096), stream=stream0)
        buf446 = buf431; del buf431  # reuse
        buf447 = reinterpret_tensor(buf423, (1, 512, 1024), (524288, 1024, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf446, buf438, primals_222, mul_94, div_82, getitem_137, buf447, 512, 1024, grid=grid(512), stream=stream0)
        del div_82
        del getitem_137
        del primals_222
        buf444 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf445 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf438, mul_94, buf444, buf445, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_94
        buf448 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (512, 1024), (1024, 1), 0), permute_612, out=buf448)
        del permute_612
        buf449 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (1024, 512), (1, 1024), 0), view_302, out=buf449)
        del view_302
        buf450 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf447, buf450, 4096, 128, grid=grid(4096), stream=stream0)
        del buf447
        buf451 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf450, buf451, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf452 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf448, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_317, getitem_318, getitem_319, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_317
        del getitem_318
        del getitem_319
        buf453 = buf452[0]
        buf454 = buf452[1]
        buf455 = buf452[2]
        del buf452
        buf456 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (512, 1024), (1024, 1), 0), permute_624, out=buf456)
        del permute_624
        buf457 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (1024, 512), (1, 1024), 0), view_286, out=buf457)
        buf458 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf455, buf458, 4096, 128, grid=grid(4096), stream=stream0)
        buf459 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf458, buf459, 1024, 4, grid=grid(1024), stream=stream0)
        buf460 = reinterpret_tensor(buf455, (512, 1024), (1024, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 1024), (1024, 1), 0), permute_629, out=buf460)
        del permute_629
        buf461 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (1024, 512), (1, 1024), 0), view_286, out=buf461)
        buf462 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf454, buf462, 4096, 128, grid=grid(4096), stream=stream0)
        buf463 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf462, buf463, 1024, 4, grid=grid(1024), stream=stream0)
        buf464 = reinterpret_tensor(buf454, (512, 1024), (1024, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 1024), (1024, 1), 0), permute_633, out=buf464)
        del permute_633
        buf465 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (1024, 512), (1, 1024), 0), view_286, out=buf465)
        del view_286
        buf466 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf453, buf466, 4096, 128, grid=grid(4096), stream=stream0)
        buf467 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf466, buf467, 1024, 4, grid=grid(1024), stream=stream0)
        buf472 = buf446; del buf446  # reuse
        buf473 = reinterpret_tensor(buf453, (1, 512, 1024), (524288, 1024, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf472, buf456, buf460, buf464, primals_212, mul_92, div_84, getitem_131, buf473, 512, 1024, grid=grid(512), stream=stream0)
        del div_84
        del getitem_131
        del primals_212
        buf470 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf471 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf456, buf460, buf464, mul_92, buf470, buf471, 1024, 512, grid=grid(1024), stream=stream0)
        del buf456
        del buf460
        del mul_92
        buf474 = reinterpret_tensor(buf437, (512, 4096), (4096, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (512, 1024), (1024, 1), 0), permute_637, out=buf474)
        del permute_637
        buf475 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1024, 512), (1, 1024), 0), view_284, out=buf475)
        del view_284
        buf476 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf473, buf476, 4096, 128, grid=grid(4096), stream=stream0)
        buf477 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf476, buf477, 1024, 4, grid=grid(1024), stream=stream0)
        buf478 = reinterpret_tensor(buf474, (1, 512, 4096), (2097152, 4096, 1), 0); del buf474  # reuse
        # Source Nodes: [intermediate_output_12], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf478, addmm_76, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_76
        buf479 = reinterpret_tensor(buf473, (512, 1024), (1024, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 4096), (4096, 1), 0), permute_641, out=buf479)
        del permute_641
        buf480 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (4096, 512), (1, 4096), 0), view_282, out=buf480)
        del view_282
        buf481 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf478, buf481, 16384, 128, grid=grid(16384), stream=stream0)
        buf482 = reinterpret_tensor(buf476, (1, 4096), (4096, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf481, buf482, 4096, 4, grid=grid(4096), stream=stream0)
        buf487 = buf472; del buf472  # reuse
        buf488 = reinterpret_tensor(buf464, (1, 512, 1024), (524288, 1024, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf487, buf479, primals_206, mul_87, div_85, getitem_127, buf488, 512, 1024, grid=grid(512), stream=stream0)
        del div_85
        del getitem_127
        del primals_206
        buf485 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf486 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf479, mul_87, buf485, buf486, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_87
        buf489 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (512, 1024), (1024, 1), 0), permute_645, out=buf489)
        del permute_645
        buf490 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (1024, 512), (1, 1024), 0), view_280, out=buf490)
        del view_280
        buf491 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf488, buf491, 4096, 128, grid=grid(4096), stream=stream0)
        del buf488
        buf492 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf491, buf492, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf493 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf489, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_324, getitem_325, getitem_326, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_324
        del getitem_325
        del getitem_326
        buf494 = buf493[0]
        buf495 = buf493[1]
        buf496 = buf493[2]
        del buf493
        buf497 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (512, 1024), (1024, 1), 0), permute_657, out=buf497)
        del permute_657
        buf498 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (1024, 512), (1, 1024), 0), view_264, out=buf498)
        buf499 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf496, buf499, 4096, 128, grid=grid(4096), stream=stream0)
        buf500 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf499, buf500, 1024, 4, grid=grid(1024), stream=stream0)
        buf501 = reinterpret_tensor(buf496, (512, 1024), (1024, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (512, 1024), (1024, 1), 0), permute_662, out=buf501)
        del permute_662
        buf502 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (1024, 512), (1, 1024), 0), view_264, out=buf502)
        buf503 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf495, buf503, 4096, 128, grid=grid(4096), stream=stream0)
        buf504 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf503, buf504, 1024, 4, grid=grid(1024), stream=stream0)
        buf505 = reinterpret_tensor(buf495, (512, 1024), (1024, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (512, 1024), (1024, 1), 0), permute_666, out=buf505)
        del permute_666
        buf506 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (1024, 512), (1, 1024), 0), view_264, out=buf506)
        del view_264
        buf507 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf494, buf507, 4096, 128, grid=grid(4096), stream=stream0)
        buf508 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf507, buf508, 1024, 4, grid=grid(1024), stream=stream0)
        buf513 = buf487; del buf487  # reuse
        buf514 = reinterpret_tensor(buf494, (1, 512, 1024), (524288, 1024, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf513, buf497, buf501, buf505, primals_196, mul_85, div_87, getitem_121, buf514, 512, 1024, grid=grid(512), stream=stream0)
        del div_87
        del getitem_121
        del primals_196
        buf511 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf512 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf497, buf501, buf505, mul_85, buf511, buf512, 1024, 512, grid=grid(1024), stream=stream0)
        del buf497
        del buf501
        del mul_85
        buf515 = reinterpret_tensor(buf478, (512, 4096), (4096, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0), permute_670, out=buf515)
        del permute_670
        buf516 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (1024, 512), (1, 1024), 0), view_262, out=buf516)
        del view_262
        buf517 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf514, buf517, 4096, 128, grid=grid(4096), stream=stream0)
        buf518 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf517, buf518, 1024, 4, grid=grid(1024), stream=stream0)
        buf519 = reinterpret_tensor(buf515, (1, 512, 4096), (2097152, 4096, 1), 0); del buf515  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf519, addmm_70, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_70
        buf520 = reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf519, (512, 4096), (4096, 1), 0), permute_674, out=buf520)
        del permute_674
        buf521 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf519, (4096, 512), (1, 4096), 0), view_260, out=buf521)
        del view_260
        buf522 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf519, buf522, 16384, 128, grid=grid(16384), stream=stream0)
        buf523 = reinterpret_tensor(buf517, (1, 4096), (4096, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf522, buf523, 4096, 4, grid=grid(4096), stream=stream0)
        buf528 = buf513; del buf513  # reuse
        buf529 = reinterpret_tensor(buf505, (1, 512, 1024), (524288, 1024, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf528, buf520, primals_190, mul_80, div_88, getitem_117, buf529, 512, 1024, grid=grid(512), stream=stream0)
        del div_88
        del getitem_117
        del primals_190
        buf526 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf527 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf520, mul_80, buf526, buf527, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_80
        buf530 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (512, 1024), (1024, 1), 0), permute_678, out=buf530)
        del permute_678
        buf531 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (1024, 512), (1, 1024), 0), view_258, out=buf531)
        del view_258
        buf532 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf529, buf532, 4096, 128, grid=grid(4096), stream=stream0)
        del buf529
        buf533 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf532, buf533, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf534 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf530, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_36, clone_default_37, clone_default_38, None, alias_default_25, getitem_331, getitem_332, getitem_333, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_25
        del clone_default_36
        del clone_default_37
        del clone_default_38
        del getitem_331
        del getitem_332
        del getitem_333
        buf535 = buf534[0]
        buf536 = buf534[1]
        buf537 = buf534[2]
        del buf534
        buf538 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (512, 1024), (1024, 1), 0), permute_690, out=buf538)
        del permute_690
        buf539 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (1024, 512), (1, 1024), 0), view_242, out=buf539)
        buf540 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf537, buf540, 4096, 128, grid=grid(4096), stream=stream0)
        buf541 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf540, buf541, 1024, 4, grid=grid(1024), stream=stream0)
        buf542 = reinterpret_tensor(buf537, (512, 1024), (1024, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (512, 1024), (1024, 1), 0), permute_695, out=buf542)
        del permute_695
        buf543 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (1024, 512), (1, 1024), 0), view_242, out=buf543)
        buf544 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf536, buf544, 4096, 128, grid=grid(4096), stream=stream0)
        buf545 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf544, buf545, 1024, 4, grid=grid(1024), stream=stream0)
        buf546 = reinterpret_tensor(buf536, (512, 1024), (1024, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (512, 1024), (1024, 1), 0), permute_699, out=buf546)
        del permute_699
        buf547 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (1024, 512), (1, 1024), 0), view_242, out=buf547)
        del view_242
        buf548 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf535, buf548, 4096, 128, grid=grid(4096), stream=stream0)
        buf549 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf548, buf549, 1024, 4, grid=grid(1024), stream=stream0)
        buf554 = buf528; del buf528  # reuse
        buf555 = reinterpret_tensor(buf535, (1, 512, 1024), (524288, 1024, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf554, buf538, buf542, buf546, primals_180, mul_78, div_90, getitem_111, buf555, 512, 1024, grid=grid(512), stream=stream0)
        del div_90
        del getitem_111
        del primals_180
        buf552 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf553 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf538, buf542, buf546, mul_78, buf552, buf553, 1024, 512, grid=grid(1024), stream=stream0)
        del buf538
        del buf542
        del mul_78
        buf556 = reinterpret_tensor(buf519, (512, 4096), (4096, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1024), (1024, 1), 0), permute_703, out=buf556)
        del permute_703
        buf557 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (1024, 512), (1, 1024), 0), view_240, out=buf557)
        del view_240
        buf558 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf555, buf558, 4096, 128, grid=grid(4096), stream=stream0)
        buf559 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf558, buf559, 1024, 4, grid=grid(1024), stream=stream0)
        buf560 = reinterpret_tensor(buf556, (1, 512, 4096), (2097152, 4096, 1), 0); del buf556  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf560, addmm_64, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_64
        buf561 = reinterpret_tensor(buf555, (512, 1024), (1024, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (512, 4096), (4096, 1), 0), permute_707, out=buf561)
        del permute_707
        buf562 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (4096, 512), (1, 4096), 0), view_238, out=buf562)
        del view_238
        buf563 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf560, buf563, 16384, 128, grid=grid(16384), stream=stream0)
        buf564 = reinterpret_tensor(buf558, (1, 4096), (4096, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf563, buf564, 4096, 4, grid=grid(4096), stream=stream0)
        buf569 = buf554; del buf554  # reuse
        buf570 = reinterpret_tensor(buf546, (1, 512, 1024), (524288, 1024, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf569, buf561, primals_174, mul_73, div_91, getitem_107, buf570, 512, 1024, grid=grid(512), stream=stream0)
        del div_91
        del getitem_107
        del primals_174
        buf567 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf568 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf561, mul_73, buf567, buf568, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_73
        buf571 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (512, 1024), (1024, 1), 0), permute_711, out=buf571)
        del permute_711
        buf572 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (1024, 512), (1, 1024), 0), view_236, out=buf572)
        del view_236
        buf573 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf570, buf573, 4096, 128, grid=grid(4096), stream=stream0)
        del buf570
        buf574 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf573, buf574, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf575 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf571, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_39, clone_default_40, clone_default_41, None, alias_default_27, getitem_338, getitem_339, getitem_340, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_27
        del clone_default_39
        del clone_default_40
        del clone_default_41
        del getitem_338
        del getitem_339
        del getitem_340
        buf576 = buf575[0]
        buf577 = buf575[1]
        buf578 = buf575[2]
        del buf575
        buf579 = buf571; del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (512, 1024), (1024, 1), 0), permute_723, out=buf579)
        del permute_723
        buf580 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (1024, 512), (1, 1024), 0), view_220, out=buf580)
        buf581 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf578, buf581, 4096, 128, grid=grid(4096), stream=stream0)
        buf582 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf581, buf582, 1024, 4, grid=grid(1024), stream=stream0)
        buf583 = reinterpret_tensor(buf578, (512, 1024), (1024, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (512, 1024), (1024, 1), 0), permute_728, out=buf583)
        del permute_728
        buf584 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (1024, 512), (1, 1024), 0), view_220, out=buf584)
        buf585 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf577, buf585, 4096, 128, grid=grid(4096), stream=stream0)
        buf586 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf585, buf586, 1024, 4, grid=grid(1024), stream=stream0)
        buf587 = reinterpret_tensor(buf577, (512, 1024), (1024, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (512, 1024), (1024, 1), 0), permute_732, out=buf587)
        del permute_732
        buf588 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (1024, 512), (1, 1024), 0), view_220, out=buf588)
        del view_220
        buf589 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf576, buf589, 4096, 128, grid=grid(4096), stream=stream0)
        buf590 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf589, buf590, 1024, 4, grid=grid(1024), stream=stream0)
        buf595 = buf569; del buf569  # reuse
        buf596 = reinterpret_tensor(buf576, (1, 512, 1024), (524288, 1024, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf595, buf579, buf583, buf587, primals_164, mul_71, div_93, getitem_101, buf596, 512, 1024, grid=grid(512), stream=stream0)
        del div_93
        del getitem_101
        del primals_164
        buf593 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf594 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf579, buf583, buf587, mul_71, buf593, buf594, 1024, 512, grid=grid(1024), stream=stream0)
        del buf579
        del buf583
        del mul_71
        buf597 = reinterpret_tensor(buf560, (512, 4096), (4096, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (512, 1024), (1024, 1), 0), permute_736, out=buf597)
        del permute_736
        buf598 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (1024, 512), (1, 1024), 0), view_218, out=buf598)
        del view_218
        buf599 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf596, buf599, 4096, 128, grid=grid(4096), stream=stream0)
        buf600 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf599, buf600, 1024, 4, grid=grid(1024), stream=stream0)
        buf601 = reinterpret_tensor(buf597, (1, 512, 4096), (2097152, 4096, 1), 0); del buf597  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf601, addmm_58, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_58
        buf602 = reinterpret_tensor(buf596, (512, 1024), (1024, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf601, (512, 4096), (4096, 1), 0), permute_740, out=buf602)
        del permute_740
        buf603 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf601, (4096, 512), (1, 4096), 0), view_216, out=buf603)
        del view_216
        buf604 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf601, buf604, 16384, 128, grid=grid(16384), stream=stream0)
        buf605 = reinterpret_tensor(buf599, (1, 4096), (4096, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf604, buf605, 4096, 4, grid=grid(4096), stream=stream0)
        buf610 = buf595; del buf595  # reuse
        buf611 = reinterpret_tensor(buf587, (1, 512, 1024), (524288, 1024, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf610, buf602, primals_158, mul_66, div_94, getitem_97, buf611, 512, 1024, grid=grid(512), stream=stream0)
        del div_94
        del getitem_97
        del primals_158
        buf608 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf609 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf602, mul_66, buf608, buf609, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_66
        buf612 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (512, 1024), (1024, 1), 0), permute_744, out=buf612)
        del permute_744
        buf613 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (1024, 512), (1, 1024), 0), view_214, out=buf613)
        del view_214
        buf614 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf611, buf614, 4096, 128, grid=grid(4096), stream=stream0)
        del buf611
        buf615 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf614, buf615, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf616 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf612, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_42, clone_default_43, clone_default_44, None, alias_default_29, getitem_345, getitem_346, getitem_347, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_29
        del clone_default_42
        del clone_default_43
        del clone_default_44
        del getitem_345
        del getitem_346
        del getitem_347
        buf617 = buf616[0]
        buf618 = buf616[1]
        buf619 = buf616[2]
        del buf616
        buf620 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (512, 1024), (1024, 1), 0), permute_756, out=buf620)
        del permute_756
        buf621 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (1024, 512), (1, 1024), 0), view_198, out=buf621)
        buf622 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf619, buf622, 4096, 128, grid=grid(4096), stream=stream0)
        buf623 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf622, buf623, 1024, 4, grid=grid(1024), stream=stream0)
        buf624 = reinterpret_tensor(buf619, (512, 1024), (1024, 1), 0); del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (512, 1024), (1024, 1), 0), permute_761, out=buf624)
        del permute_761
        buf625 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (1024, 512), (1, 1024), 0), view_198, out=buf625)
        buf626 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf618, buf626, 4096, 128, grid=grid(4096), stream=stream0)
        buf627 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf626, buf627, 1024, 4, grid=grid(1024), stream=stream0)
        buf628 = reinterpret_tensor(buf618, (512, 1024), (1024, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (512, 1024), (1024, 1), 0), permute_765, out=buf628)
        del permute_765
        buf629 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (1024, 512), (1, 1024), 0), view_198, out=buf629)
        del view_198
        buf630 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf617, buf630, 4096, 128, grid=grid(4096), stream=stream0)
        buf631 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf630, buf631, 1024, 4, grid=grid(1024), stream=stream0)
        buf636 = buf610; del buf610  # reuse
        buf637 = reinterpret_tensor(buf617, (1, 512, 1024), (524288, 1024, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf636, buf620, buf624, buf628, primals_148, mul_64, div_96, getitem_91, buf637, 512, 1024, grid=grid(512), stream=stream0)
        del div_96
        del getitem_91
        del primals_148
        buf634 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf635 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf620, buf624, buf628, mul_64, buf634, buf635, 1024, 512, grid=grid(1024), stream=stream0)
        del buf620
        del buf624
        del mul_64
        buf638 = reinterpret_tensor(buf601, (512, 4096), (4096, 1), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (512, 1024), (1024, 1), 0), permute_769, out=buf638)
        del permute_769
        buf639 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (1024, 512), (1, 1024), 0), view_196, out=buf639)
        del view_196
        buf640 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf637, buf640, 4096, 128, grid=grid(4096), stream=stream0)
        buf641 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf640, buf641, 1024, 4, grid=grid(1024), stream=stream0)
        buf642 = reinterpret_tensor(buf638, (1, 512, 4096), (2097152, 4096, 1), 0); del buf638  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf642, addmm_52, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_52
        buf643 = reinterpret_tensor(buf637, (512, 1024), (1024, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (512, 4096), (4096, 1), 0), permute_773, out=buf643)
        del permute_773
        buf644 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (4096, 512), (1, 4096), 0), view_194, out=buf644)
        del view_194
        buf645 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf642, buf645, 16384, 128, grid=grid(16384), stream=stream0)
        buf646 = reinterpret_tensor(buf640, (1, 4096), (4096, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf645, buf646, 4096, 4, grid=grid(4096), stream=stream0)
        buf651 = buf636; del buf636  # reuse
        buf652 = reinterpret_tensor(buf628, (1, 512, 1024), (524288, 1024, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf651, buf643, primals_142, mul_59, div_97, getitem_87, buf652, 512, 1024, grid=grid(512), stream=stream0)
        del div_97
        del getitem_87
        del primals_142
        buf649 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf650 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf643, mul_59, buf649, buf650, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_59
        buf653 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (512, 1024), (1024, 1), 0), permute_777, out=buf653)
        del permute_777
        buf654 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (1024, 512), (1, 1024), 0), view_192, out=buf654)
        del view_192
        buf655 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf652, buf655, 4096, 128, grid=grid(4096), stream=stream0)
        del buf652
        buf656 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf655, buf656, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf657 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf653, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_45, clone_default_46, clone_default_47, None, alias_default_31, getitem_352, getitem_353, getitem_354, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_31
        del clone_default_45
        del clone_default_46
        del clone_default_47
        del getitem_352
        del getitem_353
        del getitem_354
        buf658 = buf657[0]
        buf659 = buf657[1]
        buf660 = buf657[2]
        del buf657
        buf661 = buf653; del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (512, 1024), (1024, 1), 0), permute_789, out=buf661)
        del permute_789
        buf662 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (1024, 512), (1, 1024), 0), view_176, out=buf662)
        buf663 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf660, buf663, 4096, 128, grid=grid(4096), stream=stream0)
        buf664 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf663, buf664, 1024, 4, grid=grid(1024), stream=stream0)
        buf665 = reinterpret_tensor(buf660, (512, 1024), (1024, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (512, 1024), (1024, 1), 0), permute_794, out=buf665)
        del permute_794
        buf666 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (1024, 512), (1, 1024), 0), view_176, out=buf666)
        buf667 = buf663; del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf659, buf667, 4096, 128, grid=grid(4096), stream=stream0)
        buf668 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf667, buf668, 1024, 4, grid=grid(1024), stream=stream0)
        buf669 = reinterpret_tensor(buf659, (512, 1024), (1024, 1), 0); del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (512, 1024), (1024, 1), 0), permute_798, out=buf669)
        del permute_798
        buf670 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (1024, 512), (1, 1024), 0), view_176, out=buf670)
        del view_176
        buf671 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf658, buf671, 4096, 128, grid=grid(4096), stream=stream0)
        buf672 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf671, buf672, 1024, 4, grid=grid(1024), stream=stream0)
        buf677 = buf651; del buf651  # reuse
        buf678 = reinterpret_tensor(buf658, (1, 512, 1024), (524288, 1024, 1), 0); del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf677, buf661, buf665, buf669, primals_132, mul_57, div_99, getitem_81, buf678, 512, 1024, grid=grid(512), stream=stream0)
        del div_99
        del getitem_81
        del primals_132
        buf675 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf676 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf661, buf665, buf669, mul_57, buf675, buf676, 1024, 512, grid=grid(1024), stream=stream0)
        del buf661
        del buf665
        del mul_57
        buf679 = reinterpret_tensor(buf642, (512, 4096), (4096, 1), 0); del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (512, 1024), (1024, 1), 0), permute_802, out=buf679)
        del permute_802
        buf680 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (1024, 512), (1, 1024), 0), view_174, out=buf680)
        del view_174
        buf681 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf678, buf681, 4096, 128, grid=grid(4096), stream=stream0)
        buf682 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf681, buf682, 1024, 4, grid=grid(1024), stream=stream0)
        buf683 = reinterpret_tensor(buf679, (1, 512, 4096), (2097152, 4096, 1), 0); del buf679  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf683, addmm_46, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_46
        buf684 = reinterpret_tensor(buf678, (512, 1024), (1024, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (512, 4096), (4096, 1), 0), permute_806, out=buf684)
        del permute_806
        buf685 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (4096, 512), (1, 4096), 0), view_172, out=buf685)
        del view_172
        buf686 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf683, buf686, 16384, 128, grid=grid(16384), stream=stream0)
        buf687 = reinterpret_tensor(buf681, (1, 4096), (4096, 1), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf686, buf687, 4096, 4, grid=grid(4096), stream=stream0)
        buf692 = buf677; del buf677  # reuse
        buf693 = reinterpret_tensor(buf669, (1, 512, 1024), (524288, 1024, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf692, buf684, primals_126, mul_52, div_100, getitem_77, buf693, 512, 1024, grid=grid(512), stream=stream0)
        del div_100
        del getitem_77
        del primals_126
        buf690 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf691 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf684, mul_52, buf690, buf691, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_52
        buf694 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (512, 1024), (1024, 1), 0), permute_810, out=buf694)
        del permute_810
        buf695 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1024, 512), (1, 1024), 0), view_170, out=buf695)
        del view_170
        buf696 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf693, buf696, 4096, 128, grid=grid(4096), stream=stream0)
        del buf693
        buf697 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf696, buf697, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf698 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf694, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_48, clone_default_49, clone_default_50, None, alias_default_33, getitem_359, getitem_360, getitem_361, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_33
        del clone_default_48
        del clone_default_49
        del clone_default_50
        del getitem_359
        del getitem_360
        del getitem_361
        buf699 = buf698[0]
        buf700 = buf698[1]
        buf701 = buf698[2]
        del buf698
        buf702 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (512, 1024), (1024, 1), 0), permute_822, out=buf702)
        del permute_822
        buf703 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (1024, 512), (1, 1024), 0), view_154, out=buf703)
        buf704 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf701, buf704, 4096, 128, grid=grid(4096), stream=stream0)
        buf705 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf704, buf705, 1024, 4, grid=grid(1024), stream=stream0)
        buf706 = reinterpret_tensor(buf701, (512, 1024), (1024, 1), 0); del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (512, 1024), (1024, 1), 0), permute_827, out=buf706)
        del permute_827
        buf707 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (1024, 512), (1, 1024), 0), view_154, out=buf707)
        buf708 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf700, buf708, 4096, 128, grid=grid(4096), stream=stream0)
        buf709 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf708, buf709, 1024, 4, grid=grid(1024), stream=stream0)
        buf710 = reinterpret_tensor(buf700, (512, 1024), (1024, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf699, (512, 1024), (1024, 1), 0), permute_831, out=buf710)
        del permute_831
        buf711 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf699, (1024, 512), (1, 1024), 0), view_154, out=buf711)
        del view_154
        buf712 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf699, buf712, 4096, 128, grid=grid(4096), stream=stream0)
        buf713 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf712, buf713, 1024, 4, grid=grid(1024), stream=stream0)
        buf718 = buf692; del buf692  # reuse
        buf719 = reinterpret_tensor(buf699, (1, 512, 1024), (524288, 1024, 1), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf718, buf702, buf706, buf710, primals_116, mul_50, div_102, getitem_71, buf719, 512, 1024, grid=grid(512), stream=stream0)
        del div_102
        del getitem_71
        del primals_116
        buf716 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf717 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf702, buf706, buf710, mul_50, buf716, buf717, 1024, 512, grid=grid(1024), stream=stream0)
        del buf702
        del buf706
        del mul_50
        buf720 = reinterpret_tensor(buf683, (512, 4096), (4096, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (512, 1024), (1024, 1), 0), permute_835, out=buf720)
        del permute_835
        buf721 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (1024, 512), (1, 1024), 0), view_152, out=buf721)
        del view_152
        buf722 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf719, buf722, 4096, 128, grid=grid(4096), stream=stream0)
        buf723 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf722, buf723, 1024, 4, grid=grid(1024), stream=stream0)
        buf724 = reinterpret_tensor(buf720, (1, 512, 4096), (2097152, 4096, 1), 0); del buf720  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf724, addmm_40, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_40
        buf725 = reinterpret_tensor(buf719, (512, 1024), (1024, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (512, 4096), (4096, 1), 0), permute_839, out=buf725)
        del permute_839
        buf726 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (4096, 512), (1, 4096), 0), view_150, out=buf726)
        del view_150
        buf727 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf724, buf727, 16384, 128, grid=grid(16384), stream=stream0)
        buf728 = reinterpret_tensor(buf722, (1, 4096), (4096, 1), 0); del buf722  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf727, buf728, 4096, 4, grid=grid(4096), stream=stream0)
        buf733 = buf718; del buf718  # reuse
        buf734 = reinterpret_tensor(buf710, (1, 512, 1024), (524288, 1024, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf733, buf725, primals_110, mul_45, div_103, getitem_67, buf734, 512, 1024, grid=grid(512), stream=stream0)
        del div_103
        del getitem_67
        del primals_110
        buf731 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf732 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf725, mul_45, buf731, buf732, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_45
        buf735 = buf725; del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf734, (512, 1024), (1024, 1), 0), permute_843, out=buf735)
        del permute_843
        buf736 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf734, (1024, 512), (1, 1024), 0), view_148, out=buf736)
        del view_148
        buf737 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf734, buf737, 4096, 128, grid=grid(4096), stream=stream0)
        del buf734
        buf738 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf737, buf738, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf739 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf735, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_51, clone_default_52, clone_default_53, None, alias_default_35, getitem_366, getitem_367, getitem_368, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_35
        del clone_default_51
        del clone_default_52
        del clone_default_53
        del getitem_366
        del getitem_367
        del getitem_368
        buf740 = buf739[0]
        buf741 = buf739[1]
        buf742 = buf739[2]
        del buf739
        buf743 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (512, 1024), (1024, 1), 0), permute_855, out=buf743)
        del permute_855
        buf744 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (1024, 512), (1, 1024), 0), view_132, out=buf744)
        buf745 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf742, buf745, 4096, 128, grid=grid(4096), stream=stream0)
        buf746 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf745, buf746, 1024, 4, grid=grid(1024), stream=stream0)
        buf747 = reinterpret_tensor(buf742, (512, 1024), (1024, 1), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (512, 1024), (1024, 1), 0), permute_860, out=buf747)
        del permute_860
        buf748 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (1024, 512), (1, 1024), 0), view_132, out=buf748)
        buf749 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf741, buf749, 4096, 128, grid=grid(4096), stream=stream0)
        buf750 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf749, buf750, 1024, 4, grid=grid(1024), stream=stream0)
        buf751 = reinterpret_tensor(buf741, (512, 1024), (1024, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf740, (512, 1024), (1024, 1), 0), permute_864, out=buf751)
        del permute_864
        buf752 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf740, (1024, 512), (1, 1024), 0), view_132, out=buf752)
        del view_132
        buf753 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf740, buf753, 4096, 128, grid=grid(4096), stream=stream0)
        buf754 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf753, buf754, 1024, 4, grid=grid(1024), stream=stream0)
        buf759 = buf733; del buf733  # reuse
        buf760 = reinterpret_tensor(buf740, (1, 512, 1024), (524288, 1024, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf759, buf743, buf747, buf751, primals_100, mul_43, div_105, getitem_61, buf760, 512, 1024, grid=grid(512), stream=stream0)
        del div_105
        del getitem_61
        del primals_100
        buf757 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf758 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf743, buf747, buf751, mul_43, buf757, buf758, 1024, 512, grid=grid(1024), stream=stream0)
        del buf743
        del buf747
        del mul_43
        buf761 = reinterpret_tensor(buf724, (512, 4096), (4096, 1), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (512, 1024), (1024, 1), 0), permute_868, out=buf761)
        del permute_868
        buf762 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (1024, 512), (1, 1024), 0), view_130, out=buf762)
        del view_130
        buf763 = buf753; del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf760, buf763, 4096, 128, grid=grid(4096), stream=stream0)
        buf764 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf763, buf764, 1024, 4, grid=grid(1024), stream=stream0)
        buf765 = reinterpret_tensor(buf761, (1, 512, 4096), (2097152, 4096, 1), 0); del buf761  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf765, addmm_34, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_34
        buf766 = reinterpret_tensor(buf760, (512, 1024), (1024, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (512, 4096), (4096, 1), 0), permute_872, out=buf766)
        del permute_872
        buf767 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (4096, 512), (1, 4096), 0), view_128, out=buf767)
        del view_128
        buf768 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf765, buf768, 16384, 128, grid=grid(16384), stream=stream0)
        buf769 = reinterpret_tensor(buf763, (1, 4096), (4096, 1), 0); del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf768, buf769, 4096, 4, grid=grid(4096), stream=stream0)
        buf774 = buf759; del buf759  # reuse
        buf775 = reinterpret_tensor(buf751, (1, 512, 1024), (524288, 1024, 1), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf774, buf766, primals_94, mul_38, div_106, getitem_57, buf775, 512, 1024, grid=grid(512), stream=stream0)
        del div_106
        del getitem_57
        del primals_94
        buf772 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf773 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf766, mul_38, buf772, buf773, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_38
        buf776 = buf766; del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf775, (512, 1024), (1024, 1), 0), permute_876, out=buf776)
        del permute_876
        buf777 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf775, (1024, 512), (1, 1024), 0), view_126, out=buf777)
        del view_126
        buf778 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf775, buf778, 4096, 128, grid=grid(4096), stream=stream0)
        del buf775
        buf779 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf778, buf779, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf780 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf776, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_54, clone_default_55, clone_default_56, None, alias_default_37, getitem_373, getitem_374, getitem_375, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_37
        del clone_default_54
        del clone_default_55
        del clone_default_56
        del getitem_373
        del getitem_374
        del getitem_375
        buf781 = buf780[0]
        buf782 = buf780[1]
        buf783 = buf780[2]
        del buf780
        buf784 = buf776; del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf783, (512, 1024), (1024, 1), 0), permute_888, out=buf784)
        del permute_888
        buf785 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf783, (1024, 512), (1, 1024), 0), view_110, out=buf785)
        buf786 = buf778; del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf783, buf786, 4096, 128, grid=grid(4096), stream=stream0)
        buf787 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf786, buf787, 1024, 4, grid=grid(1024), stream=stream0)
        buf788 = reinterpret_tensor(buf783, (512, 1024), (1024, 1), 0); del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf782, (512, 1024), (1024, 1), 0), permute_893, out=buf788)
        del permute_893
        buf789 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf782, (1024, 512), (1, 1024), 0), view_110, out=buf789)
        buf790 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf782, buf790, 4096, 128, grid=grid(4096), stream=stream0)
        buf791 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf790, buf791, 1024, 4, grid=grid(1024), stream=stream0)
        buf792 = reinterpret_tensor(buf782, (512, 1024), (1024, 1), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (512, 1024), (1024, 1), 0), permute_897, out=buf792)
        del permute_897
        buf793 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (1024, 512), (1, 1024), 0), view_110, out=buf793)
        del view_110
        buf794 = buf790; del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf781, buf794, 4096, 128, grid=grid(4096), stream=stream0)
        buf795 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf794, buf795, 1024, 4, grid=grid(1024), stream=stream0)
        buf800 = buf774; del buf774  # reuse
        buf801 = reinterpret_tensor(buf781, (1, 512, 1024), (524288, 1024, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf800, buf784, buf788, buf792, primals_84, mul_36, div_108, getitem_51, buf801, 512, 1024, grid=grid(512), stream=stream0)
        del div_108
        del getitem_51
        del primals_84
        buf798 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf799 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf784, buf788, buf792, mul_36, buf798, buf799, 1024, 512, grid=grid(1024), stream=stream0)
        del buf784
        del buf788
        del mul_36
        buf802 = reinterpret_tensor(buf765, (512, 4096), (4096, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (512, 1024), (1024, 1), 0), permute_901, out=buf802)
        del permute_901
        buf803 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (1024, 512), (1, 1024), 0), view_108, out=buf803)
        del view_108
        buf804 = buf794; del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf801, buf804, 4096, 128, grid=grid(4096), stream=stream0)
        buf805 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf804, buf805, 1024, 4, grid=grid(1024), stream=stream0)
        buf806 = reinterpret_tensor(buf802, (1, 512, 4096), (2097152, 4096, 1), 0); del buf802  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf806, addmm_28, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_28
        buf807 = reinterpret_tensor(buf801, (512, 1024), (1024, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (512, 4096), (4096, 1), 0), permute_905, out=buf807)
        del permute_905
        buf808 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (4096, 512), (1, 4096), 0), view_106, out=buf808)
        del view_106
        buf809 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf806, buf809, 16384, 128, grid=grid(16384), stream=stream0)
        buf810 = reinterpret_tensor(buf804, (1, 4096), (4096, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf809, buf810, 4096, 4, grid=grid(4096), stream=stream0)
        buf815 = buf800; del buf800  # reuse
        buf816 = reinterpret_tensor(buf792, (1, 512, 1024), (524288, 1024, 1), 0); del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf815, buf807, primals_78, mul_31, div_109, getitem_47, buf816, 512, 1024, grid=grid(512), stream=stream0)
        del div_109
        del getitem_47
        del primals_78
        buf813 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf814 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf807, mul_31, buf813, buf814, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_31
        buf817 = buf807; del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (512, 1024), (1024, 1), 0), permute_909, out=buf817)
        del permute_909
        buf818 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (1024, 512), (1, 1024), 0), view_104, out=buf818)
        del view_104
        buf819 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf816, buf819, 4096, 128, grid=grid(4096), stream=stream0)
        del buf816
        buf820 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf819, buf820, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf821 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf817, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_57, clone_default_58, clone_default_59, None, alias_default_39, getitem_380, getitem_381, getitem_382, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_39
        del clone_default_57
        del clone_default_58
        del clone_default_59
        del getitem_380
        del getitem_381
        del getitem_382
        buf822 = buf821[0]
        buf823 = buf821[1]
        buf824 = buf821[2]
        del buf821
        buf825 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (512, 1024), (1024, 1), 0), permute_921, out=buf825)
        del permute_921
        buf826 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (1024, 512), (1, 1024), 0), view_88, out=buf826)
        buf827 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf824, buf827, 4096, 128, grid=grid(4096), stream=stream0)
        buf828 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf827, buf828, 1024, 4, grid=grid(1024), stream=stream0)
        buf829 = reinterpret_tensor(buf824, (512, 1024), (1024, 1), 0); del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (512, 1024), (1024, 1), 0), permute_926, out=buf829)
        del permute_926
        buf830 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (1024, 512), (1, 1024), 0), view_88, out=buf830)
        buf831 = buf827; del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf823, buf831, 4096, 128, grid=grid(4096), stream=stream0)
        buf832 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf831, buf832, 1024, 4, grid=grid(1024), stream=stream0)
        buf833 = reinterpret_tensor(buf823, (512, 1024), (1024, 1), 0); del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf822, (512, 1024), (1024, 1), 0), permute_930, out=buf833)
        del permute_930
        buf834 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf822, (1024, 512), (1, 1024), 0), view_88, out=buf834)
        del view_88
        buf835 = buf831; del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf822, buf835, 4096, 128, grid=grid(4096), stream=stream0)
        buf836 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf835, buf836, 1024, 4, grid=grid(1024), stream=stream0)
        buf841 = buf815; del buf815  # reuse
        buf842 = reinterpret_tensor(buf822, (1, 512, 1024), (524288, 1024, 1), 0); del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf841, buf825, buf829, buf833, primals_68, mul_29, div_111, getitem_41, buf842, 512, 1024, grid=grid(512), stream=stream0)
        del div_111
        del getitem_41
        del primals_68
        buf839 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf840 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf825, buf829, buf833, mul_29, buf839, buf840, 1024, 512, grid=grid(1024), stream=stream0)
        del buf825
        del buf829
        del mul_29
        buf843 = reinterpret_tensor(buf806, (512, 4096), (4096, 1), 0); del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (512, 1024), (1024, 1), 0), permute_934, out=buf843)
        del permute_934
        buf844 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (1024, 512), (1, 1024), 0), view_86, out=buf844)
        del view_86
        buf845 = buf835; del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf842, buf845, 4096, 128, grid=grid(4096), stream=stream0)
        buf846 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf845, buf846, 1024, 4, grid=grid(1024), stream=stream0)
        buf847 = reinterpret_tensor(buf843, (1, 512, 4096), (2097152, 4096, 1), 0); del buf843  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf847, addmm_22, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_22
        buf848 = reinterpret_tensor(buf842, (512, 1024), (1024, 1), 0); del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (512, 4096), (4096, 1), 0), permute_938, out=buf848)
        del permute_938
        buf849 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (4096, 512), (1, 4096), 0), view_84, out=buf849)
        del view_84
        buf850 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf847, buf850, 16384, 128, grid=grid(16384), stream=stream0)
        buf851 = reinterpret_tensor(buf845, (1, 4096), (4096, 1), 0); del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf850, buf851, 4096, 4, grid=grid(4096), stream=stream0)
        buf856 = buf841; del buf841  # reuse
        buf857 = reinterpret_tensor(buf833, (1, 512, 1024), (524288, 1024, 1), 0); del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf856, buf848, primals_62, mul_24, div_112, getitem_37, buf857, 512, 1024, grid=grid(512), stream=stream0)
        del div_112
        del getitem_37
        del primals_62
        buf854 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf855 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf848, mul_24, buf854, buf855, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_24
        buf858 = buf848; del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf857, (512, 1024), (1024, 1), 0), permute_942, out=buf858)
        del permute_942
        buf859 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf857, (1024, 512), (1, 1024), 0), view_82, out=buf859)
        del view_82
        buf860 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf857, buf860, 4096, 128, grid=grid(4096), stream=stream0)
        del buf857
        buf861 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf860, buf861, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf862 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf858, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_60, clone_default_61, clone_default_62, None, alias_default_41, getitem_387, getitem_388, getitem_389, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_41
        del clone_default_60
        del clone_default_61
        del clone_default_62
        del getitem_387
        del getitem_388
        del getitem_389
        buf863 = buf862[0]
        buf864 = buf862[1]
        buf865 = buf862[2]
        del buf862
        buf866 = buf858; del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (512, 1024), (1024, 1), 0), permute_954, out=buf866)
        del permute_954
        buf867 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (1024, 512), (1, 1024), 0), view_66, out=buf867)
        buf868 = buf860; del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf865, buf868, 4096, 128, grid=grid(4096), stream=stream0)
        buf869 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf868, buf869, 1024, 4, grid=grid(1024), stream=stream0)
        buf870 = reinterpret_tensor(buf865, (512, 1024), (1024, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf864, (512, 1024), (1024, 1), 0), permute_959, out=buf870)
        del permute_959
        buf871 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf864, (1024, 512), (1, 1024), 0), view_66, out=buf871)
        buf872 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf864, buf872, 4096, 128, grid=grid(4096), stream=stream0)
        buf873 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf872, buf873, 1024, 4, grid=grid(1024), stream=stream0)
        buf874 = reinterpret_tensor(buf864, (512, 1024), (1024, 1), 0); del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (512, 1024), (1024, 1), 0), permute_963, out=buf874)
        del permute_963
        buf875 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (1024, 512), (1, 1024), 0), view_66, out=buf875)
        del view_66
        buf876 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf863, buf876, 4096, 128, grid=grid(4096), stream=stream0)
        buf877 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf876, buf877, 1024, 4, grid=grid(1024), stream=stream0)
        buf882 = buf856; del buf856  # reuse
        buf883 = reinterpret_tensor(buf863, (1, 512, 1024), (524288, 1024, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf882, buf866, buf870, buf874, primals_52, mul_22, div_114, getitem_31, buf883, 512, 1024, grid=grid(512), stream=stream0)
        del div_114
        del getitem_31
        del primals_52
        buf880 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf881 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf866, buf870, buf874, mul_22, buf880, buf881, 1024, 512, grid=grid(1024), stream=stream0)
        del buf866
        del buf870
        del mul_22
        buf884 = reinterpret_tensor(buf847, (512, 4096), (4096, 1), 0); del buf847  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf883, (512, 1024), (1024, 1), 0), permute_967, out=buf884)
        del permute_967
        buf885 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf883, (1024, 512), (1, 1024), 0), view_64, out=buf885)
        del view_64
        buf886 = buf876; del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf883, buf886, 4096, 128, grid=grid(4096), stream=stream0)
        buf887 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf886, buf887, 1024, 4, grid=grid(1024), stream=stream0)
        buf888 = reinterpret_tensor(buf884, (1, 512, 4096), (2097152, 4096, 1), 0); del buf884  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf888, addmm_16, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_16
        buf889 = reinterpret_tensor(buf883, (512, 1024), (1024, 1), 0); del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf888, (512, 4096), (4096, 1), 0), permute_971, out=buf889)
        del permute_971
        buf890 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf888, (4096, 512), (1, 4096), 0), view_62, out=buf890)
        del view_62
        buf891 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf888, buf891, 16384, 128, grid=grid(16384), stream=stream0)
        buf892 = reinterpret_tensor(buf886, (1, 4096), (4096, 1), 0); del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf891, buf892, 4096, 4, grid=grid(4096), stream=stream0)
        buf897 = buf882; del buf882  # reuse
        buf898 = reinterpret_tensor(buf874, (1, 512, 1024), (524288, 1024, 1), 0); del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf897, buf889, primals_46, mul_17, div_115, getitem_27, buf898, 512, 1024, grid=grid(512), stream=stream0)
        del div_115
        del getitem_27
        del primals_46
        buf895 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf896 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf889, mul_17, buf895, buf896, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_17
        buf899 = buf889; del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf898, (512, 1024), (1024, 1), 0), permute_975, out=buf899)
        del permute_975
        buf900 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf898, (1024, 512), (1, 1024), 0), view_60, out=buf900)
        del view_60
        buf901 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf898, buf901, 4096, 128, grid=grid(4096), stream=stream0)
        del buf898
        buf902 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf901, buf902, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf903 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf899, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_63, clone_default_64, clone_default_65, None, alias_default_43, getitem_394, getitem_395, getitem_396, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_43
        del clone_default_63
        del clone_default_64
        del clone_default_65
        del getitem_394
        del getitem_395
        del getitem_396
        buf904 = buf903[0]
        buf905 = buf903[1]
        buf906 = buf903[2]
        del buf903
        buf907 = buf899; del buf899  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (512, 1024), (1024, 1), 0), permute_987, out=buf907)
        del permute_987
        buf908 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (1024, 512), (1, 1024), 0), view_44, out=buf908)
        buf909 = buf901; del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf906, buf909, 4096, 128, grid=grid(4096), stream=stream0)
        buf910 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf909, buf910, 1024, 4, grid=grid(1024), stream=stream0)
        buf911 = reinterpret_tensor(buf906, (512, 1024), (1024, 1), 0); del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf905, (512, 1024), (1024, 1), 0), permute_992, out=buf911)
        del permute_992
        buf912 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf905, (1024, 512), (1, 1024), 0), view_44, out=buf912)
        buf913 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf905, buf913, 4096, 128, grid=grid(4096), stream=stream0)
        buf914 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf913, buf914, 1024, 4, grid=grid(1024), stream=stream0)
        buf915 = reinterpret_tensor(buf905, (512, 1024), (1024, 1), 0); del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (512, 1024), (1024, 1), 0), permute_996, out=buf915)
        del permute_996
        buf916 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (1024, 512), (1, 1024), 0), view_44, out=buf916)
        del view_44
        buf917 = buf913; del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf904, buf917, 4096, 128, grid=grid(4096), stream=stream0)
        buf918 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf917, buf918, 1024, 4, grid=grid(1024), stream=stream0)
        buf923 = buf897; del buf897  # reuse
        buf924 = reinterpret_tensor(buf904, (1, 512, 1024), (524288, 1024, 1), 0); del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf923, buf907, buf911, buf915, primals_36, mul_15, div_117, getitem_21, buf924, 512, 1024, grid=grid(512), stream=stream0)
        del div_117
        del getitem_21
        del primals_36
        buf921 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf922 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf907, buf911, buf915, mul_15, buf921, buf922, 1024, 512, grid=grid(1024), stream=stream0)
        del buf907
        del buf911
        del mul_15
        buf925 = reinterpret_tensor(buf888, (512, 4096), (4096, 1), 0); del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf924, (512, 1024), (1024, 1), 0), permute_1000, out=buf925)
        del permute_1000
        buf926 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf924, (1024, 512), (1, 1024), 0), view_42, out=buf926)
        del view_42
        buf927 = buf917; del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf924, buf927, 4096, 128, grid=grid(4096), stream=stream0)
        buf928 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf927, buf928, 1024, 4, grid=grid(1024), stream=stream0)
        buf929 = reinterpret_tensor(buf925, (1, 512, 4096), (2097152, 4096, 1), 0); del buf925  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf929, addmm_10, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_10
        buf930 = reinterpret_tensor(buf924, (512, 1024), (1024, 1), 0); del buf924  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf929, (512, 4096), (4096, 1), 0), permute_1004, out=buf930)
        del permute_1004
        buf931 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf929, (4096, 512), (1, 4096), 0), view_40, out=buf931)
        del view_40
        buf932 = buf891; del buf891  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf929, buf932, 16384, 128, grid=grid(16384), stream=stream0)
        buf933 = reinterpret_tensor(buf927, (1, 4096), (4096, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf932, buf933, 4096, 4, grid=grid(4096), stream=stream0)
        buf938 = buf923; del buf923  # reuse
        buf939 = reinterpret_tensor(buf915, (1, 512, 1024), (524288, 1024, 1), 0); del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf938, buf930, primals_30, mul_10, div_118, getitem_17, buf939, 512, 1024, grid=grid(512), stream=stream0)
        del div_118
        del getitem_17
        del primals_30
        buf936 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf937 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf930, mul_10, buf936, buf937, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_10
        buf940 = buf930; del buf930  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf939, (512, 1024), (1024, 1), 0), permute_1008, out=buf940)
        del permute_1008
        buf941 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf939, (1024, 512), (1, 1024), 0), view_38, out=buf941)
        del view_38
        buf942 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf939, buf942, 4096, 128, grid=grid(4096), stream=stream0)
        del buf939
        buf943 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf942, buf943, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf944 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf940, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_66, clone_default_67, clone_default_68, None, alias_default_45, getitem_401, getitem_402, getitem_403, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_45
        del clone_default_66
        del clone_default_67
        del clone_default_68
        del getitem_401
        del getitem_402
        del getitem_403
        buf945 = buf944[0]
        buf946 = buf944[1]
        buf947 = buf944[2]
        del buf944
        buf948 = buf940; del buf940  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf947, (512, 1024), (1024, 1), 0), permute_1020, out=buf948)
        del permute_1020
        buf949 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf947, (1024, 512), (1, 1024), 0), view_22, out=buf949)
        buf950 = buf942; del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf947, buf950, 4096, 128, grid=grid(4096), stream=stream0)
        buf951 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf950, buf951, 1024, 4, grid=grid(1024), stream=stream0)
        buf952 = reinterpret_tensor(buf947, (512, 1024), (1024, 1), 0); del buf947  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf946, (512, 1024), (1024, 1), 0), permute_1025, out=buf952)
        del permute_1025
        buf953 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf946, (1024, 512), (1, 1024), 0), view_22, out=buf953)
        buf954 = buf950; del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf946, buf954, 4096, 128, grid=grid(4096), stream=stream0)
        buf955 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf954, buf955, 1024, 4, grid=grid(1024), stream=stream0)
        buf956 = reinterpret_tensor(buf946, (512, 1024), (1024, 1), 0); del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (512, 1024), (1024, 1), 0), permute_1029, out=buf956)
        del permute_1029
        buf957 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (1024, 512), (1, 1024), 0), view_22, out=buf957)
        del view_22
        buf958 = buf954; del buf954  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf945, buf958, 4096, 128, grid=grid(4096), stream=stream0)
        buf959 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf958, buf959, 1024, 4, grid=grid(1024), stream=stream0)
        buf964 = buf938; del buf938  # reuse
        buf965 = reinterpret_tensor(buf945, (1, 512, 1024), (524288, 1024, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf964, buf948, buf952, buf956, primals_20, mul_8, div_120, getitem_11, buf965, 512, 1024, grid=grid(512), stream=stream0)
        del div_120
        del getitem_11
        del primals_20
        buf962 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf963 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf948, buf952, buf956, mul_8, buf962, buf963, 1024, 512, grid=grid(1024), stream=stream0)
        del buf948
        del mul_8
        buf966 = reinterpret_tensor(buf929, (512, 4096), (4096, 1), 0); del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf965, (512, 1024), (1024, 1), 0), permute_1033, out=buf966)
        del permute_1033
        buf967 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf965, (1024, 512), (1, 1024), 0), view_20, out=buf967)
        del view_20
        buf968 = buf958; del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf965, buf968, 4096, 128, grid=grid(4096), stream=stream0)
        buf969 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf968, buf969, 1024, 4, grid=grid(1024), stream=stream0)
        buf970 = reinterpret_tensor(buf966, (1, 512, 4096), (2097152, 4096, 1), 0); del buf966  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf970, addmm_4, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_4
        buf971 = reinterpret_tensor(buf965, (512, 1024), (1024, 1), 0); del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf970, (512, 4096), (4096, 1), 0), permute_1037, out=buf971)
        del permute_1037
        buf972 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf970, (4096, 512), (1, 4096), 0), view_18, out=buf972)
        del view_18
        buf973 = buf932; del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf970, buf973, 16384, 128, grid=grid(16384), stream=stream0)
        del buf970
        buf974 = reinterpret_tensor(buf968, (1, 4096), (4096, 1), 0); del buf968  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf973, buf974, 4096, 4, grid=grid(4096), stream=stream0)
        del buf973
        buf979 = buf964; del buf964  # reuse
        buf980 = reinterpret_tensor(buf956, (1, 512, 1024), (524288, 1024, 1), 0); del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf979, buf971, primals_14, mul_3, div_121, getitem_7, buf980, 512, 1024, grid=grid(512), stream=stream0)
        del div_121
        del getitem_7
        del primals_14
        buf977 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf978 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf971, mul_3, buf977, buf978, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_3
        buf981 = buf971; del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (512, 1024), (1024, 1), 0), permute_1041, out=buf981)
        del permute_1041
        buf982 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (1024, 512), (1, 1024), 0), view_16, out=buf982)
        del view_16
        buf983 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf980, buf983, 4096, 128, grid=grid(4096), stream=stream0)
        buf984 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf983, buf984, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf985 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf981, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_69, clone_default_70, clone_default_71, None, alias_default_47, getitem_408, getitem_409, getitem_410, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_47
        del clone_default_69
        del clone_default_70
        del clone_default_71
        del getitem_408
        del getitem_409
        del getitem_410
        buf986 = buf985[0]
        buf987 = buf985[1]
        buf988 = buf985[2]
        del buf985
        buf989 = buf981; del buf981  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf988, (512, 1024), (1024, 1), 0), permute_1053, out=buf989)
        del permute_1053
        buf990 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf988, (1024, 512), (1, 1024), 0), view, out=buf990)
        buf991 = buf983; del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf988, buf991, 4096, 128, grid=grid(4096), stream=stream0)
        buf992 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf991, buf992, 1024, 4, grid=grid(1024), stream=stream0)
        buf993 = reinterpret_tensor(buf988, (512, 1024), (1024, 1), 0); del buf988  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf987, (512, 1024), (1024, 1), 0), permute_1058, out=buf993)
        del permute_1058
        buf994 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf987, (1024, 512), (1, 1024), 0), view, out=buf994)
        buf995 = buf991; del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf987, buf995, 4096, 128, grid=grid(4096), stream=stream0)
        buf996 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf995, buf996, 1024, 4, grid=grid(1024), stream=stream0)
        buf997 = reinterpret_tensor(buf987, (512, 1024), (1024, 1), 0); del buf987  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf986, (512, 1024), (1024, 1), 0), permute_1062, out=buf997)
        del permute_1062
        buf998 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf986, (1024, 512), (1, 1024), 0), view, out=buf998)
        del view
        buf999 = buf995; del buf995  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf986, buf999, 4096, 128, grid=grid(4096), stream=stream0)
        buf1000 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf999, buf1000, 1024, 4, grid=grid(1024), stream=stream0)
        del buf999
        buf1005 = buf979; del buf979  # reuse
        buf1007 = reinterpret_tensor(buf986, (1, 512, 1024), (524288, 1024, 1), 0); del buf986  # reuse
        buf1011 = buf980; del buf980  # reuse
        buf1015 = reinterpret_tensor(buf952, (1, 512, 1024), (524288, 1024, 1), 0); del buf952  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15.run(buf1005, buf989, buf993, buf997, primals_4, mul_1, div_123, slice_3, getitem_1, primals_398, buf1007, buf1011, buf1015, 512, 1024, grid=grid(512), stream=stream0)
        del buf1005
        del div_123
        del getitem_1
        del primals_4
        buf1003 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1004 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf989, buf993, buf997, mul_1, buf1003, buf1004, 1024, 512, grid=grid(1024), stream=stream0)
        del buf989
        del buf993
        del mul_1
        buf1006 = buf997; del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf1006, 524288, grid=grid(524288), stream=stream0)
        aten.index_put_(buf1006, [slice_3], buf1007, True)
        del buf1007
        del slice_3
        buf1010 = empty((2, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf1010, 2048, grid=grid(2048), stream=stream0)
        aten.index_put_(buf1010, [full_default], buf1011, True)
        del buf1011
        del full_default
        buf1014 = empty((29056, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf1014, 29753344, grid=grid(29753344), stream=stream0)
        aten.index_put_(buf1014, [primals_398], buf1015, True)
        del buf1015
        del primals_398
        return (buf1014, buf1010, buf1006, buf1003, buf1004, reinterpret_tensor(buf998, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf1000, (1024, ), (1, ), 0), reinterpret_tensor(buf994, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf996, (1024, ), (1, ), 0), reinterpret_tensor(buf990, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf992, (1024, ), (1, ), 0), reinterpret_tensor(buf982, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf984, (1024, ), (1, ), 0), buf977, buf978, reinterpret_tensor(buf972, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf974, (4096, ), (1, ), 0), reinterpret_tensor(buf967, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf969, (1024, ), (1, ), 0), buf962, buf963, reinterpret_tensor(buf957, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf959, (1024, ), (1, ), 0), reinterpret_tensor(buf953, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf955, (1024, ), (1, ), 0), reinterpret_tensor(buf949, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf951, (1024, ), (1, ), 0), reinterpret_tensor(buf941, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf943, (1024, ), (1, ), 0), buf936, buf937, reinterpret_tensor(buf931, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf933, (4096, ), (1, ), 0), reinterpret_tensor(buf926, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf928, (1024, ), (1, ), 0), buf921, buf922, reinterpret_tensor(buf916, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf918, (1024, ), (1, ), 0), reinterpret_tensor(buf912, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf914, (1024, ), (1, ), 0), reinterpret_tensor(buf908, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf910, (1024, ), (1, ), 0), reinterpret_tensor(buf900, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf902, (1024, ), (1, ), 0), buf895, buf896, reinterpret_tensor(buf890, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf892, (4096, ), (1, ), 0), reinterpret_tensor(buf885, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf887, (1024, ), (1, ), 0), buf880, buf881, reinterpret_tensor(buf875, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf877, (1024, ), (1, ), 0), reinterpret_tensor(buf871, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf873, (1024, ), (1, ), 0), reinterpret_tensor(buf867, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf869, (1024, ), (1, ), 0), reinterpret_tensor(buf859, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf861, (1024, ), (1, ), 0), buf854, buf855, reinterpret_tensor(buf849, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf851, (4096, ), (1, ), 0), reinterpret_tensor(buf844, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf846, (1024, ), (1, ), 0), buf839, buf840, reinterpret_tensor(buf834, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf836, (1024, ), (1, ), 0), reinterpret_tensor(buf830, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf832, (1024, ), (1, ), 0), reinterpret_tensor(buf826, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf828, (1024, ), (1, ), 0), reinterpret_tensor(buf818, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf820, (1024, ), (1, ), 0), buf813, buf814, reinterpret_tensor(buf808, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf810, (4096, ), (1, ), 0), reinterpret_tensor(buf803, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf805, (1024, ), (1, ), 0), buf798, buf799, reinterpret_tensor(buf793, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf795, (1024, ), (1, ), 0), reinterpret_tensor(buf789, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf791, (1024, ), (1, ), 0), reinterpret_tensor(buf785, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf787, (1024, ), (1, ), 0), reinterpret_tensor(buf777, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf779, (1024, ), (1, ), 0), buf772, buf773, reinterpret_tensor(buf767, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf769, (4096, ), (1, ), 0), reinterpret_tensor(buf762, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf764, (1024, ), (1, ), 0), buf757, buf758, reinterpret_tensor(buf752, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf754, (1024, ), (1, ), 0), reinterpret_tensor(buf748, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf750, (1024, ), (1, ), 0), reinterpret_tensor(buf744, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf746, (1024, ), (1, ), 0), reinterpret_tensor(buf736, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf738, (1024, ), (1, ), 0), buf731, buf732, reinterpret_tensor(buf726, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf728, (4096, ), (1, ), 0), reinterpret_tensor(buf721, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf723, (1024, ), (1, ), 0), buf716, buf717, reinterpret_tensor(buf711, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf713, (1024, ), (1, ), 0), reinterpret_tensor(buf707, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf709, (1024, ), (1, ), 0), reinterpret_tensor(buf703, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf705, (1024, ), (1, ), 0), reinterpret_tensor(buf695, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf697, (1024, ), (1, ), 0), buf690, buf691, reinterpret_tensor(buf685, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf687, (4096, ), (1, ), 0), reinterpret_tensor(buf680, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf682, (1024, ), (1, ), 0), buf675, buf676, reinterpret_tensor(buf670, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf672, (1024, ), (1, ), 0), reinterpret_tensor(buf666, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf668, (1024, ), (1, ), 0), reinterpret_tensor(buf662, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf664, (1024, ), (1, ), 0), reinterpret_tensor(buf654, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf656, (1024, ), (1, ), 0), buf649, buf650, reinterpret_tensor(buf644, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf646, (4096, ), (1, ), 0), reinterpret_tensor(buf639, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf641, (1024, ), (1, ), 0), buf634, buf635, reinterpret_tensor(buf629, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf631, (1024, ), (1, ), 0), reinterpret_tensor(buf625, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf627, (1024, ), (1, ), 0), reinterpret_tensor(buf621, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf623, (1024, ), (1, ), 0), reinterpret_tensor(buf613, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf615, (1024, ), (1, ), 0), buf608, buf609, reinterpret_tensor(buf603, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf605, (4096, ), (1, ), 0), reinterpret_tensor(buf598, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf600, (1024, ), (1, ), 0), buf593, buf594, reinterpret_tensor(buf588, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf590, (1024, ), (1, ), 0), reinterpret_tensor(buf584, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf586, (1024, ), (1, ), 0), reinterpret_tensor(buf580, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf582, (1024, ), (1, ), 0), reinterpret_tensor(buf572, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf574, (1024, ), (1, ), 0), buf567, buf568, reinterpret_tensor(buf562, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf564, (4096, ), (1, ), 0), reinterpret_tensor(buf557, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf559, (1024, ), (1, ), 0), buf552, buf553, reinterpret_tensor(buf547, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf549, (1024, ), (1, ), 0), reinterpret_tensor(buf543, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf545, (1024, ), (1, ), 0), reinterpret_tensor(buf539, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf541, (1024, ), (1, ), 0), reinterpret_tensor(buf531, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf533, (1024, ), (1, ), 0), buf526, buf527, reinterpret_tensor(buf521, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf523, (4096, ), (1, ), 0), reinterpret_tensor(buf516, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf518, (1024, ), (1, ), 0), buf511, buf512, reinterpret_tensor(buf506, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf508, (1024, ), (1, ), 0), reinterpret_tensor(buf502, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf504, (1024, ), (1, ), 0), reinterpret_tensor(buf498, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf500, (1024, ), (1, ), 0), reinterpret_tensor(buf490, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf492, (1024, ), (1, ), 0), buf485, buf486, reinterpret_tensor(buf480, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf482, (4096, ), (1, ), 0), reinterpret_tensor(buf475, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf477, (1024, ), (1, ), 0), buf470, buf471, reinterpret_tensor(buf465, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf467, (1024, ), (1, ), 0), reinterpret_tensor(buf461, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf463, (1024, ), (1, ), 0), reinterpret_tensor(buf457, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf459, (1024, ), (1, ), 0), reinterpret_tensor(buf449, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf451, (1024, ), (1, ), 0), buf444, buf445, reinterpret_tensor(buf439, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf441, (4096, ), (1, ), 0), reinterpret_tensor(buf434, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf436, (1024, ), (1, ), 0), buf429, buf430, reinterpret_tensor(buf424, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf426, (1024, ), (1, ), 0), reinterpret_tensor(buf420, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf422, (1024, ), (1, ), 0), reinterpret_tensor(buf416, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf418, (1024, ), (1, ), 0), reinterpret_tensor(buf408, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf410, (1024, ), (1, ), 0), buf403, buf404, reinterpret_tensor(buf398, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf400, (4096, ), (1, ), 0), reinterpret_tensor(buf393, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf395, (1024, ), (1, ), 0), buf388, buf389, reinterpret_tensor(buf383, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf385, (1024, ), (1, ), 0), reinterpret_tensor(buf379, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf381, (1024, ), (1, ), 0), reinterpret_tensor(buf375, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf377, (1024, ), (1, ), 0), reinterpret_tensor(buf367, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf369, (1024, ), (1, ), 0), buf362, buf363, reinterpret_tensor(buf357, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf359, (4096, ), (1, ), 0), reinterpret_tensor(buf352, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf354, (1024, ), (1, ), 0), buf347, buf348, reinterpret_tensor(buf342, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf344, (1024, ), (1, ), 0), reinterpret_tensor(buf338, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf340, (1024, ), (1, ), 0), reinterpret_tensor(buf334, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf336, (1024, ), (1, ), 0), reinterpret_tensor(buf326, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf328, (1024, ), (1, ), 0), buf321, buf322, reinterpret_tensor(buf316, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf318, (4096, ), (1, ), 0), reinterpret_tensor(buf311, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf313, (1024, ), (1, ), 0), buf306, buf307, reinterpret_tensor(buf301, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf303, (1024, ), (1, ), 0), reinterpret_tensor(buf297, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf299, (1024, ), (1, ), 0), reinterpret_tensor(buf293, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf295, (1024, ), (1, ), 0), reinterpret_tensor(buf285, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf287, (1024, ), (1, ), 0), buf280, buf281, reinterpret_tensor(buf275, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf277, (4096, ), (1, ), 0), reinterpret_tensor(buf270, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf272, (1024, ), (1, ), 0), buf265, buf266, reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf262, (1024, ), (1, ), 0), reinterpret_tensor(buf256, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf258, (1024, ), (1, ), 0), reinterpret_tensor(buf252, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf254, (1024, ), (1, ), 0), reinterpret_tensor(buf244, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf246, (1024, ), (1, ), 0), buf239, buf240, reinterpret_tensor(buf234, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf236, (4096, ), (1, ), 0), reinterpret_tensor(buf229, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf231, (1024, ), (1, ), 0), buf224, buf225, reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf221, (1024, ), (1, ), 0), reinterpret_tensor(buf215, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf217, (1024, ), (1, ), 0), reinterpret_tensor(buf211, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf213, (1024, ), (1, ), 0), reinterpret_tensor(buf203, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf205, (1024, ), (1, ), 0), buf198, buf199, reinterpret_tensor(buf193, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf195, (4096, ), (1, ), 0), reinterpret_tensor(buf188, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf190, (1024, ), (1, ), 0), buf183, buf184, reinterpret_tensor(buf178, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf180, (1024, ), (1, ), 0), reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf176, (1024, ), (1, ), 0), reinterpret_tensor(buf170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf172, (1024, ), (1, ), 0), reinterpret_tensor(buf162, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf164, (1024, ), (1, ), 0), buf157, buf158, reinterpret_tensor(buf152, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf154, (4096, ), (1, ), 0), reinterpret_tensor(buf147, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf149, (1024, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf137, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf139, (1024, ), (1, ), 0), reinterpret_tensor(buf133, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf135, (1024, ), (1, ), 0), reinterpret_tensor(buf129, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf131, (1024, ), (1, ), 0), reinterpret_tensor(buf121, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf123, (1024, ), (1, ), 0), buf116, buf117, reinterpret_tensor(buf111, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf113, (4096, ), (1, ), 0), reinterpret_tensor(buf106, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf108, (1024, ), (1, ), 0), buf101, buf102, reinterpret_tensor(buf96, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf98, (1024, ), (1, ), 0), reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf94, (1024, ), (1, ), 0), reinterpret_tensor(buf88, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf90, (1024, ), (1, ), 0), reinterpret_tensor(buf80, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf82, (1024, ), (1, ), 0), buf75, buf76, reinterpret_tensor(buf70, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (4096, ), (1, ), 0), reinterpret_tensor(buf65, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf67, (1024, ), (1, ), 0), buf60, buf61, reinterpret_tensor(buf55, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf57, (1024, ), (1, ), 0), reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf53, (1024, ), (1, ), 0), reinterpret_tensor(buf47, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf49, (1024, ), (1, ), 0), reinterpret_tensor(buf39, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf41, (1024, ), (1, ), 0), buf34, buf35, reinterpret_tensor(buf29, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf31, (4096, ), (1, ), 0), reinterpret_tensor(buf24, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf26, (1024, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf14, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf16, (1024, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf6, (29056, 1024), (1024, 1), 0), reinterpret_tensor(buf7, (29056, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    full_default = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    getitem_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_69 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_70 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_71 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_408 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_47 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_66 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_67 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_68 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_401 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_45 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_63 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_64 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_65 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_394 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_43 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_60 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_61 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_62 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_387 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_41 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_57 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_58 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_59 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_380 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_39 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_54 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_55 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_56 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_373 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_37 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_51 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_52 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_53 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_366 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_35 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_48 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_49 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_50 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_359 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_33 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_45 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_46 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_47 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_352 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_31 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_42 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_43 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_44 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_345 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_29 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_39 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_40 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_41 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_338 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_27 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_36 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_37 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_38 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_331 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_25 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_324 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_76 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_284 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_92 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_286 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_317 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_137 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_94 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_304 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_82 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_306 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_99 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_310 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_324 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_147 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_326 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_88 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_328 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_106 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_330 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_303 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_108 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_348 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_94 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_350 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_113 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_352 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_296 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_368 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_115 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_370 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_100 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_372 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_120 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_374 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_122 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_106 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_181 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_396 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_282 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_412 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_187 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_129 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_112 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_416 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_134 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_418 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_275 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_136 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_436 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_118 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_438 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_201 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_440 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_268 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_456 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_207 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_143 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_458 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_124 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_460 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_148 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_462 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_261 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_478 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_217 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_150 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_480 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_130 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_482 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_221 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_155 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_484 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_254 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_500 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_227 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_502 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_136 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_504 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_231 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_162 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_506 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_247 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_522 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_237 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_164 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_524 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_142 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_526 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_241 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_169 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_528 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_144 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_530 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    sub_76 = rand_strided((511, 29056), (29056, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_2 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_266 = rand_strided((29056, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_538 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_542 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_546 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_575 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_608 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_633 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_637 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_641 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_645 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_666 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_670 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_674 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_678 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_695 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_699 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_703 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_707 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_711 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_723 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_728 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_732 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_736 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_744 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_756 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_761 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_96 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_769 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_773 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_97 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_777 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_789 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_794 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_99 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_802 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_806 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_100 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_810 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_827 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_831 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_102 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_835 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_839 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_103 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_843 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_855 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_860 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_864 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_105 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_868 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_872 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_106 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_876 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_888 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_893 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_897 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_108 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_901 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_905 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_109 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_909 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_921 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_926 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_930 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_111 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_934 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_938 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_112 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_942 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_954 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_959 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_963 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_114 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_967 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_971 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_115 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_975 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_987 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_992 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_996 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_117 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1000 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1004 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_118 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1008 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1020 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1025 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1029 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_120 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1033 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1037 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_121 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1041 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1053 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1058 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1062 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_123 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 29056), (14876672, 29056, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_392, primals_398, full_default, slice_3, getitem_1, mul_1, view, clone_default_69, clone_default_70, clone_default_71, getitem_408, getitem_409, getitem_410, alias_default_47, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_66, clone_default_67, clone_default_68, getitem_401, getitem_402, getitem_403, alias_default_45, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_63, clone_default_64, clone_default_65, getitem_394, getitem_395, getitem_396, alias_default_43, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_60, clone_default_61, clone_default_62, getitem_387, getitem_388, getitem_389, alias_default_41, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_57, clone_default_58, clone_default_59, getitem_380, getitem_381, getitem_382, alias_default_39, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_54, clone_default_55, clone_default_56, getitem_373, getitem_374, getitem_375, alias_default_37, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_51, clone_default_52, clone_default_53, getitem_366, getitem_367, getitem_368, alias_default_35, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_48, clone_default_49, clone_default_50, getitem_359, getitem_360, getitem_361, alias_default_33, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_45, clone_default_46, clone_default_47, getitem_352, getitem_353, getitem_354, alias_default_31, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_42, clone_default_43, clone_default_44, getitem_345, getitem_346, getitem_347, alias_default_29, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_39, clone_default_40, clone_default_41, getitem_338, getitem_339, getitem_340, alias_default_27, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default_36, clone_default_37, clone_default_38, getitem_331, getitem_332, getitem_333, alias_default_25, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, clone_default_33, clone_default_34, clone_default_35, getitem_324, getitem_325, getitem_326, alias_default_23, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, clone_default_30, clone_default_31, clone_default_32, getitem_317, getitem_318, getitem_319, alias_default_21, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, clone_default_27, clone_default_28, clone_default_29, getitem_310, getitem_311, getitem_312, alias_default_19, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, clone_default_24, clone_default_25, clone_default_26, getitem_303, getitem_304, getitem_305, alias_default_17, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, clone_default_21, clone_default_22, clone_default_23, getitem_296, getitem_297, getitem_298, alias_default_15, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, clone_default_18, clone_default_19, clone_default_20, getitem_289, getitem_290, getitem_291, alias_default_13, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, clone_default_15, clone_default_16, clone_default_17, getitem_282, getitem_283, getitem_284, alias_default_11, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, clone_default_12, clone_default_13, clone_default_14, getitem_275, getitem_276, getitem_277, alias_default_9, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, clone_default_9, clone_default_10, clone_default_11, getitem_268, getitem_269, getitem_270, alias_default_7, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, clone_default_6, clone_default_7, clone_default_8, getitem_261, getitem_262, getitem_263, alias_default_5, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, clone_default_3, clone_default_4, clone_default_5, getitem_254, getitem_255, getitem_256, alias_default_3, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, clone_default, clone_default_1, clone_default_2, getitem_247, getitem_248, getitem_249, alias_default_1, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, addmm_144, mul_174, view_530, sub_76, convert_element_type, ne_3, where_2, permute_266, div_50, permute_270, div_51, permute_274, permute_278, div_52, permute_282, permute_294, permute_299, permute_303, div_54, permute_307, permute_311, div_55, permute_315, permute_327, permute_332, permute_336, div_57, permute_340, permute_344, div_58, permute_348, permute_360, permute_365, permute_369, div_60, permute_373, permute_377, div_61, permute_381, permute_393, permute_398, permute_402, div_63, permute_406, permute_410, div_64, permute_414, permute_426, permute_431, permute_435, div_66, permute_439, permute_443, div_67, permute_447, permute_459, permute_464, permute_468, div_69, permute_472, permute_476, div_70, permute_480, permute_492, permute_497, permute_501, div_72, permute_505, permute_509, div_73, permute_513, permute_525, permute_530, permute_534, div_75, permute_538, permute_542, div_76, permute_546, permute_558, permute_563, permute_567, div_78, permute_571, permute_575, div_79, permute_579, permute_591, permute_596, permute_600, div_81, permute_604, permute_608, div_82, permute_612, permute_624, permute_629, permute_633, div_84, permute_637, permute_641, div_85, permute_645, permute_657, permute_662, permute_666, div_87, permute_670, permute_674, div_88, permute_678, permute_690, permute_695, permute_699, div_90, permute_703, permute_707, div_91, permute_711, permute_723, permute_728, permute_732, div_93, permute_736, permute_740, div_94, permute_744, permute_756, permute_761, permute_765, div_96, permute_769, permute_773, div_97, permute_777, permute_789, permute_794, permute_798, div_99, permute_802, permute_806, div_100, permute_810, permute_822, permute_827, permute_831, div_102, permute_835, permute_839, div_103, permute_843, permute_855, permute_860, permute_864, div_105, permute_868, permute_872, div_106, permute_876, permute_888, permute_893, permute_897, div_108, permute_901, permute_905, div_109, permute_909, permute_921, permute_926, permute_930, div_111, permute_934, permute_938, div_112, permute_942, permute_954, permute_959, permute_963, div_114, permute_967, permute_971, div_115, permute_975, permute_987, permute_992, permute_996, div_117, permute_1000, permute_1004, div_118, permute_1008, permute_1020, permute_1025, permute_1029, div_120, permute_1033, permute_1037, div_121, permute_1041, permute_1053, permute_1058, permute_1062, div_123, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForCausalLM', benchmark_compiled_module)
