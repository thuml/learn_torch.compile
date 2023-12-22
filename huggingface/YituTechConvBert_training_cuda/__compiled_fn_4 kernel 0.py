
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


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwnnhi722eitkgpqgvh4ncmwt7a4pl2gjyinid6neh3xxgoswfq.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_13
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
    xnumel = 15627264
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjivva35nrjxummuzug4vgfwcnmnn5qo5bkjhdvca6ghm43la35h.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_13
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/cx/ccx456psnzvqlklshdo7atsic3cgdag6uloqyhr4asvt3opoxbdn.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_14
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (30522*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4nubqp3xa4ostxpcquwfuojeatw5pne3r6zcwtgsnmckk6kput.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkpnwbrxhtprzbeskjsvcewrjkjck4lhud3yar2z5yhxiyfz623.py
# Source Nodes: [hidden_states_110], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# hidden_states_110 => add_148, erf_12, mul_100
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cuj5jipddg65hcmq25dutvuyerqmzocp5m7uy3uhaqqpiq5zmj6v.py
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
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5e32ob646uag7xx6bcm77p3cbdhfinmau2k2fzb6vsgphniet6.py
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
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnu4hetcmyv2laomty752vknskibjjygllxzp5ciznvl4mugsv5.py
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
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghs7wjo7pdu56ewpdwlexdkoygzpigcwmovu2b3hljsza7id5pb.py
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmig47we7pe6qi4m6fodzi7o7v6bw622yrvfq45hd42yuxtarlpf.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_144, erf_11, mul_95
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
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yqq7sfjgaaxdkh5o3kqam47b6ywf4s5xn4mipbwryklhixmpic.py
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
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6ughq2lkyzherih2mehrcz7rlcve3vxx3kazka7vrbbtqebhve.py
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
    xnumel = 3072
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ul/culdgxck7pkwghuuz27p4eono53pxybauflzgunao7v5tfq2fggg.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgq42whvhsx5nt2p6roncq4lq6d7l3nswd42piwxjdybpr4zjgds.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cryg2knrp4j4wryimmpveah3t5jf2pilgk3wkpouimo5qfetx5iz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (6*x1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (32768*y0)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb26bgmtquhabmmsrmjm7bemsvroq6np3gjnxdztefzi5xkcb6wa.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (768*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cavcisyl4jdyayubwdloflc7wuzen6y7kllrbs6aw7vgxhgrwsol.py
# Source Nodes: [], Original ATen: [aten.col2im]

triton_poi_fused_col2im_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 199680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhfvpkcjvrbqvkhaeu7mmio66fa5kybp4rjp46ob6lufvwunvcs.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_per_fused_sum_view_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_view_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 384
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
    tmp0 = 4 + r1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 520, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (4 + r1 + (520*x0)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygm56cedkkyzfavymktdanghdze6lt7kbdc6u2vyedicgojmpq4.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (9*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnvjwodja3kqji22h2dl22ivpz4mfll37pgrk62gnqchzrpzuy35.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 54
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
    tmp0 = tl.load(in_ptr0 + (x0 + (54*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (54*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + ((6*r1) + (x0 // 9)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuckb2mztcbndufma4poxqkxeinphbi2rmcalbmliewu3zqzsvn.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]

triton_poi_fused_add_mul_view_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (512*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x1 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x1 + (384*y0)), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (512*x1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdaf4kr7xw2llcgdfxtxjp5hxzgnkpvoco7yhjceeop4c363r7py.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]

triton_per_fused_mul_sum_transpose_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_transpose_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp0, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ytsetd7rtffzmq2yq4bhvs6arvhzm4lkt4mavcbbfpzxoaotgy.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clwjyxqdgfufcwihnmf7gx7sdtkflpk4dsgplyd72omn3wbeedpe.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zdk53wlfg4bv2cvfsi4f2suclssmshrcrdvw7cw6qlxwn65zvl.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cyk4uvz2pjhtmu5cqdm7rzm5lexunwk2hdsgxsor6fnqityljpgj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp12 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 768.0
    tmp25 = tmp12 * tmp24
    tmp26 = tmp25 - tmp16
    tmp27 = tmp17 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tmp34 = tmp29 * tmp33
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp34, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vqel5mes73dp5galiq3eqnv47nxo6smpmszamf6ctipkspccma.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_14
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*i64', 12: '*i64', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp16 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp36 = tl.full([1], -1, tl.int64)
    tmp37 = tmp35 == tmp36
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp34)
    tmp41 = tmp40 == tmp36
    tmp42 = tl.where(tmp41, tmp38, tmp34)
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tmp43 == tmp44
    tmp46 = tl.where(tmp45, tmp38, tmp34)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp15, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp42, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp46, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctaazsgmj77uh3yc5ddkqekcgdvcv5nlsj63l5luj527lsvvorl2.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5ysrdtito2vqorszz5tsnkcsbshgjie4owaklhtoqowzaj47gl.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czl4bhy2yzmq2obsffxupkwwjzw3ldsthosn66snp7zqsa2jjpr2.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, expand, slice_4, mul_1, getitem_3, view, addmm, permute_3, convolution, convolution_1, permute_9, view_9, full_default_1, unsqueeze_8, clone_default_33, clone_default_34, clone_default_35, getitem_276, getitem_277, getitem_278, alias_default_23, view_30, getitem_7, mul_4, view_32, addmm_5, view_34, getitem_11, mul_9, view_36, addmm_7, permute_22, convolution_2, convolution_3, permute_28, view_45, clone_default_30, clone_default_31, clone_default_32, getitem_269, getitem_270, getitem_271, alias_default_21, view_66, getitem_17, mul_12, view_68, addmm_12, view_70, getitem_21, mul_17, view_72, addmm_14, permute_41, convolution_4, convolution_5, permute_47, view_81, clone_default_27, clone_default_28, clone_default_29, getitem_262, getitem_263, getitem_264, alias_default_19, view_102, getitem_27, mul_20, view_104, addmm_19, view_106, getitem_31, mul_25, view_108, addmm_21, permute_60, convolution_6, convolution_7, permute_66, view_117, clone_default_24, clone_default_25, clone_default_26, getitem_255, getitem_256, getitem_257, alias_default_17, view_138, getitem_37, mul_28, view_140, addmm_26, view_142, getitem_41, mul_33, view_144, addmm_28, permute_79, convolution_8, convolution_9, permute_85, view_153, clone_default_21, clone_default_22, clone_default_23, getitem_248, getitem_249, getitem_250, alias_default_15, view_174, getitem_47, mul_36, view_176, addmm_33, view_178, getitem_51, mul_41, view_180, addmm_35, permute_98, convolution_10, convolution_11, permute_104, view_189, clone_default_18, clone_default_19, clone_default_20, getitem_241, getitem_242, getitem_243, alias_default_13, view_210, getitem_57, mul_44, view_212, addmm_40, view_214, getitem_61, mul_49, view_216, addmm_42, permute_117, convolution_12, convolution_13, permute_123, view_225, clone_default_15, clone_default_16, clone_default_17, getitem_234, getitem_235, getitem_236, alias_default_11, view_246, getitem_67, mul_52, view_248, addmm_47, view_250, getitem_71, mul_57, view_252, addmm_49, permute_136, convolution_14, convolution_15, permute_142, view_261, clone_default_12, clone_default_13, clone_default_14, getitem_227, getitem_228, getitem_229, alias_default_9, view_282, getitem_77, mul_60, view_284, addmm_54, view_286, getitem_81, mul_65, view_288, addmm_56, permute_155, convolution_16, convolution_17, permute_161, view_297, clone_default_9, clone_default_10, clone_default_11, getitem_220, getitem_221, getitem_222, alias_default_7, view_318, getitem_87, mul_68, view_320, addmm_61, view_322, getitem_91, mul_73, view_324, addmm_63, permute_174, convolution_18, convolution_19, permute_180, view_333, clone_default_6, clone_default_7, clone_default_8, getitem_213, getitem_214, getitem_215, alias_default_5, view_354, getitem_97, mul_76, view_356, addmm_68, view_358, getitem_101, mul_81, view_360, addmm_70, permute_193, convolution_20, convolution_21, permute_199, view_369, clone_default_3, clone_default_4, clone_default_5, getitem_206, getitem_207, getitem_208, alias_default_3, view_390, getitem_107, mul_84, view_392, addmm_75, view_394, getitem_111, mul_89, view_396, addmm_77, permute_212, convolution_22, convolution_23, permute_218, view_405, clone_default, clone_default_1, clone_default_2, getitem_199, getitem_200, getitem_201, alias_default_1, view_426, getitem_117, mul_92, view_428, addmm_82, view_430, getitem_121, mul_97, view_432, addmm_84, mul_102, view_434, sub_52, convert_element_type, permute_230, div_38, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_257, permute_261, alias_27, permute_275, permute_279, permute_283, div_42, permute_287, permute_291, div_43, permute_295, permute_305, permute_306, permute_310, alias_29, permute_324, permute_328, permute_332, div_45, permute_336, permute_340, div_46, permute_344, permute_354, permute_355, permute_359, alias_31, permute_373, permute_377, permute_381, div_48, permute_385, permute_389, div_49, permute_393, permute_403, permute_404, permute_408, alias_33, permute_422, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_452, permute_453, permute_457, alias_35, permute_471, permute_475, permute_479, div_54, permute_483, permute_487, div_55, permute_491, permute_501, permute_502, permute_506, alias_37, permute_520, permute_524, permute_528, div_57, permute_532, permute_536, div_58, permute_540, permute_550, permute_551, permute_555, alias_39, permute_569, permute_573, permute_577, div_60, permute_581, permute_585, div_61, permute_589, permute_599, permute_600, permute_604, alias_41, permute_618, permute_622, permute_626, div_63, permute_630, permute_634, div_64, permute_638, permute_648, permute_649, permute_653, alias_43, permute_667, permute_671, permute_675, div_66, permute_679, permute_683, div_67, permute_687, permute_697, permute_698, permute_702, alias_45, permute_716, permute_720, permute_724, div_69, permute_728, permute_732, div_70, permute_736, permute_746, permute_747, permute_751, alias_47, permute_765, permute_769, permute_773, div_72, permute_777, permute_781, div_73, permute_785, permute_795, permute_796, permute_800, alias_49, permute_814, permute_818, permute_822, div_75, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (384, 1), (1, 1))
    assert_size_stride(primals_2, (384, 1), (1, 1))
    assert_size_stride(primals_3, (384, 1), (1, 1))
    assert_size_stride(primals_4, (384, 1), (1, 1))
    assert_size_stride(primals_5, (384, 1), (1, 1))
    assert_size_stride(primals_6, (384, 1), (1, 1))
    assert_size_stride(primals_7, (384, 1), (1, 1))
    assert_size_stride(primals_8, (384, 1), (1, 1))
    assert_size_stride(primals_9, (384, 1), (1, 1))
    assert_size_stride(primals_10, (384, 1), (1, 1))
    assert_size_stride(primals_11, (384, 1), (1, 1))
    assert_size_stride(primals_12, (384, 1), (1, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_24, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_25, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_46, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_47, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_68, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_69, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_91, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_112, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_113, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_134, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_135, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_157, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_178, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_179, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_200, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_201, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_222, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_223, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_244, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_245, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_266, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_267, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_274, (768, ), (1, ))
    assert_size_stride(primals_280, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_290, (1, 512), (512, 1))
    assert_size_stride(primals_291, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(addmm, (512, 384), (384, 1))
    assert_size_stride(permute_3, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_1, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_9, (384, 54), (1, 384))
    assert_size_stride(view_9, (512, 384), (1, 512))
    assert_size_stride(full_default_1, (1, 1), (1, 1))
    assert_size_stride(unsqueeze_8, (9, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(clone_default_33, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_34, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_35, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_276, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_277, (), ())
    assert_size_stride(getitem_278, (), ())
    assert_size_stride(alias_default_23, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_5, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 384), (384, 1))
    assert_size_stride(permute_22, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_2, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_3, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_28, (384, 54), (1, 384))
    assert_size_stride(view_45, (512, 384), (1, 512))
    assert_size_stride(clone_default_30, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_31, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_32, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_269, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_270, (), ())
    assert_size_stride(getitem_271, (), ())
    assert_size_stride(alias_default_21, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_12, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(addmm_14, (512, 384), (384, 1))
    assert_size_stride(permute_41, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_4, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_5, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_47, (384, 54), (1, 384))
    assert_size_stride(view_81, (512, 384), (1, 512))
    assert_size_stride(clone_default_27, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_28, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_29, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_262, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_263, (), ())
    assert_size_stride(getitem_264, (), ())
    assert_size_stride(alias_default_19, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(addmm_21, (512, 384), (384, 1))
    assert_size_stride(permute_60, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_6, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_7, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_66, (384, 54), (1, 384))
    assert_size_stride(view_117, (512, 384), (1, 512))
    assert_size_stride(clone_default_24, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_25, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_26, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_255, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_256, (), ())
    assert_size_stride(getitem_257, (), ())
    assert_size_stride(alias_default_17, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_26, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 384), (384, 1))
    assert_size_stride(permute_79, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_8, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_9, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_85, (384, 54), (1, 384))
    assert_size_stride(view_153, (512, 384), (1, 512))
    assert_size_stride(clone_default_21, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_22, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_23, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_248, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_249, (), ())
    assert_size_stride(getitem_250, (), ())
    assert_size_stride(alias_default_15, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_33, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(addmm_35, (512, 384), (384, 1))
    assert_size_stride(permute_98, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_10, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_11, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_104, (384, 54), (1, 384))
    assert_size_stride(view_189, (512, 384), (1, 512))
    assert_size_stride(clone_default_18, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_19, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_20, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_241, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_242, (), ())
    assert_size_stride(getitem_243, (), ())
    assert_size_stride(alias_default_13, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_42, (512, 384), (384, 1))
    assert_size_stride(permute_117, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_12, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_13, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_123, (384, 54), (1, 384))
    assert_size_stride(view_225, (512, 384), (1, 512))
    assert_size_stride(clone_default_15, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_16, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_17, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_234, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_235, (), ())
    assert_size_stride(getitem_236, (), ())
    assert_size_stride(alias_default_11, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_246, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_248, (512, 768), (768, 1))
    assert_size_stride(addmm_47, (512, 3072), (3072, 1))
    assert_size_stride(view_250, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_252, (512, 768), (768, 1))
    assert_size_stride(addmm_49, (512, 384), (384, 1))
    assert_size_stride(permute_136, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_14, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_15, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_142, (384, 54), (1, 384))
    assert_size_stride(view_261, (512, 384), (1, 512))
    assert_size_stride(clone_default_12, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_13, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_14, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_227, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_228, (), ())
    assert_size_stride(getitem_229, (), ())
    assert_size_stride(alias_default_9, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_282, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_60, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_284, (512, 768), (768, 1))
    assert_size_stride(addmm_54, (512, 3072), (3072, 1))
    assert_size_stride(view_286, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_288, (512, 768), (768, 1))
    assert_size_stride(addmm_56, (512, 384), (384, 1))
    assert_size_stride(permute_155, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_16, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_17, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_161, (384, 54), (1, 384))
    assert_size_stride(view_297, (512, 384), (1, 512))
    assert_size_stride(clone_default_9, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_10, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_11, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_220, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_221, (), ())
    assert_size_stride(getitem_222, (), ())
    assert_size_stride(alias_default_7, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_318, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_68, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_320, (512, 768), (768, 1))
    assert_size_stride(addmm_61, (512, 3072), (3072, 1))
    assert_size_stride(view_322, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_324, (512, 768), (768, 1))
    assert_size_stride(addmm_63, (512, 384), (384, 1))
    assert_size_stride(permute_174, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_18, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_19, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_180, (384, 54), (1, 384))
    assert_size_stride(view_333, (512, 384), (1, 512))
    assert_size_stride(clone_default_6, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_7, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_8, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_213, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_214, (), ())
    assert_size_stride(getitem_215, (), ())
    assert_size_stride(alias_default_5, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_354, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_76, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_356, (512, 768), (768, 1))
    assert_size_stride(addmm_68, (512, 3072), (3072, 1))
    assert_size_stride(view_358, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_360, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 384), (384, 1))
    assert_size_stride(permute_193, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_20, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_21, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_199, (384, 54), (1, 384))
    assert_size_stride(view_369, (512, 384), (1, 512))
    assert_size_stride(clone_default_3, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_4, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_5, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_206, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_207, (), ())
    assert_size_stride(getitem_208, (), ())
    assert_size_stride(alias_default_3, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_390, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_84, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_392, (512, 768), (768, 1))
    assert_size_stride(addmm_75, (512, 3072), (3072, 1))
    assert_size_stride(view_394, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_396, (512, 768), (768, 1))
    assert_size_stride(addmm_77, (512, 384), (384, 1))
    assert_size_stride(permute_212, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_22, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_23, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_218, (384, 54), (1, 384))
    assert_size_stride(view_405, (512, 384), (1, 512))
    assert_size_stride(clone_default, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_1, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(clone_default_2, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(getitem_199, (1, 6, 512), (3072, 512, 1))
    assert_size_stride(getitem_200, (), ())
    assert_size_stride(getitem_201, (), ())
    assert_size_stride(alias_default_1, (1, 6, 512, 64), (196608, 1, 384, 6))
    assert_size_stride(view_426, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_92, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_428, (512, 768), (768, 1))
    assert_size_stride(addmm_82, (512, 3072), (3072, 1))
    assert_size_stride(view_430, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_432, (512, 768), (768, 1))
    assert_size_stride(addmm_84, (512, 768), (768, 1))
    assert_size_stride(mul_102, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_434, (512, 768), (768, 1))
    assert_size_stride(sub_52, (512, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_230, (30522, 768), (768, 1))
    assert_size_stride(div_38, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_234, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_238, (768, 3072), (3072, 1))
    assert_size_stride(permute_242, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_246, (768, 768), (768, 1))
    assert_size_stride(permute_256, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_257, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_261, (384, 768), (768, 1))
    assert_size_stride(alias_27, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_275, (384, 768), (768, 1))
    assert_size_stride(permute_279, (384, 768), (768, 1))
    assert_size_stride(permute_283, (384, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_287, (768, 3072), (3072, 1))
    assert_size_stride(permute_291, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_295, (768, 768), (768, 1))
    assert_size_stride(permute_305, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_306, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_310, (384, 768), (768, 1))
    assert_size_stride(alias_29, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_324, (384, 768), (768, 1))
    assert_size_stride(permute_328, (384, 768), (768, 1))
    assert_size_stride(permute_332, (384, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_336, (768, 3072), (3072, 1))
    assert_size_stride(permute_340, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_344, (768, 768), (768, 1))
    assert_size_stride(permute_354, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_355, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_359, (384, 768), (768, 1))
    assert_size_stride(alias_31, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_373, (384, 768), (768, 1))
    assert_size_stride(permute_377, (384, 768), (768, 1))
    assert_size_stride(permute_381, (384, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_385, (768, 3072), (3072, 1))
    assert_size_stride(permute_389, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_403, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_404, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_408, (384, 768), (768, 1))
    assert_size_stride(alias_33, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_422, (384, 768), (768, 1))
    assert_size_stride(permute_426, (384, 768), (768, 1))
    assert_size_stride(permute_430, (384, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_434, (768, 3072), (3072, 1))
    assert_size_stride(permute_438, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_442, (768, 768), (768, 1))
    assert_size_stride(permute_452, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_453, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_457, (384, 768), (768, 1))
    assert_size_stride(alias_35, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_471, (384, 768), (768, 1))
    assert_size_stride(permute_475, (384, 768), (768, 1))
    assert_size_stride(permute_479, (384, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_483, (768, 3072), (3072, 1))
    assert_size_stride(permute_487, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_491, (768, 768), (768, 1))
    assert_size_stride(permute_501, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_502, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_506, (384, 768), (768, 1))
    assert_size_stride(alias_37, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_520, (384, 768), (768, 1))
    assert_size_stride(permute_524, (384, 768), (768, 1))
    assert_size_stride(permute_528, (384, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_532, (768, 3072), (3072, 1))
    assert_size_stride(permute_536, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_540, (768, 768), (768, 1))
    assert_size_stride(permute_550, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_551, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_555, (384, 768), (768, 1))
    assert_size_stride(alias_39, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_569, (384, 768), (768, 1))
    assert_size_stride(permute_573, (384, 768), (768, 1))
    assert_size_stride(permute_577, (384, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_581, (768, 3072), (3072, 1))
    assert_size_stride(permute_585, (3072, 768), (768, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_589, (768, 768), (768, 1))
    assert_size_stride(permute_599, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_600, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_604, (384, 768), (768, 1))
    assert_size_stride(alias_41, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_618, (384, 768), (768, 1))
    assert_size_stride(permute_622, (384, 768), (768, 1))
    assert_size_stride(permute_626, (384, 768), (768, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_630, (768, 3072), (3072, 1))
    assert_size_stride(permute_634, (3072, 768), (768, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_638, (768, 768), (768, 1))
    assert_size_stride(permute_648, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_649, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_653, (384, 768), (768, 1))
    assert_size_stride(alias_43, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_667, (384, 768), (768, 1))
    assert_size_stride(permute_671, (384, 768), (768, 1))
    assert_size_stride(permute_675, (384, 768), (768, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_679, (768, 3072), (3072, 1))
    assert_size_stride(permute_683, (3072, 768), (768, 1))
    assert_size_stride(div_67, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_687, (768, 768), (768, 1))
    assert_size_stride(permute_697, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_698, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_702, (384, 768), (768, 1))
    assert_size_stride(alias_45, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_716, (384, 768), (768, 1))
    assert_size_stride(permute_720, (384, 768), (768, 1))
    assert_size_stride(permute_724, (384, 768), (768, 1))
    assert_size_stride(div_69, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_728, (768, 3072), (3072, 1))
    assert_size_stride(permute_732, (3072, 768), (768, 1))
    assert_size_stride(div_70, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_736, (768, 768), (768, 1))
    assert_size_stride(permute_746, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_747, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_751, (384, 768), (768, 1))
    assert_size_stride(alias_47, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_765, (384, 768), (768, 1))
    assert_size_stride(permute_769, (384, 768), (768, 1))
    assert_size_stride(permute_773, (384, 768), (768, 1))
    assert_size_stride(div_72, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_777, (768, 3072), (3072, 1))
    assert_size_stride(permute_781, (3072, 768), (768, 1))
    assert_size_stride(div_73, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_785, (768, 768), (768, 1))
    assert_size_stride(permute_795, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_796, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_800, (384, 768), (768, 1))
    assert_size_stride(alias_49, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_814, (384, 768), (768, 1))
    assert_size_stride(permute_818, (384, 768), (768, 1))
    assert_size_stride(permute_822, (384, 768), (768, 1))
    assert_size_stride(div_75, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 30522), (15627264, 30522, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 15627264, grid=grid(15627264), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_291, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_291, tangents_1, convert_element_type, tangents_2, sub_52, buf5, 512, 30522, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type
        del primals_291
        del sub_52
        del tangents_1
        del tangents_2
        buf6 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 30522), (30522, 1), 0), permute_230, out=buf6)
        del permute_230
        buf7 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (30522, 512), (1, 30522), 0), view_434, out=buf7)
        del view_434
        buf8 = empty((1, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 30522, 512, grid=grid(30522), stream=stream0)
        del buf5
        buf13 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_110], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf6, primals_284, mul_102, div_38, addmm_84, buf13, 512, 768, grid=grid(512), stream=stream0)
        del addmm_84
        del div_38
        del primals_284
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf6, mul_102, buf11, buf12, 768, 512, grid=grid(768), stream=stream0)
        del mul_102
        buf14 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_234, out=buf14)
        del permute_234
        buf15 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_432, out=buf15)
        del view_432
        buf16 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf13, buf16, 3072, 128, grid=grid(3072), stream=stream0)
        buf17 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf16, buf17, 768, 4, grid=grid(768), stream=stream0)
        buf20 = buf13; del buf13  # reuse
        buf23 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf14, primals_280, mul_97, div_39, getitem_121, buf20, buf23, 512, 768, grid=grid(512), stream=stream0)
        del div_39
        del getitem_121
        del primals_280
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf14, mul_97, buf21, buf22, 768, 512, grid=grid(768), stream=stream0)
        del mul_97
        buf24 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), (768, 1), 0), permute_238, out=buf24)
        del permute_238
        buf25 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (768, 512), (1, 768), 0), view_430, out=buf25)
        del view_430
        buf26 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf23, buf26, 3072, 128, grid=grid(3072), stream=stream0)
        buf27 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf26, buf27, 768, 4, grid=grid(768), stream=stream0)
        buf28 = reinterpret_tensor(buf24, (1, 512, 3072), (1572864, 3072, 1), 0); del buf24  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf28, addmm_82, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_82
        buf29 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 3072), (3072, 1), 0), permute_242, out=buf29)
        del permute_242
        buf30 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (3072, 512), (1, 3072), 0), view_428, out=buf30)
        del view_428
        buf31 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf28, buf31, 12288, 128, grid=grid(12288), stream=stream0)
        buf32 = reinterpret_tensor(buf26, (1, 3072), (3072, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf31, buf32, 3072, 4, grid=grid(3072), stream=stream0)
        buf35 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
        buf38 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf20, buf29, primals_274, mul_92, div_40, getitem_117, buf35, buf38, 512, 768, grid=grid(512), stream=stream0)
        del div_40
        del getitem_117
        del primals_274
        buf36 = empty((768, ), device='cuda', dtype=torch.float32)
        buf37 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf20, buf29, mul_92, buf36, buf37, 768, 512, grid=grid(768), stream=stream0)
        del mul_92
        buf39 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (512, 768), (768, 1), 0), permute_246, out=buf39)
        del permute_246
        buf40 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (768, 512), (1, 768), 0), view_426, out=buf40)
        del view_426
        buf41 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf38, buf41, 3072, 128, grid=grid(3072), stream=stream0)
        buf42 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf41, buf42, 768, 4, grid=grid(768), stream=stream0)
        buf43 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default, buf43, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default
        buf44 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_1, buf44, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_1
        buf45 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_2, buf45, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_2
        buf46 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_1, buf46, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_1
        # Source Nodes: [], Original ATen: []
        buf47 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf39, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf43, buf44, buf45, None, buf46, getitem_199, getitem_200, getitem_201, 0.1, [True, True, True, False], scale=0.125)
        del buf43
        del getitem_199
        del getitem_200
        del getitem_201
        buf48 = buf47[0]
        buf49 = buf47[1]
        buf50 = buf47[2]
        del buf47
        buf51 = reinterpret_tensor(buf46, (512, 384), (384, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf39, buf51, 196608, grid=grid(196608), stream=stream0)
        buf52 = empty((3072, 9, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_256, reinterpret_tensor(buf51, (3072, 64, 1), (64, 1, 0), 0), out=buf52)
        del permute_256
        buf53 = empty((3072, 64, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (3072, 64, 1), (64, 1, 0), 0), permute_257, out=buf53)
        del permute_257
        buf54 = empty((1, 384, 520, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf54, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf54, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf53, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf57 = reinterpret_tensor(buf51, (512, 384), (1, 512), 0); del buf51  # reuse
        buf60 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf54, buf57, buf60, 384, 512, grid=grid(384), stream=stream0)
        buf58 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf57, permute_261, out=buf58)
        del permute_261
        buf59 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (384, 512), (512, 1), 0), view_396, out=buf59)
        buf61 = reinterpret_tensor(buf41, (3072, 1, 1), (1, 3072, 3072), 0); del buf41  # reuse
        buf63 = empty((3072, 9, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf52, alias_27, buf61, buf63, 3072, 9, grid=grid(3072), stream=stream0)
        buf62 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf52, alias_27, buf61, buf62, 54, 512, grid=grid(54), stream=stream0)
        del alias_27
        buf64 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (54, 512), (1, 54), 0), view_405, out=buf64)
        del view_405
        buf65 = reinterpret_tensor(buf57, (384, 512), (512, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_218, reinterpret_tensor(buf63, (54, 512), (1, 54), 0), out=buf65)
        del permute_218
        buf66 = reinterpret_tensor(buf45, (1, 512, 384), (196608, 384, 1), 0); del buf45  # reuse
        buf83 = reinterpret_tensor(buf48, (1, 512, 384), (196608, 384, 1), 0); del buf48  # reuse
        buf84 = reinterpret_tensor(buf44, (512, 384), (1, 512), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf83, buf65, addmm_77, convolution_23, primals_12, buf66, buf84, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_77
        del buf65
        del convolution_23
        del primals_12
        buf67 = reinterpret_tensor(buf83, (1, 384, 512), (196608, 512, 1), 0); del buf83  # reuse
        buf68 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf66, buf67, buf68, 384, 512, grid=grid(384), stream=stream0)
        del buf66
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = aten.convolution_backward(buf67, convolution_22, primals_267, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_22
        del primals_267
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf72 = aten.convolution_backward(buf70, permute_212, primals_266, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_212
        del primals_266
        buf73 = buf72[0]
        buf74 = buf72[1]
        del buf72
        buf75 = reinterpret_tensor(buf70, (512, 768), (768, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 384), (384, 1), 0), permute_275, out=buf75)
        del permute_275
        buf76 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (384, 512), (1, 384), 0), view_396, out=buf76)
        buf77 = empty_strided((1, 384, 4), (1536, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf50, buf77, 1536, 128, grid=grid(1536), stream=stream0)
        buf78 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf77, buf78, 384, 4, grid=grid(384), stream=stream0)
        buf79 = reinterpret_tensor(buf38, (512, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 384), (384, 1), 0), permute_279, out=buf79)
        del permute_279
        buf80 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (384, 512), (1, 384), 0), view_396, out=buf80)
        buf81 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf49, buf81, 1536, 128, grid=grid(1536), stream=stream0)
        buf82 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf81, buf82, 384, 4, grid=grid(384), stream=stream0)
        buf85 = reinterpret_tensor(buf20, (512, 768), (768, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf84, permute_283, out=buf85)
        del permute_283
        buf86 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (384, 512), (512, 1), 0), view_396, out=buf86)
        del view_396
        buf87 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf84, buf87, 384, 512, grid=grid(384), stream=stream0)
        buf88 = buf35; del buf35  # reuse
        buf91 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf94 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf88, buf58, buf73, buf75, buf79, buf85, primals_258, mul_89, div_42, getitem_111, buf91, buf94, 512, 768, grid=grid(512), stream=stream0)
        del buf58
        del buf73
        del div_42
        del getitem_111
        del primals_258
        buf92 = empty((768, ), device='cuda', dtype=torch.float32)
        buf93 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf88, mul_89, buf92, buf93, 768, 512, grid=grid(768), stream=stream0)
        del mul_89
        buf95 = reinterpret_tensor(buf28, (512, 3072), (3072, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (512, 768), (768, 1), 0), permute_287, out=buf95)
        del permute_287
        buf96 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (768, 512), (1, 768), 0), view_394, out=buf96)
        del view_394
        buf97 = reinterpret_tensor(buf61, (1, 768, 4), (3072, 1, 768), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf94, buf97, 3072, 128, grid=grid(3072), stream=stream0)
        buf98 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf97, buf98, 768, 4, grid=grid(768), stream=stream0)
        buf99 = reinterpret_tensor(buf95, (1, 512, 3072), (1572864, 3072, 1), 0); del buf95  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf99, addmm_75, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_75
        buf100 = reinterpret_tensor(buf94, (512, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 3072), (3072, 1), 0), permute_291, out=buf100)
        del permute_291
        buf101 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (3072, 512), (1, 3072), 0), view_392, out=buf101)
        del view_392
        buf102 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf99, buf102, 12288, 128, grid=grid(12288), stream=stream0)
        buf103 = reinterpret_tensor(buf97, (1, 3072), (3072, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf102, buf103, 3072, 4, grid=grid(3072), stream=stream0)
        buf106 = buf88; del buf88  # reuse
        buf109 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf91, buf100, primals_252, mul_84, div_43, getitem_107, buf106, buf109, 512, 768, grid=grid(512), stream=stream0)
        del div_43
        del getitem_107
        del primals_252
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        buf108 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf91, buf100, mul_84, buf107, buf108, 768, 512, grid=grid(768), stream=stream0)
        del mul_84
        buf110 = reinterpret_tensor(buf91, (512, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), permute_295, out=buf110)
        del permute_295
        buf111 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (768, 512), (1, 768), 0), view_390, out=buf111)
        del view_390
        buf112 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf109, buf112, 3072, 128, grid=grid(3072), stream=stream0)
        buf113 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf112, buf113, 768, 4, grid=grid(768), stream=stream0)
        buf114 = reinterpret_tensor(buf84, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_3, buf114, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_3
        buf115 = reinterpret_tensor(buf49, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_4, buf115, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_4
        buf116 = reinterpret_tensor(buf50, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_5, buf116, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_5
        buf117 = reinterpret_tensor(buf67, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_3, buf117, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_3
        # Source Nodes: [], Original ATen: []
        buf118 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf110, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf114, buf115, buf116, None, buf117, getitem_206, getitem_207, getitem_208, 0.1, [True, True, True, False], scale=0.125)
        del buf114
        del getitem_206
        del getitem_207
        del getitem_208
        buf119 = buf118[0]
        buf120 = buf118[1]
        buf121 = buf118[2]
        del buf118
        buf122 = reinterpret_tensor(buf117, (512, 384), (384, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf110, buf122, 196608, grid=grid(196608), stream=stream0)
        buf123 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_305, reinterpret_tensor(buf122, (3072, 64, 1), (64, 1, 0), 0), out=buf123)
        del permute_305
        buf124 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (3072, 64, 1), (64, 1, 0), 0), permute_306, out=buf124)
        del permute_306
        buf125 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf125, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf125, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf124, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf128 = reinterpret_tensor(buf122, (512, 384), (1, 512), 0); del buf122  # reuse
        buf131 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf125, buf128, buf131, 384, 512, grid=grid(384), stream=stream0)
        buf129 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_310, out=buf129)
        del permute_310
        buf130 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (384, 512), (512, 1), 0), view_360, out=buf130)
        buf132 = reinterpret_tensor(buf112, (3072, 1, 1), (1, 3072, 3072), 0); del buf112  # reuse
        buf134 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf123, alias_29, buf132, buf134, 3072, 9, grid=grid(3072), stream=stream0)
        buf133 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf123, alias_29, buf132, buf133, 54, 512, grid=grid(54), stream=stream0)
        del alias_29
        buf135 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (54, 512), (1, 54), 0), view_369, out=buf135)
        del view_369
        buf136 = reinterpret_tensor(buf128, (384, 512), (512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_199, reinterpret_tensor(buf134, (54, 512), (1, 54), 0), out=buf136)
        del permute_199
        buf137 = reinterpret_tensor(buf116, (1, 512, 384), (196608, 384, 1), 0); del buf116  # reuse
        buf154 = reinterpret_tensor(buf119, (1, 512, 384), (196608, 384, 1), 0); del buf119  # reuse
        buf155 = reinterpret_tensor(buf115, (512, 384), (1, 512), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf154, buf136, addmm_70, convolution_21, primals_11, buf137, buf155, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_70
        del buf136
        del convolution_21
        del primals_11
        buf138 = reinterpret_tensor(buf154, (1, 384, 512), (196608, 512, 1), 0); del buf154  # reuse
        buf139 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf137, buf138, buf139, 384, 512, grid=grid(384), stream=stream0)
        del buf137
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf140 = aten.convolution_backward(buf138, convolution_20, primals_245, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_20
        del primals_245
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf143 = aten.convolution_backward(buf141, permute_193, primals_244, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_193
        del primals_244
        buf144 = buf143[0]
        buf145 = buf143[1]
        del buf143
        buf146 = reinterpret_tensor(buf141, (512, 768), (768, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 384), (384, 1), 0), permute_324, out=buf146)
        del permute_324
        buf147 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (384, 512), (1, 384), 0), view_360, out=buf147)
        buf148 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf121, buf148, 1536, 128, grid=grid(1536), stream=stream0)
        buf149 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf148, buf149, 384, 4, grid=grid(384), stream=stream0)
        buf150 = reinterpret_tensor(buf109, (512, 768), (768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 384), (384, 1), 0), permute_328, out=buf150)
        del permute_328
        buf151 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (384, 512), (1, 384), 0), view_360, out=buf151)
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf120, buf152, 1536, 128, grid=grid(1536), stream=stream0)
        buf153 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf152, buf153, 384, 4, grid=grid(384), stream=stream0)
        buf156 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf155, permute_332, out=buf156)
        del permute_332
        buf157 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (384, 512), (512, 1), 0), view_360, out=buf157)
        del view_360
        buf158 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf155, buf158, 384, 512, grid=grid(384), stream=stream0)
        buf159 = buf106; del buf106  # reuse
        buf162 = reinterpret_tensor(buf79, (1, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
        buf165 = reinterpret_tensor(buf75, (1, 512, 768), (393216, 768, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf159, buf129, buf144, buf146, buf150, buf156, primals_236, mul_81, div_45, getitem_101, buf162, buf165, 512, 768, grid=grid(512), stream=stream0)
        del buf129
        del buf144
        del div_45
        del getitem_101
        del primals_236
        buf163 = empty((768, ), device='cuda', dtype=torch.float32)
        buf164 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf159, mul_81, buf163, buf164, 768, 512, grid=grid(768), stream=stream0)
        del mul_81
        buf166 = reinterpret_tensor(buf99, (512, 3072), (3072, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 768), (768, 1), 0), permute_336, out=buf166)
        del permute_336
        buf167 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (768, 512), (1, 768), 0), view_358, out=buf167)
        del view_358
        buf168 = reinterpret_tensor(buf132, (1, 768, 4), (3072, 1, 768), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf165, buf168, 3072, 128, grid=grid(3072), stream=stream0)
        buf169 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf168, buf169, 768, 4, grid=grid(768), stream=stream0)
        buf170 = reinterpret_tensor(buf166, (1, 512, 3072), (1572864, 3072, 1), 0); del buf166  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf170, addmm_68, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_68
        buf171 = reinterpret_tensor(buf165, (512, 768), (768, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 3072), (3072, 1), 0), permute_340, out=buf171)
        del permute_340
        buf172 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (3072, 512), (1, 3072), 0), view_356, out=buf172)
        del view_356
        buf173 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf170, buf173, 12288, 128, grid=grid(12288), stream=stream0)
        buf174 = reinterpret_tensor(buf168, (1, 3072), (3072, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf173, buf174, 3072, 4, grid=grid(3072), stream=stream0)
        buf177 = buf159; del buf159  # reuse
        buf180 = reinterpret_tensor(buf156, (1, 512, 768), (393216, 768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf162, buf171, primals_230, mul_76, div_46, getitem_97, buf177, buf180, 512, 768, grid=grid(512), stream=stream0)
        del div_46
        del getitem_97
        del primals_230
        buf178 = empty((768, ), device='cuda', dtype=torch.float32)
        buf179 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf162, buf171, mul_76, buf178, buf179, 768, 512, grid=grid(768), stream=stream0)
        del mul_76
        buf181 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (512, 768), (768, 1), 0), permute_344, out=buf181)
        del permute_344
        buf182 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (768, 512), (1, 768), 0), view_354, out=buf182)
        del view_354
        buf183 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf180, buf183, 3072, 128, grid=grid(3072), stream=stream0)
        buf184 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf183, buf184, 768, 4, grid=grid(768), stream=stream0)
        buf185 = reinterpret_tensor(buf155, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_6, buf185, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_6
        buf186 = reinterpret_tensor(buf120, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_7, buf186, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_7
        buf187 = reinterpret_tensor(buf121, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_8, buf187, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_8
        buf188 = reinterpret_tensor(buf138, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_5, buf188, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_5
        # Source Nodes: [], Original ATen: []
        buf189 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf181, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf185, buf186, buf187, None, buf188, getitem_213, getitem_214, getitem_215, 0.1, [True, True, True, False], scale=0.125)
        del buf185
        del getitem_213
        del getitem_214
        del getitem_215
        buf190 = buf189[0]
        buf191 = buf189[1]
        buf192 = buf189[2]
        del buf189
        buf193 = reinterpret_tensor(buf188, (512, 384), (384, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf181, buf193, 196608, grid=grid(196608), stream=stream0)
        buf194 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_354, reinterpret_tensor(buf193, (3072, 64, 1), (64, 1, 0), 0), out=buf194)
        del permute_354
        buf195 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (3072, 64, 1), (64, 1, 0), 0), permute_355, out=buf195)
        del permute_355
        buf196 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf196, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf196, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf195, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf199 = reinterpret_tensor(buf193, (512, 384), (1, 512), 0); del buf193  # reuse
        buf202 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf196, buf199, buf202, 384, 512, grid=grid(384), stream=stream0)
        buf200 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf199, permute_359, out=buf200)
        del permute_359
        buf201 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (384, 512), (512, 1), 0), view_324, out=buf201)
        buf203 = reinterpret_tensor(buf183, (3072, 1, 1), (1, 3072, 3072), 0); del buf183  # reuse
        buf205 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf194, alias_31, buf203, buf205, 3072, 9, grid=grid(3072), stream=stream0)
        buf204 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf194, alias_31, buf203, buf204, 54, 512, grid=grid(54), stream=stream0)
        del alias_31
        buf206 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (54, 512), (1, 54), 0), view_333, out=buf206)
        del view_333
        buf207 = reinterpret_tensor(buf199, (384, 512), (512, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_180, reinterpret_tensor(buf205, (54, 512), (1, 54), 0), out=buf207)
        del permute_180
        buf208 = reinterpret_tensor(buf187, (1, 512, 384), (196608, 384, 1), 0); del buf187  # reuse
        buf225 = reinterpret_tensor(buf190, (1, 512, 384), (196608, 384, 1), 0); del buf190  # reuse
        buf226 = reinterpret_tensor(buf186, (512, 384), (1, 512), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf225, buf207, addmm_63, convolution_19, primals_10, buf208, buf226, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_63
        del buf207
        del convolution_19
        del primals_10
        buf209 = reinterpret_tensor(buf225, (1, 384, 512), (196608, 512, 1), 0); del buf225  # reuse
        buf210 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf208, buf209, buf210, 384, 512, grid=grid(384), stream=stream0)
        del buf208
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf211 = aten.convolution_backward(buf209, convolution_18, primals_223, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_18
        del primals_223
        buf212 = buf211[0]
        buf213 = buf211[1]
        del buf211
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf214 = aten.convolution_backward(buf212, permute_174, primals_222, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_174
        del primals_222
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = reinterpret_tensor(buf212, (512, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (512, 384), (384, 1), 0), permute_373, out=buf217)
        del permute_373
        buf218 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (384, 512), (1, 384), 0), view_324, out=buf218)
        buf219 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf192, buf219, 1536, 128, grid=grid(1536), stream=stream0)
        buf220 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf219, buf220, 384, 4, grid=grid(384), stream=stream0)
        buf221 = reinterpret_tensor(buf180, (512, 768), (768, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 384), (384, 1), 0), permute_377, out=buf221)
        del permute_377
        buf222 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (384, 512), (1, 384), 0), view_324, out=buf222)
        buf223 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf191, buf223, 1536, 128, grid=grid(1536), stream=stream0)
        buf224 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf223, buf224, 384, 4, grid=grid(384), stream=stream0)
        buf227 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf226, permute_381, out=buf227)
        del permute_381
        buf228 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (384, 512), (512, 1), 0), view_324, out=buf228)
        del view_324
        buf229 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf226, buf229, 384, 512, grid=grid(384), stream=stream0)
        buf230 = buf177; del buf177  # reuse
        buf233 = reinterpret_tensor(buf150, (1, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
        buf236 = reinterpret_tensor(buf146, (1, 512, 768), (393216, 768, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf230, buf200, buf215, buf217, buf221, buf227, primals_214, mul_73, div_48, getitem_91, buf233, buf236, 512, 768, grid=grid(512), stream=stream0)
        del buf200
        del buf215
        del div_48
        del getitem_91
        del primals_214
        buf234 = empty((768, ), device='cuda', dtype=torch.float32)
        buf235 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf230, mul_73, buf234, buf235, 768, 512, grid=grid(768), stream=stream0)
        del mul_73
        buf237 = reinterpret_tensor(buf170, (512, 3072), (3072, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (512, 768), (768, 1), 0), permute_385, out=buf237)
        del permute_385
        buf238 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (768, 512), (1, 768), 0), view_322, out=buf238)
        del view_322
        buf239 = reinterpret_tensor(buf203, (1, 768, 4), (3072, 1, 768), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf236, buf239, 3072, 128, grid=grid(3072), stream=stream0)
        buf240 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf239, buf240, 768, 4, grid=grid(768), stream=stream0)
        buf241 = reinterpret_tensor(buf237, (1, 512, 3072), (1572864, 3072, 1), 0); del buf237  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf241, addmm_61, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_61
        buf242 = reinterpret_tensor(buf236, (512, 768), (768, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0), permute_389, out=buf242)
        del permute_389
        buf243 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (3072, 512), (1, 3072), 0), view_320, out=buf243)
        del view_320
        buf244 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf241, buf244, 12288, 128, grid=grid(12288), stream=stream0)
        buf245 = reinterpret_tensor(buf239, (1, 3072), (3072, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf244, buf245, 3072, 4, grid=grid(3072), stream=stream0)
        buf248 = buf230; del buf230  # reuse
        buf251 = reinterpret_tensor(buf227, (1, 512, 768), (393216, 768, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf233, buf242, primals_208, mul_68, div_49, getitem_87, buf248, buf251, 512, 768, grid=grid(512), stream=stream0)
        del div_49
        del getitem_87
        del primals_208
        buf249 = empty((768, ), device='cuda', dtype=torch.float32)
        buf250 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf233, buf242, mul_68, buf249, buf250, 768, 512, grid=grid(768), stream=stream0)
        del mul_68
        buf252 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (512, 768), (768, 1), 0), permute_393, out=buf252)
        del permute_393
        buf253 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (768, 512), (1, 768), 0), view_318, out=buf253)
        del view_318
        buf254 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf251, buf254, 3072, 128, grid=grid(3072), stream=stream0)
        buf255 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf254, buf255, 768, 4, grid=grid(768), stream=stream0)
        buf256 = reinterpret_tensor(buf226, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_9, buf256, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_9
        buf257 = reinterpret_tensor(buf191, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_10, buf257, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_10
        buf258 = reinterpret_tensor(buf192, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_11, buf258, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_11
        buf259 = reinterpret_tensor(buf209, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_7, buf259, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_7
        # Source Nodes: [], Original ATen: []
        buf260 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf252, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf256, buf257, buf258, None, buf259, getitem_220, getitem_221, getitem_222, 0.1, [True, True, True, False], scale=0.125)
        del buf256
        del getitem_220
        del getitem_221
        del getitem_222
        buf261 = buf260[0]
        buf262 = buf260[1]
        buf263 = buf260[2]
        del buf260
        buf264 = reinterpret_tensor(buf259, (512, 384), (384, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf252, buf264, 196608, grid=grid(196608), stream=stream0)
        buf265 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_403, reinterpret_tensor(buf264, (3072, 64, 1), (64, 1, 0), 0), out=buf265)
        del permute_403
        buf266 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf264, (3072, 64, 1), (64, 1, 0), 0), permute_404, out=buf266)
        del permute_404
        buf267 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf267, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf267, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf266, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf270 = reinterpret_tensor(buf264, (512, 384), (1, 512), 0); del buf264  # reuse
        buf273 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf267, buf270, buf273, 384, 512, grid=grid(384), stream=stream0)
        buf271 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, permute_408, out=buf271)
        del permute_408
        buf272 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (384, 512), (512, 1), 0), view_288, out=buf272)
        buf274 = reinterpret_tensor(buf254, (3072, 1, 1), (1, 3072, 3072), 0); del buf254  # reuse
        buf276 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf265, alias_33, buf274, buf276, 3072, 9, grid=grid(3072), stream=stream0)
        buf275 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf265, alias_33, buf274, buf275, 54, 512, grid=grid(54), stream=stream0)
        del alias_33
        buf277 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (54, 512), (1, 54), 0), view_297, out=buf277)
        del view_297
        buf278 = reinterpret_tensor(buf270, (384, 512), (512, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_161, reinterpret_tensor(buf276, (54, 512), (1, 54), 0), out=buf278)
        del permute_161
        buf279 = reinterpret_tensor(buf258, (1, 512, 384), (196608, 384, 1), 0); del buf258  # reuse
        buf296 = reinterpret_tensor(buf261, (1, 512, 384), (196608, 384, 1), 0); del buf261  # reuse
        buf297 = reinterpret_tensor(buf257, (512, 384), (1, 512), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf296, buf278, addmm_56, convolution_17, primals_9, buf279, buf297, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_56
        del buf278
        del convolution_17
        del primals_9
        buf280 = reinterpret_tensor(buf296, (1, 384, 512), (196608, 512, 1), 0); del buf296  # reuse
        buf281 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf279, buf280, buf281, 384, 512, grid=grid(384), stream=stream0)
        del buf279
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf282 = aten.convolution_backward(buf280, convolution_16, primals_201, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_16
        del primals_201
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf285 = aten.convolution_backward(buf283, permute_155, primals_200, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_155
        del primals_200
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf288 = reinterpret_tensor(buf283, (512, 768), (768, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (512, 384), (384, 1), 0), permute_422, out=buf288)
        del permute_422
        buf289 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (384, 512), (1, 384), 0), view_288, out=buf289)
        buf290 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf263, buf290, 1536, 128, grid=grid(1536), stream=stream0)
        buf291 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf290, buf291, 384, 4, grid=grid(384), stream=stream0)
        buf292 = reinterpret_tensor(buf251, (512, 768), (768, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (512, 384), (384, 1), 0), permute_426, out=buf292)
        del permute_426
        buf293 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (384, 512), (1, 384), 0), view_288, out=buf293)
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf262, buf294, 1536, 128, grid=grid(1536), stream=stream0)
        buf295 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf294, buf295, 384, 4, grid=grid(384), stream=stream0)
        buf298 = reinterpret_tensor(buf233, (512, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, permute_430, out=buf298)
        del permute_430
        buf299 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (384, 512), (512, 1), 0), view_288, out=buf299)
        del view_288
        buf300 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf297, buf300, 384, 512, grid=grid(384), stream=stream0)
        buf301 = buf248; del buf248  # reuse
        buf304 = reinterpret_tensor(buf221, (1, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        buf307 = reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf301, buf271, buf286, buf288, buf292, buf298, primals_192, mul_65, div_51, getitem_81, buf304, buf307, 512, 768, grid=grid(512), stream=stream0)
        del buf271
        del buf286
        del div_51
        del getitem_81
        del primals_192
        buf305 = empty((768, ), device='cuda', dtype=torch.float32)
        buf306 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf301, mul_65, buf305, buf306, 768, 512, grid=grid(768), stream=stream0)
        del mul_65
        buf308 = reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (512, 768), (768, 1), 0), permute_434, out=buf308)
        del permute_434
        buf309 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (768, 512), (1, 768), 0), view_286, out=buf309)
        del view_286
        buf310 = reinterpret_tensor(buf274, (1, 768, 4), (3072, 1, 768), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf307, buf310, 3072, 128, grid=grid(3072), stream=stream0)
        buf311 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf310, buf311, 768, 4, grid=grid(768), stream=stream0)
        buf312 = reinterpret_tensor(buf308, (1, 512, 3072), (1572864, 3072, 1), 0); del buf308  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf312, addmm_54, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_54
        buf313 = reinterpret_tensor(buf307, (512, 768), (768, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (512, 3072), (3072, 1), 0), permute_438, out=buf313)
        del permute_438
        buf314 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (3072, 512), (1, 3072), 0), view_284, out=buf314)
        del view_284
        buf315 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf312, buf315, 12288, 128, grid=grid(12288), stream=stream0)
        buf316 = reinterpret_tensor(buf310, (1, 3072), (3072, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf315, buf316, 3072, 4, grid=grid(3072), stream=stream0)
        buf319 = buf301; del buf301  # reuse
        buf322 = reinterpret_tensor(buf298, (1, 512, 768), (393216, 768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf304, buf313, primals_186, mul_60, div_52, getitem_77, buf319, buf322, 512, 768, grid=grid(512), stream=stream0)
        del div_52
        del getitem_77
        del primals_186
        buf320 = empty((768, ), device='cuda', dtype=torch.float32)
        buf321 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf304, buf313, mul_60, buf320, buf321, 768, 512, grid=grid(768), stream=stream0)
        del mul_60
        buf323 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (512, 768), (768, 1), 0), permute_442, out=buf323)
        del permute_442
        buf324 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (768, 512), (1, 768), 0), view_282, out=buf324)
        del view_282
        buf325 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf322, buf325, 3072, 128, grid=grid(3072), stream=stream0)
        buf326 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf325, buf326, 768, 4, grid=grid(768), stream=stream0)
        buf327 = reinterpret_tensor(buf297, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_12, buf327, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_12
        buf328 = reinterpret_tensor(buf262, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_13, buf328, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_13
        buf329 = reinterpret_tensor(buf263, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_14, buf329, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_14
        buf330 = reinterpret_tensor(buf280, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_9, buf330, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_9
        # Source Nodes: [], Original ATen: []
        buf331 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf323, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf327, buf328, buf329, None, buf330, getitem_227, getitem_228, getitem_229, 0.1, [True, True, True, False], scale=0.125)
        del buf327
        del getitem_227
        del getitem_228
        del getitem_229
        buf332 = buf331[0]
        buf333 = buf331[1]
        buf334 = buf331[2]
        del buf331
        buf335 = reinterpret_tensor(buf330, (512, 384), (384, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf323, buf335, 196608, grid=grid(196608), stream=stream0)
        buf336 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_452, reinterpret_tensor(buf335, (3072, 64, 1), (64, 1, 0), 0), out=buf336)
        del permute_452
        buf337 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (3072, 64, 1), (64, 1, 0), 0), permute_453, out=buf337)
        del permute_453
        buf338 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf338, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf338, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf337, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf341 = reinterpret_tensor(buf335, (512, 384), (1, 512), 0); del buf335  # reuse
        buf344 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf338, buf341, buf344, 384, 512, grid=grid(384), stream=stream0)
        buf342 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf341, permute_457, out=buf342)
        del permute_457
        buf343 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (384, 512), (512, 1), 0), view_252, out=buf343)
        buf345 = reinterpret_tensor(buf325, (3072, 1, 1), (1, 3072, 3072), 0); del buf325  # reuse
        buf347 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf336, alias_35, buf345, buf347, 3072, 9, grid=grid(3072), stream=stream0)
        buf346 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf336, alias_35, buf345, buf346, 54, 512, grid=grid(54), stream=stream0)
        del alias_35
        buf348 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (54, 512), (1, 54), 0), view_261, out=buf348)
        del view_261
        buf349 = reinterpret_tensor(buf341, (384, 512), (512, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_142, reinterpret_tensor(buf347, (54, 512), (1, 54), 0), out=buf349)
        del permute_142
        buf350 = reinterpret_tensor(buf329, (1, 512, 384), (196608, 384, 1), 0); del buf329  # reuse
        buf367 = reinterpret_tensor(buf332, (1, 512, 384), (196608, 384, 1), 0); del buf332  # reuse
        buf368 = reinterpret_tensor(buf328, (512, 384), (1, 512), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf367, buf349, addmm_49, convolution_15, primals_8, buf350, buf368, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_49
        del buf349
        del convolution_15
        del primals_8
        buf351 = reinterpret_tensor(buf367, (1, 384, 512), (196608, 512, 1), 0); del buf367  # reuse
        buf352 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf350, buf351, buf352, 384, 512, grid=grid(384), stream=stream0)
        del buf350
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf353 = aten.convolution_backward(buf351, convolution_14, primals_179, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_14
        del primals_179
        buf354 = buf353[0]
        buf355 = buf353[1]
        del buf353
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf356 = aten.convolution_backward(buf354, permute_136, primals_178, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_136
        del primals_178
        buf357 = buf356[0]
        buf358 = buf356[1]
        del buf356
        buf359 = reinterpret_tensor(buf354, (512, 768), (768, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (512, 384), (384, 1), 0), permute_471, out=buf359)
        del permute_471
        buf360 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (384, 512), (1, 384), 0), view_252, out=buf360)
        buf361 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf334, buf361, 1536, 128, grid=grid(1536), stream=stream0)
        buf362 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf361, buf362, 384, 4, grid=grid(384), stream=stream0)
        buf363 = reinterpret_tensor(buf322, (512, 768), (768, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (512, 384), (384, 1), 0), permute_475, out=buf363)
        del permute_475
        buf364 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (384, 512), (1, 384), 0), view_252, out=buf364)
        buf365 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf333, buf365, 1536, 128, grid=grid(1536), stream=stream0)
        buf366 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf365, buf366, 384, 4, grid=grid(384), stream=stream0)
        buf369 = reinterpret_tensor(buf304, (512, 768), (768, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf368, permute_479, out=buf369)
        del permute_479
        buf370 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (384, 512), (512, 1), 0), view_252, out=buf370)
        del view_252
        buf371 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf368, buf371, 384, 512, grid=grid(384), stream=stream0)
        buf372 = buf319; del buf319  # reuse
        buf375 = reinterpret_tensor(buf292, (1, 512, 768), (393216, 768, 1), 0); del buf292  # reuse
        buf378 = reinterpret_tensor(buf288, (1, 512, 768), (393216, 768, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf372, buf342, buf357, buf359, buf363, buf369, primals_170, mul_57, div_54, getitem_71, buf375, buf378, 512, 768, grid=grid(512), stream=stream0)
        del buf342
        del buf357
        del div_54
        del getitem_71
        del primals_170
        buf376 = empty((768, ), device='cuda', dtype=torch.float32)
        buf377 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf372, mul_57, buf376, buf377, 768, 512, grid=grid(768), stream=stream0)
        del mul_57
        buf379 = reinterpret_tensor(buf312, (512, 3072), (3072, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 768), (768, 1), 0), permute_483, out=buf379)
        del permute_483
        buf380 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (768, 512), (1, 768), 0), view_250, out=buf380)
        del view_250
        buf381 = reinterpret_tensor(buf345, (1, 768, 4), (3072, 1, 768), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf378, buf381, 3072, 128, grid=grid(3072), stream=stream0)
        buf382 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf381, buf382, 768, 4, grid=grid(768), stream=stream0)
        buf383 = reinterpret_tensor(buf379, (1, 512, 3072), (1572864, 3072, 1), 0); del buf379  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf383, addmm_47, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_47
        buf384 = reinterpret_tensor(buf378, (512, 768), (768, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (512, 3072), (3072, 1), 0), permute_487, out=buf384)
        del permute_487
        buf385 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (3072, 512), (1, 3072), 0), view_248, out=buf385)
        del view_248
        buf386 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf383, buf386, 12288, 128, grid=grid(12288), stream=stream0)
        buf387 = reinterpret_tensor(buf381, (1, 3072), (3072, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf386, buf387, 3072, 4, grid=grid(3072), stream=stream0)
        buf390 = buf372; del buf372  # reuse
        buf393 = reinterpret_tensor(buf369, (1, 512, 768), (393216, 768, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf375, buf384, primals_164, mul_52, div_55, getitem_67, buf390, buf393, 512, 768, grid=grid(512), stream=stream0)
        del div_55
        del getitem_67
        del primals_164
        buf391 = empty((768, ), device='cuda', dtype=torch.float32)
        buf392 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf375, buf384, mul_52, buf391, buf392, 768, 512, grid=grid(768), stream=stream0)
        del mul_52
        buf394 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf393, (512, 768), (768, 1), 0), permute_491, out=buf394)
        del permute_491
        buf395 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf393, (768, 512), (1, 768), 0), view_246, out=buf395)
        del view_246
        buf396 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf393, buf396, 3072, 128, grid=grid(3072), stream=stream0)
        buf397 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf396, buf397, 768, 4, grid=grid(768), stream=stream0)
        buf398 = reinterpret_tensor(buf368, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_15, buf398, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_15
        buf399 = reinterpret_tensor(buf333, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_16, buf399, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_16
        buf400 = reinterpret_tensor(buf334, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_17, buf400, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_17
        buf401 = reinterpret_tensor(buf351, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_11, buf401, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_11
        # Source Nodes: [], Original ATen: []
        buf402 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf394, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf398, buf399, buf400, None, buf401, getitem_234, getitem_235, getitem_236, 0.1, [True, True, True, False], scale=0.125)
        del buf398
        del getitem_234
        del getitem_235
        del getitem_236
        buf403 = buf402[0]
        buf404 = buf402[1]
        buf405 = buf402[2]
        del buf402
        buf406 = reinterpret_tensor(buf401, (512, 384), (384, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf394, buf406, 196608, grid=grid(196608), stream=stream0)
        buf407 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_501, reinterpret_tensor(buf406, (3072, 64, 1), (64, 1, 0), 0), out=buf407)
        del permute_501
        buf408 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (3072, 64, 1), (64, 1, 0), 0), permute_502, out=buf408)
        del permute_502
        buf409 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf409, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf409, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf408, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf412 = reinterpret_tensor(buf406, (512, 384), (1, 512), 0); del buf406  # reuse
        buf415 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf409, buf412, buf415, 384, 512, grid=grid(384), stream=stream0)
        buf413 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf412, permute_506, out=buf413)
        del permute_506
        buf414 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (384, 512), (512, 1), 0), view_216, out=buf414)
        buf416 = reinterpret_tensor(buf396, (3072, 1, 1), (1, 3072, 3072), 0); del buf396  # reuse
        buf418 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf407, alias_37, buf416, buf418, 3072, 9, grid=grid(3072), stream=stream0)
        buf417 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf407, alias_37, buf416, buf417, 54, 512, grid=grid(54), stream=stream0)
        del alias_37
        buf419 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (54, 512), (1, 54), 0), view_225, out=buf419)
        del view_225
        buf420 = reinterpret_tensor(buf412, (384, 512), (512, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_123, reinterpret_tensor(buf418, (54, 512), (1, 54), 0), out=buf420)
        del permute_123
        buf421 = reinterpret_tensor(buf400, (1, 512, 384), (196608, 384, 1), 0); del buf400  # reuse
        buf438 = reinterpret_tensor(buf403, (1, 512, 384), (196608, 384, 1), 0); del buf403  # reuse
        buf439 = reinterpret_tensor(buf399, (512, 384), (1, 512), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf438, buf420, addmm_42, convolution_13, primals_7, buf421, buf439, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_42
        del buf420
        del convolution_13
        del primals_7
        buf422 = reinterpret_tensor(buf438, (1, 384, 512), (196608, 512, 1), 0); del buf438  # reuse
        buf423 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf421, buf422, buf423, 384, 512, grid=grid(384), stream=stream0)
        del buf421
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf424 = aten.convolution_backward(buf422, convolution_12, primals_157, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_12
        del primals_157
        buf425 = buf424[0]
        buf426 = buf424[1]
        del buf424
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf427 = aten.convolution_backward(buf425, permute_117, primals_156, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_117
        del primals_156
        buf428 = buf427[0]
        buf429 = buf427[1]
        del buf427
        buf430 = reinterpret_tensor(buf425, (512, 768), (768, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 384), (384, 1), 0), permute_520, out=buf430)
        del permute_520
        buf431 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (384, 512), (1, 384), 0), view_216, out=buf431)
        buf432 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf405, buf432, 1536, 128, grid=grid(1536), stream=stream0)
        buf433 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf432, buf433, 384, 4, grid=grid(384), stream=stream0)
        buf434 = reinterpret_tensor(buf393, (512, 768), (768, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (512, 384), (384, 1), 0), permute_524, out=buf434)
        del permute_524
        buf435 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (384, 512), (1, 384), 0), view_216, out=buf435)
        buf436 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf404, buf436, 1536, 128, grid=grid(1536), stream=stream0)
        buf437 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf436, buf437, 384, 4, grid=grid(384), stream=stream0)
        buf440 = reinterpret_tensor(buf375, (512, 768), (768, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf439, permute_528, out=buf440)
        del permute_528
        buf441 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (384, 512), (512, 1), 0), view_216, out=buf441)
        del view_216
        buf442 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf439, buf442, 384, 512, grid=grid(384), stream=stream0)
        buf443 = buf390; del buf390  # reuse
        buf446 = reinterpret_tensor(buf363, (1, 512, 768), (393216, 768, 1), 0); del buf363  # reuse
        buf449 = reinterpret_tensor(buf359, (1, 512, 768), (393216, 768, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf443, buf413, buf428, buf430, buf434, buf440, primals_148, mul_49, div_57, getitem_61, buf446, buf449, 512, 768, grid=grid(512), stream=stream0)
        del buf413
        del buf428
        del div_57
        del getitem_61
        del primals_148
        buf447 = empty((768, ), device='cuda', dtype=torch.float32)
        buf448 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf443, mul_49, buf447, buf448, 768, 512, grid=grid(768), stream=stream0)
        del mul_49
        buf450 = reinterpret_tensor(buf383, (512, 3072), (3072, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (512, 768), (768, 1), 0), permute_532, out=buf450)
        del permute_532
        buf451 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (768, 512), (1, 768), 0), view_214, out=buf451)
        del view_214
        buf452 = reinterpret_tensor(buf416, (1, 768, 4), (3072, 1, 768), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf449, buf452, 3072, 128, grid=grid(3072), stream=stream0)
        buf453 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf452, buf453, 768, 4, grid=grid(768), stream=stream0)
        buf454 = reinterpret_tensor(buf450, (1, 512, 3072), (1572864, 3072, 1), 0); del buf450  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf454, addmm_40, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_40
        buf455 = reinterpret_tensor(buf449, (512, 768), (768, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 3072), (3072, 1), 0), permute_536, out=buf455)
        del permute_536
        buf456 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (3072, 512), (1, 3072), 0), view_212, out=buf456)
        del view_212
        buf457 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf454, buf457, 12288, 128, grid=grid(12288), stream=stream0)
        buf458 = reinterpret_tensor(buf452, (1, 3072), (3072, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf457, buf458, 3072, 4, grid=grid(3072), stream=stream0)
        buf461 = buf443; del buf443  # reuse
        buf464 = reinterpret_tensor(buf440, (1, 512, 768), (393216, 768, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf446, buf455, primals_142, mul_44, div_58, getitem_57, buf461, buf464, 512, 768, grid=grid(512), stream=stream0)
        del div_58
        del getitem_57
        del primals_142
        buf462 = empty((768, ), device='cuda', dtype=torch.float32)
        buf463 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf446, buf455, mul_44, buf462, buf463, 768, 512, grid=grid(768), stream=stream0)
        del mul_44
        buf465 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 768), (768, 1), 0), permute_540, out=buf465)
        del permute_540
        buf466 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (768, 512), (1, 768), 0), view_210, out=buf466)
        del view_210
        buf467 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf464, buf467, 3072, 128, grid=grid(3072), stream=stream0)
        buf468 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf467, buf468, 768, 4, grid=grid(768), stream=stream0)
        buf469 = reinterpret_tensor(buf439, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_18, buf469, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_18
        buf470 = reinterpret_tensor(buf404, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_19, buf470, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_19
        buf471 = reinterpret_tensor(buf405, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_20, buf471, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_20
        buf472 = reinterpret_tensor(buf422, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_13, buf472, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_13
        # Source Nodes: [], Original ATen: []
        buf473 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf465, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf469, buf470, buf471, None, buf472, getitem_241, getitem_242, getitem_243, 0.1, [True, True, True, False], scale=0.125)
        del buf469
        del getitem_241
        del getitem_242
        del getitem_243
        buf474 = buf473[0]
        buf475 = buf473[1]
        buf476 = buf473[2]
        del buf473
        buf477 = reinterpret_tensor(buf472, (512, 384), (384, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf465, buf477, 196608, grid=grid(196608), stream=stream0)
        buf478 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_550, reinterpret_tensor(buf477, (3072, 64, 1), (64, 1, 0), 0), out=buf478)
        del permute_550
        buf479 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf477, (3072, 64, 1), (64, 1, 0), 0), permute_551, out=buf479)
        del permute_551
        buf480 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf480, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf480, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf479, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf483 = reinterpret_tensor(buf477, (512, 384), (1, 512), 0); del buf477  # reuse
        buf486 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf480, buf483, buf486, 384, 512, grid=grid(384), stream=stream0)
        buf484 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf483, permute_555, out=buf484)
        del permute_555
        buf485 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (384, 512), (512, 1), 0), view_180, out=buf485)
        buf487 = reinterpret_tensor(buf467, (3072, 1, 1), (1, 3072, 3072), 0); del buf467  # reuse
        buf489 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf478, alias_39, buf487, buf489, 3072, 9, grid=grid(3072), stream=stream0)
        buf488 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf478, alias_39, buf487, buf488, 54, 512, grid=grid(54), stream=stream0)
        del alias_39
        buf490 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (54, 512), (1, 54), 0), view_189, out=buf490)
        del view_189
        buf491 = reinterpret_tensor(buf483, (384, 512), (512, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_104, reinterpret_tensor(buf489, (54, 512), (1, 54), 0), out=buf491)
        del permute_104
        buf492 = reinterpret_tensor(buf471, (1, 512, 384), (196608, 384, 1), 0); del buf471  # reuse
        buf509 = reinterpret_tensor(buf474, (1, 512, 384), (196608, 384, 1), 0); del buf474  # reuse
        buf510 = reinterpret_tensor(buf470, (512, 384), (1, 512), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf509, buf491, addmm_35, convolution_11, primals_6, buf492, buf510, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_35
        del buf491
        del convolution_11
        del primals_6
        buf493 = reinterpret_tensor(buf509, (1, 384, 512), (196608, 512, 1), 0); del buf509  # reuse
        buf494 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf492, buf493, buf494, 384, 512, grid=grid(384), stream=stream0)
        del buf492
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf495 = aten.convolution_backward(buf493, convolution_10, primals_135, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_10
        del primals_135
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf498 = aten.convolution_backward(buf496, permute_98, primals_134, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_98
        del primals_134
        buf499 = buf498[0]
        buf500 = buf498[1]
        del buf498
        buf501 = reinterpret_tensor(buf496, (512, 768), (768, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (512, 384), (384, 1), 0), permute_569, out=buf501)
        del permute_569
        buf502 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (384, 512), (1, 384), 0), view_180, out=buf502)
        buf503 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf476, buf503, 1536, 128, grid=grid(1536), stream=stream0)
        buf504 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf503, buf504, 384, 4, grid=grid(384), stream=stream0)
        buf505 = reinterpret_tensor(buf464, (512, 768), (768, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (512, 384), (384, 1), 0), permute_573, out=buf505)
        del permute_573
        buf506 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (384, 512), (1, 384), 0), view_180, out=buf506)
        buf507 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf475, buf507, 1536, 128, grid=grid(1536), stream=stream0)
        buf508 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf507, buf508, 384, 4, grid=grid(384), stream=stream0)
        buf511 = reinterpret_tensor(buf446, (512, 768), (768, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf510, permute_577, out=buf511)
        del permute_577
        buf512 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (384, 512), (512, 1), 0), view_180, out=buf512)
        del view_180
        buf513 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf510, buf513, 384, 512, grid=grid(384), stream=stream0)
        buf514 = buf461; del buf461  # reuse
        buf517 = reinterpret_tensor(buf434, (1, 512, 768), (393216, 768, 1), 0); del buf434  # reuse
        buf520 = reinterpret_tensor(buf430, (1, 512, 768), (393216, 768, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf514, buf484, buf499, buf501, buf505, buf511, primals_126, mul_41, div_60, getitem_51, buf517, buf520, 512, 768, grid=grid(512), stream=stream0)
        del buf484
        del buf499
        del div_60
        del getitem_51
        del primals_126
        buf518 = empty((768, ), device='cuda', dtype=torch.float32)
        buf519 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf514, mul_41, buf518, buf519, 768, 512, grid=grid(768), stream=stream0)
        del mul_41
        buf521 = reinterpret_tensor(buf454, (512, 3072), (3072, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (512, 768), (768, 1), 0), permute_581, out=buf521)
        del permute_581
        buf522 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (768, 512), (1, 768), 0), view_178, out=buf522)
        del view_178
        buf523 = reinterpret_tensor(buf487, (1, 768, 4), (3072, 1, 768), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf520, buf523, 3072, 128, grid=grid(3072), stream=stream0)
        buf524 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf523, buf524, 768, 4, grid=grid(768), stream=stream0)
        buf525 = reinterpret_tensor(buf521, (1, 512, 3072), (1572864, 3072, 1), 0); del buf521  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf525, addmm_33, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_33
        buf526 = reinterpret_tensor(buf520, (512, 768), (768, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (512, 3072), (3072, 1), 0), permute_585, out=buf526)
        del permute_585
        buf527 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (3072, 512), (1, 3072), 0), view_176, out=buf527)
        del view_176
        buf528 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf525, buf528, 12288, 128, grid=grid(12288), stream=stream0)
        buf529 = reinterpret_tensor(buf523, (1, 3072), (3072, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf528, buf529, 3072, 4, grid=grid(3072), stream=stream0)
        buf532 = buf514; del buf514  # reuse
        buf535 = reinterpret_tensor(buf511, (1, 512, 768), (393216, 768, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf517, buf526, primals_120, mul_36, div_61, getitem_47, buf532, buf535, 512, 768, grid=grid(512), stream=stream0)
        del div_61
        del getitem_47
        del primals_120
        buf533 = empty((768, ), device='cuda', dtype=torch.float32)
        buf534 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf517, buf526, mul_36, buf533, buf534, 768, 512, grid=grid(768), stream=stream0)
        del mul_36
        buf536 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (512, 768), (768, 1), 0), permute_589, out=buf536)
        del permute_589
        buf537 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (768, 512), (1, 768), 0), view_174, out=buf537)
        del view_174
        buf538 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf535, buf538, 3072, 128, grid=grid(3072), stream=stream0)
        buf539 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf538, buf539, 768, 4, grid=grid(768), stream=stream0)
        buf540 = reinterpret_tensor(buf510, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_21, buf540, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_21
        buf541 = reinterpret_tensor(buf475, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_22, buf541, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_22
        buf542 = reinterpret_tensor(buf476, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_23, buf542, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_23
        buf543 = reinterpret_tensor(buf493, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_15, buf543, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_15
        # Source Nodes: [], Original ATen: []
        buf544 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf536, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf540, buf541, buf542, None, buf543, getitem_248, getitem_249, getitem_250, 0.1, [True, True, True, False], scale=0.125)
        del buf540
        del getitem_248
        del getitem_249
        del getitem_250
        buf545 = buf544[0]
        buf546 = buf544[1]
        buf547 = buf544[2]
        del buf544
        buf548 = reinterpret_tensor(buf543, (512, 384), (384, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf536, buf548, 196608, grid=grid(196608), stream=stream0)
        buf549 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_599, reinterpret_tensor(buf548, (3072, 64, 1), (64, 1, 0), 0), out=buf549)
        del permute_599
        buf550 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (3072, 64, 1), (64, 1, 0), 0), permute_600, out=buf550)
        del permute_600
        buf551 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf551, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf551, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf550, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf554 = reinterpret_tensor(buf548, (512, 384), (1, 512), 0); del buf548  # reuse
        buf557 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf551, buf554, buf557, 384, 512, grid=grid(384), stream=stream0)
        buf555 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf554, permute_604, out=buf555)
        del permute_604
        buf556 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf554, (384, 512), (512, 1), 0), view_144, out=buf556)
        buf558 = reinterpret_tensor(buf538, (3072, 1, 1), (1, 3072, 3072), 0); del buf538  # reuse
        buf560 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf549, alias_41, buf558, buf560, 3072, 9, grid=grid(3072), stream=stream0)
        buf559 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf549, alias_41, buf558, buf559, 54, 512, grid=grid(54), stream=stream0)
        del alias_41
        buf561 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (54, 512), (1, 54), 0), view_153, out=buf561)
        del view_153
        buf562 = reinterpret_tensor(buf554, (384, 512), (512, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_85, reinterpret_tensor(buf560, (54, 512), (1, 54), 0), out=buf562)
        del permute_85
        buf563 = reinterpret_tensor(buf542, (1, 512, 384), (196608, 384, 1), 0); del buf542  # reuse
        buf580 = reinterpret_tensor(buf545, (1, 512, 384), (196608, 384, 1), 0); del buf545  # reuse
        buf581 = reinterpret_tensor(buf541, (512, 384), (1, 512), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf580, buf562, addmm_28, convolution_9, primals_5, buf563, buf581, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_28
        del buf562
        del convolution_9
        del primals_5
        buf564 = reinterpret_tensor(buf580, (1, 384, 512), (196608, 512, 1), 0); del buf580  # reuse
        buf565 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf563, buf564, buf565, 384, 512, grid=grid(384), stream=stream0)
        del buf563
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf566 = aten.convolution_backward(buf564, convolution_8, primals_113, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_8
        del primals_113
        buf567 = buf566[0]
        buf568 = buf566[1]
        del buf566
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf569 = aten.convolution_backward(buf567, permute_79, primals_112, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_79
        del primals_112
        buf570 = buf569[0]
        buf571 = buf569[1]
        del buf569
        buf572 = reinterpret_tensor(buf567, (512, 768), (768, 1), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf547, (512, 384), (384, 1), 0), permute_618, out=buf572)
        del permute_618
        buf573 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf547, (384, 512), (1, 384), 0), view_144, out=buf573)
        buf574 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf547, buf574, 1536, 128, grid=grid(1536), stream=stream0)
        buf575 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf574, buf575, 384, 4, grid=grid(384), stream=stream0)
        buf576 = reinterpret_tensor(buf535, (512, 768), (768, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (512, 384), (384, 1), 0), permute_622, out=buf576)
        del permute_622
        buf577 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (384, 512), (1, 384), 0), view_144, out=buf577)
        buf578 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf546, buf578, 1536, 128, grid=grid(1536), stream=stream0)
        buf579 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf578, buf579, 384, 4, grid=grid(384), stream=stream0)
        buf582 = reinterpret_tensor(buf517, (512, 768), (768, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf581, permute_626, out=buf582)
        del permute_626
        buf583 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (384, 512), (512, 1), 0), view_144, out=buf583)
        del view_144
        buf584 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf581, buf584, 384, 512, grid=grid(384), stream=stream0)
        buf585 = buf532; del buf532  # reuse
        buf588 = reinterpret_tensor(buf505, (1, 512, 768), (393216, 768, 1), 0); del buf505  # reuse
        buf591 = reinterpret_tensor(buf501, (1, 512, 768), (393216, 768, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf585, buf555, buf570, buf572, buf576, buf582, primals_104, mul_33, div_63, getitem_41, buf588, buf591, 512, 768, grid=grid(512), stream=stream0)
        del buf555
        del buf570
        del div_63
        del getitem_41
        del primals_104
        buf589 = empty((768, ), device='cuda', dtype=torch.float32)
        buf590 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf585, mul_33, buf589, buf590, 768, 512, grid=grid(768), stream=stream0)
        del mul_33
        buf592 = reinterpret_tensor(buf525, (512, 3072), (3072, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (512, 768), (768, 1), 0), permute_630, out=buf592)
        del permute_630
        buf593 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (768, 512), (1, 768), 0), view_142, out=buf593)
        del view_142
        buf594 = reinterpret_tensor(buf558, (1, 768, 4), (3072, 1, 768), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf591, buf594, 3072, 128, grid=grid(3072), stream=stream0)
        buf595 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf594, buf595, 768, 4, grid=grid(768), stream=stream0)
        buf596 = reinterpret_tensor(buf592, (1, 512, 3072), (1572864, 3072, 1), 0); del buf592  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf596, addmm_26, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_26
        buf597 = reinterpret_tensor(buf591, (512, 768), (768, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (512, 3072), (3072, 1), 0), permute_634, out=buf597)
        del permute_634
        buf598 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (3072, 512), (1, 3072), 0), view_140, out=buf598)
        del view_140
        buf599 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf596, buf599, 12288, 128, grid=grid(12288), stream=stream0)
        buf600 = reinterpret_tensor(buf594, (1, 3072), (3072, 1), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf599, buf600, 3072, 4, grid=grid(3072), stream=stream0)
        buf603 = buf585; del buf585  # reuse
        buf606 = reinterpret_tensor(buf582, (1, 512, 768), (393216, 768, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf588, buf597, primals_98, mul_28, div_64, getitem_37, buf603, buf606, 512, 768, grid=grid(512), stream=stream0)
        del div_64
        del getitem_37
        del primals_98
        buf604 = empty((768, ), device='cuda', dtype=torch.float32)
        buf605 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf588, buf597, mul_28, buf604, buf605, 768, 512, grid=grid(768), stream=stream0)
        del mul_28
        buf607 = buf597; del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (512, 768), (768, 1), 0), permute_638, out=buf607)
        del permute_638
        buf608 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (768, 512), (1, 768), 0), view_138, out=buf608)
        del view_138
        buf609 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf606, buf609, 3072, 128, grid=grid(3072), stream=stream0)
        buf610 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf609, buf610, 768, 4, grid=grid(768), stream=stream0)
        buf611 = reinterpret_tensor(buf581, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_24, buf611, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_24
        buf612 = reinterpret_tensor(buf546, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_25, buf612, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_25
        buf613 = reinterpret_tensor(buf547, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_26, buf613, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_26
        buf614 = reinterpret_tensor(buf564, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_17, buf614, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_17
        # Source Nodes: [], Original ATen: []
        buf615 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf607, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf611, buf612, buf613, None, buf614, getitem_255, getitem_256, getitem_257, 0.1, [True, True, True, False], scale=0.125)
        del buf611
        del getitem_255
        del getitem_256
        del getitem_257
        buf616 = buf615[0]
        buf617 = buf615[1]
        buf618 = buf615[2]
        del buf615
        buf619 = reinterpret_tensor(buf614, (512, 384), (384, 1), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf607, buf619, 196608, grid=grid(196608), stream=stream0)
        buf620 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_648, reinterpret_tensor(buf619, (3072, 64, 1), (64, 1, 0), 0), out=buf620)
        del permute_648
        buf621 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf619, (3072, 64, 1), (64, 1, 0), 0), permute_649, out=buf621)
        del permute_649
        buf622 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf622, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf622, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf621, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf625 = reinterpret_tensor(buf619, (512, 384), (1, 512), 0); del buf619  # reuse
        buf628 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf622, buf625, buf628, 384, 512, grid=grid(384), stream=stream0)
        buf626 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf625, permute_653, out=buf626)
        del permute_653
        buf627 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (384, 512), (512, 1), 0), view_108, out=buf627)
        buf629 = reinterpret_tensor(buf609, (3072, 1, 1), (1, 3072, 3072), 0); del buf609  # reuse
        buf631 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf620, alias_43, buf629, buf631, 3072, 9, grid=grid(3072), stream=stream0)
        buf630 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf620, alias_43, buf629, buf630, 54, 512, grid=grid(54), stream=stream0)
        del alias_43
        buf632 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf631, (54, 512), (1, 54), 0), view_117, out=buf632)
        del view_117
        buf633 = reinterpret_tensor(buf625, (384, 512), (512, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_66, reinterpret_tensor(buf631, (54, 512), (1, 54), 0), out=buf633)
        del permute_66
        buf634 = reinterpret_tensor(buf613, (1, 512, 384), (196608, 384, 1), 0); del buf613  # reuse
        buf651 = reinterpret_tensor(buf616, (1, 512, 384), (196608, 384, 1), 0); del buf616  # reuse
        buf652 = reinterpret_tensor(buf612, (512, 384), (1, 512), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf651, buf633, addmm_21, convolution_7, primals_4, buf634, buf652, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_21
        del buf633
        del convolution_7
        del primals_4
        buf635 = reinterpret_tensor(buf651, (1, 384, 512), (196608, 512, 1), 0); del buf651  # reuse
        buf636 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf634, buf635, buf636, 384, 512, grid=grid(384), stream=stream0)
        del buf634
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf637 = aten.convolution_backward(buf635, convolution_6, primals_91, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_6
        del primals_91
        buf638 = buf637[0]
        buf639 = buf637[1]
        del buf637
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf640 = aten.convolution_backward(buf638, permute_60, primals_90, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_60
        del primals_90
        buf641 = buf640[0]
        buf642 = buf640[1]
        del buf640
        buf643 = reinterpret_tensor(buf638, (512, 768), (768, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (512, 384), (384, 1), 0), permute_667, out=buf643)
        del permute_667
        buf644 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (384, 512), (1, 384), 0), view_108, out=buf644)
        buf645 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf618, buf645, 1536, 128, grid=grid(1536), stream=stream0)
        buf646 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf645, buf646, 384, 4, grid=grid(384), stream=stream0)
        buf647 = reinterpret_tensor(buf606, (512, 768), (768, 1), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (512, 384), (384, 1), 0), permute_671, out=buf647)
        del permute_671
        buf648 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (384, 512), (1, 384), 0), view_108, out=buf648)
        buf649 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf617, buf649, 1536, 128, grid=grid(1536), stream=stream0)
        buf650 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf649, buf650, 384, 4, grid=grid(384), stream=stream0)
        buf653 = reinterpret_tensor(buf588, (512, 768), (768, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf652, permute_675, out=buf653)
        del permute_675
        buf654 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (384, 512), (512, 1), 0), view_108, out=buf654)
        del view_108
        buf655 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf652, buf655, 384, 512, grid=grid(384), stream=stream0)
        buf656 = buf603; del buf603  # reuse
        buf659 = reinterpret_tensor(buf576, (1, 512, 768), (393216, 768, 1), 0); del buf576  # reuse
        buf662 = reinterpret_tensor(buf572, (1, 512, 768), (393216, 768, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf656, buf626, buf641, buf643, buf647, buf653, primals_82, mul_25, div_66, getitem_31, buf659, buf662, 512, 768, grid=grid(512), stream=stream0)
        del buf626
        del buf641
        del div_66
        del getitem_31
        del primals_82
        buf660 = empty((768, ), device='cuda', dtype=torch.float32)
        buf661 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf656, mul_25, buf660, buf661, 768, 512, grid=grid(768), stream=stream0)
        del mul_25
        buf663 = reinterpret_tensor(buf596, (512, 3072), (3072, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf662, (512, 768), (768, 1), 0), permute_679, out=buf663)
        del permute_679
        buf664 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf662, (768, 512), (1, 768), 0), view_106, out=buf664)
        del view_106
        buf665 = reinterpret_tensor(buf629, (1, 768, 4), (3072, 1, 768), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf662, buf665, 3072, 128, grid=grid(3072), stream=stream0)
        buf666 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf665, buf666, 768, 4, grid=grid(768), stream=stream0)
        buf667 = reinterpret_tensor(buf663, (1, 512, 3072), (1572864, 3072, 1), 0); del buf663  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf667, addmm_19, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_19
        buf668 = reinterpret_tensor(buf662, (512, 768), (768, 1), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (512, 3072), (3072, 1), 0), permute_683, out=buf668)
        del permute_683
        buf669 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (3072, 512), (1, 3072), 0), view_104, out=buf669)
        del view_104
        buf670 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf667, buf670, 12288, 128, grid=grid(12288), stream=stream0)
        buf671 = reinterpret_tensor(buf665, (1, 3072), (3072, 1), 0); del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf670, buf671, 3072, 4, grid=grid(3072), stream=stream0)
        buf674 = buf656; del buf656  # reuse
        buf677 = reinterpret_tensor(buf653, (1, 512, 768), (393216, 768, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf659, buf668, primals_76, mul_20, div_67, getitem_27, buf674, buf677, 512, 768, grid=grid(512), stream=stream0)
        del div_67
        del getitem_27
        del primals_76
        buf675 = empty((768, ), device='cuda', dtype=torch.float32)
        buf676 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf659, buf668, mul_20, buf675, buf676, 768, 512, grid=grid(768), stream=stream0)
        del mul_20
        buf678 = buf668; del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (512, 768), (768, 1), 0), permute_687, out=buf678)
        del permute_687
        buf679 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (768, 512), (1, 768), 0), view_102, out=buf679)
        del view_102
        buf680 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf677, buf680, 3072, 128, grid=grid(3072), stream=stream0)
        buf681 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf680, buf681, 768, 4, grid=grid(768), stream=stream0)
        buf682 = reinterpret_tensor(buf652, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_27, buf682, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_27
        buf683 = reinterpret_tensor(buf617, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_28, buf683, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_28
        buf684 = reinterpret_tensor(buf618, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_29, buf684, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_29
        buf685 = reinterpret_tensor(buf635, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_19, buf685, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_19
        # Source Nodes: [], Original ATen: []
        buf686 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf678, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf682, buf683, buf684, None, buf685, getitem_262, getitem_263, getitem_264, 0.1, [True, True, True, False], scale=0.125)
        del buf682
        del getitem_262
        del getitem_263
        del getitem_264
        buf687 = buf686[0]
        buf688 = buf686[1]
        buf689 = buf686[2]
        del buf686
        buf690 = reinterpret_tensor(buf685, (512, 384), (384, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf678, buf690, 196608, grid=grid(196608), stream=stream0)
        buf691 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_697, reinterpret_tensor(buf690, (3072, 64, 1), (64, 1, 0), 0), out=buf691)
        del permute_697
        buf692 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf690, (3072, 64, 1), (64, 1, 0), 0), permute_698, out=buf692)
        del permute_698
        buf693 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf693, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf693, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf692, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf696 = reinterpret_tensor(buf690, (512, 384), (1, 512), 0); del buf690  # reuse
        buf699 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf693, buf696, buf699, 384, 512, grid=grid(384), stream=stream0)
        buf697 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf696, permute_702, out=buf697)
        del permute_702
        buf698 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (384, 512), (512, 1), 0), view_72, out=buf698)
        buf700 = reinterpret_tensor(buf680, (3072, 1, 1), (1, 3072, 3072), 0); del buf680  # reuse
        buf702 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf691, alias_45, buf700, buf702, 3072, 9, grid=grid(3072), stream=stream0)
        buf701 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf691, alias_45, buf700, buf701, 54, 512, grid=grid(54), stream=stream0)
        del alias_45
        buf703 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf702, (54, 512), (1, 54), 0), view_81, out=buf703)
        del view_81
        buf704 = reinterpret_tensor(buf696, (384, 512), (512, 1), 0); del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_47, reinterpret_tensor(buf702, (54, 512), (1, 54), 0), out=buf704)
        del permute_47
        buf705 = reinterpret_tensor(buf684, (1, 512, 384), (196608, 384, 1), 0); del buf684  # reuse
        buf722 = reinterpret_tensor(buf687, (1, 512, 384), (196608, 384, 1), 0); del buf687  # reuse
        buf723 = reinterpret_tensor(buf683, (512, 384), (1, 512), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf722, buf704, addmm_14, convolution_5, primals_3, buf705, buf723, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_14
        del buf704
        del convolution_5
        del primals_3
        buf706 = reinterpret_tensor(buf722, (1, 384, 512), (196608, 512, 1), 0); del buf722  # reuse
        buf707 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf705, buf706, buf707, 384, 512, grid=grid(384), stream=stream0)
        del buf705
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf708 = aten.convolution_backward(buf706, convolution_4, primals_69, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_4
        del primals_69
        buf709 = buf708[0]
        buf710 = buf708[1]
        del buf708
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf711 = aten.convolution_backward(buf709, permute_41, primals_68, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_41
        del primals_68
        buf712 = buf711[0]
        buf713 = buf711[1]
        del buf711
        buf714 = reinterpret_tensor(buf709, (512, 768), (768, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (512, 384), (384, 1), 0), permute_716, out=buf714)
        del permute_716
        buf715 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (384, 512), (1, 384), 0), view_72, out=buf715)
        buf716 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf689, buf716, 1536, 128, grid=grid(1536), stream=stream0)
        buf717 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf716, buf717, 384, 4, grid=grid(384), stream=stream0)
        buf718 = reinterpret_tensor(buf677, (512, 768), (768, 1), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (512, 384), (384, 1), 0), permute_720, out=buf718)
        del permute_720
        buf719 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (384, 512), (1, 384), 0), view_72, out=buf719)
        buf720 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf688, buf720, 1536, 128, grid=grid(1536), stream=stream0)
        buf721 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf720, buf721, 384, 4, grid=grid(384), stream=stream0)
        buf724 = reinterpret_tensor(buf659, (512, 768), (768, 1), 0); del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf723, permute_724, out=buf724)
        del permute_724
        buf725 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (384, 512), (512, 1), 0), view_72, out=buf725)
        del view_72
        buf726 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf723, buf726, 384, 512, grid=grid(384), stream=stream0)
        buf727 = buf674; del buf674  # reuse
        buf730 = reinterpret_tensor(buf647, (1, 512, 768), (393216, 768, 1), 0); del buf647  # reuse
        buf733 = reinterpret_tensor(buf643, (1, 512, 768), (393216, 768, 1), 0); del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf727, buf697, buf712, buf714, buf718, buf724, primals_60, mul_17, div_69, getitem_21, buf730, buf733, 512, 768, grid=grid(512), stream=stream0)
        del buf697
        del buf712
        del div_69
        del getitem_21
        del primals_60
        buf731 = empty((768, ), device='cuda', dtype=torch.float32)
        buf732 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf727, mul_17, buf731, buf732, 768, 512, grid=grid(768), stream=stream0)
        del mul_17
        buf734 = reinterpret_tensor(buf667, (512, 3072), (3072, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf733, (512, 768), (768, 1), 0), permute_728, out=buf734)
        del permute_728
        buf735 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf733, (768, 512), (1, 768), 0), view_70, out=buf735)
        del view_70
        buf736 = reinterpret_tensor(buf700, (1, 768, 4), (3072, 1, 768), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf733, buf736, 3072, 128, grid=grid(3072), stream=stream0)
        buf737 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf736, buf737, 768, 4, grid=grid(768), stream=stream0)
        buf738 = reinterpret_tensor(buf734, (1, 512, 3072), (1572864, 3072, 1), 0); del buf734  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf738, addmm_12, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_12
        buf739 = reinterpret_tensor(buf733, (512, 768), (768, 1), 0); del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (512, 3072), (3072, 1), 0), permute_732, out=buf739)
        del permute_732
        buf740 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (3072, 512), (1, 3072), 0), view_68, out=buf740)
        del view_68
        buf741 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf738, buf741, 12288, 128, grid=grid(12288), stream=stream0)
        buf742 = reinterpret_tensor(buf736, (1, 3072), (3072, 1), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf741, buf742, 3072, 4, grid=grid(3072), stream=stream0)
        buf745 = buf727; del buf727  # reuse
        buf748 = reinterpret_tensor(buf724, (1, 512, 768), (393216, 768, 1), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf730, buf739, primals_54, mul_12, div_70, getitem_17, buf745, buf748, 512, 768, grid=grid(512), stream=stream0)
        del div_70
        del getitem_17
        del primals_54
        buf746 = empty((768, ), device='cuda', dtype=torch.float32)
        buf747 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf730, buf739, mul_12, buf746, buf747, 768, 512, grid=grid(768), stream=stream0)
        del mul_12
        buf749 = buf739; del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf748, (512, 768), (768, 1), 0), permute_736, out=buf749)
        del permute_736
        buf750 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf748, (768, 512), (1, 768), 0), view_66, out=buf750)
        del view_66
        buf751 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf748, buf751, 3072, 128, grid=grid(3072), stream=stream0)
        buf752 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf751, buf752, 768, 4, grid=grid(768), stream=stream0)
        buf753 = reinterpret_tensor(buf723, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_30, buf753, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_30
        buf754 = reinterpret_tensor(buf688, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_31, buf754, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_31
        buf755 = reinterpret_tensor(buf689, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf689  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_32, buf755, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_32
        buf756 = reinterpret_tensor(buf706, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_21, buf756, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_21
        # Source Nodes: [], Original ATen: []
        buf757 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf749, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf753, buf754, buf755, None, buf756, getitem_269, getitem_270, getitem_271, 0.1, [True, True, True, False], scale=0.125)
        del buf753
        del getitem_269
        del getitem_270
        del getitem_271
        buf758 = buf757[0]
        buf759 = buf757[1]
        buf760 = buf757[2]
        del buf757
        buf761 = reinterpret_tensor(buf756, (512, 384), (384, 1), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf749, buf761, 196608, grid=grid(196608), stream=stream0)
        buf762 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_746, reinterpret_tensor(buf761, (3072, 64, 1), (64, 1, 0), 0), out=buf762)
        del permute_746
        buf763 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf761, (3072, 64, 1), (64, 1, 0), 0), permute_747, out=buf763)
        del permute_747
        buf764 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf764, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf764, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf763, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        buf767 = reinterpret_tensor(buf761, (512, 384), (1, 512), 0); del buf761  # reuse
        buf770 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf764, buf767, buf770, 384, 512, grid=grid(384), stream=stream0)
        buf768 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf767, permute_751, out=buf768)
        del permute_751
        buf769 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf767, (384, 512), (512, 1), 0), view_36, out=buf769)
        buf771 = reinterpret_tensor(buf751, (3072, 1, 1), (1, 3072, 3072), 0); del buf751  # reuse
        buf773 = buf691; del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf762, alias_47, buf771, buf773, 3072, 9, grid=grid(3072), stream=stream0)
        buf772 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf762, alias_47, buf771, buf772, 54, 512, grid=grid(54), stream=stream0)
        del alias_47
        buf774 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf773, (54, 512), (1, 54), 0), view_45, out=buf774)
        del view_45
        buf775 = reinterpret_tensor(buf767, (384, 512), (512, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_28, reinterpret_tensor(buf773, (54, 512), (1, 54), 0), out=buf775)
        del permute_28
        buf776 = reinterpret_tensor(buf755, (1, 512, 384), (196608, 384, 1), 0); del buf755  # reuse
        buf793 = reinterpret_tensor(buf758, (1, 512, 384), (196608, 384, 1), 0); del buf758  # reuse
        buf794 = reinterpret_tensor(buf754, (512, 384), (1, 512), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf793, buf775, addmm_7, convolution_3, primals_2, buf776, buf794, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm_7
        del buf775
        del convolution_3
        del primals_2
        buf777 = reinterpret_tensor(buf793, (1, 384, 512), (196608, 512, 1), 0); del buf793  # reuse
        buf778 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf776, buf777, buf778, 384, 512, grid=grid(384), stream=stream0)
        del buf776
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf779 = aten.convolution_backward(buf777, convolution_2, primals_47, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del convolution_2
        del primals_47
        buf780 = buf779[0]
        buf781 = buf779[1]
        del buf779
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf782 = aten.convolution_backward(buf780, permute_22, primals_46, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_22
        del primals_46
        buf783 = buf782[0]
        buf784 = buf782[1]
        del buf782
        buf785 = reinterpret_tensor(buf780, (512, 768), (768, 1), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (512, 384), (384, 1), 0), permute_765, out=buf785)
        del permute_765
        buf786 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (384, 512), (1, 384), 0), view_36, out=buf786)
        buf787 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf760, buf787, 1536, 128, grid=grid(1536), stream=stream0)
        buf788 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf787, buf788, 384, 4, grid=grid(384), stream=stream0)
        buf789 = reinterpret_tensor(buf748, (512, 768), (768, 1), 0); del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf759, (512, 384), (384, 1), 0), permute_769, out=buf789)
        del permute_769
        buf790 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf759, (384, 512), (1, 384), 0), view_36, out=buf790)
        buf791 = buf787; del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf759, buf791, 1536, 128, grid=grid(1536), stream=stream0)
        buf792 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf791, buf792, 384, 4, grid=grid(384), stream=stream0)
        buf795 = reinterpret_tensor(buf730, (512, 768), (768, 1), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf794, permute_773, out=buf795)
        del permute_773
        buf796 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf794, (384, 512), (512, 1), 0), view_36, out=buf796)
        del view_36
        buf797 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf794, buf797, 384, 512, grid=grid(384), stream=stream0)
        buf798 = buf745; del buf745  # reuse
        buf801 = reinterpret_tensor(buf718, (1, 512, 768), (393216, 768, 1), 0); del buf718  # reuse
        buf804 = reinterpret_tensor(buf714, (1, 512, 768), (393216, 768, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_25.run(buf798, buf768, buf783, buf785, buf789, buf795, primals_38, mul_9, div_72, getitem_11, buf801, buf804, 512, 768, grid=grid(512), stream=stream0)
        del buf768
        del div_72
        del getitem_11
        del primals_38
        buf802 = empty((768, ), device='cuda', dtype=torch.float32)
        buf803 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf798, mul_9, buf802, buf803, 768, 512, grid=grid(768), stream=stream0)
        del mul_9
        buf805 = reinterpret_tensor(buf738, (512, 3072), (3072, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (512, 768), (768, 1), 0), permute_777, out=buf805)
        del permute_777
        buf806 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (768, 512), (1, 768), 0), view_34, out=buf806)
        del view_34
        buf807 = reinterpret_tensor(buf771, (1, 768, 4), (3072, 1, 768), 0); del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf804, buf807, 3072, 128, grid=grid(3072), stream=stream0)
        buf808 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf807, buf808, 768, 4, grid=grid(768), stream=stream0)
        buf809 = reinterpret_tensor(buf805, (1, 512, 3072), (1572864, 3072, 1), 0); del buf805  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf809, addmm_5, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_5
        buf810 = reinterpret_tensor(buf804, (512, 768), (768, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (512, 3072), (3072, 1), 0), permute_781, out=buf810)
        del permute_781
        buf811 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (3072, 512), (1, 3072), 0), view_32, out=buf811)
        del view_32
        buf812 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf809, buf812, 12288, 128, grid=grid(12288), stream=stream0)
        del buf809
        buf813 = reinterpret_tensor(buf807, (1, 3072), (3072, 1), 0); del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf812, buf813, 3072, 4, grid=grid(3072), stream=stream0)
        del buf812
        buf816 = buf798; del buf798  # reuse
        buf819 = reinterpret_tensor(buf795, (1, 512, 768), (393216, 768, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf801, buf810, primals_32, mul_4, div_73, getitem_7, buf816, buf819, 512, 768, grid=grid(512), stream=stream0)
        del div_73
        del getitem_7
        del primals_32
        buf817 = empty((768, ), device='cuda', dtype=torch.float32)
        buf818 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf801, buf810, mul_4, buf817, buf818, 768, 512, grid=grid(768), stream=stream0)
        del mul_4
        buf820 = buf810; del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (512, 768), (768, 1), 0), permute_785, out=buf820)
        del permute_785
        buf821 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (768, 512), (1, 768), 0), view_30, out=buf821)
        del view_30
        buf822 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf819, buf822, 3072, 128, grid=grid(3072), stream=stream0)
        buf823 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf822, buf823, 768, 4, grid=grid(768), stream=stream0)
        buf824 = reinterpret_tensor(buf794, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf794  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_33, buf824, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_33
        buf825 = reinterpret_tensor(buf759, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_34, buf825, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_34
        buf826 = reinterpret_tensor(buf760, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(clone_default_35, buf826, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del clone_default_35
        buf827 = reinterpret_tensor(buf777, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(alias_default_23, buf827, 6, 32768, grid=grid(6, 32768), stream=stream0)
        del alias_default_23
        # Source Nodes: [], Original ATen: []
        buf828 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf820, (1, 6, 512, 64), (393216, 64, 768, 1), 0), buf824, buf825, buf826, None, buf827, getitem_276, getitem_277, getitem_278, 0.1, [True, True, True, False], scale=0.125)
        del buf824
        del getitem_276
        del getitem_277
        del getitem_278
        buf829 = buf828[0]
        buf830 = buf828[1]
        buf831 = buf828[2]
        del buf828
        buf832 = reinterpret_tensor(buf827, (512, 384), (384, 1), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf820, buf832, 196608, grid=grid(196608), stream=stream0)
        buf833 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_795, reinterpret_tensor(buf832, (3072, 64, 1), (64, 1, 0), 0), out=buf833)
        del permute_795
        buf834 = buf763; del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf832, (3072, 64, 1), (64, 1, 0), 0), permute_796, out=buf834)
        del permute_796
        buf835 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_16.run(buf835, 199680, grid=grid(199680), stream=stream0)
        aten.index_put_(buf835, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf834, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
        del buf834
        del full_default_1
        del unsqueeze_8
        buf838 = reinterpret_tensor(buf832, (512, 384), (1, 512), 0); del buf832  # reuse
        buf841 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_per_fused_sum_view_17.run(buf835, buf838, buf841, 384, 512, grid=grid(384), stream=stream0)
        del buf835
        buf839 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf838, permute_800, out=buf839)
        del permute_800
        buf840 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (384, 512), (512, 1), 0), view, out=buf840)
        buf842 = reinterpret_tensor(buf822, (3072, 1, 1), (1, 3072, 3072), 0); del buf822  # reuse
        buf844 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_18.run(buf833, alias_49, buf842, buf844, 3072, 9, grid=grid(3072), stream=stream0)
        buf843 = empty((1, 1, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_19.run(buf833, alias_49, buf842, buf843, 54, 512, grid=grid(54), stream=stream0)
        del alias_49
        del buf833
        del buf842
        buf845 = empty((54, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf844, (54, 512), (1, 54), 0), view_9, out=buf845)
        del view_9
        buf846 = reinterpret_tensor(buf838, (384, 512), (512, 1), 0); del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_9, reinterpret_tensor(buf844, (54, 512), (1, 54), 0), out=buf846)
        del buf844
        del permute_9
        buf847 = reinterpret_tensor(buf826, (1, 512, 384), (196608, 384, 1), 0); del buf826  # reuse
        buf864 = reinterpret_tensor(buf829, (1, 512, 384), (196608, 384, 1), 0); del buf829  # reuse
        buf865 = reinterpret_tensor(buf825, (512, 384), (1, 512), 0); del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_20.run(buf864, buf846, addmm, convolution_1, primals_1, buf847, buf865, 512, 384, grid=grid(512, 384), stream=stream0)
        del addmm
        del buf846
        del convolution_1
        del primals_1
        buf848 = reinterpret_tensor(buf864, (1, 384, 512), (196608, 512, 1), 0); del buf864  # reuse
        buf849 = empty((1, 384, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.transpose]
        triton_per_fused_mul_sum_transpose_21.run(buf847, buf848, buf849, 384, 512, grid=grid(384), stream=stream0)
        del buf847
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf850 = aten.convolution_backward(buf848, convolution, primals_25, [0], [1], [0], [1], False, [0], 1, [True, True, False])
        del buf848
        del convolution
        del primals_25
        buf851 = buf850[0]
        buf852 = buf850[1]
        del buf850
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf853 = aten.convolution_backward(buf851, permute_3, primals_24, [0], [1], [4], [1], False, [0], 768, [True, True, False])
        del permute_3
        del primals_24
        buf854 = buf853[0]
        buf855 = buf853[1]
        del buf853
        buf856 = reinterpret_tensor(buf851, (512, 768), (768, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (512, 384), (384, 1), 0), permute_814, out=buf856)
        del permute_814
        buf857 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (384, 512), (1, 384), 0), view, out=buf857)
        buf858 = buf791; del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf831, buf858, 1536, 128, grid=grid(1536), stream=stream0)
        del buf831
        buf859 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf858, buf859, 384, 4, grid=grid(384), stream=stream0)
        buf860 = reinterpret_tensor(buf819, (512, 768), (768, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (512, 384), (384, 1), 0), permute_818, out=buf860)
        del permute_818
        buf861 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (384, 512), (1, 384), 0), view, out=buf861)
        buf862 = buf858; del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf830, buf862, 1536, 128, grid=grid(1536), stream=stream0)
        del buf830
        buf863 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf862, buf863, 384, 4, grid=grid(384), stream=stream0)
        buf866 = reinterpret_tensor(buf801, (512, 768), (768, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf865, permute_822, out=buf866)
        del permute_822
        buf867 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (384, 512), (512, 1), 0), view, out=buf867)
        del view
        buf868 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf865, buf868, 384, 512, grid=grid(384), stream=stream0)
        del buf865
        buf869 = buf816; del buf816  # reuse
        buf876 = reinterpret_tensor(buf789, (1, 512, 768), (393216, 768, 1), 0); del buf789  # reuse
        buf880 = reinterpret_tensor(buf785, (1, 512, 768), (393216, 768, 1), 0); del buf785  # reuse
        buf884 = reinterpret_tensor(buf783, (1, 512, 768), (393216, 768, 1), 0); del buf783  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_26.run(buf869, buf839, buf854, buf856, buf860, buf866, getitem_3, primals_16, mul_1, div_75, expand, slice_4, primals_290, buf876, buf880, buf884, 512, 768, grid=grid(512), stream=stream0)
        del buf839
        del buf854
        del buf856
        del buf860
        del buf866
        del div_75
        del getitem_3
        del primals_16
        buf873 = empty((768, ), device='cuda', dtype=torch.float32)
        buf874 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf869, mul_1, buf873, buf874, 768, 512, grid=grid(768), stream=stream0)
        del buf869
        del mul_1
        buf875 = reinterpret_tensor(buf862, (2, 768), (768, 1), 0); del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_27.run(buf875, 1536, grid=grid(1536), stream=stream0)
        aten.index_put_(buf875, [expand], buf876, True)
        del expand
        buf879 = reinterpret_tensor(buf876, (512, 768), (768, 1), 0); del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_28.run(buf879, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf879, [slice_4], buf880, True)
        del buf880
        del slice_4
        buf883 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_29.run(buf883, 23440896, grid=grid(23440896), stream=stream0)
        aten.index_put_(buf883, [primals_290], buf884, True)
        del buf884
        del primals_290
        return (reinterpret_tensor(buf849, (384, 1), (1, 1), 0), reinterpret_tensor(buf778, (384, 1), (1, 1), 0), reinterpret_tensor(buf707, (384, 1), (1, 1), 0), reinterpret_tensor(buf636, (384, 1), (1, 1), 0), reinterpret_tensor(buf565, (384, 1), (1, 1), 0), reinterpret_tensor(buf494, (384, 1), (1, 1), 0), reinterpret_tensor(buf423, (384, 1), (1, 1), 0), reinterpret_tensor(buf352, (384, 1), (1, 1), 0), reinterpret_tensor(buf281, (384, 1), (1, 1), 0), reinterpret_tensor(buf210, (384, 1), (1, 1), 0), reinterpret_tensor(buf139, (384, 1), (1, 1), 0), reinterpret_tensor(buf68, (384, 1), (1, 1), 0), buf883, buf879, buf875, buf873, buf874, reinterpret_tensor(buf867, (384, 768), (768, 1), 0), reinterpret_tensor(buf868, (384, ), (1, ), 0), reinterpret_tensor(buf861, (384, 768), (768, 1), 0), reinterpret_tensor(buf863, (384, ), (1, ), 0), reinterpret_tensor(buf857, (384, 768), (768, 1), 0), reinterpret_tensor(buf859, (384, ), (1, ), 0), buf855, buf852, reinterpret_tensor(buf845, (54, 384), (384, 1), 0), reinterpret_tensor(buf843, (54, ), (1, ), 0), reinterpret_tensor(buf840, (384, 768), (768, 1), 0), reinterpret_tensor(buf841, (384, ), (1, ), 0), reinterpret_tensor(buf821, (768, 768), (768, 1), 0), reinterpret_tensor(buf823, (768, ), (1, ), 0), buf817, buf818, reinterpret_tensor(buf811, (3072, 768), (768, 1), 0), reinterpret_tensor(buf813, (3072, ), (1, ), 0), reinterpret_tensor(buf806, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf808, (768, ), (1, ), 0), buf802, buf803, reinterpret_tensor(buf796, (384, 768), (768, 1), 0), reinterpret_tensor(buf797, (384, ), (1, ), 0), reinterpret_tensor(buf790, (384, 768), (768, 1), 0), reinterpret_tensor(buf792, (384, ), (1, ), 0), reinterpret_tensor(buf786, (384, 768), (768, 1), 0), reinterpret_tensor(buf788, (384, ), (1, ), 0), buf784, buf781, reinterpret_tensor(buf774, (54, 384), (384, 1), 0), reinterpret_tensor(buf772, (54, ), (1, ), 0), reinterpret_tensor(buf769, (384, 768), (768, 1), 0), reinterpret_tensor(buf770, (384, ), (1, ), 0), reinterpret_tensor(buf750, (768, 768), (768, 1), 0), reinterpret_tensor(buf752, (768, ), (1, ), 0), buf746, buf747, reinterpret_tensor(buf740, (3072, 768), (768, 1), 0), reinterpret_tensor(buf742, (3072, ), (1, ), 0), reinterpret_tensor(buf735, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf737, (768, ), (1, ), 0), buf731, buf732, reinterpret_tensor(buf725, (384, 768), (768, 1), 0), reinterpret_tensor(buf726, (384, ), (1, ), 0), reinterpret_tensor(buf719, (384, 768), (768, 1), 0), reinterpret_tensor(buf721, (384, ), (1, ), 0), reinterpret_tensor(buf715, (384, 768), (768, 1), 0), reinterpret_tensor(buf717, (384, ), (1, ), 0), buf713, buf710, reinterpret_tensor(buf703, (54, 384), (384, 1), 0), reinterpret_tensor(buf701, (54, ), (1, ), 0), reinterpret_tensor(buf698, (384, 768), (768, 1), 0), reinterpret_tensor(buf699, (384, ), (1, ), 0), reinterpret_tensor(buf679, (768, 768), (768, 1), 0), reinterpret_tensor(buf681, (768, ), (1, ), 0), buf675, buf676, reinterpret_tensor(buf669, (3072, 768), (768, 1), 0), reinterpret_tensor(buf671, (3072, ), (1, ), 0), reinterpret_tensor(buf664, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf666, (768, ), (1, ), 0), buf660, buf661, reinterpret_tensor(buf654, (384, 768), (768, 1), 0), reinterpret_tensor(buf655, (384, ), (1, ), 0), reinterpret_tensor(buf648, (384, 768), (768, 1), 0), reinterpret_tensor(buf650, (384, ), (1, ), 0), reinterpret_tensor(buf644, (384, 768), (768, 1), 0), reinterpret_tensor(buf646, (384, ), (1, ), 0), buf642, buf639, reinterpret_tensor(buf632, (54, 384), (384, 1), 0), reinterpret_tensor(buf630, (54, ), (1, ), 0), reinterpret_tensor(buf627, (384, 768), (768, 1), 0), reinterpret_tensor(buf628, (384, ), (1, ), 0), reinterpret_tensor(buf608, (768, 768), (768, 1), 0), reinterpret_tensor(buf610, (768, ), (1, ), 0), buf604, buf605, reinterpret_tensor(buf598, (3072, 768), (768, 1), 0), reinterpret_tensor(buf600, (3072, ), (1, ), 0), reinterpret_tensor(buf593, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf595, (768, ), (1, ), 0), buf589, buf590, reinterpret_tensor(buf583, (384, 768), (768, 1), 0), reinterpret_tensor(buf584, (384, ), (1, ), 0), reinterpret_tensor(buf577, (384, 768), (768, 1), 0), reinterpret_tensor(buf579, (384, ), (1, ), 0), reinterpret_tensor(buf573, (384, 768), (768, 1), 0), reinterpret_tensor(buf575, (384, ), (1, ), 0), buf571, buf568, reinterpret_tensor(buf561, (54, 384), (384, 1), 0), reinterpret_tensor(buf559, (54, ), (1, ), 0), reinterpret_tensor(buf556, (384, 768), (768, 1), 0), reinterpret_tensor(buf557, (384, ), (1, ), 0), reinterpret_tensor(buf537, (768, 768), (768, 1), 0), reinterpret_tensor(buf539, (768, ), (1, ), 0), buf533, buf534, reinterpret_tensor(buf527, (3072, 768), (768, 1), 0), reinterpret_tensor(buf529, (3072, ), (1, ), 0), reinterpret_tensor(buf522, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf524, (768, ), (1, ), 0), buf518, buf519, reinterpret_tensor(buf512, (384, 768), (768, 1), 0), reinterpret_tensor(buf513, (384, ), (1, ), 0), reinterpret_tensor(buf506, (384, 768), (768, 1), 0), reinterpret_tensor(buf508, (384, ), (1, ), 0), reinterpret_tensor(buf502, (384, 768), (768, 1), 0), reinterpret_tensor(buf504, (384, ), (1, ), 0), buf500, buf497, reinterpret_tensor(buf490, (54, 384), (384, 1), 0), reinterpret_tensor(buf488, (54, ), (1, ), 0), reinterpret_tensor(buf485, (384, 768), (768, 1), 0), reinterpret_tensor(buf486, (384, ), (1, ), 0), reinterpret_tensor(buf466, (768, 768), (768, 1), 0), reinterpret_tensor(buf468, (768, ), (1, ), 0), buf462, buf463, reinterpret_tensor(buf456, (3072, 768), (768, 1), 0), reinterpret_tensor(buf458, (3072, ), (1, ), 0), reinterpret_tensor(buf451, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf453, (768, ), (1, ), 0), buf447, buf448, reinterpret_tensor(buf441, (384, 768), (768, 1), 0), reinterpret_tensor(buf442, (384, ), (1, ), 0), reinterpret_tensor(buf435, (384, 768), (768, 1), 0), reinterpret_tensor(buf437, (384, ), (1, ), 0), reinterpret_tensor(buf431, (384, 768), (768, 1), 0), reinterpret_tensor(buf433, (384, ), (1, ), 0), buf429, buf426, reinterpret_tensor(buf419, (54, 384), (384, 1), 0), reinterpret_tensor(buf417, (54, ), (1, ), 0), reinterpret_tensor(buf414, (384, 768), (768, 1), 0), reinterpret_tensor(buf415, (384, ), (1, ), 0), reinterpret_tensor(buf395, (768, 768), (768, 1), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), buf391, buf392, reinterpret_tensor(buf385, (3072, 768), (768, 1), 0), reinterpret_tensor(buf387, (3072, ), (1, ), 0), reinterpret_tensor(buf380, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf382, (768, ), (1, ), 0), buf376, buf377, reinterpret_tensor(buf370, (384, 768), (768, 1), 0), reinterpret_tensor(buf371, (384, ), (1, ), 0), reinterpret_tensor(buf364, (384, 768), (768, 1), 0), reinterpret_tensor(buf366, (384, ), (1, ), 0), reinterpret_tensor(buf360, (384, 768), (768, 1), 0), reinterpret_tensor(buf362, (384, ), (1, ), 0), buf358, buf355, reinterpret_tensor(buf348, (54, 384), (384, 1), 0), reinterpret_tensor(buf346, (54, ), (1, ), 0), reinterpret_tensor(buf343, (384, 768), (768, 1), 0), reinterpret_tensor(buf344, (384, ), (1, ), 0), reinterpret_tensor(buf324, (768, 768), (768, 1), 0), reinterpret_tensor(buf326, (768, ), (1, ), 0), buf320, buf321, reinterpret_tensor(buf314, (3072, 768), (768, 1), 0), reinterpret_tensor(buf316, (3072, ), (1, ), 0), reinterpret_tensor(buf309, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf311, (768, ), (1, ), 0), buf305, buf306, reinterpret_tensor(buf299, (384, 768), (768, 1), 0), reinterpret_tensor(buf300, (384, ), (1, ), 0), reinterpret_tensor(buf293, (384, 768), (768, 1), 0), reinterpret_tensor(buf295, (384, ), (1, ), 0), reinterpret_tensor(buf289, (384, 768), (768, 1), 0), reinterpret_tensor(buf291, (384, ), (1, ), 0), buf287, buf284, reinterpret_tensor(buf277, (54, 384), (384, 1), 0), reinterpret_tensor(buf275, (54, ), (1, ), 0), reinterpret_tensor(buf272, (384, 768), (768, 1), 0), reinterpret_tensor(buf273, (384, ), (1, ), 0), reinterpret_tensor(buf253, (768, 768), (768, 1), 0), reinterpret_tensor(buf255, (768, ), (1, ), 0), buf249, buf250, reinterpret_tensor(buf243, (3072, 768), (768, 1), 0), reinterpret_tensor(buf245, (3072, ), (1, ), 0), reinterpret_tensor(buf238, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf240, (768, ), (1, ), 0), buf234, buf235, reinterpret_tensor(buf228, (384, 768), (768, 1), 0), reinterpret_tensor(buf229, (384, ), (1, ), 0), reinterpret_tensor(buf222, (384, 768), (768, 1), 0), reinterpret_tensor(buf224, (384, ), (1, ), 0), reinterpret_tensor(buf218, (384, 768), (768, 1), 0), reinterpret_tensor(buf220, (384, ), (1, ), 0), buf216, buf213, reinterpret_tensor(buf206, (54, 384), (384, 1), 0), reinterpret_tensor(buf204, (54, ), (1, ), 0), reinterpret_tensor(buf201, (384, 768), (768, 1), 0), reinterpret_tensor(buf202, (384, ), (1, ), 0), reinterpret_tensor(buf182, (768, 768), (768, 1), 0), reinterpret_tensor(buf184, (768, ), (1, ), 0), buf178, buf179, reinterpret_tensor(buf172, (3072, 768), (768, 1), 0), reinterpret_tensor(buf174, (3072, ), (1, ), 0), reinterpret_tensor(buf167, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), buf163, buf164, reinterpret_tensor(buf157, (384, 768), (768, 1), 0), reinterpret_tensor(buf158, (384, ), (1, ), 0), reinterpret_tensor(buf151, (384, 768), (768, 1), 0), reinterpret_tensor(buf153, (384, ), (1, ), 0), reinterpret_tensor(buf147, (384, 768), (768, 1), 0), reinterpret_tensor(buf149, (384, ), (1, ), 0), buf145, buf142, reinterpret_tensor(buf135, (54, 384), (384, 1), 0), reinterpret_tensor(buf133, (54, ), (1, ), 0), reinterpret_tensor(buf130, (384, 768), (768, 1), 0), reinterpret_tensor(buf131, (384, ), (1, ), 0), reinterpret_tensor(buf111, (768, 768), (768, 1), 0), reinterpret_tensor(buf113, (768, ), (1, ), 0), buf107, buf108, reinterpret_tensor(buf101, (3072, 768), (768, 1), 0), reinterpret_tensor(buf103, (3072, ), (1, ), 0), reinterpret_tensor(buf96, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf98, (768, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf86, (384, 768), (768, 1), 0), reinterpret_tensor(buf87, (384, ), (1, ), 0), reinterpret_tensor(buf80, (384, 768), (768, 1), 0), reinterpret_tensor(buf82, (384, ), (1, ), 0), reinterpret_tensor(buf76, (384, 768), (768, 1), 0), reinterpret_tensor(buf78, (384, ), (1, ), 0), buf74, buf71, reinterpret_tensor(buf64, (54, 384), (384, 1), 0), reinterpret_tensor(buf62, (54, ), (1, ), 0), reinterpret_tensor(buf59, (384, 768), (768, 1), 0), reinterpret_tensor(buf60, (384, ), (1, ), 0), reinterpret_tensor(buf40, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), buf36, buf37, reinterpret_tensor(buf30, (3072, 768), (768, 1), 0), reinterpret_tensor(buf32, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf27, (768, ), (1, ), 0), buf21, buf22, reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (30522, 768), (768, 1), 0), reinterpret_tensor(buf8, (30522, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_9 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    full_default_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_8 = rand_strided((9, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.int64)
    clone_default_33 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_22 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_269 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_41 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_262 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_255 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_138 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_153 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_248 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_33 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_98 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_189 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_241 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_225 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_234 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_246 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_49 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_261 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_227 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_229 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_60 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_284 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_286 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_288 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_56 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_297 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_220 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_318 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_68 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_320 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_61 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_324 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_63 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_333 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_213 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_354 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_76 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_356 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_68 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_358 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_360 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_369 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_206 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_208 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_84 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_75 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_396 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_77 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((1, 768, 512), (393216, 1, 768), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((1, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((1, 384, 512), (196608, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((384, 54), (1, 384), device='cuda:0', dtype=torch.float32)
    view_405 = rand_strided((512, 384), (1, 512), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    getitem_199 = rand_strided((1, 6, 512), (3072, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda:0', dtype=torch.float32)
    view_426 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_92 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_428 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_82 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_430 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_432 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_52 = rand_strided((512, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_457 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_479 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_491 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_528 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_532 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_536 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_569 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_573 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_581 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_589 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_599 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_622 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_626 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_630 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_634 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_638 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_648 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_649 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_653 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_671 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_679 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_683 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_697 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_698 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_702 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_716 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_720 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_724 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_728 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_732 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_736 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_747 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_751 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_769 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_773 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_777 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_781 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_785 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_795 = rand_strided((3072, 9, 64), (576, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_796 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cuda:0', dtype=torch.float32)
    permute_800 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cuda:0', dtype=torch.float32)
    permute_814 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_818 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 30522), (15627264, 30522, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, expand, slice_4, mul_1, getitem_3, view, addmm, permute_3, convolution, convolution_1, permute_9, view_9, full_default_1, unsqueeze_8, clone_default_33, clone_default_34, clone_default_35, getitem_276, getitem_277, getitem_278, alias_default_23, view_30, getitem_7, mul_4, view_32, addmm_5, view_34, getitem_11, mul_9, view_36, addmm_7, permute_22, convolution_2, convolution_3, permute_28, view_45, clone_default_30, clone_default_31, clone_default_32, getitem_269, getitem_270, getitem_271, alias_default_21, view_66, getitem_17, mul_12, view_68, addmm_12, view_70, getitem_21, mul_17, view_72, addmm_14, permute_41, convolution_4, convolution_5, permute_47, view_81, clone_default_27, clone_default_28, clone_default_29, getitem_262, getitem_263, getitem_264, alias_default_19, view_102, getitem_27, mul_20, view_104, addmm_19, view_106, getitem_31, mul_25, view_108, addmm_21, permute_60, convolution_6, convolution_7, permute_66, view_117, clone_default_24, clone_default_25, clone_default_26, getitem_255, getitem_256, getitem_257, alias_default_17, view_138, getitem_37, mul_28, view_140, addmm_26, view_142, getitem_41, mul_33, view_144, addmm_28, permute_79, convolution_8, convolution_9, permute_85, view_153, clone_default_21, clone_default_22, clone_default_23, getitem_248, getitem_249, getitem_250, alias_default_15, view_174, getitem_47, mul_36, view_176, addmm_33, view_178, getitem_51, mul_41, view_180, addmm_35, permute_98, convolution_10, convolution_11, permute_104, view_189, clone_default_18, clone_default_19, clone_default_20, getitem_241, getitem_242, getitem_243, alias_default_13, view_210, getitem_57, mul_44, view_212, addmm_40, view_214, getitem_61, mul_49, view_216, addmm_42, permute_117, convolution_12, convolution_13, permute_123, view_225, clone_default_15, clone_default_16, clone_default_17, getitem_234, getitem_235, getitem_236, alias_default_11, view_246, getitem_67, mul_52, view_248, addmm_47, view_250, getitem_71, mul_57, view_252, addmm_49, permute_136, convolution_14, convolution_15, permute_142, view_261, clone_default_12, clone_default_13, clone_default_14, getitem_227, getitem_228, getitem_229, alias_default_9, view_282, getitem_77, mul_60, view_284, addmm_54, view_286, getitem_81, mul_65, view_288, addmm_56, permute_155, convolution_16, convolution_17, permute_161, view_297, clone_default_9, clone_default_10, clone_default_11, getitem_220, getitem_221, getitem_222, alias_default_7, view_318, getitem_87, mul_68, view_320, addmm_61, view_322, getitem_91, mul_73, view_324, addmm_63, permute_174, convolution_18, convolution_19, permute_180, view_333, clone_default_6, clone_default_7, clone_default_8, getitem_213, getitem_214, getitem_215, alias_default_5, view_354, getitem_97, mul_76, view_356, addmm_68, view_358, getitem_101, mul_81, view_360, addmm_70, permute_193, convolution_20, convolution_21, permute_199, view_369, clone_default_3, clone_default_4, clone_default_5, getitem_206, getitem_207, getitem_208, alias_default_3, view_390, getitem_107, mul_84, view_392, addmm_75, view_394, getitem_111, mul_89, view_396, addmm_77, permute_212, convolution_22, convolution_23, permute_218, view_405, clone_default, clone_default_1, clone_default_2, getitem_199, getitem_200, getitem_201, alias_default_1, view_426, getitem_117, mul_92, view_428, addmm_82, view_430, getitem_121, mul_97, view_432, addmm_84, mul_102, view_434, sub_52, convert_element_type, permute_230, div_38, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_257, permute_261, alias_27, permute_275, permute_279, permute_283, div_42, permute_287, permute_291, div_43, permute_295, permute_305, permute_306, permute_310, alias_29, permute_324, permute_328, permute_332, div_45, permute_336, permute_340, div_46, permute_344, permute_354, permute_355, permute_359, alias_31, permute_373, permute_377, permute_381, div_48, permute_385, permute_389, div_49, permute_393, permute_403, permute_404, permute_408, alias_33, permute_422, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_452, permute_453, permute_457, alias_35, permute_471, permute_475, permute_479, div_54, permute_483, permute_487, div_55, permute_491, permute_501, permute_502, permute_506, alias_37, permute_520, permute_524, permute_528, div_57, permute_532, permute_536, div_58, permute_540, permute_550, permute_551, permute_555, alias_39, permute_569, permute_573, permute_577, div_60, permute_581, permute_585, div_61, permute_589, permute_599, permute_600, permute_604, alias_41, permute_618, permute_622, permute_626, div_63, permute_630, permute_634, div_64, permute_638, permute_648, permute_649, permute_653, alias_43, permute_667, permute_671, permute_675, div_66, permute_679, permute_683, div_67, permute_687, permute_697, permute_698, permute_702, alias_45, permute_716, permute_720, permute_724, div_69, permute_728, permute_732, div_70, permute_736, permute_746, permute_747, permute_751, alias_47, permute_765, permute_769, permute_773, div_72, permute_777, permute_781, div_73, permute_785, permute_795, permute_796, permute_800, alias_49, permute_814, permute_818, permute_822, div_75, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
