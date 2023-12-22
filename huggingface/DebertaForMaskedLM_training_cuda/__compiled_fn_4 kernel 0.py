
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnu72rwb6yxvjibsbh7ckihv67pmm2mesmzry2gs3jhrlvfb3eza.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_86
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25735680
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
# loss => full_default_86
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


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzduxc5l4d7dymxqrulc2bu2beulyy2mbysl4ijkxkrew6dkhjk.py
# Source Nodes: [query_states], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.masked_fill, aten.nll_loss_backward]
# query_states => full_default_1
triton_red_fused__log_softmax_backward_data_add_masked_fill_nll_loss_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_masked_fill_nll_loss_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp15 = tl.load(in_ptr4 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr1 + (r1 + (50265*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz25inbd7pplbr67prd5pllqe56csj56ho4u5sxlslihmfdbdgz2.py
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (x0 + (50265*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkpnwbrxhtprzbeskjsvcewrjkjck4lhud3yar2z5yhxiyfz623.py
# Source Nodes: [hidden_states_184], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# hidden_states_184 => add_111, erf_12, mul_113
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


# kernel path: /tmp/torchinductor_youkaichao/42/c423j2jt5ug27hwe4wwnmrxgqaiyqehpolifb743yvdz6ixdnzc6.py
# Source Nodes: [hidden_states_179], Original ATen: [aten.div, aten.mul, aten.sum]
# hidden_states_179 => div_48
triton_per_fused_div_mul_sum_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_sum_8', 'mutated_arg_names': []}
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
    tmp5 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aa/caaww45wmz2tms2yh45lebovgkgspqhkbuezbgouhmqtlwvr6pu2.py
# Source Nodes: [hidden_states_179, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_179 => div_48
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp2 / tmp5
    tmp14 = -tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 2.0
    tmp20 = tmp5 * tmp19
    tmp21 = tmp12 / tmp20
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = tmp4 * tmp19
    tmp25 = tmp23 * tmp24
    tmp26 = -tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp13 + tmp25
    tmp32 = tmp18 + tmp30
    tmp33 = tmp32 / tmp22
    tmp34 = tmp31 + tmp33
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp34)
    tmp38 = 1.1111111111111112
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp34, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdyqxf4rmyjejgfu7z4q6jxahwezy2r7d6dxqchoqxbu7uqcof7.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_107, erf_11, mul_108
triton_poi_fused_gelu_gelu_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/kt/cktrwvgygrd3kz7supv7x5233c2dosxfo4igb4bfss2q2fngjluh.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvjnsn3ioyupe24akxslz5ldqeoeqr7qq4mfbzepegr6qarsqh2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sc/csc5woggn6ndaej5ekrb3jsbdyzppd23wlub47vwakovpc7b5zx5.py
# Source Nodes: [hidden_states_171], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# hidden_states_171 => div_47
triton_per_fused_add_div_mul_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 / tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6hqwqjxqwvybun6zbvqh5q6drvd2iftafwnql7ojw5rk5x75mj.py
# Source Nodes: [hidden_states_171, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_171 => div_47
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp6 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = -tmp4
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 / tmp7
    tmp10 = tmp5 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp4 / tmp7
    tmp16 = -tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 2.0
    tmp22 = tmp7 * tmp21
    tmp23 = tmp14 / tmp22
    tmp24 = 768.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp6 * tmp21
    tmp27 = tmp25 * tmp26
    tmp28 = -tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp15 + tmp27
    tmp34 = tmp20 + tmp32
    tmp35 = tmp34 / tmp24
    tmp36 = tmp33 + tmp35
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp36)
    tmp40 = 1.1111111111111112
    tmp41 = tmp39 * tmp40
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp36, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2tou77evqfubdkukpbk6566oqaa2x7hjh2fvtvvyu34w6jvo7x.py
# Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
# query_states => full_default_1
triton_per_fused__softmax_backward_data_masked_fill_mul_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_masked_fill_mul_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 6144
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybctjqyzarxrgn7pjj4odgui7eoalg5ra4evfuwsggecsvwj7od.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7fcvzdzry426gmhgvhjfzhhjxklmvm4rl5woas4delgibqfdid.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
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
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (256*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpkn5oyn7m3dxmwzkofavcpjtr5gino7mjl3thmyrockk4l75ta.py
# Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
# scale => full_default_2
triton_red_fused_div_sqrt_sum_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_sqrt_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszdchk56rnh4u2dmyww7r6u3tyvb2aq4dibbyapljtjfjzbsjwr.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 8.0
    tmp7 = tmp5 / tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-32768) + y0 + (512*x2) + (32768*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 192, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-128) + x2 + (64*y3)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x2 + (192*y1) + (2304*y0)), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicumaamk5i3zrzqzproyxumoi6siabvyzla3cskefpv33aavca2.py
# Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.sum]
# hidden_states_1 => div
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = tmp12 / tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjw53ldspl6akidxe62da2f5f5gn2fa3zcqnpu2laoqaa2yywsz.py
# Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_1 => div
# query_states => full_default_1
triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr5, out_ptr6, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 / tmp10
    tmp12 = -tmp9
    tmp14 = tmp13 / tmp10
    tmp15 = tmp14 / tmp10
    tmp16 = tmp12 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 2.0
    tmp22 = tmp10 * tmp21
    tmp23 = tmp20 / tmp22
    tmp24 = 768.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp13 * tmp21
    tmp27 = tmp25 * tmp26
    tmp28 = -tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = -tmp11
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp11 + tmp27
    tmp39 = tmp37 + tmp32
    tmp40 = tmp39 / tmp24
    tmp41 = tmp38 + tmp40
    tmp43 = tl.full([1], -1, tl.int64)
    tmp44 = tmp42 == tmp43
    tmp45 = tl.where(tmp44, tmp4, tmp41)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tmp46 == tmp47
    tmp49 = tl.where(tmp48, tmp4, tmp41)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp45, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c373wqik4ocfgmyp6rxsgpmcurd4cvgqmydqheoiplfkz4xuk4xm.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_22', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwah6st6ce6mpz4jdjqugpzr2qnhoy74cntiu6b7t3e6xbunlse.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_23', 'mutated_arg_names': []},
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
    primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_163, primals_168, primals_169, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, addmm_36, mul_115, view_218, sub_101, convert_element_type_49, permute_147, div_51, permute_151, permute_155, permute_159, permute_163, permute_168, permute_169, alias_43, permute_170, permute_171, permute_178, permute_180, permute_184, permute_188, permute_193, permute_194, alias_48, permute_195, permute_196, permute_203, permute_205, permute_209, permute_213, permute_218, permute_219, alias_53, permute_220, permute_221, permute_228, permute_230, permute_234, permute_238, permute_243, permute_244, alias_58, permute_245, permute_246, permute_253, permute_255, permute_259, permute_263, permute_268, permute_269, alias_63, permute_270, permute_271, permute_278, permute_280, permute_284, permute_288, permute_293, permute_294, alias_68, permute_295, permute_296, permute_303, permute_305, permute_309, permute_313, permute_318, permute_319, alias_73, permute_320, permute_321, permute_328, permute_330, permute_334, permute_338, permute_343, permute_344, alias_78, permute_345, permute_346, permute_353, permute_355, permute_359, permute_363, permute_368, permute_369, alias_83, permute_370, permute_371, permute_378, permute_380, permute_384, permute_388, permute_393, permute_394, alias_88, permute_395, permute_396, permute_403, permute_405, permute_409, permute_413, permute_418, permute_419, alias_93, permute_420, permute_421, permute_428, permute_430, permute_434, permute_438, permute_443, permute_444, alias_98, permute_445, permute_446, permute_453, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_168, (1, 512), (512, 1))
    assert_size_stride(primals_169, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(sub, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt, (1, 512, 1), (512, 1, 1))
    assert_size_stride(convert_element_type, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_2, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_12, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_6, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_2, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_6, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_10, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_48, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(view_52, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_54, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_14, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_30, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_18, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_86, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(view_88, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_90, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_22, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_46, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_26, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_120, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_54, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_122, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_124, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_30, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_156, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_35, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_70, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_158, (512, 768), (768, 1))
    assert_size_stride(addmm_25, (512, 3072), (3072, 1))
    assert_size_stride(view_160, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_162, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_38, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_39, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_42, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_86, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_31, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_46, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_94, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_35, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_36, (512, 768), (768, 1))
    assert_size_stride(mul_115, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_218, (512, 768), (768, 1))
    assert_size_stride(sub_101, (512, 50265), (50265, 1))
    assert_size_stride(convert_element_type_49, (), ())
    assert_size_stride(permute_147, (50265, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_155, (768, 3072), (3072, 1))
    assert_size_stride(permute_159, (3072, 768), (768, 1))
    assert_size_stride(permute_163, (768, 768), (768, 1))
    assert_size_stride(permute_168, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_169, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_43, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_170, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_171, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_178, (2304, 768), (768, 1))
    assert_size_stride(permute_180, (768, 3072), (3072, 1))
    assert_size_stride(permute_184, (3072, 768), (768, 1))
    assert_size_stride(permute_188, (768, 768), (768, 1))
    assert_size_stride(permute_193, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_194, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_48, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_195, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_196, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_203, (2304, 768), (768, 1))
    assert_size_stride(permute_205, (768, 3072), (3072, 1))
    assert_size_stride(permute_209, (3072, 768), (768, 1))
    assert_size_stride(permute_213, (768, 768), (768, 1))
    assert_size_stride(permute_218, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_219, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_53, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_220, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_221, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_228, (2304, 768), (768, 1))
    assert_size_stride(permute_230, (768, 3072), (3072, 1))
    assert_size_stride(permute_234, (3072, 768), (768, 1))
    assert_size_stride(permute_238, (768, 768), (768, 1))
    assert_size_stride(permute_243, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_244, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_58, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_245, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_246, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_253, (2304, 768), (768, 1))
    assert_size_stride(permute_255, (768, 3072), (3072, 1))
    assert_size_stride(permute_259, (3072, 768), (768, 1))
    assert_size_stride(permute_263, (768, 768), (768, 1))
    assert_size_stride(permute_268, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_269, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_63, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_270, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_271, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_278, (2304, 768), (768, 1))
    assert_size_stride(permute_280, (768, 3072), (3072, 1))
    assert_size_stride(permute_284, (3072, 768), (768, 1))
    assert_size_stride(permute_288, (768, 768), (768, 1))
    assert_size_stride(permute_293, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_294, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_68, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_295, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_296, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_303, (2304, 768), (768, 1))
    assert_size_stride(permute_305, (768, 3072), (3072, 1))
    assert_size_stride(permute_309, (3072, 768), (768, 1))
    assert_size_stride(permute_313, (768, 768), (768, 1))
    assert_size_stride(permute_318, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_319, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_73, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_320, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_321, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_328, (2304, 768), (768, 1))
    assert_size_stride(permute_330, (768, 3072), (3072, 1))
    assert_size_stride(permute_334, (3072, 768), (768, 1))
    assert_size_stride(permute_338, (768, 768), (768, 1))
    assert_size_stride(permute_343, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_344, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_78, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_345, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_346, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_353, (2304, 768), (768, 1))
    assert_size_stride(permute_355, (768, 3072), (3072, 1))
    assert_size_stride(permute_359, (3072, 768), (768, 1))
    assert_size_stride(permute_363, (768, 768), (768, 1))
    assert_size_stride(permute_368, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_369, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_83, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_370, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_371, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_378, (2304, 768), (768, 1))
    assert_size_stride(permute_380, (768, 3072), (3072, 1))
    assert_size_stride(permute_384, (3072, 768), (768, 1))
    assert_size_stride(permute_388, (768, 768), (768, 1))
    assert_size_stride(permute_393, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_394, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_88, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_395, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_396, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_403, (2304, 768), (768, 1))
    assert_size_stride(permute_405, (768, 3072), (3072, 1))
    assert_size_stride(permute_409, (3072, 768), (768, 1))
    assert_size_stride(permute_413, (768, 768), (768, 1))
    assert_size_stride(permute_418, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_419, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_93, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_420, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_421, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_428, (2304, 768), (768, 1))
    assert_size_stride(permute_430, (768, 3072), (3072, 1))
    assert_size_stride(permute_434, (3072, 768), (768, 1))
    assert_size_stride(permute_438, (768, 768), (768, 1))
    assert_size_stride(permute_443, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_444, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_98, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_445, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_446, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_453, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 50265), (25735680, 50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 25735680, grid=grid(25735680), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_169, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_states], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.masked_fill, aten.nll_loss_backward]
        triton_red_fused__log_softmax_backward_data_add_masked_fill_nll_loss_backward_2.run(buf0, primals_169, tangents_1, convert_element_type_49, tangents_2, sub_101, buf5, 512, 50265, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type_49
        del primals_169
        del sub_101
        del tangents_1
        del tangents_2
        buf6 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 50265), (50265, 1), 0), permute_147, out=buf6)
        del permute_147
        buf7 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (50265, 512), (1, 50265), 0), view_218, out=buf7)
        del view_218
        buf8 = empty((1, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 50265, 512, grid=grid(50265), stream=stream0)
        del buf5
        buf13 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_184], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf6, primals_163, mul_115, div_51, addmm_36, buf13, 512, 768, grid=grid(512), stream=stream0)
        del addmm_36
        del div_51
        del primals_163
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf6, mul_115, buf11, buf12, 768, 512, grid=grid(768), stream=stream0)
        del mul_115
        buf14 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_151, out=buf14)
        del permute_151
        buf15 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_216, out=buf15)
        del view_216
        buf16 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf13, buf16, 3072, 128, grid=grid(3072), stream=stream0)
        buf17 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf16, buf17, 768, 4, grid=grid(768), stream=stream0)
        buf18 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf19 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_179], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_8.run(buf14, sub_97, sqrt_36, buf18, buf19, 768, 512, grid=grid(768), stream=stream0)
        buf23 = buf13; del buf13  # reuse
        buf24 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_179, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9.run(buf14, primals_73, sub_97, sqrt_36, convert_element_type_48, buf23, buf24, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_48
        del primals_73
        del sqrt_36
        del sub_97
        buf25 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 768), (768, 1), 0), permute_155, out=buf25)
        del permute_155
        buf26 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (768, 512), (1, 768), 0), view_214, out=buf26)
        del view_214
        buf27 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf24, buf27, 3072, 128, grid=grid(3072), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf27, buf28, 768, 4, grid=grid(768), stream=stream0)
        buf29 = reinterpret_tensor(buf25, (1, 512, 3072), (1572864, 3072, 1), 0); del buf25  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf29, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf30 = reinterpret_tensor(buf24, (512, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0), permute_159, out=buf30)
        del permute_159
        buf31 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (3072, 512), (1, 3072), 0), view_212, out=buf31)
        del view_212
        buf32 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf29, buf32, 12288, 128, grid=grid(12288), stream=stream0)
        buf33 = reinterpret_tensor(buf27, (1, 3072), (3072, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf32, buf33, 3072, 4, grid=grid(3072), stream=stream0)
        buf34 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf35 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf23, buf30, sub_94, sqrt_35, buf34, buf35, 768, 512, grid=grid(768), stream=stream0)
        buf39 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
        buf40 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf23, buf30, primals_71, sub_94, sqrt_35, convert_element_type_47, buf39, buf40, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_47
        del primals_71
        del sqrt_35
        del sub_94
        buf41 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 768), (768, 1), 0), permute_163, out=buf41)
        del permute_163
        buf42 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (768, 512), (1, 768), 0), view_210, out=buf42)
        del view_210
        buf43 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf40, buf43, 3072, 128, grid=grid(3072), stream=stream0)
        buf44 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf43, buf44, 768, 4, grid=grid(768), stream=stream0)
        buf45 = reinterpret_tensor(buf40, (12, 512, 64), (32768, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_168, reinterpret_tensor(buf41, (12, 512, 64), (64, 768, 1), 0), out=buf45)
        del permute_168
        buf46 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (12, 512, 64), (64, 768, 1), 0), permute_169, out=buf46)
        del permute_169
        buf48 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_46, buf46, alias_43, buf48, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_43
        del convert_element_type_46
        buf49 = reinterpret_tensor(buf41, (12, 64, 512), (32768, 512, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_170, reinterpret_tensor(buf48, (12, 512, 512), (262144, 512, 1), 0), out=buf49)
        del permute_170
        buf50 = reinterpret_tensor(buf23, (12, 512, 64), (32768, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (12, 512, 512), (262144, 512, 1), 0), permute_171, out=buf50)
        del permute_171
        buf51 = reinterpret_tensor(buf43, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf45, buf51, 3072, 128, grid=grid(3072), stream=stream0)
        buf52 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf51, buf52, 768, 4, grid=grid(768), stream=stream0)
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf50, buf53, 3072, 128, grid=grid(3072), stream=stream0)
        buf54 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf53, buf54, 768, 4, grid=grid(768), stream=stream0)
        buf55 = empty((1, 512, 12, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf50, buf49, buf45, buf55, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf56 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (2304, 512), (1, 2304), 0), view_198, out=buf56)
        del view_198
        buf57 = reinterpret_tensor(buf50, (512, 768), (768, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (512, 2304), (2304, 1), 0), permute_178, out=buf57)
        del permute_178
        buf58 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf59 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_164], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf39, buf57, sub_89, sqrt_33, buf58, buf59, 768, 512, grid=grid(768), stream=stream0)
        buf63 = reinterpret_tensor(buf49, (1, 512, 768), (393216, 768, 1), 0); del buf49  # reuse
        buf64 = reinterpret_tensor(buf45, (1, 512, 768), (393216, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_164, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf39, buf57, primals_67, sub_89, sqrt_33, convert_element_type_44, buf63, buf64, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_44
        del primals_67
        del sqrt_33
        del sub_89
        buf65 = reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 768), (768, 1), 0), permute_180, out=buf65)
        del permute_180
        buf66 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (768, 512), (1, 768), 0), view_196, out=buf66)
        del view_196
        buf67 = reinterpret_tensor(buf53, (1, 768, 4), (3072, 1, 768), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf64, buf67, 3072, 128, grid=grid(3072), stream=stream0)
        buf68 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf67, buf68, 768, 4, grid=grid(768), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (1, 512, 3072), (1572864, 3072, 1), 0); del buf65  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf69, addmm_31, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_31
        buf70 = reinterpret_tensor(buf64, (512, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), permute_184, out=buf70)
        del permute_184
        buf71 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (3072, 512), (1, 3072), 0), view_194, out=buf71)
        del view_194
        buf72 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf69, buf72, 12288, 128, grid=grid(12288), stream=stream0)
        buf73 = reinterpret_tensor(buf67, (1, 3072), (3072, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf72, buf73, 3072, 4, grid=grid(3072), stream=stream0)
        buf74 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf75 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf63, buf70, sub_86, sqrt_32, buf74, buf75, 768, 512, grid=grid(768), stream=stream0)
        buf79 = reinterpret_tensor(buf57, (1, 512, 768), (393216, 768, 1), 0); del buf57  # reuse
        buf80 = buf39; del buf39  # reuse
        # Source Nodes: [hidden_states_156, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf63, buf70, primals_65, sub_86, sqrt_32, convert_element_type_43, buf79, buf80, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_43
        del primals_65
        del sqrt_32
        del sub_86
        buf81 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (512, 768), (768, 1), 0), permute_188, out=buf81)
        del permute_188
        buf82 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (768, 512), (1, 768), 0), view_192, out=buf82)
        del view_192
        buf83 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf80, buf83, 3072, 128, grid=grid(3072), stream=stream0)
        buf84 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf83, buf84, 768, 4, grid=grid(768), stream=stream0)
        buf85 = reinterpret_tensor(buf80, (12, 512, 64), (32768, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_193, reinterpret_tensor(buf81, (12, 512, 64), (64, 768, 1), 0), out=buf85)
        del permute_193
        buf86 = reinterpret_tensor(buf48, (12, 512, 512), (262144, 512, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf81, (12, 512, 64), (64, 768, 1), 0), permute_194, out=buf86)
        del permute_194
        buf88 = reinterpret_tensor(buf46, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf46  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_42, buf86, alias_48, buf88, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_48
        del convert_element_type_42
        buf89 = reinterpret_tensor(buf81, (12, 64, 512), (32768, 512, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_195, reinterpret_tensor(buf88, (12, 512, 512), (262144, 512, 1), 0), out=buf89)
        del permute_195
        buf90 = reinterpret_tensor(buf63, (12, 512, 64), (32768, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (12, 512, 512), (262144, 512, 1), 0), permute_196, out=buf90)
        del permute_196
        buf91 = reinterpret_tensor(buf83, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf85, buf91, 3072, 128, grid=grid(3072), stream=stream0)
        buf92 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf91, buf92, 768, 4, grid=grid(768), stream=stream0)
        buf93 = buf91; del buf91  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf90, buf93, 3072, 128, grid=grid(3072), stream=stream0)
        buf94 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf93, buf94, 768, 4, grid=grid(768), stream=stream0)
        buf95 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf90, buf89, buf85, buf95, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf96 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (2304, 512), (1, 2304), 0), view_180, out=buf96)
        del view_180
        buf97 = reinterpret_tensor(buf90, (512, 768), (768, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (512, 2304), (2304, 1), 0), permute_203, out=buf97)
        del permute_203
        buf98 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf99 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_149], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf79, buf97, sub_81, sqrt_30, buf98, buf99, 768, 512, grid=grid(768), stream=stream0)
        buf103 = reinterpret_tensor(buf89, (1, 512, 768), (393216, 768, 1), 0); del buf89  # reuse
        buf104 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
        # Source Nodes: [hidden_states_149, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf79, buf97, primals_61, sub_81, sqrt_30, convert_element_type_40, buf103, buf104, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_40
        del primals_61
        del sqrt_30
        del sub_81
        buf105 = reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 768), (768, 1), 0), permute_205, out=buf105)
        del permute_205
        buf106 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (768, 512), (1, 768), 0), view_178, out=buf106)
        del view_178
        buf107 = reinterpret_tensor(buf93, (1, 768, 4), (3072, 1, 768), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf104, buf107, 3072, 128, grid=grid(3072), stream=stream0)
        buf108 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf107, buf108, 768, 4, grid=grid(768), stream=stream0)
        buf109 = reinterpret_tensor(buf105, (1, 512, 3072), (1572864, 3072, 1), 0); del buf105  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf109, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf110 = reinterpret_tensor(buf104, (512, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 3072), (3072, 1), 0), permute_209, out=buf110)
        del permute_209
        buf111 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (3072, 512), (1, 3072), 0), view_176, out=buf111)
        del view_176
        buf112 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf109, buf112, 12288, 128, grid=grid(12288), stream=stream0)
        buf113 = reinterpret_tensor(buf107, (1, 3072), (3072, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf112, buf113, 3072, 4, grid=grid(3072), stream=stream0)
        buf114 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf115 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_141], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf103, buf110, sub_78, sqrt_29, buf114, buf115, 768, 512, grid=grid(768), stream=stream0)
        buf119 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        buf120 = buf79; del buf79  # reuse
        # Source Nodes: [hidden_states_141, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf103, buf110, primals_59, sub_78, sqrt_29, convert_element_type_39, buf119, buf120, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_39
        del primals_59
        del sqrt_29
        del sub_78
        buf121 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 768), (768, 1), 0), permute_213, out=buf121)
        del permute_213
        buf122 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (768, 512), (1, 768), 0), view_174, out=buf122)
        del view_174
        buf123 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf120, buf123, 3072, 128, grid=grid(3072), stream=stream0)
        buf124 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf123, buf124, 768, 4, grid=grid(768), stream=stream0)
        buf125 = reinterpret_tensor(buf120, (12, 512, 64), (32768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_218, reinterpret_tensor(buf121, (12, 512, 64), (64, 768, 1), 0), out=buf125)
        del permute_218
        buf126 = reinterpret_tensor(buf88, (12, 512, 512), (262144, 512, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (12, 512, 64), (64, 768, 1), 0), permute_219, out=buf126)
        del permute_219
        buf128 = reinterpret_tensor(buf86, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf86  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_38, buf126, alias_53, buf128, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_53
        del convert_element_type_38
        buf129 = reinterpret_tensor(buf121, (12, 64, 512), (32768, 512, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_220, reinterpret_tensor(buf128, (12, 512, 512), (262144, 512, 1), 0), out=buf129)
        del permute_220
        buf130 = reinterpret_tensor(buf103, (12, 512, 64), (32768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (12, 512, 512), (262144, 512, 1), 0), permute_221, out=buf130)
        del permute_221
        buf131 = reinterpret_tensor(buf123, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf125, buf131, 3072, 128, grid=grid(3072), stream=stream0)
        buf132 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf131, buf132, 768, 4, grid=grid(768), stream=stream0)
        buf133 = buf131; del buf131  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf130, buf133, 3072, 128, grid=grid(3072), stream=stream0)
        buf134 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf133, buf134, 768, 4, grid=grid(768), stream=stream0)
        buf135 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf130, buf129, buf125, buf135, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf136 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (2304, 512), (1, 2304), 0), view_162, out=buf136)
        del view_162
        buf137 = reinterpret_tensor(buf130, (512, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (512, 2304), (2304, 1), 0), permute_228, out=buf137)
        del permute_228
        buf138 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf139 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf119, buf137, sub_73, sqrt_27, buf138, buf139, 768, 512, grid=grid(768), stream=stream0)
        buf143 = reinterpret_tensor(buf129, (1, 512, 768), (393216, 768, 1), 0); del buf129  # reuse
        buf144 = reinterpret_tensor(buf125, (1, 512, 768), (393216, 768, 1), 0); del buf125  # reuse
        # Source Nodes: [hidden_states_134, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf119, buf137, primals_55, sub_73, sqrt_27, convert_element_type_36, buf143, buf144, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_36
        del primals_55
        del sqrt_27
        del sub_73
        buf145 = reinterpret_tensor(buf109, (512, 3072), (3072, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), permute_230, out=buf145)
        del permute_230
        buf146 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (768, 512), (1, 768), 0), view_160, out=buf146)
        del view_160
        buf147 = reinterpret_tensor(buf133, (1, 768, 4), (3072, 1, 768), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf144, buf147, 3072, 128, grid=grid(3072), stream=stream0)
        buf148 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf147, buf148, 768, 4, grid=grid(768), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (1, 512, 3072), (1572864, 3072, 1), 0); del buf145  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf149, addmm_25, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_25
        buf150 = reinterpret_tensor(buf144, (512, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0), permute_234, out=buf150)
        del permute_234
        buf151 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (3072, 512), (1, 3072), 0), view_158, out=buf151)
        del view_158
        buf152 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf149, buf152, 12288, 128, grid=grid(12288), stream=stream0)
        buf153 = reinterpret_tensor(buf147, (1, 3072), (3072, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf152, buf153, 3072, 4, grid=grid(3072), stream=stream0)
        buf154 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf155 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_126], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf143, buf150, sub_70, sqrt_26, buf154, buf155, 768, 512, grid=grid(768), stream=stream0)
        buf159 = reinterpret_tensor(buf137, (1, 512, 768), (393216, 768, 1), 0); del buf137  # reuse
        buf160 = buf119; del buf119  # reuse
        # Source Nodes: [hidden_states_126, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf143, buf150, primals_53, sub_70, sqrt_26, convert_element_type_35, buf159, buf160, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_35
        del primals_53
        del sqrt_26
        del sub_70
        buf161 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 768), (768, 1), 0), permute_238, out=buf161)
        del permute_238
        buf162 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (768, 512), (1, 768), 0), view_156, out=buf162)
        del view_156
        buf163 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf160, buf163, 3072, 128, grid=grid(3072), stream=stream0)
        buf164 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf163, buf164, 768, 4, grid=grid(768), stream=stream0)
        buf165 = reinterpret_tensor(buf160, (12, 512, 64), (32768, 64, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_243, reinterpret_tensor(buf161, (12, 512, 64), (64, 768, 1), 0), out=buf165)
        del permute_243
        buf166 = reinterpret_tensor(buf128, (12, 512, 512), (262144, 512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (12, 512, 64), (64, 768, 1), 0), permute_244, out=buf166)
        del permute_244
        buf168 = reinterpret_tensor(buf126, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf126  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_34, buf166, alias_58, buf168, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_58
        del convert_element_type_34
        buf169 = reinterpret_tensor(buf161, (12, 64, 512), (32768, 512, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_245, reinterpret_tensor(buf168, (12, 512, 512), (262144, 512, 1), 0), out=buf169)
        del permute_245
        buf170 = reinterpret_tensor(buf143, (12, 512, 64), (32768, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (12, 512, 512), (262144, 512, 1), 0), permute_246, out=buf170)
        del permute_246
        buf171 = reinterpret_tensor(buf163, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf165, buf171, 3072, 128, grid=grid(3072), stream=stream0)
        buf172 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf171, buf172, 768, 4, grid=grid(768), stream=stream0)
        buf173 = buf171; del buf171  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf170, buf173, 3072, 128, grid=grid(3072), stream=stream0)
        buf174 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf173, buf174, 768, 4, grid=grid(768), stream=stream0)
        buf175 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf170, buf169, buf165, buf175, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf176 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (2304, 512), (1, 2304), 0), view_144, out=buf176)
        del view_144
        buf177 = reinterpret_tensor(buf170, (512, 768), (768, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (512, 2304), (2304, 1), 0), permute_253, out=buf177)
        del permute_253
        buf178 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf179 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_119], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf159, buf177, sub_65, sqrt_24, buf178, buf179, 768, 512, grid=grid(768), stream=stream0)
        buf183 = reinterpret_tensor(buf169, (1, 512, 768), (393216, 768, 1), 0); del buf169  # reuse
        buf184 = reinterpret_tensor(buf165, (1, 512, 768), (393216, 768, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_119, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf159, buf177, primals_49, sub_65, sqrt_24, convert_element_type_32, buf183, buf184, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_32
        del primals_49
        del sqrt_24
        del sub_65
        buf185 = reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (512, 768), (768, 1), 0), permute_255, out=buf185)
        del permute_255
        buf186 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (768, 512), (1, 768), 0), view_142, out=buf186)
        del view_142
        buf187 = reinterpret_tensor(buf173, (1, 768, 4), (3072, 1, 768), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf184, buf187, 3072, 128, grid=grid(3072), stream=stream0)
        buf188 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf187, buf188, 768, 4, grid=grid(768), stream=stream0)
        buf189 = reinterpret_tensor(buf185, (1, 512, 3072), (1572864, 3072, 1), 0); del buf185  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf189, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf190 = reinterpret_tensor(buf184, (512, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 3072), (3072, 1), 0), permute_259, out=buf190)
        del permute_259
        buf191 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (3072, 512), (1, 3072), 0), view_140, out=buf191)
        del view_140
        buf192 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf189, buf192, 12288, 128, grid=grid(12288), stream=stream0)
        buf193 = reinterpret_tensor(buf187, (1, 3072), (3072, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf192, buf193, 3072, 4, grid=grid(3072), stream=stream0)
        buf194 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf195 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_111], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf183, buf190, sub_62, sqrt_23, buf194, buf195, 768, 512, grid=grid(768), stream=stream0)
        buf199 = reinterpret_tensor(buf177, (1, 512, 768), (393216, 768, 1), 0); del buf177  # reuse
        buf200 = buf159; del buf159  # reuse
        # Source Nodes: [hidden_states_111, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf183, buf190, primals_47, sub_62, sqrt_23, convert_element_type_31, buf199, buf200, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_31
        del primals_47
        del sqrt_23
        del sub_62
        buf201 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (512, 768), (768, 1), 0), permute_263, out=buf201)
        del permute_263
        buf202 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (768, 512), (1, 768), 0), view_138, out=buf202)
        del view_138
        buf203 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf200, buf203, 3072, 128, grid=grid(3072), stream=stream0)
        buf204 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf203, buf204, 768, 4, grid=grid(768), stream=stream0)
        buf205 = reinterpret_tensor(buf200, (12, 512, 64), (32768, 64, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_268, reinterpret_tensor(buf201, (12, 512, 64), (64, 768, 1), 0), out=buf205)
        del permute_268
        buf206 = reinterpret_tensor(buf168, (12, 512, 512), (262144, 512, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (12, 512, 64), (64, 768, 1), 0), permute_269, out=buf206)
        del permute_269
        buf208 = reinterpret_tensor(buf166, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf166  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_30, buf206, alias_63, buf208, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_63
        del convert_element_type_30
        buf209 = reinterpret_tensor(buf201, (12, 64, 512), (32768, 512, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_270, reinterpret_tensor(buf208, (12, 512, 512), (262144, 512, 1), 0), out=buf209)
        del permute_270
        buf210 = reinterpret_tensor(buf183, (12, 512, 64), (32768, 64, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (12, 512, 512), (262144, 512, 1), 0), permute_271, out=buf210)
        del permute_271
        buf211 = reinterpret_tensor(buf203, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf205, buf211, 3072, 128, grid=grid(3072), stream=stream0)
        buf212 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf211, buf212, 768, 4, grid=grid(768), stream=stream0)
        buf213 = buf211; del buf211  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf210, buf213, 3072, 128, grid=grid(3072), stream=stream0)
        buf214 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf213, buf214, 768, 4, grid=grid(768), stream=stream0)
        buf215 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf210, buf209, buf205, buf215, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf216 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (2304, 512), (1, 2304), 0), view_126, out=buf216)
        del view_126
        buf217 = reinterpret_tensor(buf210, (512, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 2304), (2304, 1), 0), permute_278, out=buf217)
        del permute_278
        buf218 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf219 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf199, buf217, sub_57, sqrt_21, buf218, buf219, 768, 512, grid=grid(768), stream=stream0)
        buf223 = reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        buf224 = reinterpret_tensor(buf205, (1, 512, 768), (393216, 768, 1), 0); del buf205  # reuse
        # Source Nodes: [hidden_states_104, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf199, buf217, primals_43, sub_57, sqrt_21, convert_element_type_28, buf223, buf224, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_28
        del primals_43
        del sqrt_21
        del sub_57
        buf225 = reinterpret_tensor(buf189, (512, 3072), (3072, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (512, 768), (768, 1), 0), permute_280, out=buf225)
        del permute_280
        buf226 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (768, 512), (1, 768), 0), view_124, out=buf226)
        del view_124
        buf227 = reinterpret_tensor(buf213, (1, 768, 4), (3072, 1, 768), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf224, buf227, 3072, 128, grid=grid(3072), stream=stream0)
        buf228 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf227, buf228, 768, 4, grid=grid(768), stream=stream0)
        buf229 = reinterpret_tensor(buf225, (1, 512, 3072), (1572864, 3072, 1), 0); del buf225  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf229, addmm_19, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_19
        buf230 = reinterpret_tensor(buf224, (512, 768), (768, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (512, 3072), (3072, 1), 0), permute_284, out=buf230)
        del permute_284
        buf231 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (3072, 512), (1, 3072), 0), view_122, out=buf231)
        del view_122
        buf232 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf229, buf232, 12288, 128, grid=grid(12288), stream=stream0)
        buf233 = reinterpret_tensor(buf227, (1, 3072), (3072, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf232, buf233, 3072, 4, grid=grid(3072), stream=stream0)
        buf234 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf235 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_96], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf223, buf230, sub_54, sqrt_20, buf234, buf235, 768, 512, grid=grid(768), stream=stream0)
        buf239 = reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0); del buf217  # reuse
        buf240 = buf199; del buf199  # reuse
        # Source Nodes: [hidden_states_96, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf223, buf230, primals_41, sub_54, sqrt_20, convert_element_type_27, buf239, buf240, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_27
        del primals_41
        del sqrt_20
        del sub_54
        buf241 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (512, 768), (768, 1), 0), permute_288, out=buf241)
        del permute_288
        buf242 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (768, 512), (1, 768), 0), view_120, out=buf242)
        del view_120
        buf243 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf240, buf243, 3072, 128, grid=grid(3072), stream=stream0)
        buf244 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf243, buf244, 768, 4, grid=grid(768), stream=stream0)
        buf245 = reinterpret_tensor(buf240, (12, 512, 64), (32768, 64, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_293, reinterpret_tensor(buf241, (12, 512, 64), (64, 768, 1), 0), out=buf245)
        del permute_293
        buf246 = reinterpret_tensor(buf208, (12, 512, 512), (262144, 512, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (12, 512, 64), (64, 768, 1), 0), permute_294, out=buf246)
        del permute_294
        buf248 = reinterpret_tensor(buf206, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf206  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_26, buf246, alias_68, buf248, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_68
        del convert_element_type_26
        buf249 = reinterpret_tensor(buf241, (12, 64, 512), (32768, 512, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_295, reinterpret_tensor(buf248, (12, 512, 512), (262144, 512, 1), 0), out=buf249)
        del permute_295
        buf250 = reinterpret_tensor(buf223, (12, 512, 64), (32768, 64, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (12, 512, 512), (262144, 512, 1), 0), permute_296, out=buf250)
        del permute_296
        buf251 = reinterpret_tensor(buf243, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf245, buf251, 3072, 128, grid=grid(3072), stream=stream0)
        buf252 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf251, buf252, 768, 4, grid=grid(768), stream=stream0)
        buf253 = buf251; del buf251  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf250, buf253, 3072, 128, grid=grid(3072), stream=stream0)
        buf254 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf253, buf254, 768, 4, grid=grid(768), stream=stream0)
        buf255 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf250, buf249, buf245, buf255, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf256 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (2304, 512), (1, 2304), 0), view_108, out=buf256)
        del view_108
        buf257 = reinterpret_tensor(buf250, (512, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 2304), (2304, 1), 0), permute_303, out=buf257)
        del permute_303
        buf258 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf259 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_89], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf239, buf257, sub_49, sqrt_18, buf258, buf259, 768, 512, grid=grid(768), stream=stream0)
        buf263 = reinterpret_tensor(buf249, (1, 512, 768), (393216, 768, 1), 0); del buf249  # reuse
        buf264 = reinterpret_tensor(buf245, (1, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        # Source Nodes: [hidden_states_89, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf239, buf257, primals_37, sub_49, sqrt_18, convert_element_type_24, buf263, buf264, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_24
        del primals_37
        del sqrt_18
        del sub_49
        buf265 = reinterpret_tensor(buf229, (512, 3072), (3072, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (512, 768), (768, 1), 0), permute_305, out=buf265)
        del permute_305
        buf266 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (768, 512), (1, 768), 0), view_106, out=buf266)
        del view_106
        buf267 = reinterpret_tensor(buf253, (1, 768, 4), (3072, 1, 768), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf264, buf267, 3072, 128, grid=grid(3072), stream=stream0)
        buf268 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf267, buf268, 768, 4, grid=grid(768), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (1, 512, 3072), (1572864, 3072, 1), 0); del buf265  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf269, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf270 = reinterpret_tensor(buf264, (512, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0), permute_309, out=buf270)
        del permute_309
        buf271 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (3072, 512), (1, 3072), 0), view_104, out=buf271)
        del view_104
        buf272 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf269, buf272, 12288, 128, grid=grid(12288), stream=stream0)
        buf273 = reinterpret_tensor(buf267, (1, 3072), (3072, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf272, buf273, 3072, 4, grid=grid(3072), stream=stream0)
        buf274 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf275 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf263, buf270, sub_46, sqrt_17, buf274, buf275, 768, 512, grid=grid(768), stream=stream0)
        buf279 = reinterpret_tensor(buf257, (1, 512, 768), (393216, 768, 1), 0); del buf257  # reuse
        buf280 = buf239; del buf239  # reuse
        # Source Nodes: [hidden_states_81, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf263, buf270, primals_35, sub_46, sqrt_17, convert_element_type_23, buf279, buf280, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_23
        del primals_35
        del sqrt_17
        del sub_46
        buf281 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 768), (768, 1), 0), permute_313, out=buf281)
        del permute_313
        buf282 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (768, 512), (1, 768), 0), view_102, out=buf282)
        del view_102
        buf283 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf280, buf283, 3072, 128, grid=grid(3072), stream=stream0)
        buf284 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf283, buf284, 768, 4, grid=grid(768), stream=stream0)
        buf285 = reinterpret_tensor(buf280, (12, 512, 64), (32768, 64, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_318, reinterpret_tensor(buf281, (12, 512, 64), (64, 768, 1), 0), out=buf285)
        del permute_318
        buf286 = reinterpret_tensor(buf248, (12, 512, 512), (262144, 512, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (12, 512, 64), (64, 768, 1), 0), permute_319, out=buf286)
        del permute_319
        buf288 = reinterpret_tensor(buf246, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf246  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_22, buf286, alias_73, buf288, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_73
        del convert_element_type_22
        buf289 = reinterpret_tensor(buf281, (12, 64, 512), (32768, 512, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_320, reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0), out=buf289)
        del permute_320
        buf290 = reinterpret_tensor(buf263, (12, 512, 64), (32768, 64, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0), permute_321, out=buf290)
        del permute_321
        buf291 = reinterpret_tensor(buf283, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf285, buf291, 3072, 128, grid=grid(3072), stream=stream0)
        buf292 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf291, buf292, 768, 4, grid=grid(768), stream=stream0)
        buf293 = buf291; del buf291  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf290, buf293, 3072, 128, grid=grid(3072), stream=stream0)
        buf294 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf293, buf294, 768, 4, grid=grid(768), stream=stream0)
        buf295 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf290, buf289, buf285, buf295, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf296 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (2304, 512), (1, 2304), 0), view_90, out=buf296)
        del view_90
        buf297 = reinterpret_tensor(buf290, (512, 768), (768, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (512, 2304), (2304, 1), 0), permute_328, out=buf297)
        del permute_328
        buf298 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf299 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf279, buf297, sub_41, sqrt_15, buf298, buf299, 768, 512, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0); del buf289  # reuse
        buf304 = reinterpret_tensor(buf285, (1, 512, 768), (393216, 768, 1), 0); del buf285  # reuse
        # Source Nodes: [hidden_states_74, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf279, buf297, primals_31, sub_41, sqrt_15, convert_element_type_20, buf303, buf304, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_20
        del primals_31
        del sqrt_15
        del sub_41
        buf305 = reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 768), (768, 1), 0), permute_330, out=buf305)
        del permute_330
        buf306 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (768, 512), (1, 768), 0), view_88, out=buf306)
        del view_88
        buf307 = reinterpret_tensor(buf293, (1, 768, 4), (3072, 1, 768), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf304, buf307, 3072, 128, grid=grid(3072), stream=stream0)
        buf308 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf307, buf308, 768, 4, grid=grid(768), stream=stream0)
        buf309 = reinterpret_tensor(buf305, (1, 512, 3072), (1572864, 3072, 1), 0); del buf305  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf309, addmm_13, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_13
        buf310 = reinterpret_tensor(buf304, (512, 768), (768, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 3072), (3072, 1), 0), permute_334, out=buf310)
        del permute_334
        buf311 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (3072, 512), (1, 3072), 0), view_86, out=buf311)
        del view_86
        buf312 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf309, buf312, 12288, 128, grid=grid(12288), stream=stream0)
        buf313 = reinterpret_tensor(buf307, (1, 3072), (3072, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf312, buf313, 3072, 4, grid=grid(3072), stream=stream0)
        buf314 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf315 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf303, buf310, sub_38, sqrt_14, buf314, buf315, 768, 512, grid=grid(768), stream=stream0)
        buf319 = reinterpret_tensor(buf297, (1, 512, 768), (393216, 768, 1), 0); del buf297  # reuse
        buf320 = buf279; del buf279  # reuse
        # Source Nodes: [hidden_states_66, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf303, buf310, primals_29, sub_38, sqrt_14, convert_element_type_19, buf319, buf320, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_19
        del primals_29
        del sqrt_14
        del sub_38
        buf321 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (512, 768), (768, 1), 0), permute_338, out=buf321)
        del permute_338
        buf322 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (768, 512), (1, 768), 0), view_84, out=buf322)
        del view_84
        buf323 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf320, buf323, 3072, 128, grid=grid(3072), stream=stream0)
        buf324 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf323, buf324, 768, 4, grid=grid(768), stream=stream0)
        buf325 = reinterpret_tensor(buf320, (12, 512, 64), (32768, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_343, reinterpret_tensor(buf321, (12, 512, 64), (64, 768, 1), 0), out=buf325)
        del permute_343
        buf326 = reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (12, 512, 64), (64, 768, 1), 0), permute_344, out=buf326)
        del permute_344
        buf328 = reinterpret_tensor(buf286, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf286  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_18, buf326, alias_78, buf328, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_78
        del convert_element_type_18
        buf329 = reinterpret_tensor(buf321, (12, 64, 512), (32768, 512, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_345, reinterpret_tensor(buf328, (12, 512, 512), (262144, 512, 1), 0), out=buf329)
        del permute_345
        buf330 = reinterpret_tensor(buf303, (12, 512, 64), (32768, 64, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (12, 512, 512), (262144, 512, 1), 0), permute_346, out=buf330)
        del permute_346
        buf331 = reinterpret_tensor(buf323, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf325, buf331, 3072, 128, grid=grid(3072), stream=stream0)
        buf332 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf331, buf332, 768, 4, grid=grid(768), stream=stream0)
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf330, buf333, 3072, 128, grid=grid(3072), stream=stream0)
        buf334 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf333, buf334, 768, 4, grid=grid(768), stream=stream0)
        buf335 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf330, buf329, buf325, buf335, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf336 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (2304, 512), (1, 2304), 0), view_72, out=buf336)
        del view_72
        buf337 = reinterpret_tensor(buf330, (512, 768), (768, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (512, 2304), (2304, 1), 0), permute_353, out=buf337)
        del permute_353
        buf338 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf339 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf319, buf337, sub_33, sqrt_12, buf338, buf339, 768, 512, grid=grid(768), stream=stream0)
        buf343 = reinterpret_tensor(buf329, (1, 512, 768), (393216, 768, 1), 0); del buf329  # reuse
        buf344 = reinterpret_tensor(buf325, (1, 512, 768), (393216, 768, 1), 0); del buf325  # reuse
        # Source Nodes: [hidden_states_59, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf319, buf337, primals_25, sub_33, sqrt_12, convert_element_type_16, buf343, buf344, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_16
        del primals_25
        del sqrt_12
        del sub_33
        buf345 = reinterpret_tensor(buf309, (512, 3072), (3072, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (512, 768), (768, 1), 0), permute_355, out=buf345)
        del permute_355
        buf346 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (768, 512), (1, 768), 0), view_70, out=buf346)
        del view_70
        buf347 = reinterpret_tensor(buf333, (1, 768, 4), (3072, 1, 768), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf344, buf347, 3072, 128, grid=grid(3072), stream=stream0)
        buf348 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf347, buf348, 768, 4, grid=grid(768), stream=stream0)
        buf349 = reinterpret_tensor(buf345, (1, 512, 3072), (1572864, 3072, 1), 0); del buf345  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf349, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf350 = reinterpret_tensor(buf344, (512, 768), (768, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0), permute_359, out=buf350)
        del permute_359
        buf351 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (3072, 512), (1, 3072), 0), view_68, out=buf351)
        del view_68
        buf352 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf349, buf352, 12288, 128, grid=grid(12288), stream=stream0)
        buf353 = reinterpret_tensor(buf347, (1, 3072), (3072, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf352, buf353, 3072, 4, grid=grid(3072), stream=stream0)
        buf354 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf355 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf343, buf350, sub_30, sqrt_11, buf354, buf355, 768, 512, grid=grid(768), stream=stream0)
        buf359 = reinterpret_tensor(buf337, (1, 512, 768), (393216, 768, 1), 0); del buf337  # reuse
        buf360 = buf319; del buf319  # reuse
        # Source Nodes: [hidden_states_51, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf343, buf350, primals_23, sub_30, sqrt_11, convert_element_type_15, buf359, buf360, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_15
        del primals_23
        del sqrt_11
        del sub_30
        buf361 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (512, 768), (768, 1), 0), permute_363, out=buf361)
        del permute_363
        buf362 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (768, 512), (1, 768), 0), view_66, out=buf362)
        del view_66
        buf363 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf360, buf363, 3072, 128, grid=grid(3072), stream=stream0)
        buf364 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf363, buf364, 768, 4, grid=grid(768), stream=stream0)
        buf365 = reinterpret_tensor(buf360, (12, 512, 64), (32768, 64, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_368, reinterpret_tensor(buf361, (12, 512, 64), (64, 768, 1), 0), out=buf365)
        del permute_368
        buf366 = reinterpret_tensor(buf328, (12, 512, 512), (262144, 512, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf361, (12, 512, 64), (64, 768, 1), 0), permute_369, out=buf366)
        del permute_369
        buf368 = reinterpret_tensor(buf326, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf326  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_14, buf366, alias_83, buf368, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_83
        del convert_element_type_14
        buf369 = reinterpret_tensor(buf361, (12, 64, 512), (32768, 512, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_370, reinterpret_tensor(buf368, (12, 512, 512), (262144, 512, 1), 0), out=buf369)
        del permute_370
        buf370 = reinterpret_tensor(buf343, (12, 512, 64), (32768, 64, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf368, (12, 512, 512), (262144, 512, 1), 0), permute_371, out=buf370)
        del permute_371
        buf371 = reinterpret_tensor(buf363, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf365, buf371, 3072, 128, grid=grid(3072), stream=stream0)
        buf372 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf371, buf372, 768, 4, grid=grid(768), stream=stream0)
        buf373 = buf371; del buf371  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf370, buf373, 3072, 128, grid=grid(3072), stream=stream0)
        buf374 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf373, buf374, 768, 4, grid=grid(768), stream=stream0)
        buf375 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf370, buf369, buf365, buf375, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf376 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (2304, 512), (1, 2304), 0), view_54, out=buf376)
        del view_54
        buf377 = reinterpret_tensor(buf370, (512, 768), (768, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (512, 2304), (2304, 1), 0), permute_378, out=buf377)
        del permute_378
        buf378 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf379 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_44], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf359, buf377, sub_25, sqrt_9, buf378, buf379, 768, 512, grid=grid(768), stream=stream0)
        buf383 = reinterpret_tensor(buf369, (1, 512, 768), (393216, 768, 1), 0); del buf369  # reuse
        buf384 = reinterpret_tensor(buf365, (1, 512, 768), (393216, 768, 1), 0); del buf365  # reuse
        # Source Nodes: [hidden_states_44, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf359, buf377, primals_19, sub_25, sqrt_9, convert_element_type_12, buf383, buf384, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_12
        del primals_19
        del sqrt_9
        del sub_25
        buf385 = reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (512, 768), (768, 1), 0), permute_380, out=buf385)
        del permute_380
        buf386 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (768, 512), (1, 768), 0), view_52, out=buf386)
        del view_52
        buf387 = reinterpret_tensor(buf373, (1, 768, 4), (3072, 1, 768), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf384, buf387, 3072, 128, grid=grid(3072), stream=stream0)
        buf388 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf387, buf388, 768, 4, grid=grid(768), stream=stream0)
        buf389 = reinterpret_tensor(buf385, (1, 512, 3072), (1572864, 3072, 1), 0); del buf385  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf389, addmm_7, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_7
        buf390 = reinterpret_tensor(buf384, (512, 768), (768, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 3072), (3072, 1), 0), permute_384, out=buf390)
        del permute_384
        buf391 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (3072, 512), (1, 3072), 0), view_50, out=buf391)
        del view_50
        buf392 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf389, buf392, 12288, 128, grid=grid(12288), stream=stream0)
        buf393 = reinterpret_tensor(buf387, (1, 3072), (3072, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf392, buf393, 3072, 4, grid=grid(3072), stream=stream0)
        buf394 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf395 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf383, buf390, sub_22, sqrt_8, buf394, buf395, 768, 512, grid=grid(768), stream=stream0)
        buf399 = reinterpret_tensor(buf377, (1, 512, 768), (393216, 768, 1), 0); del buf377  # reuse
        buf400 = buf359; del buf359  # reuse
        # Source Nodes: [hidden_states_36, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf383, buf390, primals_17, sub_22, sqrt_8, convert_element_type_11, buf399, buf400, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_11
        del primals_17
        del sqrt_8
        del sub_22
        buf401 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (512, 768), (768, 1), 0), permute_388, out=buf401)
        del permute_388
        buf402 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (768, 512), (1, 768), 0), view_48, out=buf402)
        del view_48
        buf403 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf400, buf403, 3072, 128, grid=grid(3072), stream=stream0)
        buf404 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf403, buf404, 768, 4, grid=grid(768), stream=stream0)
        buf405 = reinterpret_tensor(buf400, (12, 512, 64), (32768, 64, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_393, reinterpret_tensor(buf401, (12, 512, 64), (64, 768, 1), 0), out=buf405)
        del permute_393
        buf406 = reinterpret_tensor(buf368, (12, 512, 512), (262144, 512, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (12, 512, 64), (64, 768, 1), 0), permute_394, out=buf406)
        del permute_394
        buf408 = reinterpret_tensor(buf366, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf366  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_10, buf406, alias_88, buf408, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_88
        del convert_element_type_10
        buf409 = reinterpret_tensor(buf401, (12, 64, 512), (32768, 512, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_395, reinterpret_tensor(buf408, (12, 512, 512), (262144, 512, 1), 0), out=buf409)
        del permute_395
        buf410 = reinterpret_tensor(buf383, (12, 512, 64), (32768, 64, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (12, 512, 512), (262144, 512, 1), 0), permute_396, out=buf410)
        del permute_396
        buf411 = reinterpret_tensor(buf403, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf405, buf411, 3072, 128, grid=grid(3072), stream=stream0)
        buf412 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf411, buf412, 768, 4, grid=grid(768), stream=stream0)
        buf413 = buf411; del buf411  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf410, buf413, 3072, 128, grid=grid(3072), stream=stream0)
        buf414 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf413, buf414, 768, 4, grid=grid(768), stream=stream0)
        buf415 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf410, buf409, buf405, buf415, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf416 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (2304, 512), (1, 2304), 0), view_36, out=buf416)
        del view_36
        buf417 = reinterpret_tensor(buf410, (512, 768), (768, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 2304), (2304, 1), 0), permute_403, out=buf417)
        del permute_403
        buf418 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf419 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_29], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf399, buf417, sub_17, sqrt_6, buf418, buf419, 768, 512, grid=grid(768), stream=stream0)
        buf423 = reinterpret_tensor(buf409, (1, 512, 768), (393216, 768, 1), 0); del buf409  # reuse
        buf424 = reinterpret_tensor(buf405, (1, 512, 768), (393216, 768, 1), 0); del buf405  # reuse
        # Source Nodes: [hidden_states_29, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf399, buf417, primals_13, sub_17, sqrt_6, convert_element_type_8, buf423, buf424, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_8
        del primals_13
        del sqrt_6
        del sub_17
        buf425 = reinterpret_tensor(buf389, (512, 3072), (3072, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (512, 768), (768, 1), 0), permute_405, out=buf425)
        del permute_405
        buf426 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (768, 512), (1, 768), 0), view_34, out=buf426)
        del view_34
        buf427 = reinterpret_tensor(buf413, (1, 768, 4), (3072, 1, 768), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf424, buf427, 3072, 128, grid=grid(3072), stream=stream0)
        buf428 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf427, buf428, 768, 4, grid=grid(768), stream=stream0)
        buf429 = reinterpret_tensor(buf425, (1, 512, 3072), (1572864, 3072, 1), 0); del buf425  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf429, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf430 = reinterpret_tensor(buf424, (512, 768), (768, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (512, 3072), (3072, 1), 0), permute_409, out=buf430)
        del permute_409
        buf431 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (3072, 512), (1, 3072), 0), view_32, out=buf431)
        del view_32
        buf432 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf429, buf432, 12288, 128, grid=grid(12288), stream=stream0)
        buf433 = reinterpret_tensor(buf427, (1, 3072), (3072, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf432, buf433, 3072, 4, grid=grid(3072), stream=stream0)
        buf434 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf435 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf423, buf430, sub_14, sqrt_5, buf434, buf435, 768, 512, grid=grid(768), stream=stream0)
        buf439 = reinterpret_tensor(buf417, (1, 512, 768), (393216, 768, 1), 0); del buf417  # reuse
        buf440 = buf399; del buf399  # reuse
        # Source Nodes: [hidden_states_21, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf423, buf430, primals_11, sub_14, sqrt_5, convert_element_type_7, buf439, buf440, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_7
        del primals_11
        del sqrt_5
        del sub_14
        buf441 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (512, 768), (768, 1), 0), permute_413, out=buf441)
        del permute_413
        buf442 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (768, 512), (1, 768), 0), view_30, out=buf442)
        del view_30
        buf443 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf440, buf443, 3072, 128, grid=grid(3072), stream=stream0)
        buf444 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf443, buf444, 768, 4, grid=grid(768), stream=stream0)
        buf445 = reinterpret_tensor(buf440, (12, 512, 64), (32768, 64, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_418, reinterpret_tensor(buf441, (12, 512, 64), (64, 768, 1), 0), out=buf445)
        del permute_418
        buf446 = reinterpret_tensor(buf408, (12, 512, 512), (262144, 512, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (12, 512, 64), (64, 768, 1), 0), permute_419, out=buf446)
        del permute_419
        buf448 = reinterpret_tensor(buf406, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf406  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_6, buf446, alias_93, buf448, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_93
        del convert_element_type_6
        buf449 = reinterpret_tensor(buf441, (12, 64, 512), (32768, 512, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_420, reinterpret_tensor(buf448, (12, 512, 512), (262144, 512, 1), 0), out=buf449)
        del permute_420
        buf450 = reinterpret_tensor(buf423, (12, 512, 64), (32768, 64, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf448, (12, 512, 512), (262144, 512, 1), 0), permute_421, out=buf450)
        del permute_421
        buf451 = reinterpret_tensor(buf443, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf445, buf451, 3072, 128, grid=grid(3072), stream=stream0)
        buf452 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf451, buf452, 768, 4, grid=grid(768), stream=stream0)
        buf453 = buf451; del buf451  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf450, buf453, 3072, 128, grid=grid(3072), stream=stream0)
        buf454 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf453, buf454, 768, 4, grid=grid(768), stream=stream0)
        buf455 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf450, buf449, buf445, buf455, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf456 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (2304, 512), (1, 2304), 0), view_18, out=buf456)
        del view_18
        buf457 = reinterpret_tensor(buf450, (512, 768), (768, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (512, 2304), (2304, 1), 0), permute_428, out=buf457)
        del permute_428
        buf458 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf459 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf439, buf457, sub_9, sqrt_3, buf458, buf459, 768, 512, grid=grid(768), stream=stream0)
        buf463 = reinterpret_tensor(buf449, (1, 512, 768), (393216, 768, 1), 0); del buf449  # reuse
        buf464 = reinterpret_tensor(buf445, (1, 512, 768), (393216, 768, 1), 0); del buf445  # reuse
        # Source Nodes: [hidden_states_14, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf439, buf457, primals_7, sub_9, sqrt_3, convert_element_type_4, buf463, buf464, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_4
        del primals_7
        del sqrt_3
        del sub_9
        buf465 = reinterpret_tensor(buf429, (512, 3072), (3072, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 768), (768, 1), 0), permute_430, out=buf465)
        del permute_430
        buf466 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (768, 512), (1, 768), 0), view_16, out=buf466)
        del view_16
        buf467 = reinterpret_tensor(buf453, (1, 768, 4), (3072, 1, 768), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf464, buf467, 3072, 128, grid=grid(3072), stream=stream0)
        buf468 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf467, buf468, 768, 4, grid=grid(768), stream=stream0)
        buf469 = reinterpret_tensor(buf465, (1, 512, 3072), (1572864, 3072, 1), 0); del buf465  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf469, addmm_1, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_1
        buf470 = reinterpret_tensor(buf464, (512, 768), (768, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (512, 3072), (3072, 1), 0), permute_434, out=buf470)
        del permute_434
        buf471 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (3072, 512), (1, 3072), 0), view_14, out=buf471)
        del view_14
        buf472 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf469, buf472, 12288, 128, grid=grid(12288), stream=stream0)
        del buf469
        buf473 = reinterpret_tensor(buf467, (1, 3072), (3072, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf472, buf473, 3072, 4, grid=grid(3072), stream=stream0)
        del buf472
        buf474 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf475 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_13.run(buf463, buf470, sub_6, sqrt_2, buf474, buf475, 768, 512, grid=grid(768), stream=stream0)
        buf479 = reinterpret_tensor(buf457, (1, 512, 768), (393216, 768, 1), 0); del buf457  # reuse
        buf480 = buf439; del buf439  # reuse
        # Source Nodes: [hidden_states_6, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_14.run(buf463, buf470, primals_5, sub_6, sqrt_2, convert_element_type_3, buf479, buf480, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_3
        del primals_5
        del sqrt_2
        del sub_6
        buf481 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (512, 768), (768, 1), 0), permute_438, out=buf481)
        del permute_438
        buf482 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (768, 512), (1, 768), 0), view_12, out=buf482)
        del view_12
        buf483 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf480, buf483, 3072, 128, grid=grid(3072), stream=stream0)
        buf484 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf483, buf484, 768, 4, grid=grid(768), stream=stream0)
        buf485 = reinterpret_tensor(buf480, (12, 512, 64), (32768, 64, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_443, reinterpret_tensor(buf481, (12, 512, 64), (64, 768, 1), 0), out=buf485)
        del permute_443
        buf486 = reinterpret_tensor(buf448, (12, 512, 512), (262144, 512, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf481, (12, 512, 64), (64, 768, 1), 0), permute_444, out=buf486)
        del permute_444
        buf488 = reinterpret_tensor(buf446, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf446  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_15.run(convert_element_type_2, buf486, alias_98, buf488, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_98
        del buf486
        del convert_element_type_2
        buf489 = reinterpret_tensor(buf481, (12, 64, 512), (32768, 512, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_445, reinterpret_tensor(buf488, (12, 512, 512), (262144, 512, 1), 0), out=buf489)
        del permute_445
        buf490 = reinterpret_tensor(buf463, (12, 512, 64), (32768, 64, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf488, (12, 512, 512), (262144, 512, 1), 0), permute_446, out=buf490)
        del buf488
        del permute_446
        buf491 = reinterpret_tensor(buf483, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf485, buf491, 3072, 128, grid=grid(3072), stream=stream0)
        buf492 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf491, buf492, 768, 4, grid=grid(768), stream=stream0)
        buf493 = buf491; del buf491  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_18.run(buf490, buf493, 3072, 128, grid=grid(3072), stream=stream0)
        buf494 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_17.run(buf493, buf494, 768, 4, grid=grid(768), stream=stream0)
        del buf493
        buf495 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf490, buf489, buf485, buf495, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf496 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (2304, 512), (1, 2304), 0), view, out=buf496)
        del view
        buf497 = reinterpret_tensor(buf490, (512, 768), (768, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (512, 2304), (2304, 1), 0), permute_453, out=buf497)
        del buf495
        del permute_453
        buf498 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf499 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_sum_20.run(convert_element_type, buf479, buf497, sub, sqrt, buf498, buf499, 768, 512, grid=grid(768), stream=stream0)
        buf506 = reinterpret_tensor(buf489, (1, 512, 768), (393216, 768, 1), 0); del buf489  # reuse
        buf510 = reinterpret_tensor(buf485, (1, 512, 768), (393216, 768, 1), 0); del buf485  # reuse
        # Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_21.run(convert_element_type, buf479, buf497, primals_1, sqrt, sub, slice_1, primals_168, buf506, buf510, 512, 768, grid=grid(512), stream=stream0)
        del buf479
        del convert_element_type
        del primals_1
        del sqrt
        del sub
        buf505 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf505, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf505, [slice_1], buf506, True)
        del buf506
        del slice_1
        buf509 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_23.run(buf509, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf509, [primals_168], buf510, True)
        del buf510
        del primals_168
        return (reinterpret_tensor(buf499, (768, ), (1, ), 0), reinterpret_tensor(buf498, (768, ), (1, ), 0), reinterpret_tensor(buf494, (768, ), (1, ), 0), reinterpret_tensor(buf492, (768, ), (1, ), 0), reinterpret_tensor(buf475, (768, ), (1, ), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), reinterpret_tensor(buf459, (768, ), (1, ), 0), reinterpret_tensor(buf458, (768, ), (1, ), 0), reinterpret_tensor(buf454, (768, ), (1, ), 0), reinterpret_tensor(buf452, (768, ), (1, ), 0), reinterpret_tensor(buf435, (768, ), (1, ), 0), reinterpret_tensor(buf434, (768, ), (1, ), 0), reinterpret_tensor(buf419, (768, ), (1, ), 0), reinterpret_tensor(buf418, (768, ), (1, ), 0), reinterpret_tensor(buf414, (768, ), (1, ), 0), reinterpret_tensor(buf412, (768, ), (1, ), 0), reinterpret_tensor(buf395, (768, ), (1, ), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), reinterpret_tensor(buf379, (768, ), (1, ), 0), reinterpret_tensor(buf378, (768, ), (1, ), 0), reinterpret_tensor(buf374, (768, ), (1, ), 0), reinterpret_tensor(buf372, (768, ), (1, ), 0), reinterpret_tensor(buf355, (768, ), (1, ), 0), reinterpret_tensor(buf354, (768, ), (1, ), 0), reinterpret_tensor(buf339, (768, ), (1, ), 0), reinterpret_tensor(buf338, (768, ), (1, ), 0), reinterpret_tensor(buf334, (768, ), (1, ), 0), reinterpret_tensor(buf332, (768, ), (1, ), 0), reinterpret_tensor(buf315, (768, ), (1, ), 0), reinterpret_tensor(buf314, (768, ), (1, ), 0), reinterpret_tensor(buf299, (768, ), (1, ), 0), reinterpret_tensor(buf298, (768, ), (1, ), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), reinterpret_tensor(buf292, (768, ), (1, ), 0), reinterpret_tensor(buf275, (768, ), (1, ), 0), reinterpret_tensor(buf274, (768, ), (1, ), 0), reinterpret_tensor(buf259, (768, ), (1, ), 0), reinterpret_tensor(buf258, (768, ), (1, ), 0), reinterpret_tensor(buf254, (768, ), (1, ), 0), reinterpret_tensor(buf252, (768, ), (1, ), 0), reinterpret_tensor(buf235, (768, ), (1, ), 0), reinterpret_tensor(buf234, (768, ), (1, ), 0), reinterpret_tensor(buf219, (768, ), (1, ), 0), reinterpret_tensor(buf218, (768, ), (1, ), 0), reinterpret_tensor(buf214, (768, ), (1, ), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf195, (768, ), (1, ), 0), reinterpret_tensor(buf194, (768, ), (1, ), 0), reinterpret_tensor(buf179, (768, ), (1, ), 0), reinterpret_tensor(buf178, (768, ), (1, ), 0), reinterpret_tensor(buf174, (768, ), (1, ), 0), reinterpret_tensor(buf172, (768, ), (1, ), 0), reinterpret_tensor(buf155, (768, ), (1, ), 0), reinterpret_tensor(buf154, (768, ), (1, ), 0), reinterpret_tensor(buf139, (768, ), (1, ), 0), reinterpret_tensor(buf138, (768, ), (1, ), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), reinterpret_tensor(buf132, (768, ), (1, ), 0), reinterpret_tensor(buf115, (768, ), (1, ), 0), reinterpret_tensor(buf114, (768, ), (1, ), 0), reinterpret_tensor(buf99, (768, ), (1, ), 0), reinterpret_tensor(buf98, (768, ), (1, ), 0), reinterpret_tensor(buf94, (768, ), (1, ), 0), reinterpret_tensor(buf92, (768, ), (1, ), 0), reinterpret_tensor(buf75, (768, ), (1, ), 0), reinterpret_tensor(buf74, (768, ), (1, ), 0), reinterpret_tensor(buf59, (768, ), (1, ), 0), reinterpret_tensor(buf58, (768, ), (1, ), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), reinterpret_tensor(buf52, (768, ), (1, ), 0), reinterpret_tensor(buf35, (768, ), (1, ), 0), reinterpret_tensor(buf34, (768, ), (1, ), 0), reinterpret_tensor(buf19, (768, ), (1, ), 0), reinterpret_tensor(buf18, (768, ), (1, ), 0), buf509, buf505, reinterpret_tensor(buf496, (2304, 768), (768, 1), 0), reinterpret_tensor(buf482, (768, 768), (768, 1), 0), reinterpret_tensor(buf484, (768, ), (1, ), 0), reinterpret_tensor(buf471, (3072, 768), (768, 1), 0), reinterpret_tensor(buf473, (3072, ), (1, ), 0), reinterpret_tensor(buf466, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf468, (768, ), (1, ), 0), reinterpret_tensor(buf456, (2304, 768), (768, 1), 0), reinterpret_tensor(buf442, (768, 768), (768, 1), 0), reinterpret_tensor(buf444, (768, ), (1, ), 0), reinterpret_tensor(buf431, (3072, 768), (768, 1), 0), reinterpret_tensor(buf433, (3072, ), (1, ), 0), reinterpret_tensor(buf426, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf428, (768, ), (1, ), 0), reinterpret_tensor(buf416, (2304, 768), (768, 1), 0), reinterpret_tensor(buf402, (768, 768), (768, 1), 0), reinterpret_tensor(buf404, (768, ), (1, ), 0), reinterpret_tensor(buf391, (3072, 768), (768, 1), 0), reinterpret_tensor(buf393, (3072, ), (1, ), 0), reinterpret_tensor(buf386, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf388, (768, ), (1, ), 0), reinterpret_tensor(buf376, (2304, 768), (768, 1), 0), reinterpret_tensor(buf362, (768, 768), (768, 1), 0), reinterpret_tensor(buf364, (768, ), (1, ), 0), reinterpret_tensor(buf351, (3072, 768), (768, 1), 0), reinterpret_tensor(buf353, (3072, ), (1, ), 0), reinterpret_tensor(buf346, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf348, (768, ), (1, ), 0), reinterpret_tensor(buf336, (2304, 768), (768, 1), 0), reinterpret_tensor(buf322, (768, 768), (768, 1), 0), reinterpret_tensor(buf324, (768, ), (1, ), 0), reinterpret_tensor(buf311, (3072, 768), (768, 1), 0), reinterpret_tensor(buf313, (3072, ), (1, ), 0), reinterpret_tensor(buf306, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf308, (768, ), (1, ), 0), reinterpret_tensor(buf296, (2304, 768), (768, 1), 0), reinterpret_tensor(buf282, (768, 768), (768, 1), 0), reinterpret_tensor(buf284, (768, ), (1, ), 0), reinterpret_tensor(buf271, (3072, 768), (768, 1), 0), reinterpret_tensor(buf273, (3072, ), (1, ), 0), reinterpret_tensor(buf266, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf268, (768, ), (1, ), 0), reinterpret_tensor(buf256, (2304, 768), (768, 1), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, ), (1, ), 0), reinterpret_tensor(buf231, (3072, 768), (768, 1), 0), reinterpret_tensor(buf233, (3072, ), (1, ), 0), reinterpret_tensor(buf226, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf228, (768, ), (1, ), 0), reinterpret_tensor(buf216, (2304, 768), (768, 1), 0), reinterpret_tensor(buf202, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (768, ), (1, ), 0), reinterpret_tensor(buf191, (3072, 768), (768, 1), 0), reinterpret_tensor(buf193, (3072, ), (1, ), 0), reinterpret_tensor(buf186, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf188, (768, ), (1, ), 0), reinterpret_tensor(buf176, (2304, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, 768), (768, 1), 0), reinterpret_tensor(buf164, (768, ), (1, ), 0), reinterpret_tensor(buf151, (3072, 768), (768, 1), 0), reinterpret_tensor(buf153, (3072, ), (1, ), 0), reinterpret_tensor(buf146, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf148, (768, ), (1, ), 0), reinterpret_tensor(buf136, (2304, 768), (768, 1), 0), reinterpret_tensor(buf122, (768, 768), (768, 1), 0), reinterpret_tensor(buf124, (768, ), (1, ), 0), reinterpret_tensor(buf111, (3072, 768), (768, 1), 0), reinterpret_tensor(buf113, (3072, ), (1, ), 0), reinterpret_tensor(buf106, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf108, (768, ), (1, ), 0), reinterpret_tensor(buf96, (2304, 768), (768, 1), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), reinterpret_tensor(buf71, (3072, 768), (768, 1), 0), reinterpret_tensor(buf73, (3072, ), (1, ), 0), reinterpret_tensor(buf66, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf68, (768, ), (1, ), 0), reinterpret_tensor(buf56, (2304, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, 768), (768, 1), 0), reinterpret_tensor(buf44, (768, ), (1, ), 0), reinterpret_tensor(buf31, (3072, 768), (768, 1), 0), reinterpret_tensor(buf33, (3072, ), (1, ), 0), reinterpret_tensor(buf26, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (50265, 768), (768, 1), 0), reinterpret_tensor(buf8, (50265, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_169 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    sub = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_2 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_12 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_6 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_2 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_6 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_30 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_10 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_48 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_14 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_30 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_18 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_22 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_102 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_46 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_26 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_120 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_54 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_30 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_138 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_156 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_35 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_70 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_38 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_174 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_39 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_42 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_192 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_86 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_46 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_210 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_94 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_35 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_115 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_101 = rand_strided((512, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_49 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_48 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_196 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_53 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_243 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_58 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_63 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_271 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_293 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_68 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_309 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_73 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_338 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_78 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_353 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_363 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_83 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_384 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_88 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_395 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_409 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_93 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_428 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_98 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_445 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 50265), (25735680, 50265, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_163, primals_168, primals_169, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, addmm_36, mul_115, view_218, sub_101, convert_element_type_49, permute_147, div_51, permute_151, permute_155, permute_159, permute_163, permute_168, permute_169, alias_43, permute_170, permute_171, permute_178, permute_180, permute_184, permute_188, permute_193, permute_194, alias_48, permute_195, permute_196, permute_203, permute_205, permute_209, permute_213, permute_218, permute_219, alias_53, permute_220, permute_221, permute_228, permute_230, permute_234, permute_238, permute_243, permute_244, alias_58, permute_245, permute_246, permute_253, permute_255, permute_259, permute_263, permute_268, permute_269, alias_63, permute_270, permute_271, permute_278, permute_280, permute_284, permute_288, permute_293, permute_294, alias_68, permute_295, permute_296, permute_303, permute_305, permute_309, permute_313, permute_318, permute_319, alias_73, permute_320, permute_321, permute_328, permute_330, permute_334, permute_338, permute_343, permute_344, alias_78, permute_345, permute_346, permute_353, permute_355, permute_359, permute_363, permute_368, permute_369, alias_83, permute_370, permute_371, permute_378, permute_380, permute_384, permute_388, permute_393, permute_394, alias_88, permute_395, permute_396, permute_403, permute_405, permute_409, permute_413, permute_418, permute_419, alias_93, permute_420, permute_421, permute_428, permute_430, permute_434, permute_438, permute_443, permute_444, alias_98, permute_445, permute_446, permute_453, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForMaskedLM', benchmark_compiled_module)
