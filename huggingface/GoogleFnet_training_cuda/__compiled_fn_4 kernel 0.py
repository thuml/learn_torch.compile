
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


# kernel path: /tmp/torchinductor_youkaichao/jn/cjn6f2hn4432x5rhepdlprilugmw5fma5zqzihufyvz5h443vvdo.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
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
    xnumel = 16384000
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjivva35nrjxummuzug4vgfwcnmnn5qo5bkjhdvca6ghm43la35h.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
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


# kernel path: /tmp/torchinductor_youkaichao/lc/clchnoe3fron4od7fl56ynorv2yzhgtgfzrp5qea6aoicmgmc5cv.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp15 = tl.load(in_ptr4 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr1 + (r1 + (32000*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjz2s4ms7saklb32tw6y74qpaytdd5cb4u4r2wku65gnpuhfmk7r.py
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
    xnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32000*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3sqitatxifpmfjjy4ikxu2ir6sv5sp7rckyanahuspykgir24z.py
# Source Nodes: [add_50, hidden_states_85, hidden_states_87, mul_48], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_50 => add_101
# hidden_states_85 => mul_101
# hidden_states_87 => mul_102, sub_25
# mul_48 => mul_98
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
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
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp2 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = 768.0
    tmp24 = tmp16 / tmp23
    tmp25 = tmp2 * tmp23
    tmp26 = tmp25 - tmp6
    tmp27 = tmp17 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp9
    tmp31 = tmp10 * tmp10
    tmp32 = tmp11 - tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = 0.7978845608028654
    tmp35 = tmp33 * tmp34
    tmp36 = 0.044715
    tmp37 = tmp35 * tmp36
    tmp38 = tmp7 * tmp7
    tmp39 = 3.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = tmp29 * tmp12
    tmp44 = tmp43 * tmp8
    tmp45 = tmp42 + tmp44
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkzzprp4eqzb37neagkqtfhwh5klhpieanytommunzalgh37ml3.py
# Source Nodes: [add_50, hidden_states_85, hidden_states_87, mul_48], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_50 => add_101
# hidden_states_85 => mul_101
# hidden_states_87 => mul_102, sub_25
# mul_48 => mul_98
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/oo/coohfpb5vxwr477orqeycx63olcwfa6khk3tpg2r3smw7x6ewxaj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/za/czaiqkygranlbtsy57uwg7wsmw642ncsjrluynpoicnxfungp5ji.py
# Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_47 => add_96
# mul_44 => mul_92
triton_poi_fused_add_mul_pow_tanh_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp4 * tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = 0.044715
    tmp13 = tmp11 * tmp12
    tmp14 = tmp1 * tmp1
    tmp15 = 3.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp11 + tmp17
    tmp19 = tmp5 + tmp7
    tmp20 = tmp0 * tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tmp18 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
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


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6ulltkdwqstsyeypqeqvdxbp255csju5vi7rjabrzlc4jvxrbf.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvafdc6wmgkt3obk5xuvetome2g2njjcilyuunu5xsa7ipobyks.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2qheq3coynkqhjiuo7jv2526abthiweumxea2eetf7wsbfoqlc.py
# Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]

triton_poi_fused_select_backward_view_as_complex_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_backward_view_as_complex_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5wjb3jcfsnen5ttpbpbf4fvlguy66hu4ikmvpyiijprlhp4nwh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 6
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((2*r2) + (256*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbmozcqpobhelzcs2qcbnqmjlkncd6cuxgx76ouzbxmwv4vieo3.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dj42f4474qy54okhq54us26ytrrfdgxrevmomjhxlme3zseg3t.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + ((2*r1) + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 768.0
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 - tmp14
    tmp16 = tmp5 * tmp10
    tmp17 = tmp15 - tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.1111111111111112
    tmp22 = tmp20 * tmp21
    tmp23 = tmp18 * tmp22
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czhxkgxdg3n2ihvydqrl6kkms7rkp3nt4u3qmyfzvknwh3gekjjh.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_19', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + ((2*x0) + (1536*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrxqexdtunbc3tyyytoondzocflplkob5vdfjki3hyfkgzwxqnd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward]

triton_poi_fused_add_native_dropout_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (2*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caq5rp2xk7sh2aiq5wylywyctcrpn46hz2psenqfcixndeedunxo.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
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
    tmp21 = tl.full([1], -1, tl.int64)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp19)
    tmp26 = tmp25 == tmp21
    tmp27 = tl.where(tmp26, tmp23, tmp19)
    tmp29 = tl.full([1], 3, tl.int64)
    tmp30 = tmp28 == tmp29
    tmp31 = tl.where(tmp30, tmp23, tmp19)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp24, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp31, rmask & xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/sw/cswultqlabmc7kxse2y2dru4kkh2f75vbocgncgdpknbqa5nbifu.py
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
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36pnelmrbqewgsg2ycyekrig3i4pkk2pa4xc5vwc3s3zq7es3yq.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576000
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
    primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, expand, slice_2, mul, view, getitem_3, mul_2, view_2, addmm_1, tanh, view_4, getitem_7, mul_8, mul_10, view_6, addmm_3, tanh_1, view_8, getitem_13, mul_16, mul_18, view_10, addmm_5, tanh_2, view_12, getitem_19, mul_24, mul_26, view_14, addmm_7, tanh_3, view_16, getitem_25, mul_32, mul_34, view_18, addmm_9, tanh_4, view_20, getitem_31, mul_40, mul_42, view_22, addmm_11, tanh_5, view_24, getitem_37, mul_48, mul_50, view_26, addmm_13, tanh_6, view_28, getitem_43, mul_56, mul_58, view_30, addmm_15, tanh_7, view_32, getitem_49, mul_64, mul_66, view_34, addmm_17, tanh_8, view_36, getitem_55, mul_72, mul_74, view_38, addmm_19, tanh_9, view_40, getitem_61, mul_80, mul_82, view_42, addmm_21, tanh_10, view_44, getitem_67, mul_88, mul_90, view_46, addmm_23, tanh_11, view_48, getitem_73, mul_96, view_50, addmm_26, tanh_13, getitem_77, rsqrt_25, view_52, sub_27, convert_element_type_12, permute_28, permute_32, div_3, permute_36, permute_40, div_4, div_5, permute_44, permute_48, div_6, div_7, permute_52, permute_56, div_8, div_9, permute_60, permute_64, div_10, div_11, permute_68, permute_72, div_12, div_13, permute_76, permute_80, div_14, div_15, permute_84, permute_88, div_16, div_17, permute_92, permute_96, div_18, div_19, permute_100, permute_104, div_20, div_21, permute_108, permute_112, div_22, div_23, permute_116, permute_120, div_24, div_25, permute_124, permute_128, div_26, permute_132, div_27, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_114, (1, 512), (512, 1))
    assert_size_stride(primals_115, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_2, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_2, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_4, (512, 3072), (3072, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_6, (512, 768), (768, 1))
    assert_size_stride(addmm_3, (512, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_8, (512, 3072), (3072, 1))
    assert_size_stride(getitem_13, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_18, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_10, (512, 768), (768, 1))
    assert_size_stride(addmm_5, (512, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_12, (512, 3072), (3072, 1))
    assert_size_stride(getitem_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_26, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_34, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_9, (512, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_42, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(addmm_11, (512, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_24, (512, 3072), (3072, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_50, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_26, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(tanh_6, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_28, (512, 3072), (3072, 1))
    assert_size_stride(getitem_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_56, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_58, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(addmm_15, (512, 3072), (3072, 1))
    assert_size_stride(tanh_7, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_32, (512, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_64, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_34, (512, 768), (768, 1))
    assert_size_stride(addmm_17, (512, 3072), (3072, 1))
    assert_size_stride(tanh_8, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_36, (512, 3072), (3072, 1))
    assert_size_stride(getitem_55, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_72, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_74, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(tanh_9, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_40, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_80, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_82, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_42, (512, 768), (768, 1))
    assert_size_stride(addmm_21, (512, 3072), (3072, 1))
    assert_size_stride(tanh_10, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_44, (512, 3072), (3072, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_88, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_90, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_46, (512, 768), (768, 1))
    assert_size_stride(addmm_23, (512, 3072), (3072, 1))
    assert_size_stride(tanh_11, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_48, (512, 3072), (3072, 1))
    assert_size_stride(getitem_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_96, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_26, (512, 768), (768, 1))
    assert_size_stride(tanh_13, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_77, (1, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_52, (512, 768), (768, 1))
    assert_size_stride(sub_27, (512, 32000), (32000, 1))
    assert_size_stride(convert_element_type_12, (), ())
    assert_size_stride(permute_28, (32000, 768), (768, 1))
    assert_size_stride(permute_32, (768, 768), (768, 1))
    assert_size_stride(div_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_36, (768, 3072), (3072, 1))
    assert_size_stride(permute_40, (3072, 768), (768, 1))
    assert_size_stride(div_4, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_44, (768, 3072), (3072, 1))
    assert_size_stride(permute_48, (3072, 768), (768, 1))
    assert_size_stride(div_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_7, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_52, (768, 3072), (3072, 1))
    assert_size_stride(permute_56, (3072, 768), (768, 1))
    assert_size_stride(div_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_60, (768, 3072), (3072, 1))
    assert_size_stride(permute_64, (3072, 768), (768, 1))
    assert_size_stride(div_10, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_68, (768, 3072), (3072, 1))
    assert_size_stride(permute_72, (3072, 768), (768, 1))
    assert_size_stride(div_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_13, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_76, (768, 3072), (3072, 1))
    assert_size_stride(permute_80, (3072, 768), (768, 1))
    assert_size_stride(div_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_84, (768, 3072), (3072, 1))
    assert_size_stride(permute_88, (3072, 768), (768, 1))
    assert_size_stride(div_16, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_92, (768, 3072), (3072, 1))
    assert_size_stride(permute_96, (3072, 768), (768, 1))
    assert_size_stride(div_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_19, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_100, (768, 3072), (3072, 1))
    assert_size_stride(permute_104, (3072, 768), (768, 1))
    assert_size_stride(div_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_108, (768, 3072), (3072, 1))
    assert_size_stride(permute_112, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (3072, 1))
    assert_size_stride(permute_120, (3072, 768), (768, 1))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_124, (768, 3072), (3072, 1))
    assert_size_stride(permute_128, (3072, 768), (768, 1))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_132, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 32000), (16384000, 32000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 16384000, grid=grid(16384000), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_115, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_115, tangents_1, convert_element_type_12, tangents_2, sub_27, buf5, 512, 32000, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type_12
        del primals_115
        del sub_27
        del tangents_1
        del tangents_2
        buf6 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 32000), (32000, 1), 0), permute_28, out=buf6)
        del permute_28
        buf7 = empty((32000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (32000, 512), (1, 32000), 0), view_52, out=buf7)
        del view_52
        buf8 = empty((1, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 32000, 512, grid=grid(32000), stream=stream0)
        del buf5
        buf11 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf14 = buf11; del buf11  # reuse
        # Source Nodes: [add_50, hidden_states_85, hidden_states_87, mul_48], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4.run(buf14, buf6, primals_108, addmm_26, tanh_13, getitem_77, rsqrt_25, 512, 768, grid=grid(512), stream=stream0)
        del primals_108
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        buf13 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, hidden_states_85, hidden_states_87, mul_48], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5.run(buf6, addmm_26, tanh_13, getitem_77, rsqrt_25, buf12, buf13, 768, 512, grid=grid(768), stream=stream0)
        del addmm_26
        del getitem_77
        del rsqrt_25
        del tanh_13
        buf15 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (512, 768), (768, 1), 0), permute_32, out=buf15)
        del permute_32
        buf16 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (768, 512), (1, 768), 0), view_50, out=buf16)
        del view_50
        buf17 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf14, buf17, 3072, 128, grid=grid(3072), stream=stream0)
        buf18 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf17, buf18, 768, 4, grid=grid(768), stream=stream0)
        buf21 = buf14; del buf14  # reuse
        buf24 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf15, primals_102, mul_96, div_3, getitem_73, buf21, buf24, 512, 768, grid=grid(512), stream=stream0)
        del div_3
        del getitem_73
        del primals_102
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf15, mul_96, buf22, buf23, 768, 512, grid=grid(768), stream=stream0)
        del mul_96
        buf25 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 768), (768, 1), 0), permute_36, out=buf25)
        del permute_36
        buf26 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (768, 512), (1, 768), 0), view_48, out=buf26)
        del view_48
        buf27 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf24, buf27, 3072, 128, grid=grid(3072), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf27, buf28, 768, 4, grid=grid(768), stream=stream0)
        buf29 = reinterpret_tensor(buf25, (1, 512, 3072), (1572864, 3072, 1), 0); del buf25  # reuse
        # Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf29, addmm_23, tanh_11, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_23
        del tanh_11
        buf30 = reinterpret_tensor(buf24, (512, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0), permute_40, out=buf30)
        del permute_40
        buf31 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (3072, 512), (1, 3072), 0), view_46, out=buf31)
        del view_46
        buf32 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf29, buf32, 12288, 128, grid=grid(12288), stream=stream0)
        buf33 = reinterpret_tensor(buf27, (1, 3072), (3072, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf32, buf33, 3072, 4, grid=grid(3072), stream=stream0)
        buf36 = reinterpret_tensor(buf15, (1, 512, 768), (393216, 768, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf21, buf30, primals_96, mul_90, div_4, buf36, 512, 768, grid=grid(512), stream=stream0)
        del div_4
        del primals_96
        buf37 = empty((768, ), device='cuda', dtype=torch.float32)
        buf38 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf21, buf30, mul_90, buf37, buf38, 768, 512, grid=grid(768), stream=stream0)
        del mul_90
        buf39 = empty((1, 512, 768, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf36, buf39, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf40 = aten.view_as_complex(buf39)
        del buf39
        buf41 = buf40
        del buf40
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf42 = aten._fft_c2c(buf41, [1, 2], 0, False)
        del buf41
        buf43 = buf42
        del buf42
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf44 = aten.view_as_real(buf43)
        del buf43
        buf45 = buf44
        del buf44
        buf46 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf36, buf45, primals_94, buf46, 3072, 128, grid=grid(3072), stream=stream0)
        buf47 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf46, buf47, 512, 6, grid=grid(512), stream=stream0)
        buf49 = reinterpret_tensor(buf30, (1, 512, 768), (393216, 768, 1), 0); del buf30  # reuse
        buf52 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf36, buf45, primals_94, mul_88, div_5, buf47, getitem_67, buf49, buf52, 512, 768, grid=grid(512), stream=stream0)
        del div_5
        del getitem_67
        del primals_94
        buf50 = empty((768, ), device='cuda', dtype=torch.float32)
        buf51 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf36, buf45, mul_88, buf50, buf51, 768, 512, grid=grid(768), stream=stream0)
        del mul_88
        buf53 = reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (512, 768), (768, 1), 0), permute_44, out=buf53)
        del permute_44
        buf54 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (768, 512), (1, 768), 0), view_44, out=buf54)
        del view_44
        buf55 = reinterpret_tensor(buf46, (1, 768, 4), (3072, 1, 768), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf52, buf55, 3072, 128, grid=grid(3072), stream=stream0)
        buf56 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf55, buf56, 768, 4, grid=grid(768), stream=stream0)
        buf57 = reinterpret_tensor(buf53, (1, 512, 3072), (1572864, 3072, 1), 0); del buf53  # reuse
        # Source Nodes: [add_43, mul_40], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf57, addmm_21, tanh_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_21
        del tanh_10
        buf58 = reinterpret_tensor(buf52, (512, 768), (768, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0), permute_48, out=buf58)
        del permute_48
        buf59 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (3072, 512), (1, 3072), 0), view_42, out=buf59)
        del view_42
        buf60 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf57, buf60, 12288, 128, grid=grid(12288), stream=stream0)
        buf61 = reinterpret_tensor(buf55, (1, 3072), (3072, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf60, buf61, 3072, 4, grid=grid(3072), stream=stream0)
        buf64 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf49, buf58, primals_88, mul_82, div_6, buf64, 512, 768, grid=grid(512), stream=stream0)
        del div_6
        del primals_88
        buf65 = empty((768, ), device='cuda', dtype=torch.float32)
        buf66 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf49, buf58, mul_82, buf65, buf66, 768, 512, grid=grid(768), stream=stream0)
        del mul_82
        buf67 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf64, buf67, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf68 = aten.view_as_complex(buf67)
        del buf67
        buf69 = buf68
        del buf68
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf70 = aten._fft_c2c(buf69, [1, 2], 0, False)
        del buf69
        buf71 = buf70
        del buf70
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf72 = aten.view_as_real(buf71)
        del buf71
        buf73 = buf72
        del buf72
        buf74 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf64, buf73, primals_86, buf74, 3072, 128, grid=grid(3072), stream=stream0)
        buf75 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf74, buf75, 512, 6, grid=grid(512), stream=stream0)
        buf77 = reinterpret_tensor(buf58, (1, 512, 768), (393216, 768, 1), 0); del buf58  # reuse
        buf80 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf64, buf73, primals_86, mul_80, div_7, buf75, getitem_61, buf77, buf80, 512, 768, grid=grid(512), stream=stream0)
        del div_7
        del getitem_61
        del primals_86
        buf78 = empty((768, ), device='cuda', dtype=torch.float32)
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf64, buf73, mul_80, buf78, buf79, 768, 512, grid=grid(768), stream=stream0)
        del mul_80
        buf81 = reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (512, 768), (768, 1), 0), permute_52, out=buf81)
        del permute_52
        buf82 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (768, 512), (1, 768), 0), view_40, out=buf82)
        del view_40
        buf83 = reinterpret_tensor(buf74, (1, 768, 4), (3072, 1, 768), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf80, buf83, 3072, 128, grid=grid(3072), stream=stream0)
        buf84 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf83, buf84, 768, 4, grid=grid(768), stream=stream0)
        buf85 = reinterpret_tensor(buf81, (1, 512, 3072), (1572864, 3072, 1), 0); del buf81  # reuse
        # Source Nodes: [add_39, mul_36], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf85, addmm_19, tanh_9, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_19
        del tanh_9
        buf86 = reinterpret_tensor(buf80, (512, 768), (768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 3072), (3072, 1), 0), permute_56, out=buf86)
        del permute_56
        buf87 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (3072, 512), (1, 3072), 0), view_38, out=buf87)
        del view_38
        buf88 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf85, buf88, 12288, 128, grid=grid(12288), stream=stream0)
        buf89 = reinterpret_tensor(buf83, (1, 3072), (3072, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf88, buf89, 3072, 4, grid=grid(3072), stream=stream0)
        buf92 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf77, buf86, primals_80, mul_74, div_8, buf92, 512, 768, grid=grid(512), stream=stream0)
        del div_8
        del primals_80
        buf93 = empty((768, ), device='cuda', dtype=torch.float32)
        buf94 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf77, buf86, mul_74, buf93, buf94, 768, 512, grid=grid(768), stream=stream0)
        del mul_74
        buf95 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf92, buf95, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf96 = aten.view_as_complex(buf95)
        del buf95
        buf97 = buf96
        del buf96
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf98 = aten._fft_c2c(buf97, [1, 2], 0, False)
        del buf97
        buf99 = buf98
        del buf98
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf100 = aten.view_as_real(buf99)
        del buf99
        buf101 = buf100
        del buf100
        buf102 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf92, buf101, primals_78, buf102, 3072, 128, grid=grid(3072), stream=stream0)
        buf103 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf102, buf103, 512, 6, grid=grid(512), stream=stream0)
        buf105 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf108 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf92, buf101, primals_78, mul_72, div_9, buf103, getitem_55, buf105, buf108, 512, 768, grid=grid(512), stream=stream0)
        del div_9
        del getitem_55
        del primals_78
        buf106 = empty((768, ), device='cuda', dtype=torch.float32)
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf92, buf101, mul_72, buf106, buf107, 768, 512, grid=grid(768), stream=stream0)
        del mul_72
        buf109 = reinterpret_tensor(buf85, (512, 3072), (3072, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (512, 768), (768, 1), 0), permute_60, out=buf109)
        del permute_60
        buf110 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (768, 512), (1, 768), 0), view_36, out=buf110)
        del view_36
        buf111 = reinterpret_tensor(buf102, (1, 768, 4), (3072, 1, 768), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf108, buf111, 3072, 128, grid=grid(3072), stream=stream0)
        buf112 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf111, buf112, 768, 4, grid=grid(768), stream=stream0)
        buf113 = reinterpret_tensor(buf109, (1, 512, 3072), (1572864, 3072, 1), 0); del buf109  # reuse
        # Source Nodes: [add_35, mul_32], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf113, addmm_17, tanh_8, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_17
        del tanh_8
        buf114 = reinterpret_tensor(buf108, (512, 768), (768, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0), permute_64, out=buf114)
        del permute_64
        buf115 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (3072, 512), (1, 3072), 0), view_34, out=buf115)
        del view_34
        buf116 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf113, buf116, 12288, 128, grid=grid(12288), stream=stream0)
        buf117 = reinterpret_tensor(buf111, (1, 3072), (3072, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf116, buf117, 3072, 4, grid=grid(3072), stream=stream0)
        buf120 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf105, buf114, primals_72, mul_66, div_10, buf120, 512, 768, grid=grid(512), stream=stream0)
        del div_10
        del primals_72
        buf121 = empty((768, ), device='cuda', dtype=torch.float32)
        buf122 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf105, buf114, mul_66, buf121, buf122, 768, 512, grid=grid(768), stream=stream0)
        del mul_66
        buf123 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf120, buf123, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf124 = aten.view_as_complex(buf123)
        del buf123
        buf125 = buf124
        del buf124
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf126 = aten._fft_c2c(buf125, [1, 2], 0, False)
        del buf125
        buf127 = buf126
        del buf126
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf128 = aten.view_as_real(buf127)
        del buf127
        buf129 = buf128
        del buf128
        buf130 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf120, buf129, primals_70, buf130, 3072, 128, grid=grid(3072), stream=stream0)
        buf131 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf130, buf131, 512, 6, grid=grid(512), stream=stream0)
        buf133 = reinterpret_tensor(buf114, (1, 512, 768), (393216, 768, 1), 0); del buf114  # reuse
        buf136 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf120, buf129, primals_70, mul_64, div_11, buf131, getitem_49, buf133, buf136, 512, 768, grid=grid(512), stream=stream0)
        del div_11
        del getitem_49
        del primals_70
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        buf135 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf120, buf129, mul_64, buf134, buf135, 768, 512, grid=grid(768), stream=stream0)
        del mul_64
        buf137 = reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (512, 768), (768, 1), 0), permute_68, out=buf137)
        del permute_68
        buf138 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (768, 512), (1, 768), 0), view_32, out=buf138)
        del view_32
        buf139 = reinterpret_tensor(buf130, (1, 768, 4), (3072, 1, 768), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf136, buf139, 3072, 128, grid=grid(3072), stream=stream0)
        buf140 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf139, buf140, 768, 4, grid=grid(768), stream=stream0)
        buf141 = reinterpret_tensor(buf137, (1, 512, 3072), (1572864, 3072, 1), 0); del buf137  # reuse
        # Source Nodes: [add_31, mul_28], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf141, addmm_15, tanh_7, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_15
        del tanh_7
        buf142 = reinterpret_tensor(buf136, (512, 768), (768, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 3072), (3072, 1), 0), permute_72, out=buf142)
        del permute_72
        buf143 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (3072, 512), (1, 3072), 0), view_30, out=buf143)
        del view_30
        buf144 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf141, buf144, 12288, 128, grid=grid(12288), stream=stream0)
        buf145 = reinterpret_tensor(buf139, (1, 3072), (3072, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf144, buf145, 3072, 4, grid=grid(3072), stream=stream0)
        buf148 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf133, buf142, primals_64, mul_58, div_12, buf148, 512, 768, grid=grid(512), stream=stream0)
        del div_12
        del primals_64
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        buf150 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf133, buf142, mul_58, buf149, buf150, 768, 512, grid=grid(768), stream=stream0)
        del mul_58
        buf151 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf148, buf151, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf152 = aten.view_as_complex(buf151)
        del buf151
        buf153 = buf152
        del buf152
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf154 = aten._fft_c2c(buf153, [1, 2], 0, False)
        del buf153
        buf155 = buf154
        del buf154
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf156 = aten.view_as_real(buf155)
        del buf155
        buf157 = buf156
        del buf156
        buf158 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf148, buf157, primals_62, buf158, 3072, 128, grid=grid(3072), stream=stream0)
        buf159 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf158, buf159, 512, 6, grid=grid(512), stream=stream0)
        buf161 = reinterpret_tensor(buf142, (1, 512, 768), (393216, 768, 1), 0); del buf142  # reuse
        buf164 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf148, buf157, primals_62, mul_56, div_13, buf159, getitem_43, buf161, buf164, 512, 768, grid=grid(512), stream=stream0)
        del div_13
        del getitem_43
        del primals_62
        buf162 = empty((768, ), device='cuda', dtype=torch.float32)
        buf163 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf148, buf157, mul_56, buf162, buf163, 768, 512, grid=grid(768), stream=stream0)
        del mul_56
        buf165 = reinterpret_tensor(buf141, (512, 3072), (3072, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (512, 768), (768, 1), 0), permute_76, out=buf165)
        del permute_76
        buf166 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (768, 512), (1, 768), 0), view_28, out=buf166)
        del view_28
        buf167 = reinterpret_tensor(buf158, (1, 768, 4), (3072, 1, 768), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf164, buf167, 3072, 128, grid=grid(3072), stream=stream0)
        buf168 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf167, buf168, 768, 4, grid=grid(768), stream=stream0)
        buf169 = reinterpret_tensor(buf165, (1, 512, 3072), (1572864, 3072, 1), 0); del buf165  # reuse
        # Source Nodes: [add_27, mul_24], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf169, addmm_13, tanh_6, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_13
        del tanh_6
        buf170 = reinterpret_tensor(buf164, (512, 768), (768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 3072), (3072, 1), 0), permute_80, out=buf170)
        del permute_80
        buf171 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (3072, 512), (1, 3072), 0), view_26, out=buf171)
        del view_26
        buf172 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf169, buf172, 12288, 128, grid=grid(12288), stream=stream0)
        buf173 = reinterpret_tensor(buf167, (1, 3072), (3072, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf172, buf173, 3072, 4, grid=grid(3072), stream=stream0)
        buf176 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf161, buf170, primals_56, mul_50, div_14, buf176, 512, 768, grid=grid(512), stream=stream0)
        del div_14
        del primals_56
        buf177 = empty((768, ), device='cuda', dtype=torch.float32)
        buf178 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf161, buf170, mul_50, buf177, buf178, 768, 512, grid=grid(768), stream=stream0)
        del mul_50
        buf179 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf176, buf179, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf180 = aten.view_as_complex(buf179)
        del buf179
        buf181 = buf180
        del buf180
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf182 = aten._fft_c2c(buf181, [1, 2], 0, False)
        del buf181
        buf183 = buf182
        del buf182
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf184 = aten.view_as_real(buf183)
        del buf183
        buf185 = buf184
        del buf184
        buf186 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf176, buf185, primals_54, buf186, 3072, 128, grid=grid(3072), stream=stream0)
        buf187 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf186, buf187, 512, 6, grid=grid(512), stream=stream0)
        buf189 = reinterpret_tensor(buf170, (1, 512, 768), (393216, 768, 1), 0); del buf170  # reuse
        buf192 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf176, buf185, primals_54, mul_48, div_15, buf187, getitem_37, buf189, buf192, 512, 768, grid=grid(512), stream=stream0)
        del div_15
        del getitem_37
        del primals_54
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf176, buf185, mul_48, buf190, buf191, 768, 512, grid=grid(768), stream=stream0)
        del mul_48
        buf193 = reinterpret_tensor(buf169, (512, 3072), (3072, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (512, 768), (768, 1), 0), permute_84, out=buf193)
        del permute_84
        buf194 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (768, 512), (1, 768), 0), view_24, out=buf194)
        del view_24
        buf195 = reinterpret_tensor(buf186, (1, 768, 4), (3072, 1, 768), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf192, buf195, 3072, 128, grid=grid(3072), stream=stream0)
        buf196 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf195, buf196, 768, 4, grid=grid(768), stream=stream0)
        buf197 = reinterpret_tensor(buf193, (1, 512, 3072), (1572864, 3072, 1), 0); del buf193  # reuse
        # Source Nodes: [add_23, mul_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf197, addmm_11, tanh_5, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_11
        del tanh_5
        buf198 = reinterpret_tensor(buf192, (512, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (512, 3072), (3072, 1), 0), permute_88, out=buf198)
        del permute_88
        buf199 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (3072, 512), (1, 3072), 0), view_22, out=buf199)
        del view_22
        buf200 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf197, buf200, 12288, 128, grid=grid(12288), stream=stream0)
        buf201 = reinterpret_tensor(buf195, (1, 3072), (3072, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf200, buf201, 3072, 4, grid=grid(3072), stream=stream0)
        buf204 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf189, buf198, primals_48, mul_42, div_16, buf204, 512, 768, grid=grid(512), stream=stream0)
        del div_16
        del primals_48
        buf205 = empty((768, ), device='cuda', dtype=torch.float32)
        buf206 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf189, buf198, mul_42, buf205, buf206, 768, 512, grid=grid(768), stream=stream0)
        del mul_42
        buf207 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf204, buf207, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf208 = aten.view_as_complex(buf207)
        del buf207
        buf209 = buf208
        del buf208
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf210 = aten._fft_c2c(buf209, [1, 2], 0, False)
        del buf209
        buf211 = buf210
        del buf210
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf212 = aten.view_as_real(buf211)
        del buf211
        buf213 = buf212
        del buf212
        buf214 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf204, buf213, primals_46, buf214, 3072, 128, grid=grid(3072), stream=stream0)
        buf215 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf214, buf215, 512, 6, grid=grid(512), stream=stream0)
        buf217 = reinterpret_tensor(buf198, (1, 512, 768), (393216, 768, 1), 0); del buf198  # reuse
        buf220 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf204, buf213, primals_46, mul_40, div_17, buf215, getitem_31, buf217, buf220, 512, 768, grid=grid(512), stream=stream0)
        del div_17
        del getitem_31
        del primals_46
        buf218 = empty((768, ), device='cuda', dtype=torch.float32)
        buf219 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf204, buf213, mul_40, buf218, buf219, 768, 512, grid=grid(768), stream=stream0)
        del mul_40
        buf221 = reinterpret_tensor(buf197, (512, 3072), (3072, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 768), (768, 1), 0), permute_92, out=buf221)
        del permute_92
        buf222 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (768, 512), (1, 768), 0), view_20, out=buf222)
        del view_20
        buf223 = reinterpret_tensor(buf214, (1, 768, 4), (3072, 1, 768), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf220, buf223, 3072, 128, grid=grid(3072), stream=stream0)
        buf224 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf223, buf224, 768, 4, grid=grid(768), stream=stream0)
        buf225 = reinterpret_tensor(buf221, (1, 512, 3072), (1572864, 3072, 1), 0); del buf221  # reuse
        # Source Nodes: [add_19, mul_16], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf225, addmm_9, tanh_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_9
        del tanh_4
        buf226 = reinterpret_tensor(buf220, (512, 768), (768, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (512, 3072), (3072, 1), 0), permute_96, out=buf226)
        del permute_96
        buf227 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (3072, 512), (1, 3072), 0), view_18, out=buf227)
        del view_18
        buf228 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf225, buf228, 12288, 128, grid=grid(12288), stream=stream0)
        buf229 = reinterpret_tensor(buf223, (1, 3072), (3072, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf228, buf229, 3072, 4, grid=grid(3072), stream=stream0)
        buf232 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf217, buf226, primals_40, mul_34, div_18, buf232, 512, 768, grid=grid(512), stream=stream0)
        del div_18
        del primals_40
        buf233 = empty((768, ), device='cuda', dtype=torch.float32)
        buf234 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf217, buf226, mul_34, buf233, buf234, 768, 512, grid=grid(768), stream=stream0)
        del mul_34
        buf235 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf232, buf235, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf236 = aten.view_as_complex(buf235)
        del buf235
        buf237 = buf236
        del buf236
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf238 = aten._fft_c2c(buf237, [1, 2], 0, False)
        del buf237
        buf239 = buf238
        del buf238
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf240 = aten.view_as_real(buf239)
        del buf239
        buf241 = buf240
        del buf240
        buf242 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf232, buf241, primals_38, buf242, 3072, 128, grid=grid(3072), stream=stream0)
        buf243 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf242, buf243, 512, 6, grid=grid(512), stream=stream0)
        buf245 = reinterpret_tensor(buf226, (1, 512, 768), (393216, 768, 1), 0); del buf226  # reuse
        buf248 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf232, buf241, primals_38, mul_32, div_19, buf243, getitem_25, buf245, buf248, 512, 768, grid=grid(512), stream=stream0)
        del div_19
        del getitem_25
        del primals_38
        buf246 = empty((768, ), device='cuda', dtype=torch.float32)
        buf247 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf232, buf241, mul_32, buf246, buf247, 768, 512, grid=grid(768), stream=stream0)
        del mul_32
        buf249 = reinterpret_tensor(buf225, (512, 3072), (3072, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 768), (768, 1), 0), permute_100, out=buf249)
        del permute_100
        buf250 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (768, 512), (1, 768), 0), view_16, out=buf250)
        del view_16
        buf251 = reinterpret_tensor(buf242, (1, 768, 4), (3072, 1, 768), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf248, buf251, 3072, 128, grid=grid(3072), stream=stream0)
        buf252 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf251, buf252, 768, 4, grid=grid(768), stream=stream0)
        buf253 = reinterpret_tensor(buf249, (1, 512, 3072), (1572864, 3072, 1), 0); del buf249  # reuse
        # Source Nodes: [add_15, mul_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf253, addmm_7, tanh_3, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_7
        del tanh_3
        buf254 = reinterpret_tensor(buf248, (512, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 3072), (3072, 1), 0), permute_104, out=buf254)
        del permute_104
        buf255 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (3072, 512), (1, 3072), 0), view_14, out=buf255)
        del view_14
        buf256 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf253, buf256, 12288, 128, grid=grid(12288), stream=stream0)
        buf257 = reinterpret_tensor(buf251, (1, 3072), (3072, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf256, buf257, 3072, 4, grid=grid(3072), stream=stream0)
        buf260 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf245, buf254, primals_32, mul_26, div_20, buf260, 512, 768, grid=grid(512), stream=stream0)
        del div_20
        del primals_32
        buf261 = empty((768, ), device='cuda', dtype=torch.float32)
        buf262 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf245, buf254, mul_26, buf261, buf262, 768, 512, grid=grid(768), stream=stream0)
        del mul_26
        buf263 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf260, buf263, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf264 = aten.view_as_complex(buf263)
        del buf263
        buf265 = buf264
        del buf264
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf266 = aten._fft_c2c(buf265, [1, 2], 0, False)
        del buf265
        buf267 = buf266
        del buf266
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf268 = aten.view_as_real(buf267)
        del buf267
        buf269 = buf268
        del buf268
        buf270 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf260, buf269, primals_30, buf270, 3072, 128, grid=grid(3072), stream=stream0)
        buf271 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf270, buf271, 512, 6, grid=grid(512), stream=stream0)
        buf273 = reinterpret_tensor(buf254, (1, 512, 768), (393216, 768, 1), 0); del buf254  # reuse
        buf276 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf260, buf269, primals_30, mul_24, div_21, buf271, getitem_19, buf273, buf276, 512, 768, grid=grid(512), stream=stream0)
        del div_21
        del getitem_19
        del primals_30
        buf274 = empty((768, ), device='cuda', dtype=torch.float32)
        buf275 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf260, buf269, mul_24, buf274, buf275, 768, 512, grid=grid(768), stream=stream0)
        del mul_24
        buf277 = reinterpret_tensor(buf253, (512, 3072), (3072, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 768), (768, 1), 0), permute_108, out=buf277)
        del permute_108
        buf278 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (768, 512), (1, 768), 0), view_12, out=buf278)
        del view_12
        buf279 = reinterpret_tensor(buf270, (1, 768, 4), (3072, 1, 768), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf276, buf279, 3072, 128, grid=grid(3072), stream=stream0)
        buf280 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf279, buf280, 768, 4, grid=grid(768), stream=stream0)
        buf281 = reinterpret_tensor(buf277, (1, 512, 3072), (1572864, 3072, 1), 0); del buf277  # reuse
        # Source Nodes: [add_11, mul_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf281, addmm_5, tanh_2, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_5
        del tanh_2
        buf282 = reinterpret_tensor(buf276, (512, 768), (768, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (512, 3072), (3072, 1), 0), permute_112, out=buf282)
        del permute_112
        buf283 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (3072, 512), (1, 3072), 0), view_10, out=buf283)
        del view_10
        buf284 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf281, buf284, 12288, 128, grid=grid(12288), stream=stream0)
        buf285 = reinterpret_tensor(buf279, (1, 3072), (3072, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf284, buf285, 3072, 4, grid=grid(3072), stream=stream0)
        buf288 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf273, buf282, primals_24, mul_18, div_22, buf288, 512, 768, grid=grid(512), stream=stream0)
        del div_22
        del primals_24
        buf289 = empty((768, ), device='cuda', dtype=torch.float32)
        buf290 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf273, buf282, mul_18, buf289, buf290, 768, 512, grid=grid(768), stream=stream0)
        del mul_18
        buf291 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf288, buf291, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf292 = aten.view_as_complex(buf291)
        del buf291
        buf293 = buf292
        del buf292
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf294 = aten._fft_c2c(buf293, [1, 2], 0, False)
        del buf293
        buf295 = buf294
        del buf294
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf296 = aten.view_as_real(buf295)
        del buf295
        buf297 = buf296
        del buf296
        buf298 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf288, buf297, primals_22, buf298, 3072, 128, grid=grid(3072), stream=stream0)
        buf299 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf298, buf299, 512, 6, grid=grid(512), stream=stream0)
        buf301 = reinterpret_tensor(buf282, (1, 512, 768), (393216, 768, 1), 0); del buf282  # reuse
        buf304 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf288, buf297, primals_22, mul_16, div_23, buf299, getitem_13, buf301, buf304, 512, 768, grid=grid(512), stream=stream0)
        del div_23
        del getitem_13
        del primals_22
        buf302 = empty((768, ), device='cuda', dtype=torch.float32)
        buf303 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf288, buf297, mul_16, buf302, buf303, 768, 512, grid=grid(768), stream=stream0)
        del mul_16
        buf305 = reinterpret_tensor(buf281, (512, 3072), (3072, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 768), (768, 1), 0), permute_116, out=buf305)
        del permute_116
        buf306 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (768, 512), (1, 768), 0), view_8, out=buf306)
        del view_8
        buf307 = reinterpret_tensor(buf298, (1, 768, 4), (3072, 1, 768), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf304, buf307, 3072, 128, grid=grid(3072), stream=stream0)
        buf308 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf307, buf308, 768, 4, grid=grid(768), stream=stream0)
        buf309 = reinterpret_tensor(buf305, (1, 512, 3072), (1572864, 3072, 1), 0); del buf305  # reuse
        # Source Nodes: [add_7, mul_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf309, addmm_3, tanh_1, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_3
        del tanh_1
        buf310 = reinterpret_tensor(buf304, (512, 768), (768, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 3072), (3072, 1), 0), permute_120, out=buf310)
        del permute_120
        buf311 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (3072, 512), (1, 3072), 0), view_6, out=buf311)
        del view_6
        buf312 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf309, buf312, 12288, 128, grid=grid(12288), stream=stream0)
        buf313 = reinterpret_tensor(buf307, (1, 3072), (3072, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf312, buf313, 3072, 4, grid=grid(3072), stream=stream0)
        buf316 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf301, buf310, primals_16, mul_10, div_24, buf316, 512, 768, grid=grid(512), stream=stream0)
        del div_24
        del primals_16
        buf317 = empty((768, ), device='cuda', dtype=torch.float32)
        buf318 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf301, buf310, mul_10, buf317, buf318, 768, 512, grid=grid(768), stream=stream0)
        del mul_10
        buf319 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf316, buf319, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf320 = aten.view_as_complex(buf319)
        del buf319
        buf321 = buf320
        del buf320
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf322 = aten._fft_c2c(buf321, [1, 2], 0, False)
        del buf321
        buf323 = buf322
        del buf322
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf324 = aten.view_as_real(buf323)
        del buf323
        buf325 = buf324
        del buf324
        buf326 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf316, buf325, primals_14, buf326, 3072, 128, grid=grid(3072), stream=stream0)
        buf327 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf326, buf327, 512, 6, grid=grid(512), stream=stream0)
        buf329 = reinterpret_tensor(buf310, (1, 512, 768), (393216, 768, 1), 0); del buf310  # reuse
        buf332 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18.run(buf316, buf325, primals_14, mul_8, div_25, buf327, getitem_7, buf329, buf332, 512, 768, grid=grid(512), stream=stream0)
        del buf327
        del div_25
        del getitem_7
        del primals_14
        buf330 = empty((768, ), device='cuda', dtype=torch.float32)
        buf331 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf316, buf325, mul_8, buf330, buf331, 768, 512, grid=grid(768), stream=stream0)
        del mul_8
        buf333 = reinterpret_tensor(buf309, (512, 3072), (3072, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (512, 768), (768, 1), 0), permute_124, out=buf333)
        del permute_124
        buf334 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 512), (1, 768), 0), view_4, out=buf334)
        del view_4
        buf335 = reinterpret_tensor(buf326, (1, 768, 4), (3072, 1, 768), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf332, buf335, 3072, 128, grid=grid(3072), stream=stream0)
        buf336 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf335, buf336, 768, 4, grid=grid(768), stream=stream0)
        buf337 = reinterpret_tensor(buf333, (1, 512, 3072), (1572864, 3072, 1), 0); del buf333  # reuse
        # Source Nodes: [add_3, mul], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_10.run(buf337, addmm_1, tanh, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_1
        del tanh
        buf338 = reinterpret_tensor(buf332, (512, 768), (768, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (512, 3072), (3072, 1), 0), permute_128, out=buf338)
        del permute_128
        buf339 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (3072, 512), (1, 3072), 0), view_2, out=buf339)
        del view_2
        buf340 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf337, buf340, 12288, 128, grid=grid(12288), stream=stream0)
        del buf337
        buf341 = reinterpret_tensor(buf335, (1, 3072), (3072, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf340, buf341, 3072, 4, grid=grid(3072), stream=stream0)
        del buf340
        buf344 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf329, buf338, primals_8, mul_2, div_26, buf344, 512, 768, grid=grid(512), stream=stream0)
        del div_26
        del primals_8
        buf345 = empty((768, ), device='cuda', dtype=torch.float32)
        buf346 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf329, buf338, mul_2, buf345, buf346, 768, 512, grid=grid(768), stream=stream0)
        del mul_2
        buf347 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        triton_poi_fused_select_backward_view_as_complex_15.run(buf344, buf347, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
        buf348 = aten.view_as_complex(buf347)
        del buf347
        buf349 = buf348
        del buf348
        # Source Nodes: [], Original ATen: [aten._fft_c2c]
        buf350 = aten._fft_c2c(buf349, [1, 2], 0, False)
        del buf349
        buf351 = buf350
        del buf350
        # Source Nodes: [], Original ATen: [aten.view_as_real]
        buf352 = aten.view_as_real(buf351)
        del buf351
        buf353 = buf352
        del buf352
        buf354 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward]
        triton_poi_fused_add_native_dropout_backward_20.run(buf354, buf353, getitem_3, 393216, grid=grid(393216), stream=stream0)
        del buf353
        del getitem_3
        buf355 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 768), (768, 1), 0), permute_132, out=buf355)
        del permute_132
        buf356 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (768, 512), (1, 768), 0), view, out=buf356)
        del view
        buf357 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf354, buf357, 3072, 128, grid=grid(3072), stream=stream0)
        buf358 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf357, buf358, 768, 4, grid=grid(768), stream=stream0)
        buf365 = buf354; del buf354  # reuse
        buf369 = buf329; del buf329  # reuse
        buf373 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_21.run(buf355, primals_4, mul, div_27, slice_2, expand, primals_114, buf365, buf369, buf373, 512, 768, grid=grid(512), stream=stream0)
        del div_27
        del primals_4
        buf362 = empty((768, ), device='cuda', dtype=torch.float32)
        buf363 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf355, mul, buf362, buf363, 768, 512, grid=grid(768), stream=stream0)
        del mul
        buf364 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf364, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf364, [slice_2], buf365, True)
        del buf365
        del slice_2
        buf368 = reinterpret_tensor(buf357, (4, 768), (768, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_23.run(buf368, 3072, grid=grid(3072), stream=stream0)
        aten.index_put_(buf368, [expand], buf369, True)
        del buf369
        del expand
        buf372 = empty((32000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_24.run(buf372, 24576000, grid=grid(24576000), stream=stream0)
        aten.index_put_(buf372, [primals_114], buf373, True)
        del buf373
        del primals_114
        return (buf372, buf368, buf364, buf362, buf363, reinterpret_tensor(buf356, (768, 768), (768, 1), 0), reinterpret_tensor(buf358, (768, ), (1, ), 0), buf345, buf346, reinterpret_tensor(buf339, (3072, 768), (768, 1), 0), reinterpret_tensor(buf341, (3072, ), (1, ), 0), reinterpret_tensor(buf334, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf336, (768, ), (1, ), 0), buf330, buf331, buf317, buf318, reinterpret_tensor(buf311, (3072, 768), (768, 1), 0), reinterpret_tensor(buf313, (3072, ), (1, ), 0), reinterpret_tensor(buf306, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf308, (768, ), (1, ), 0), buf302, buf303, buf289, buf290, reinterpret_tensor(buf283, (3072, 768), (768, 1), 0), reinterpret_tensor(buf285, (3072, ), (1, ), 0), reinterpret_tensor(buf278, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf280, (768, ), (1, ), 0), buf274, buf275, buf261, buf262, reinterpret_tensor(buf255, (3072, 768), (768, 1), 0), reinterpret_tensor(buf257, (3072, ), (1, ), 0), reinterpret_tensor(buf250, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf252, (768, ), (1, ), 0), buf246, buf247, buf233, buf234, reinterpret_tensor(buf227, (3072, 768), (768, 1), 0), reinterpret_tensor(buf229, (3072, ), (1, ), 0), reinterpret_tensor(buf222, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf224, (768, ), (1, ), 0), buf218, buf219, buf205, buf206, reinterpret_tensor(buf199, (3072, 768), (768, 1), 0), reinterpret_tensor(buf201, (3072, ), (1, ), 0), reinterpret_tensor(buf194, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf196, (768, ), (1, ), 0), buf190, buf191, buf177, buf178, reinterpret_tensor(buf171, (3072, 768), (768, 1), 0), reinterpret_tensor(buf173, (3072, ), (1, ), 0), reinterpret_tensor(buf166, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), buf162, buf163, buf149, buf150, reinterpret_tensor(buf143, (3072, 768), (768, 1), 0), reinterpret_tensor(buf145, (3072, ), (1, ), 0), reinterpret_tensor(buf138, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), buf134, buf135, buf121, buf122, reinterpret_tensor(buf115, (3072, 768), (768, 1), 0), reinterpret_tensor(buf117, (3072, ), (1, ), 0), reinterpret_tensor(buf110, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf112, (768, ), (1, ), 0), buf106, buf107, buf93, buf94, reinterpret_tensor(buf87, (3072, 768), (768, 1), 0), reinterpret_tensor(buf89, (3072, ), (1, ), 0), reinterpret_tensor(buf82, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), buf78, buf79, buf65, buf66, reinterpret_tensor(buf59, (3072, 768), (768, 1), 0), reinterpret_tensor(buf61, (3072, ), (1, ), 0), reinterpret_tensor(buf54, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf56, (768, ), (1, ), 0), buf50, buf51, buf37, buf38, reinterpret_tensor(buf31, (3072, 768), (768, 1), 0), reinterpret_tensor(buf33, (3072, ), (1, ), 0), reinterpret_tensor(buf26, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), buf22, buf23, None, None, reinterpret_tensor(buf16, (768, 768), (768, 1), 0), reinterpret_tensor(buf18, (768, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf7, (32000, 768), (768, 1), 0), reinterpret_tensor(buf8, (32000, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_115 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_18 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_10 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_26 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_34 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_9 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_56 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_15 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_55 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_72 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_88 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_96 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tanh_13 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_27 = rand_strided((512, 32000), (32000, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_12 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_32 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_44 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_52 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_56 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_88 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_92 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_112 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_120 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 32000), (16384000, 32000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, expand, slice_2, mul, view, getitem_3, mul_2, view_2, addmm_1, tanh, view_4, getitem_7, mul_8, mul_10, view_6, addmm_3, tanh_1, view_8, getitem_13, mul_16, mul_18, view_10, addmm_5, tanh_2, view_12, getitem_19, mul_24, mul_26, view_14, addmm_7, tanh_3, view_16, getitem_25, mul_32, mul_34, view_18, addmm_9, tanh_4, view_20, getitem_31, mul_40, mul_42, view_22, addmm_11, tanh_5, view_24, getitem_37, mul_48, mul_50, view_26, addmm_13, tanh_6, view_28, getitem_43, mul_56, mul_58, view_30, addmm_15, tanh_7, view_32, getitem_49, mul_64, mul_66, view_34, addmm_17, tanh_8, view_36, getitem_55, mul_72, mul_74, view_38, addmm_19, tanh_9, view_40, getitem_61, mul_80, mul_82, view_42, addmm_21, tanh_10, view_44, getitem_67, mul_88, mul_90, view_46, addmm_23, tanh_11, view_48, getitem_73, mul_96, view_50, addmm_26, tanh_13, getitem_77, rsqrt_25, view_52, sub_27, convert_element_type_12, permute_28, permute_32, div_3, permute_36, permute_40, div_4, div_5, permute_44, permute_48, div_6, div_7, permute_52, permute_56, div_8, div_9, permute_60, permute_64, div_10, div_11, permute_68, permute_72, div_12, div_13, permute_76, permute_80, div_14, div_15, permute_84, permute_88, div_16, div_17, permute_92, permute_96, div_18, div_19, permute_100, permute_104, div_20, div_21, permute_108, permute_112, div_22, div_23, permute_116, permute_120, div_24, div_25, permute_124, permute_128, div_26, permute_132, div_27, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GoogleFnet', benchmark_compiled_module)
