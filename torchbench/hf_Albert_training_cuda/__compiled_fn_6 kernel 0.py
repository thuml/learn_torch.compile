
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


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkke5r3j3qgfeilwodl2n5s3a7o2r2jspnvciltakjg3ypk3n3x.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30000
    rnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30000*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/po/cpobwsoay4ktjm36gi2b67lws3c4lnttebn5cdinqb47lzc6cftz.py
# Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => mul_103, sub_38
# mul_49 => mul_99
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp2 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 128.0
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
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp45, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cpruottt2q4tn75rvlrlsbiek4vpvgohkiowi3ak5zcznhq6r2ke.py
# Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => mul_103, sub_38
# mul_49 => mul_99
triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 * tmp2
        tmp5 = 1.0
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp0 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxuaalzdtdcxpdz4wq7dp4xpecrj7aku76h72hpmiuxqlggi42g7.py
# Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => mul_103, sub_38
# mul_49 => mul_99
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz43zepcp7m5u3phsuppfq7uj5lt6r3nx4bioiyd2fqwkxmuqsoa.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvih22dh4pfnvghcwvh4wlfzbozh5c3cn6ptbonm5uirshpx3vub.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5nsej2vil2r4ggfy4c3ai6sgs2lep7k2jcpz6oou6nmbyvxu3t.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qd6di7sf7puvwhf6yk5mgmi2nzktrm3v5esejj6qwadumthbl5.py
# Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_59 => add_108
# mul_45 => mul_93
triton_poi_fused_add_mul_pow_tanh_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/jr/cjre7ikwtpkjtbz7xofakot7smmc5mtg2i2fm6vgk6und7i4bbbw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctoklwbctv2vu6bxnmbnwtr2vao6xrhs432mkb2qccr3wubjonm4.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cseqis3bmex7h5vpyzyrjkosmwj67dano3ldu3i2dxvap42lnjvg.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcu2q5zv3mzyuf77feac4dhj7okrifjgs5sza3jt52jiiizwvkl.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]

triton_per_fused__softmax_backward_data_div_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 24576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp10, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjnw7vdbcdo2pgz7mvhr4jupu2gx7un7ywjmj5kd4hmqazrrs4u.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((512*x1) + (393216*(y0 // 512)) + (y0 % 512)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggp7bfhmlag7qa2fvsfwd4kxdfubmy7evvepmptqai475krgjsr.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cru7djpmyiv2bvgqnnkkjbrhrxtfsmw3wteas3zar6i4gkau2ej5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3lz2zmmn65bkfa3ndgqx6ve7cnzei3cx4ysqlo6zbw6rhr4xe7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbvljwtthrw2cdvapdjrg3jytsdcagctfwectyemiosvmrxncgu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr7 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp40 = tl.load(in_ptr8 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr9 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp50 = tl.load(in_ptr10 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp55 = tl.load(in_ptr11 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.where(rmask & xmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.where(rmask & xmask, tmp46, 0)
    tmp49 = tl.sum(tmp48, 1)[:, None]
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp53 = tl.where(rmask & xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp58 = tl.where(rmask & xmask, tmp56, 0)
    tmp59 = tl.sum(tmp58, 1)[:, None]
    tmp60 = tmp4 + tmp9
    tmp61 = tmp60 + tmp14
    tmp62 = tmp61 + tmp19
    tmp63 = tmp62 + tmp24
    tmp64 = tmp63 + tmp29
    tmp65 = tmp64 + tmp39
    tmp66 = tmp65 + tmp49
    tmp67 = tmp66 + tmp59
    tmp68 = tmp67 + tmp34
    tmp69 = tmp68 + tmp44
    tmp70 = tmp69 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqsd5ecvgo4z4rh3lagfw5ijjoxogpelg66oujkpbfp46kg5g3il.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctk653ckjymtxeuyq4d6ncbnq33vcvivknu3w2gik3l3asvkvffl.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvrctytfw2uflwx6qu3os2a4zcrkztn7mtfmlq53kxegutwvq6f.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutqfvsxslx24nsh3spu5i3t37v7rvhpvfbrp6ruqpu7u2nzmzuq.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_per_fused_add_sum_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr7 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp40 = tl.load(in_ptr8 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr9 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp50 = tl.load(in_ptr10 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp55 = tl.load(in_ptr11 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.where(rmask & xmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.where(rmask & xmask, tmp46, 0)
    tmp49 = tl.sum(tmp48, 1)[:, None]
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp53 = tl.where(rmask & xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp58 = tl.where(rmask & xmask, tmp56, 0)
    tmp59 = tl.sum(tmp58, 1)[:, None]
    tmp60 = tmp4 + tmp9
    tmp61 = tmp60 + tmp14
    tmp62 = tmp61 + tmp19
    tmp63 = tmp62 + tmp24
    tmp64 = tmp63 + tmp29
    tmp65 = tmp64 + tmp39
    tmp66 = tmp65 + tmp49
    tmp67 = tmp66 + tmp59
    tmp68 = tmp67 + tmp34
    tmp69 = tmp68 + tmp44
    tmp70 = tmp69 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuxfs4ojhh2ggnqzbgt7jbc3itsmjpggiwb7mbytq5asf6dcv6u.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3yy65teka7chtepy5o4yqjmjlft63np565eac7ve2vwzc4qzke.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33zyz3etqlrgljs4h5hz7uiuk54xzbi4nf6pf4mumnwnhh7yp5h.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
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
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpk6z2qvwgkh5imbsirwoh5oefvhurrtwyh3efhnazedokico4hk.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
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
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkhh5our425ui25mic2lzieveg45uhqasgabvjwnamadsd4kmw4.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
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
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yu/cyuq7i5nq5acyitvejoef3jonnehczle5mhkhyip7vxuzg5lyvp5.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqrxizra72hgdqeqtp74g6sbcs2zyymojjy5yu5yyn4ckm3j3qw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/o4/co4ipeu4nibkihtwtaakoeo5ewg34hbvbj5rzfofxgyzlns7h52f.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward]

triton_per_fused_embedding_dense_backward_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_layer_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 128.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tl.full([1, 1], -1, tl.int64)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp19)
    tmp26 = tl.full([1, 1], 0, tl.int64)
    tmp27 = tmp25 == tmp26
    tmp28 = tl.where(tmp27, tmp23, tmp19)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp19, rmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp24, rmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp28, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czhzvyw6hfkam7n7flitkugd3ysoe4np4e7aafnswq3l7f62ml6g.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfngr573bewdkxtazreav4pjkatuuptz4mjl6dyrakij76xk65rb.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqmdvdocs43v3g3s2uawf6yphzhomh4hjdx3cilupxvebp4q4w4.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]

triton_poi_fused_embedding_dense_backward_sum_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_sum_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (65536 + x2), None)
    tmp6 = tl.load(in_ptr1 + (131072 + x2), None)
    tmp8 = tl.load(in_ptr1 + (196608 + x2), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2ktwhiuyq556642rhnusvnh2rbir2epvuekdr4jwyerrr6k3hn4.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfu663dela6g2rillacuoutdsuuleyryr7im3lm32zp2he5aaeis.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840000
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
    primals_4, primals_16, primals_22, primals_26, primals_32, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, permute_135, permute_139, div_25, permute_143, permute_147, div_26, permute_151, permute_156, permute_157, alias_27, permute_158, permute_159, permute_164, permute_168, permute_172, div_28, div_29, permute_189, permute_190, alias_29, permute_191, permute_192, div_31, div_32, permute_222, permute_223, alias_31, permute_224, permute_225, div_34, div_35, permute_255, permute_256, alias_33, permute_257, permute_258, div_37, div_38, permute_288, permute_289, alias_35, permute_290, permute_291, div_40, div_41, permute_321, permute_322, alias_37, permute_323, permute_324, div_43, div_44, permute_354, permute_355, alias_39, permute_356, permute_357, div_46, div_47, permute_387, permute_388, alias_41, permute_389, permute_390, div_49, div_50, permute_420, permute_421, alias_43, permute_422, permute_423, div_52, div_53, permute_453, permute_454, alias_45, permute_455, permute_456, div_55, div_56, permute_486, permute_487, alias_47, permute_488, permute_489, div_58, div_59, permute_519, permute_520, alias_49, permute_521, permute_522, permute_539, div_61, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_32, (4, 512), (512, 1))
    assert_size_stride(expand, (4, 512), (0, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul_1, (4, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (2048, 128), (128, 1))
    assert_size_stride(view_2, (2048, 768), (768, 1))
    assert_size_stride(view_18, (2048, 768), (768, 1))
    assert_size_stride(mul_3, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_20, (2048, 768), (768, 1))
    assert_size_stride(addmm_5, (2048, 3072), (3072, 1))
    assert_size_stride(tanh, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_22, (2048, 3072), (3072, 1))
    assert_size_stride(mul_9, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_24, (2048, 768), (768, 1))
    assert_size_stride(view_40, (2048, 768), (768, 1))
    assert_size_stride(mul_11, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_42, (2048, 768), (768, 1))
    assert_size_stride(addmm_11, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_1, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_44, (2048, 3072), (3072, 1))
    assert_size_stride(mul_17, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_46, (2048, 768), (768, 1))
    assert_size_stride(view_62, (2048, 768), (768, 1))
    assert_size_stride(mul_19, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_64, (2048, 768), (768, 1))
    assert_size_stride(addmm_17, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_2, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_66, (2048, 3072), (3072, 1))
    assert_size_stride(mul_25, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_68, (2048, 768), (768, 1))
    assert_size_stride(view_84, (2048, 768), (768, 1))
    assert_size_stride(mul_27, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_86, (2048, 768), (768, 1))
    assert_size_stride(addmm_23, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_3, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_88, (2048, 3072), (3072, 1))
    assert_size_stride(mul_33, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_90, (2048, 768), (768, 1))
    assert_size_stride(view_106, (2048, 768), (768, 1))
    assert_size_stride(mul_35, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_108, (2048, 768), (768, 1))
    assert_size_stride(addmm_29, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_4, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_110, (2048, 3072), (3072, 1))
    assert_size_stride(mul_41, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_112, (2048, 768), (768, 1))
    assert_size_stride(view_128, (2048, 768), (768, 1))
    assert_size_stride(mul_43, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_130, (2048, 768), (768, 1))
    assert_size_stride(addmm_35, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_5, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_132, (2048, 3072), (3072, 1))
    assert_size_stride(mul_49, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_134, (2048, 768), (768, 1))
    assert_size_stride(view_150, (2048, 768), (768, 1))
    assert_size_stride(mul_51, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_152, (2048, 768), (768, 1))
    assert_size_stride(addmm_41, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_6, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_154, (2048, 3072), (3072, 1))
    assert_size_stride(mul_57, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_156, (2048, 768), (768, 1))
    assert_size_stride(view_172, (2048, 768), (768, 1))
    assert_size_stride(mul_59, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_174, (2048, 768), (768, 1))
    assert_size_stride(addmm_47, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_7, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_176, (2048, 3072), (3072, 1))
    assert_size_stride(mul_65, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_178, (2048, 768), (768, 1))
    assert_size_stride(view_194, (2048, 768), (768, 1))
    assert_size_stride(mul_67, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_196, (2048, 768), (768, 1))
    assert_size_stride(addmm_53, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_8, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_198, (2048, 3072), (3072, 1))
    assert_size_stride(mul_73, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_200, (2048, 768), (768, 1))
    assert_size_stride(view_216, (2048, 768), (768, 1))
    assert_size_stride(mul_75, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_218, (2048, 768), (768, 1))
    assert_size_stride(addmm_59, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_9, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_220, (2048, 3072), (3072, 1))
    assert_size_stride(mul_81, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_222, (2048, 768), (768, 1))
    assert_size_stride(view_238, (2048, 768), (768, 1))
    assert_size_stride(mul_83, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_240, (2048, 768), (768, 1))
    assert_size_stride(addmm_65, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_10, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_242, (2048, 3072), (3072, 1))
    assert_size_stride(mul_89, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_244, (2048, 768), (768, 1))
    assert_size_stride(view_260, (2048, 768), (768, 1))
    assert_size_stride(mul_91, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_262, (2048, 768), (768, 1))
    assert_size_stride(addmm_71, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_11, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_264, (2048, 3072), (3072, 1))
    assert_size_stride(mul_97, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_266, (2048, 768), (768, 1))
    assert_size_stride(addmm_73, (2048, 128), (128, 1))
    assert_size_stride(tanh_12, (4, 512, 128), (65536, 128, 1))
    assert_size_stride(getitem_51, (4, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(view_268, (2048, 128), (128, 1))
    assert_size_stride(permute_135, (30000, 128), (128, 1))
    assert_size_stride(permute_139, (128, 768), (768, 1))
    assert_size_stride(div_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_143, (768, 3072), (3072, 1))
    assert_size_stride(permute_147, (3072, 768), (768, 1))
    assert_size_stride(div_26, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_156, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_157, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_27, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_158, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_159, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_164, (768, 768), (768, 1))
    assert_size_stride(permute_168, (768, 768), (768, 1))
    assert_size_stride(permute_172, (768, 768), (768, 1))
    assert_size_stride(div_28, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_29, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_189, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_190, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_29, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_191, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_192, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_31, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_32, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_222, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_223, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_31, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_224, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_225, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_34, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_35, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_255, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_256, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_33, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_257, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_258, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_37, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_38, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_288, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_289, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_35, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_290, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_291, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_40, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_41, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_321, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_322, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_37, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_323, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_324, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_43, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_44, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_354, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_355, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_39, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_356, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_357, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_46, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_47, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_387, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_388, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_41, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_389, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_390, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_49, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_50, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_420, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_421, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_43, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_422, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_423, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_52, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_53, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_453, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_454, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_45, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_455, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_456, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_55, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_56, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_486, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_487, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_47, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_488, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_489, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_58, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_59, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_519, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_520, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_49, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_521, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_522, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_539, (768, 128), (128, 1))
    assert_size_stride(div_61, (4, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (4, 512, 30000), (15360000, 30000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (2048, 30000), (30000, 1), 0), permute_135, out=buf0)
        del permute_135
        buf1 = empty((30000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (30000, 2048), (1, 30000), 0), view_268, out=buf1)
        del view_268
        buf2 = empty((1, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 30000, 2048, grid=grid(30000), stream=stream0)
        del tangents_1
        buf5 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf10 = buf5; del buf5  # reuse
        # Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_1.run(buf10, buf0, primals_26, addmm_73, tanh_12, getitem_51, rsqrt_25, 2048, 128, grid=grid(2048), stream=stream0)
        del primals_26
        buf6 = empty_strided((128, 16), (1, 128), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((128, 16), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2.run(buf0, addmm_73, tanh_12, getitem_51, rsqrt_25, buf6, buf8, 2048, 128, grid=grid(2048), stream=stream0)
        del addmm_73
        del getitem_51
        del rsqrt_25
        del tanh_12
        buf7 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf6, buf7, 128, 16, grid=grid(128), stream=stream0)
        buf9 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf8, buf9, 128, 16, grid=grid(128), stream=stream0)
        buf11 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (2048, 128), (128, 1), 0), permute_139, out=buf11)
        del permute_139
        buf12 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (128, 2048), (1, 128), 0), view_266, out=buf12)
        del view_266
        buf13 = reinterpret_tensor(buf8, (1, 128, 16), (2048, 1, 128), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf10, buf13, 2048, 128, grid=grid(2048), stream=stream0)
        buf14 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf13, buf14, 128, 16, grid=grid(128), stream=stream0)
        buf17 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf11, primals_22, mul_97, div_25, buf17, 2048, 768, grid=grid(2048), stream=stream0)
        del div_25
        buf18 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_6.run(buf11, mul_97, buf18, buf20, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_97
        buf22 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 768), (768, 1), 0), permute_143, out=buf22)
        buf26 = reinterpret_tensor(buf22, (4, 512, 3072), (1572864, 3072, 1), 0); del buf22  # reuse
        # Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf26, addmm_71, tanh_11, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_71
        del tanh_11
        buf27 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (2048, 3072), (3072, 1), 0), permute_147, out=buf27)
        buf33 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf17, buf27, primals_16, mul_91, div_26, buf33, 2048, 768, grid=grid(2048), stream=stream0)
        del div_26
        buf38 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (2048, 768), (768, 1), 0), permute_151, out=buf38)
        buf42 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf38, buf42, 1572864, grid=grid(1572864), stream=stream0)
        buf43 = reinterpret_tensor(buf38, (48, 512, 64), (32768, 64, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_156, reinterpret_tensor(buf42, (48, 512, 64), (32768, 64, 1), 0), out=buf43)
        del permute_156
        buf49 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf43, buf49, 1572864, grid=grid(1572864), stream=stream0)
        buf50 = reinterpret_tensor(buf43, (2048, 768), (768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf49, permute_164, out=buf50)
        buf44 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (48, 512, 64), (32768, 64, 1), 0), permute_157, out=buf44)
        del permute_157
        buf46 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf44, alias_27, buf46, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_27
        buf47 = reinterpret_tensor(buf42, (48, 64, 512), (32768, 512, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_158, reinterpret_tensor(buf46, (48, 512, 512), (262144, 512, 1), 0), out=buf47)
        del permute_158
        buf54 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf47, buf54, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf55 = reinterpret_tensor(buf47, (2048, 768), (768, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, permute_168, out=buf55)
        buf48 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (48, 512, 512), (262144, 512, 1), 0), permute_159, out=buf48)
        del permute_159
        buf59 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf48, buf59, 1572864, grid=grid(1572864), stream=stream0)
        buf60 = reinterpret_tensor(buf48, (2048, 768), (768, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf59, permute_172, out=buf60)
        buf67 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf33, buf50, buf55, buf60, primals_22, mul_89, div_28, buf67, 2048, 768, grid=grid(2048), stream=stream0)
        del div_28
        buf72 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (2048, 768), (768, 1), 0), permute_143, out=buf72)
        buf76 = reinterpret_tensor(buf72, (4, 512, 3072), (1572864, 3072, 1), 0); del buf72  # reuse
        # Source Nodes: [add_54, mul_41], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf76, addmm_65, tanh_10, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_65
        del tanh_10
        buf77 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (2048, 3072), (3072, 1), 0), permute_147, out=buf77)
        buf83 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf67, buf77, primals_16, mul_83, div_29, buf83, 2048, 768, grid=grid(2048), stream=stream0)
        del div_29
        buf88 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (2048, 768), (768, 1), 0), permute_151, out=buf88)
        buf92 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf88, buf92, 1572864, grid=grid(1572864), stream=stream0)
        buf93 = reinterpret_tensor(buf88, (48, 512, 64), (32768, 64, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_189, reinterpret_tensor(buf92, (48, 512, 64), (32768, 64, 1), 0), out=buf93)
        del permute_189
        buf99 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf93, buf99, 1572864, grid=grid(1572864), stream=stream0)
        buf100 = reinterpret_tensor(buf93, (2048, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf99, permute_164, out=buf100)
        buf94 = reinterpret_tensor(buf46, (48, 512, 512), (262144, 512, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (48, 512, 64), (32768, 64, 1), 0), permute_190, out=buf94)
        del permute_190
        buf96 = reinterpret_tensor(buf44, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf94, alias_29, buf96, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_29
        buf97 = reinterpret_tensor(buf92, (48, 64, 512), (32768, 512, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_191, reinterpret_tensor(buf96, (48, 512, 512), (262144, 512, 1), 0), out=buf97)
        del permute_191
        buf104 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf97, buf104, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf105 = reinterpret_tensor(buf97, (2048, 768), (768, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf104, permute_168, out=buf105)
        buf98 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (48, 512, 512), (262144, 512, 1), 0), permute_192, out=buf98)
        del permute_192
        buf109 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf98, buf109, 1572864, grid=grid(1572864), stream=stream0)
        buf110 = reinterpret_tensor(buf98, (2048, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf109, permute_172, out=buf110)
        buf90 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf120 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf83, buf100, buf105, buf110, mul_81, buf90, buf118, buf120, 12288, 128, grid=grid(12288), stream=stream0)
        buf114 = reinterpret_tensor(buf100, (4, 512, 768), (393216, 768, 1), 0); del buf100  # reuse
        buf117 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf114, buf83, buf105, buf110, primals_22, mul_81, div_31, buf117, 2048, 768, grid=grid(2048), stream=stream0)
        del div_31
        del mul_81
        buf122 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (2048, 768), (768, 1), 0), permute_143, out=buf122)
        buf126 = reinterpret_tensor(buf122, (4, 512, 3072), (1572864, 3072, 1), 0); del buf122  # reuse
        # Source Nodes: [add_49, mul_37], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf126, addmm_59, tanh_9, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_59
        del tanh_9
        buf127 = reinterpret_tensor(buf114, (2048, 768), (768, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (2048, 3072), (3072, 1), 0), permute_147, out=buf127)
        buf133 = reinterpret_tensor(buf110, (4, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf117, buf127, primals_16, mul_75, div_32, buf133, 2048, 768, grid=grid(2048), stream=stream0)
        del div_32
        buf138 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (2048, 768), (768, 1), 0), permute_151, out=buf138)
        buf142 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf138, buf142, 1572864, grid=grid(1572864), stream=stream0)
        buf143 = reinterpret_tensor(buf138, (48, 512, 64), (32768, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_222, reinterpret_tensor(buf142, (48, 512, 64), (32768, 64, 1), 0), out=buf143)
        del permute_222
        buf149 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf143, buf149, 1572864, grid=grid(1572864), stream=stream0)
        buf150 = reinterpret_tensor(buf143, (2048, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf149, permute_164, out=buf150)
        buf144 = reinterpret_tensor(buf96, (48, 512, 512), (262144, 512, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (48, 512, 64), (32768, 64, 1), 0), permute_223, out=buf144)
        del permute_223
        buf146 = reinterpret_tensor(buf94, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf144, alias_31, buf146, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_31
        buf147 = reinterpret_tensor(buf142, (48, 64, 512), (32768, 512, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_224, reinterpret_tensor(buf146, (48, 512, 512), (262144, 512, 1), 0), out=buf147)
        del permute_224
        buf154 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf147, buf154, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf155 = reinterpret_tensor(buf147, (2048, 768), (768, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, permute_168, out=buf155)
        buf148 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (48, 512, 512), (262144, 512, 1), 0), permute_225, out=buf148)
        del permute_225
        buf159 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf148, buf159, 1572864, grid=grid(1572864), stream=stream0)
        buf160 = reinterpret_tensor(buf148, (2048, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf159, permute_172, out=buf160)
        buf140 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf133, buf150, buf155, buf160, mul_73, buf140, buf168, buf170, 12288, 128, grid=grid(12288), stream=stream0)
        buf164 = reinterpret_tensor(buf150, (4, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
        buf167 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf164, buf133, buf155, buf160, primals_22, mul_73, div_34, buf167, 2048, 768, grid=grid(2048), stream=stream0)
        del div_34
        del mul_73
        buf172 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 768), (768, 1), 0), permute_143, out=buf172)
        buf176 = reinterpret_tensor(buf172, (4, 512, 3072), (1572864, 3072, 1), 0); del buf172  # reuse
        # Source Nodes: [add_44, mul_33], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf176, addmm_53, tanh_8, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_53
        del tanh_8
        buf177 = reinterpret_tensor(buf164, (2048, 768), (768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (2048, 3072), (3072, 1), 0), permute_147, out=buf177)
        buf183 = reinterpret_tensor(buf160, (4, 512, 768), (393216, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf167, buf177, primals_16, mul_67, div_35, buf183, 2048, 768, grid=grid(2048), stream=stream0)
        del div_35
        buf188 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (2048, 768), (768, 1), 0), permute_151, out=buf188)
        buf192 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf188, buf192, 1572864, grid=grid(1572864), stream=stream0)
        buf193 = reinterpret_tensor(buf188, (48, 512, 64), (32768, 64, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_255, reinterpret_tensor(buf192, (48, 512, 64), (32768, 64, 1), 0), out=buf193)
        del permute_255
        buf199 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf193, buf199, 1572864, grid=grid(1572864), stream=stream0)
        buf200 = reinterpret_tensor(buf193, (2048, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf199, permute_164, out=buf200)
        buf194 = reinterpret_tensor(buf146, (48, 512, 512), (262144, 512, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (48, 512, 64), (32768, 64, 1), 0), permute_256, out=buf194)
        del permute_256
        buf196 = reinterpret_tensor(buf144, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf194, alias_33, buf196, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_33
        buf197 = reinterpret_tensor(buf192, (48, 64, 512), (32768, 512, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_257, reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0), out=buf197)
        del permute_257
        buf204 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf197, buf204, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf205 = reinterpret_tensor(buf197, (2048, 768), (768, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf204, permute_168, out=buf205)
        buf198 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0), permute_258, out=buf198)
        del permute_258
        buf209 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf198, buf209, 1572864, grid=grid(1572864), stream=stream0)
        buf210 = reinterpret_tensor(buf198, (2048, 768), (768, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf209, permute_172, out=buf210)
        buf190 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf183, buf200, buf205, buf210, mul_65, buf190, buf218, buf220, 12288, 128, grid=grid(12288), stream=stream0)
        buf214 = reinterpret_tensor(buf200, (4, 512, 768), (393216, 768, 1), 0); del buf200  # reuse
        buf217 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf214, buf183, buf205, buf210, primals_22, mul_65, div_37, buf217, 2048, 768, grid=grid(2048), stream=stream0)
        del div_37
        del mul_65
        buf222 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (2048, 768), (768, 1), 0), permute_143, out=buf222)
        buf226 = reinterpret_tensor(buf222, (4, 512, 3072), (1572864, 3072, 1), 0); del buf222  # reuse
        # Source Nodes: [add_39, mul_29], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf226, addmm_47, tanh_7, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_47
        del tanh_7
        buf227 = reinterpret_tensor(buf214, (2048, 768), (768, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (2048, 3072), (3072, 1), 0), permute_147, out=buf227)
        buf233 = reinterpret_tensor(buf210, (4, 512, 768), (393216, 768, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf217, buf227, primals_16, mul_59, div_38, buf233, 2048, 768, grid=grid(2048), stream=stream0)
        del div_38
        buf238 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 768), (768, 1), 0), permute_151, out=buf238)
        buf242 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf238, buf242, 1572864, grid=grid(1572864), stream=stream0)
        buf243 = reinterpret_tensor(buf238, (48, 512, 64), (32768, 64, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_288, reinterpret_tensor(buf242, (48, 512, 64), (32768, 64, 1), 0), out=buf243)
        del permute_288
        buf249 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf243, buf249, 1572864, grid=grid(1572864), stream=stream0)
        buf250 = reinterpret_tensor(buf243, (2048, 768), (768, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf249, permute_164, out=buf250)
        buf244 = reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (48, 512, 64), (32768, 64, 1), 0), permute_289, out=buf244)
        del permute_289
        buf246 = reinterpret_tensor(buf194, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf244, alias_35, buf246, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_35
        buf247 = reinterpret_tensor(buf242, (48, 64, 512), (32768, 512, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_290, reinterpret_tensor(buf246, (48, 512, 512), (262144, 512, 1), 0), out=buf247)
        del permute_290
        buf254 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf247, buf254, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf255 = reinterpret_tensor(buf247, (2048, 768), (768, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf254, permute_168, out=buf255)
        buf248 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (48, 512, 512), (262144, 512, 1), 0), permute_291, out=buf248)
        del permute_291
        buf259 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf248, buf259, 1572864, grid=grid(1572864), stream=stream0)
        buf260 = reinterpret_tensor(buf248, (2048, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf259, permute_172, out=buf260)
        buf240 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf268 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf233, buf250, buf255, buf260, mul_57, buf240, buf268, buf270, 12288, 128, grid=grid(12288), stream=stream0)
        buf264 = reinterpret_tensor(buf250, (4, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
        buf267 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf264, buf233, buf255, buf260, primals_22, mul_57, div_40, buf267, 2048, 768, grid=grid(2048), stream=stream0)
        del div_40
        del mul_57
        buf272 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (2048, 768), (768, 1), 0), permute_143, out=buf272)
        buf276 = reinterpret_tensor(buf272, (4, 512, 3072), (1572864, 3072, 1), 0); del buf272  # reuse
        # Source Nodes: [add_34, mul_25], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf276, addmm_41, tanh_6, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_41
        del tanh_6
        buf277 = reinterpret_tensor(buf264, (2048, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (2048, 3072), (3072, 1), 0), permute_147, out=buf277)
        buf283 = reinterpret_tensor(buf260, (4, 512, 768), (393216, 768, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf267, buf277, primals_16, mul_51, div_41, buf283, 2048, 768, grid=grid(2048), stream=stream0)
        del div_41
        buf288 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (2048, 768), (768, 1), 0), permute_151, out=buf288)
        buf292 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf288, buf292, 1572864, grid=grid(1572864), stream=stream0)
        buf293 = reinterpret_tensor(buf288, (48, 512, 64), (32768, 64, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_321, reinterpret_tensor(buf292, (48, 512, 64), (32768, 64, 1), 0), out=buf293)
        del permute_321
        buf299 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf293, buf299, 1572864, grid=grid(1572864), stream=stream0)
        buf300 = reinterpret_tensor(buf293, (2048, 768), (768, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf299, permute_164, out=buf300)
        buf294 = reinterpret_tensor(buf246, (48, 512, 512), (262144, 512, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (48, 512, 64), (32768, 64, 1), 0), permute_322, out=buf294)
        del permute_322
        buf296 = reinterpret_tensor(buf244, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf294, alias_37, buf296, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_37
        buf297 = reinterpret_tensor(buf292, (48, 64, 512), (32768, 512, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_323, reinterpret_tensor(buf296, (48, 512, 512), (262144, 512, 1), 0), out=buf297)
        del permute_323
        buf304 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf297, buf304, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf305 = reinterpret_tensor(buf297, (2048, 768), (768, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf304, permute_168, out=buf305)
        buf298 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (48, 512, 512), (262144, 512, 1), 0), permute_324, out=buf298)
        del permute_324
        buf309 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf298, buf309, 1572864, grid=grid(1572864), stream=stream0)
        buf310 = reinterpret_tensor(buf298, (2048, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, permute_172, out=buf310)
        buf290 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf320 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf283, buf300, buf305, buf310, mul_49, buf290, buf318, buf320, 12288, 128, grid=grid(12288), stream=stream0)
        buf314 = reinterpret_tensor(buf300, (4, 512, 768), (393216, 768, 1), 0); del buf300  # reuse
        buf317 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf314, buf283, buf305, buf310, primals_22, mul_49, div_43, buf317, 2048, 768, grid=grid(2048), stream=stream0)
        del div_43
        del mul_49
        buf322 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 768), (768, 1), 0), permute_143, out=buf322)
        buf326 = reinterpret_tensor(buf322, (4, 512, 3072), (1572864, 3072, 1), 0); del buf322  # reuse
        # Source Nodes: [add_29, mul_21], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf326, addmm_35, tanh_5, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_35
        del tanh_5
        buf327 = reinterpret_tensor(buf314, (2048, 768), (768, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (2048, 3072), (3072, 1), 0), permute_147, out=buf327)
        buf333 = reinterpret_tensor(buf310, (4, 512, 768), (393216, 768, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf317, buf327, primals_16, mul_43, div_44, buf333, 2048, 768, grid=grid(2048), stream=stream0)
        del div_44
        buf338 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (2048, 768), (768, 1), 0), permute_151, out=buf338)
        buf342 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf338, buf342, 1572864, grid=grid(1572864), stream=stream0)
        buf343 = reinterpret_tensor(buf338, (48, 512, 64), (32768, 64, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_354, reinterpret_tensor(buf342, (48, 512, 64), (32768, 64, 1), 0), out=buf343)
        del permute_354
        buf349 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf343, buf349, 1572864, grid=grid(1572864), stream=stream0)
        buf350 = reinterpret_tensor(buf343, (2048, 768), (768, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf349, permute_164, out=buf350)
        buf344 = reinterpret_tensor(buf296, (48, 512, 512), (262144, 512, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (48, 512, 64), (32768, 64, 1), 0), permute_355, out=buf344)
        del permute_355
        buf346 = reinterpret_tensor(buf294, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf344, alias_39, buf346, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_39
        buf347 = reinterpret_tensor(buf342, (48, 64, 512), (32768, 512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_356, reinterpret_tensor(buf346, (48, 512, 512), (262144, 512, 1), 0), out=buf347)
        del permute_356
        buf354 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf347, buf354, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf355 = reinterpret_tensor(buf347, (2048, 768), (768, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf354, permute_168, out=buf355)
        buf348 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (48, 512, 512), (262144, 512, 1), 0), permute_357, out=buf348)
        del permute_357
        buf359 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf348, buf359, 1572864, grid=grid(1572864), stream=stream0)
        buf360 = reinterpret_tensor(buf348, (2048, 768), (768, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf359, permute_172, out=buf360)
        buf340 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf370 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf333, buf350, buf355, buf360, mul_41, buf340, buf368, buf370, 12288, 128, grid=grid(12288), stream=stream0)
        buf364 = reinterpret_tensor(buf350, (4, 512, 768), (393216, 768, 1), 0); del buf350  # reuse
        buf367 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf364, buf333, buf355, buf360, primals_22, mul_41, div_46, buf367, 2048, 768, grid=grid(2048), stream=stream0)
        del div_46
        del mul_41
        buf372 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (2048, 768), (768, 1), 0), permute_143, out=buf372)
        buf376 = reinterpret_tensor(buf372, (4, 512, 3072), (1572864, 3072, 1), 0); del buf372  # reuse
        # Source Nodes: [add_24, mul_17], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf376, addmm_29, tanh_4, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_29
        del tanh_4
        buf377 = reinterpret_tensor(buf364, (2048, 768), (768, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (2048, 3072), (3072, 1), 0), permute_147, out=buf377)
        buf383 = reinterpret_tensor(buf360, (4, 512, 768), (393216, 768, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf367, buf377, primals_16, mul_35, div_47, buf383, 2048, 768, grid=grid(2048), stream=stream0)
        del div_47
        buf388 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (2048, 768), (768, 1), 0), permute_151, out=buf388)
        buf392 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf388, buf392, 1572864, grid=grid(1572864), stream=stream0)
        buf393 = reinterpret_tensor(buf388, (48, 512, 64), (32768, 64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_387, reinterpret_tensor(buf392, (48, 512, 64), (32768, 64, 1), 0), out=buf393)
        del permute_387
        buf399 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf393, buf399, 1572864, grid=grid(1572864), stream=stream0)
        buf400 = reinterpret_tensor(buf393, (2048, 768), (768, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, permute_164, out=buf400)
        buf394 = reinterpret_tensor(buf346, (48, 512, 512), (262144, 512, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (48, 512, 64), (32768, 64, 1), 0), permute_388, out=buf394)
        del permute_388
        buf396 = reinterpret_tensor(buf344, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf394, alias_41, buf396, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_41
        buf397 = reinterpret_tensor(buf392, (48, 64, 512), (32768, 512, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_389, reinterpret_tensor(buf396, (48, 512, 512), (262144, 512, 1), 0), out=buf397)
        del permute_389
        buf404 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf397, buf404, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf405 = reinterpret_tensor(buf397, (2048, 768), (768, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, permute_168, out=buf405)
        buf398 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (48, 512, 512), (262144, 512, 1), 0), permute_390, out=buf398)
        del permute_390
        buf409 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf398, buf409, 1572864, grid=grid(1572864), stream=stream0)
        buf410 = reinterpret_tensor(buf398, (2048, 768), (768, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf409, permute_172, out=buf410)
        buf390 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf418 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf420 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf383, buf400, buf405, buf410, mul_33, buf390, buf418, buf420, 12288, 128, grid=grid(12288), stream=stream0)
        buf414 = reinterpret_tensor(buf400, (4, 512, 768), (393216, 768, 1), 0); del buf400  # reuse
        buf417 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf414, buf383, buf405, buf410, primals_22, mul_33, div_49, buf417, 2048, 768, grid=grid(2048), stream=stream0)
        del div_49
        del mul_33
        buf424 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (2048, 768), (768, 1), 0), permute_143, out=buf424)
        buf430 = reinterpret_tensor(buf424, (4, 512, 3072), (1572864, 3072, 1), 0); del buf424  # reuse
        # Source Nodes: [add_19, mul_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf430, addmm_23, tanh_3, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_23
        del tanh_3
        buf431 = reinterpret_tensor(buf414, (2048, 768), (768, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf430, (2048, 3072), (3072, 1), 0), permute_147, out=buf431)
        buf439 = reinterpret_tensor(buf410, (4, 512, 768), (393216, 768, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf417, buf431, primals_16, mul_27, div_50, buf439, 2048, 768, grid=grid(2048), stream=stream0)
        del div_50
        buf446 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (2048, 768), (768, 1), 0), permute_151, out=buf446)
        buf452 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf446, buf452, 1572864, grid=grid(1572864), stream=stream0)
        buf453 = reinterpret_tensor(buf446, (48, 512, 64), (32768, 64, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_420, reinterpret_tensor(buf452, (48, 512, 64), (32768, 64, 1), 0), out=buf453)
        del permute_420
        buf459 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf453, buf459, 1572864, grid=grid(1572864), stream=stream0)
        buf460 = reinterpret_tensor(buf453, (2048, 768), (768, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf459, permute_164, out=buf460)
        buf454 = reinterpret_tensor(buf396, (48, 512, 512), (262144, 512, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (48, 512, 64), (32768, 64, 1), 0), permute_421, out=buf454)
        del permute_421
        buf456 = reinterpret_tensor(buf394, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf454, alias_43, buf456, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_43
        buf457 = reinterpret_tensor(buf452, (48, 64, 512), (32768, 512, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_422, reinterpret_tensor(buf456, (48, 512, 512), (262144, 512, 1), 0), out=buf457)
        del permute_422
        buf466 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf457, buf466, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf467 = reinterpret_tensor(buf457, (2048, 768), (768, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf466, permute_168, out=buf467)
        buf458 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf456, (48, 512, 512), (262144, 512, 1), 0), permute_423, out=buf458)
        del permute_423
        buf473 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf458, buf473, 1572864, grid=grid(1572864), stream=stream0)
        buf474 = reinterpret_tensor(buf458, (2048, 768), (768, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf473, permute_172, out=buf474)
        buf448 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf484 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf486 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf439, buf460, buf467, buf474, mul_25, buf448, buf484, buf486, 12288, 128, grid=grid(12288), stream=stream0)
        buf480 = reinterpret_tensor(buf460, (4, 512, 768), (393216, 768, 1), 0); del buf460  # reuse
        buf483 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf480, buf439, buf467, buf474, primals_22, mul_25, div_52, buf483, 2048, 768, grid=grid(2048), stream=stream0)
        del div_52
        del mul_25
        buf488 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (2048, 768), (768, 1), 0), permute_143, out=buf488)
        buf492 = reinterpret_tensor(buf488, (4, 512, 3072), (1572864, 3072, 1), 0); del buf488  # reuse
        # Source Nodes: [add_14, mul_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf492, addmm_17, tanh_2, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_17
        del tanh_2
        buf493 = reinterpret_tensor(buf480, (2048, 768), (768, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf492, (2048, 3072), (3072, 1), 0), permute_147, out=buf493)
        buf499 = reinterpret_tensor(buf474, (4, 512, 768), (393216, 768, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf483, buf493, primals_16, mul_19, div_53, buf499, 2048, 768, grid=grid(2048), stream=stream0)
        del div_53
        buf504 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (2048, 768), (768, 1), 0), permute_151, out=buf504)
        buf508 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf504, buf508, 1572864, grid=grid(1572864), stream=stream0)
        buf509 = reinterpret_tensor(buf504, (48, 512, 64), (32768, 64, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_453, reinterpret_tensor(buf508, (48, 512, 64), (32768, 64, 1), 0), out=buf509)
        del permute_453
        buf515 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf509, buf515, 1572864, grid=grid(1572864), stream=stream0)
        buf516 = reinterpret_tensor(buf509, (2048, 768), (768, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf515, permute_164, out=buf516)
        buf510 = reinterpret_tensor(buf456, (48, 512, 512), (262144, 512, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf508, (48, 512, 64), (32768, 64, 1), 0), permute_454, out=buf510)
        del permute_454
        buf512 = reinterpret_tensor(buf454, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf510, alias_45, buf512, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_45
        buf513 = reinterpret_tensor(buf508, (48, 64, 512), (32768, 512, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_455, reinterpret_tensor(buf512, (48, 512, 512), (262144, 512, 1), 0), out=buf513)
        del permute_455
        buf520 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf513, buf520, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf521 = reinterpret_tensor(buf513, (2048, 768), (768, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf520, permute_168, out=buf521)
        buf514 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf512, (48, 512, 512), (262144, 512, 1), 0), permute_456, out=buf514)
        del permute_456
        buf525 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf514, buf525, 1572864, grid=grid(1572864), stream=stream0)
        buf526 = reinterpret_tensor(buf514, (2048, 768), (768, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf525, permute_172, out=buf526)
        buf506 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf534 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf536 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf499, buf516, buf521, buf526, mul_17, buf506, buf534, buf536, 12288, 128, grid=grid(12288), stream=stream0)
        buf530 = reinterpret_tensor(buf516, (4, 512, 768), (393216, 768, 1), 0); del buf516  # reuse
        buf533 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf530, buf499, buf521, buf526, primals_22, mul_17, div_55, buf533, 2048, 768, grid=grid(2048), stream=stream0)
        del div_55
        del mul_17
        buf538 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (2048, 768), (768, 1), 0), permute_143, out=buf538)
        buf542 = reinterpret_tensor(buf538, (4, 512, 3072), (1572864, 3072, 1), 0); del buf538  # reuse
        # Source Nodes: [add_9, mul_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf542, addmm_11, tanh_1, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_11
        del tanh_1
        buf543 = reinterpret_tensor(buf530, (2048, 768), (768, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (2048, 3072), (3072, 1), 0), permute_147, out=buf543)
        buf549 = reinterpret_tensor(buf526, (4, 512, 768), (393216, 768, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf533, buf543, primals_16, mul_11, div_56, buf549, 2048, 768, grid=grid(2048), stream=stream0)
        del div_56
        buf554 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (2048, 768), (768, 1), 0), permute_151, out=buf554)
        buf558 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf554, buf558, 1572864, grid=grid(1572864), stream=stream0)
        buf559 = reinterpret_tensor(buf554, (48, 512, 64), (32768, 64, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_486, reinterpret_tensor(buf558, (48, 512, 64), (32768, 64, 1), 0), out=buf559)
        del permute_486
        buf565 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf559, buf565, 1572864, grid=grid(1572864), stream=stream0)
        buf566 = reinterpret_tensor(buf559, (2048, 768), (768, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf565, permute_164, out=buf566)
        buf560 = reinterpret_tensor(buf512, (48, 512, 512), (262144, 512, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf558, (48, 512, 64), (32768, 64, 1), 0), permute_487, out=buf560)
        del permute_487
        buf562 = reinterpret_tensor(buf510, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf560, alias_47, buf562, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_47
        buf563 = reinterpret_tensor(buf558, (48, 64, 512), (32768, 512, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_488, reinterpret_tensor(buf562, (48, 512, 512), (262144, 512, 1), 0), out=buf563)
        del permute_488
        buf570 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf563, buf570, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf571 = reinterpret_tensor(buf563, (2048, 768), (768, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf570, permute_168, out=buf571)
        buf564 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf562, (48, 512, 512), (262144, 512, 1), 0), permute_489, out=buf564)
        del permute_489
        buf575 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf564, buf575, 1572864, grid=grid(1572864), stream=stream0)
        buf576 = reinterpret_tensor(buf564, (2048, 768), (768, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf575, permute_172, out=buf576)
        buf556 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf584 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf586 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf549, buf566, buf571, buf576, mul_9, buf556, buf584, buf586, 12288, 128, grid=grid(12288), stream=stream0)
        buf40 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf33, buf50, buf55, buf60, mul_89, buf40, buf68, buf70, 12288, 128, grid=grid(12288), stream=stream0)
        del buf50
        del buf55
        del buf60
        del mul_89
        buf119 = empty((768, ), device='cuda', dtype=torch.float32)
        buf422 = buf119; del buf119  # reuse
        buf588 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf588, buf18, buf68, buf118, buf168, buf218, buf268, buf484, buf318, buf534, buf368, buf584, buf418, 768, 16, grid=grid(768), stream=stream0)
        buf121 = empty((768, ), device='cuda', dtype=torch.float32)
        buf423 = buf121; del buf121  # reuse
        buf589 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf589, buf20, buf70, buf120, buf170, buf220, buf270, buf486, buf320, buf536, buf370, buf586, buf420, 768, 16, grid=grid(768), stream=stream0)
        buf23 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (768, 2048), (1, 768), 0), view_264, out=buf23)
        del view_264
        buf24 = reinterpret_tensor(buf70, (1, 768, 16), (12288, 1, 768), 0); del buf70  # reuse
        buf34 = buf586; del buf586  # reuse
        buf36 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf17, buf27, mul_91, buf24, buf34, buf36, 12288, 128, grid=grid(12288), stream=stream0)
        del buf17
        del buf27
        del mul_91
        buf124 = reinterpret_tensor(buf486, (1, 768, 16), (12288, 1, 768), 0); del buf486  # reuse
        buf134 = buf420; del buf420  # reuse
        buf136 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf117, buf127, mul_75, buf124, buf134, buf136, 12288, 128, grid=grid(12288), stream=stream0)
        del buf127
        del mul_75
        buf174 = reinterpret_tensor(buf320, (1, 768, 16), (12288, 1, 768), 0); del buf320  # reuse
        buf184 = buf270; del buf270  # reuse
        buf186 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf167, buf177, mul_67, buf174, buf184, buf186, 12288, 128, grid=grid(12288), stream=stream0)
        del buf177
        del mul_67
        buf224 = reinterpret_tensor(buf20, (1, 768, 16), (12288, 1, 768), 0); del buf20  # reuse
        buf234 = buf170; del buf170  # reuse
        buf236 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf217, buf227, mul_59, buf224, buf234, buf236, 12288, 128, grid=grid(12288), stream=stream0)
        del buf227
        del mul_59
        buf274 = reinterpret_tensor(buf68, (1, 768, 16), (12288, 1, 768), 0); del buf68  # reuse
        buf284 = buf584; del buf584  # reuse
        buf286 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf267, buf277, mul_51, buf274, buf284, buf286, 12288, 128, grid=grid(12288), stream=stream0)
        del buf277
        del mul_51
        buf324 = reinterpret_tensor(buf484, (1, 768, 16), (12288, 1, 768), 0); del buf484  # reuse
        buf334 = buf418; del buf418  # reuse
        buf336 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf317, buf327, mul_43, buf324, buf334, buf336, 12288, 128, grid=grid(12288), stream=stream0)
        del buf327
        del mul_43
        buf374 = reinterpret_tensor(buf318, (1, 768, 16), (12288, 1, 768), 0); del buf318  # reuse
        buf384 = buf268; del buf268  # reuse
        buf386 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf367, buf377, mul_35, buf374, buf384, buf386, 12288, 128, grid=grid(12288), stream=stream0)
        del buf377
        del mul_35
        buf426 = reinterpret_tensor(buf18, (1, 768, 16), (12288, 1, 768), 0); del buf18  # reuse
        buf440 = buf168; del buf168  # reuse
        buf442 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf417, buf431, mul_27, buf426, buf440, buf442, 12288, 128, grid=grid(12288), stream=stream0)
        del buf431
        del mul_27
        buf490 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf500 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf502 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf483, buf493, mul_19, buf490, buf500, buf502, 12288, 128, grid=grid(12288), stream=stream0)
        del buf493
        del mul_19
        buf540 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf550 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf552 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf533, buf543, mul_11, buf540, buf550, buf552, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_11
        buf580 = reinterpret_tensor(buf566, (4, 512, 768), (393216, 768, 1), 0); del buf566  # reuse
        buf583 = reinterpret_tensor(buf543, (4, 512, 768), (393216, 768, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf580, buf549, buf571, buf576, primals_22, mul_9, div_58, buf583, 2048, 768, grid=grid(2048), stream=stream0)
        del buf571
        del buf576
        del div_58
        del mul_9
        del primals_22
        buf590 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (2048, 768), (768, 1), 0), permute_143, out=buf590)
        del permute_143
        buf596 = reinterpret_tensor(buf590, (4, 512, 3072), (1572864, 3072, 1), 0); del buf590  # reuse
        # Source Nodes: [add_4, mul_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_7.run(buf596, addmm_5, tanh, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_5
        del tanh
        buf597 = reinterpret_tensor(buf580, (2048, 768), (768, 1), 0); del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (2048, 3072), (3072, 1), 0), permute_147, out=buf597)
        del permute_147
        buf592 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf606 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf608 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf583, buf597, mul_3, buf592, buf606, buf608, 12288, 128, grid=grid(12288), stream=stream0)
        buf74 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf86 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf67, buf77, mul_83, buf74, buf84, buf86, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_83
        buf125 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf428 = reinterpret_tensor(buf125, (768, ), (1, ), 0); del buf125  # reuse
        buf594 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf594, buf24, buf74, buf124, buf174, buf224, buf274, buf490, buf324, buf540, buf374, buf592, buf426, 768, 16, grid=grid(768), stream=stream0)
        del buf124
        del buf174
        del buf224
        del buf24
        del buf274
        del buf324
        del buf374
        del buf426
        del buf490
        del buf540
        del buf592
        del buf74
        buf28 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (3072, 2048), (1, 3072), 0), view_262, out=buf28)
        del view_262
        buf29 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf26, buf29, 49152, 128, grid=grid(49152), stream=stream0)
        del buf26
        buf129 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf126, buf129, 49152, 128, grid=grid(49152), stream=stream0)
        buf179 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf176, buf179, 49152, 128, grid=grid(49152), stream=stream0)
        buf229 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf226, buf229, 49152, 128, grid=grid(49152), stream=stream0)
        buf279 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf276, buf279, 49152, 128, grid=grid(49152), stream=stream0)
        buf329 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf326, buf329, 49152, 128, grid=grid(49152), stream=stream0)
        buf379 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf376, buf379, 49152, 128, grid=grid(49152), stream=stream0)
        buf433 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf430, buf433, 49152, 128, grid=grid(49152), stream=stream0)
        buf495 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf492, buf495, 49152, 128, grid=grid(49152), stream=stream0)
        buf545 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf542, buf545, 49152, 128, grid=grid(49152), stream=stream0)
        buf599 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf596, buf599, 49152, 128, grid=grid(49152), stream=stream0)
        buf79 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf76, buf79, 49152, 128, grid=grid(49152), stream=stream0)
        buf130 = empty((1, 3072), device='cuda', dtype=torch.float32)
        buf435 = reinterpret_tensor(buf130, (3072, ), (1, ), 0); del buf130  # reuse
        buf601 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_20.run(buf601, buf29, buf79, buf129, buf179, buf229, buf279, buf495, buf329, buf545, buf379, buf599, buf433, 3072, 16, grid=grid(3072), stream=stream0)
        del buf129
        del buf179
        del buf229
        del buf279
        del buf29
        del buf329
        del buf379
        del buf433
        del buf495
        del buf545
        del buf599
        del buf79
        buf135 = empty((768, ), device='cuda', dtype=torch.float32)
        buf444 = buf135; del buf135  # reuse
        buf610 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf610, buf34, buf84, buf134, buf184, buf234, buf284, buf500, buf334, buf550, buf384, buf606, buf440, 768, 16, grid=grid(768), stream=stream0)
        del buf134
        del buf184
        del buf234
        del buf284
        del buf334
        del buf34
        del buf384
        del buf440
        del buf500
        del buf550
        del buf606
        del buf84
        buf137 = empty((768, ), device='cuda', dtype=torch.float32)
        buf445 = buf137; del buf137  # reuse
        buf611 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf611, buf36, buf86, buf136, buf186, buf236, buf286, buf502, buf336, buf552, buf386, buf608, buf442, 768, 16, grid=grid(768), stream=stream0)
        del buf136
        del buf186
        del buf236
        del buf286
        del buf336
        del buf36
        del buf386
        del buf442
        del buf502
        del buf552
        del buf608
        buf39 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (768, 2048), (1, 768), 0), view_260, out=buf39)
        del view_260
        buf605 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf583, buf597, primals_16, mul_3, div_59, buf605, 2048, 768, grid=grid(2048), stream=stream0)
        del div_59
        del mul_3
        del primals_16
        buf614 = reinterpret_tensor(buf86, (1, 768, 16), (12288, 1, 768), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf605, buf614, 12288, 128, grid=grid(12288), stream=stream0)
        buf141 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf450 = reinterpret_tensor(buf141, (768, ), (1, ), 0); del buf141  # reuse
        buf616 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf616, buf40, buf90, buf140, buf190, buf240, buf290, buf506, buf340, buf556, buf390, buf614, buf448, 768, 16, grid=grid(768), stream=stream0)
        buf51 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (768, 2048), (1, 768), 0), view_244, out=buf51)
        buf52 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf49, buf52, 12288, 128, grid=grid(12288), stream=stream0)
        buf102 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf99, buf102, 12288, 128, grid=grid(12288), stream=stream0)
        buf152 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf149, buf152, 12288, 128, grid=grid(12288), stream=stream0)
        buf202 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf199, buf202, 12288, 128, grid=grid(12288), stream=stream0)
        buf252 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf249, buf252, 12288, 128, grid=grid(12288), stream=stream0)
        buf302 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf299, buf302, 12288, 128, grid=grid(12288), stream=stream0)
        buf352 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf349, buf352, 12288, 128, grid=grid(12288), stream=stream0)
        buf402 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf399, buf402, 12288, 128, grid=grid(12288), stream=stream0)
        buf462 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf459, buf462, 12288, 128, grid=grid(12288), stream=stream0)
        buf518 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf515, buf518, 12288, 128, grid=grid(12288), stream=stream0)
        buf568 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf565, buf568, 12288, 128, grid=grid(12288), stream=stream0)
        buf612 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf605, (2048, 768), (768, 1), 0), permute_151, out=buf612)
        del permute_151
        buf618 = reinterpret_tensor(buf597, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf612, buf618, 1572864, grid=grid(1572864), stream=stream0)
        buf619 = reinterpret_tensor(buf612, (48, 512, 64), (32768, 64, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_519, reinterpret_tensor(buf618, (48, 512, 64), (32768, 64, 1), 0), out=buf619)
        del permute_519
        buf625 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf619, buf625, 1572864, grid=grid(1572864), stream=stream0)
        del buf619
        buf628 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf625, buf628, 12288, 128, grid=grid(12288), stream=stream0)
        buf103 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf464 = reinterpret_tensor(buf103, (768, ), (1, ), 0); del buf103  # reuse
        buf630 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf630, buf52, buf102, buf152, buf202, buf252, buf302, buf518, buf352, buf568, buf402, buf628, buf462, 768, 16, grid=grid(768), stream=stream0)
        buf56 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (768, 2048), (1, 768), 0), view_244, out=buf56)
        buf57 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf54, buf57, 12288, 128, grid=grid(12288), stream=stream0)
        buf107 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf104, buf107, 12288, 128, grid=grid(12288), stream=stream0)
        buf157 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf154, buf157, 12288, 128, grid=grid(12288), stream=stream0)
        buf207 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf204, buf207, 12288, 128, grid=grid(12288), stream=stream0)
        buf257 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf254, buf257, 12288, 128, grid=grid(12288), stream=stream0)
        buf307 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf304, buf307, 12288, 128, grid=grid(12288), stream=stream0)
        buf357 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf354, buf357, 12288, 128, grid=grid(12288), stream=stream0)
        buf407 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf404, buf407, 12288, 128, grid=grid(12288), stream=stream0)
        buf469 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf466, buf469, 12288, 128, grid=grid(12288), stream=stream0)
        buf523 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf520, buf523, 12288, 128, grid=grid(12288), stream=stream0)
        buf573 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf570, buf573, 12288, 128, grid=grid(12288), stream=stream0)
        buf620 = reinterpret_tensor(buf562, (48, 512, 512), (262144, 512, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf618, (48, 512, 64), (32768, 64, 1), 0), permute_520, out=buf620)
        del permute_520
        buf622 = reinterpret_tensor(buf560, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_11.run(buf620, alias_49, buf622, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_49
        del buf620
        buf623 = reinterpret_tensor(buf618, (48, 64, 512), (32768, 512, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_521, reinterpret_tensor(buf622, (48, 512, 512), (262144, 512, 1), 0), out=buf623)
        del permute_521
        buf632 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf623, buf632, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf635 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf632, buf635, 12288, 128, grid=grid(12288), stream=stream0)
        buf108 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf471 = reinterpret_tensor(buf108, (768, ), (1, ), 0); del buf108  # reuse
        buf637 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf637, buf57, buf107, buf157, buf207, buf257, buf307, buf523, buf357, buf573, buf407, buf635, buf469, 768, 16, grid=grid(768), stream=stream0)
        buf61 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (768, 2048), (1, 768), 0), view_244, out=buf61)
        del view_244
        buf62 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf59, buf62, 12288, 128, grid=grid(12288), stream=stream0)
        buf112 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf109, buf112, 12288, 128, grid=grid(12288), stream=stream0)
        buf162 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf159, buf162, 12288, 128, grid=grid(12288), stream=stream0)
        buf212 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf209, buf212, 12288, 128, grid=grid(12288), stream=stream0)
        buf262 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf259, buf262, 12288, 128, grid=grid(12288), stream=stream0)
        buf312 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf309, buf312, 12288, 128, grid=grid(12288), stream=stream0)
        buf362 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf359, buf362, 12288, 128, grid=grid(12288), stream=stream0)
        buf412 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf409, buf412, 12288, 128, grid=grid(12288), stream=stream0)
        buf476 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf473, buf476, 12288, 128, grid=grid(12288), stream=stream0)
        buf528 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf525, buf528, 12288, 128, grid=grid(12288), stream=stream0)
        buf578 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf575, buf578, 12288, 128, grid=grid(12288), stream=stream0)
        buf624 = reinterpret_tensor(buf59, (48, 512, 64), (32768, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf622, (48, 512, 512), (262144, 512, 1), 0), permute_522, out=buf624)
        del buf622
        del permute_522
        buf639 = reinterpret_tensor(buf623, (2048, 768), (768, 1), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf624, buf639, 1572864, grid=grid(1572864), stream=stream0)
        del buf624
        buf642 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_21.run(buf639, buf642, 12288, 128, grid=grid(12288), stream=stream0)
        buf113 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf478 = reinterpret_tensor(buf113, (768, ), (1, ), 0); del buf113  # reuse
        buf644 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf644, buf62, buf112, buf162, buf212, buf262, buf312, buf528, buf362, buf578, buf412, buf642, buf476, 768, 16, grid=grid(768), stream=stream0)
        del buf112
        del buf162
        del buf212
        del buf262
        del buf312
        del buf362
        del buf412
        del buf476
        del buf528
        del buf578
        del buf62
        buf73 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (768, 2048), (1, 768), 0), view_242, out=buf73)
        del buf67
        del view_242
        buf78 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (3072, 2048), (1, 3072), 0), view_240, out=buf78)
        del buf76
        del view_240
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 2048), (1, 768), 0), view_238, out=buf89)
        del buf83
        del view_238
        buf101 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 2048), (1, 768), 0), view_222, out=buf101)
        del buf99
        buf106 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (768, 2048), (1, 768), 0), view_222, out=buf106)
        del buf104
        buf111 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (768, 2048), (1, 768), 0), view_222, out=buf111)
        del buf109
        del view_222
        buf123 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (768, 2048), (1, 768), 0), view_220, out=buf123)
        del buf117
        del view_220
        buf128 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (3072, 2048), (1, 3072), 0), view_218, out=buf128)
        del buf126
        del view_218
        buf139 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (768, 2048), (1, 768), 0), view_216, out=buf139)
        del buf133
        del view_216
        buf151 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (768, 2048), (1, 768), 0), view_200, out=buf151)
        del buf149
        buf156 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (768, 2048), (1, 768), 0), view_200, out=buf156)
        del buf154
        buf161 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (768, 2048), (1, 768), 0), view_200, out=buf161)
        del buf159
        del view_200
        buf173 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (768, 2048), (1, 768), 0), view_198, out=buf173)
        del buf167
        del view_198
        buf178 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (3072, 2048), (1, 3072), 0), view_196, out=buf178)
        del buf176
        del view_196
        buf189 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (768, 2048), (1, 768), 0), view_194, out=buf189)
        del buf183
        del view_194
        buf201 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (768, 2048), (1, 768), 0), view_178, out=buf201)
        del buf199
        buf206 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (768, 2048), (1, 768), 0), view_178, out=buf206)
        del buf204
        buf211 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (768, 2048), (1, 768), 0), view_178, out=buf211)
        del buf209
        del view_178
        buf223 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (768, 2048), (1, 768), 0), view_176, out=buf223)
        del buf217
        del view_176
        buf228 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (3072, 2048), (1, 3072), 0), view_174, out=buf228)
        del buf226
        del view_174
        buf239 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (768, 2048), (1, 768), 0), view_172, out=buf239)
        del buf233
        del view_172
        buf251 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (768, 2048), (1, 768), 0), view_156, out=buf251)
        del buf249
        buf256 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (768, 2048), (1, 768), 0), view_156, out=buf256)
        del buf254
        buf261 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (768, 2048), (1, 768), 0), view_156, out=buf261)
        del buf259
        del view_156
        buf273 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (768, 2048), (1, 768), 0), view_154, out=buf273)
        del buf267
        del view_154
        buf278 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (3072, 2048), (1, 3072), 0), view_152, out=buf278)
        del buf276
        del view_152
        buf289 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (768, 2048), (1, 768), 0), view_150, out=buf289)
        del buf283
        del view_150
        buf301 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (768, 2048), (1, 768), 0), view_134, out=buf301)
        del buf299
        buf306 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (768, 2048), (1, 768), 0), view_134, out=buf306)
        del buf304
        buf311 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (768, 2048), (1, 768), 0), view_134, out=buf311)
        del buf309
        del view_134
        buf323 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (768, 2048), (1, 768), 0), view_132, out=buf323)
        del buf317
        del view_132
        buf328 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (3072, 2048), (1, 3072), 0), view_130, out=buf328)
        del buf326
        del view_130
        buf339 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (768, 2048), (1, 768), 0), view_128, out=buf339)
        del buf333
        del view_128
        buf351 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (768, 2048), (1, 768), 0), view_112, out=buf351)
        del buf349
        buf356 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (768, 2048), (1, 768), 0), view_112, out=buf356)
        del buf354
        buf361 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (768, 2048), (1, 768), 0), view_112, out=buf361)
        del buf359
        del view_112
        buf373 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (768, 2048), (1, 768), 0), view_110, out=buf373)
        del buf367
        del view_110
        buf378 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (3072, 2048), (1, 3072), 0), view_108, out=buf378)
        del buf376
        del view_108
        buf389 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (768, 2048), (1, 768), 0), view_106, out=buf389)
        del buf383
        del view_106
        buf401 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (768, 2048), (1, 768), 0), view_90, out=buf401)
        del buf399
        buf406 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (768, 2048), (1, 768), 0), view_90, out=buf406)
        del buf404
        buf411 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (768, 2048), (1, 768), 0), view_90, out=buf411)
        del buf409
        del view_90
        buf425 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (768, 2048), (1, 768), 0), view_88, out=buf425)
        del buf417
        del view_88
        buf489 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (768, 2048), (1, 768), 0), view_66, out=buf489)
        del buf483
        del view_66
        buf539 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (768, 2048), (1, 768), 0), view_44, out=buf539)
        del buf533
        del view_44
        buf591 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (768, 2048), (1, 768), 0), view_22, out=buf591)
        del buf583
        del view_22
        buf429 = buf123; del buf123  # reuse
        buf595 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf595, buf23, buf73, buf173, buf223, buf273, buf323, buf373, buf425, buf489, buf539, buf591, 2359296, grid=grid(2359296), stream=stream0)
        del buf173
        del buf223
        del buf23
        del buf273
        del buf323
        del buf373
        del buf425
        buf432 = reinterpret_tensor(buf73, (3072, 768), (768, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf430, (3072, 2048), (1, 3072), 0), view_86, out=buf432)
        del buf430
        del view_86
        buf494 = reinterpret_tensor(buf591, (3072, 768), (768, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf492, (3072, 2048), (1, 3072), 0), view_64, out=buf494)
        del buf492
        del view_64
        buf544 = reinterpret_tensor(buf539, (3072, 768), (768, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (3072, 2048), (1, 3072), 0), view_42, out=buf544)
        del buf542
        del view_42
        buf598 = reinterpret_tensor(buf489, (3072, 768), (768, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (3072, 2048), (1, 3072), 0), view_20, out=buf598)
        del buf596
        del view_20
        buf436 = buf128; del buf128  # reuse
        buf602 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf602, buf28, buf78, buf178, buf228, buf278, buf328, buf378, buf432, buf494, buf544, buf598, 2359296, grid=grid(2359296), stream=stream0)
        del buf178
        del buf228
        del buf278
        del buf28
        del buf328
        del buf378
        del buf432
        del buf494
        del buf544
        del buf598
        del buf78
        buf447 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (768, 2048), (1, 768), 0), view_84, out=buf447)
        del buf439
        del view_84
        buf505 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (768, 2048), (1, 768), 0), view_62, out=buf505)
        del buf499
        del view_62
        buf555 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (768, 2048), (1, 768), 0), view_40, out=buf555)
        del buf549
        del view_40
        buf613 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf605, (768, 2048), (1, 768), 0), view_18, out=buf613)
        del view_18
        buf451 = buf139; del buf139  # reuse
        buf617 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_24.run(buf617, buf39, buf89, buf189, buf239, buf289, buf339, buf389, buf447, buf505, buf555, buf613, 589824, grid=grid(589824), stream=stream0)
        del buf189
        del buf239
        del buf289
        del buf339
        del buf389
        del buf39
        del buf447
        buf461 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (768, 2048), (1, 768), 0), view_68, out=buf461)
        del buf459
        buf517 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (768, 2048), (1, 768), 0), view_46, out=buf517)
        del buf515
        buf567 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (768, 2048), (1, 768), 0), view_24, out=buf567)
        del buf565
        buf627 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (768, 2048), (1, 768), 0), view_2, out=buf627)
        buf465 = buf101; del buf101  # reuse
        buf631 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf631, buf51, buf151, buf201, buf251, buf301, buf351, buf401, buf461, buf517, buf567, buf627, 589824, grid=grid(589824), stream=stream0)
        del buf151
        del buf201
        del buf251
        del buf301
        del buf351
        del buf401
        del buf461
        buf468 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (768, 2048), (1, 768), 0), view_68, out=buf468)
        del buf466
        buf522 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (768, 2048), (1, 768), 0), view_46, out=buf522)
        del buf520
        buf572 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (768, 2048), (1, 768), 0), view_24, out=buf572)
        del buf570
        buf634 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf632, (768, 2048), (1, 768), 0), view_2, out=buf634)
        buf472 = buf106; del buf106  # reuse
        buf638 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf638, buf56, buf156, buf206, buf256, buf306, buf356, buf406, buf468, buf522, buf572, buf634, 589824, grid=grid(589824), stream=stream0)
        del buf156
        del buf206
        del buf256
        del buf306
        del buf356
        del buf406
        del buf468
        buf475 = buf634; del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (768, 2048), (1, 768), 0), view_68, out=buf475)
        del buf473
        del view_68
        buf527 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (768, 2048), (1, 768), 0), view_46, out=buf527)
        del buf525
        del view_46
        buf577 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (768, 2048), (1, 768), 0), view_24, out=buf577)
        del view_24
        buf641 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf639, (768, 2048), (1, 768), 0), view_2, out=buf641)
        del view_2
        buf479 = buf111; del buf111  # reuse
        buf645 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf645, buf61, buf161, buf211, buf261, buf311, buf361, buf411, buf475, buf527, buf577, buf641, 589824, grid=grid(589824), stream=stream0)
        del buf161
        del buf211
        del buf261
        del buf311
        del buf361
        del buf411
        del buf475
        del buf527
        del buf577
        del buf61
        del buf641
        buf626 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf625, permute_164, out=buf626)
        del permute_164
        buf633 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf632, permute_168, out=buf633)
        del permute_168
        buf640 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf639, permute_172, out=buf640)
        del buf639
        del permute_172
        buf646 = buf605; del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf646, buf626, buf633, buf640, 1572864, grid=grid(1572864), stream=stream0)
        del buf626
        del buf633
        del buf640
        buf647 = reinterpret_tensor(buf10, (2048, 128), (128, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf646, (2048, 768), (768, 1), 0), permute_539, out=buf647)
        del permute_539
        buf648 = empty((768, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf646, (768, 2048), (1, 768), 0), view, out=buf648)
        del view
        buf649 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf646, buf649, 12288, 128, grid=grid(12288), stream=stream0)
        del buf646
        buf650 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf649, buf650, 768, 16, grid=grid(768), stream=stream0)
        del buf649
        buf653 = reinterpret_tensor(buf0, (4, 512, 128), (65536, 128, 1), 0); del buf0  # reuse
        buf663 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf667 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward]
        triton_per_fused_embedding_dense_backward_native_layer_norm_backward_28.run(buf647, primals_4, mul_1, div_61, expand, primals_32, buf653, buf663, buf667, 2048, 128, grid=grid(2048), stream=stream0)
        del div_61
        del primals_4
        buf654 = reinterpret_tensor(buf13, (128, 16), (1, 128), 0); del buf13  # reuse
        buf656 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf647, mul_1, buf654, buf656, 2048, 128, grid=grid(2048), stream=stream0)
        del buf647
        del mul_1
        buf655 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf654, buf655, 128, 16, grid=grid(128), stream=stream0)
        del buf654
        buf657 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf656, buf657, 128, 16, grid=grid(128), stream=stream0)
        del buf656
        buf658 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_30.run(buf658, 65536, grid=grid(65536), stream=stream0)
        buf659 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]
        triton_poi_fused_embedding_dense_backward_sum_31.run(slice_2, buf653, buf659, 65536, grid=grid(65536), stream=stream0)
        del buf653
        aten.index_put_(buf658, [slice_2], buf659, True)
        del buf659
        del slice_2
        buf662 = empty((2, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_32.run(buf662, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf662, [expand], buf663, True)
        del buf663
        del expand
        buf666 = empty((30000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_33.run(buf666, 3840000, grid=grid(3840000), stream=stream0)
        aten.index_put_(buf666, [primals_32], buf667, True)
        del buf667
        del primals_32
        return (buf666, buf662, buf658, buf655, buf657, reinterpret_tensor(buf648, (768, 128), (128, 1), 0), reinterpret_tensor(buf650, (768, ), (1, ), 0), buf645, buf644, buf638, buf637, buf631, buf630, buf617, buf616, buf610, buf611, buf602, buf601, buf595, buf594, buf588, buf589, reinterpret_tensor(buf12, (128, 768), (768, 1), 0), reinterpret_tensor(buf14, (128, ), (1, ), 0), buf7, buf9, reinterpret_tensor(buf1, (30000, 128), (128, 1), 0), reinterpret_tensor(buf2, (30000, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((4, 512), (0, 1), device='cuda:0', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((4, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_19 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_156 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_59 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_83 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_73 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tanh_12 = rand_strided((4, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_268 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_357 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_489 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 512, 30000), (15360000, 30000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_26, primals_32, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, permute_135, permute_139, div_25, permute_143, permute_147, div_26, permute_151, permute_156, permute_157, alias_27, permute_158, permute_159, permute_164, permute_168, permute_172, div_28, div_29, permute_189, permute_190, alias_29, permute_191, permute_192, div_31, div_32, permute_222, permute_223, alias_31, permute_224, permute_225, div_34, div_35, permute_255, permute_256, alias_33, permute_257, permute_258, div_37, div_38, permute_288, permute_289, alias_35, permute_290, permute_291, div_40, div_41, permute_321, permute_322, alias_37, permute_323, permute_324, div_43, div_44, permute_354, permute_355, alias_39, permute_356, permute_357, div_46, div_47, permute_387, permute_388, alias_41, permute_389, permute_390, div_49, div_50, permute_420, permute_421, alias_43, permute_422, permute_423, div_52, div_53, permute_453, permute_454, alias_45, permute_455, permute_456, div_55, div_56, permute_486, permute_487, alias_47, permute_488, permute_489, div_58, div_59, permute_519, permute_520, alias_49, permute_521, permute_522, permute_539, div_61, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Albert', benchmark_compiled_module)
