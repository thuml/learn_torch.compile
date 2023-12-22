
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


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyr37rvw654u36me7fsxvbdo4zvzsxll42d6xmxwuancaod2fij.py
# Source Nodes: [inputs_embeds], Original ATen: [aten.embedding]
# inputs_embeds => embedding
triton_poi_fused_embedding_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (x0 + (512*tmp3)), None)
    tl.store(out_ptr0 + (x2), tmp4, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmdmdajda5wzjtyhffrneehduozkzxipt4ok7mnokoi6a4cubbn.py
# Source Nodes: [add, hidden_states_1, l__mod___encoder_block_0_layer_0_self_attention_q, normed_hidden_states, pow_1, rsqrt, variance], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add => add
# hidden_states_1 => mul_1
# l__mod___encoder_block_0_layer_0_self_attention_q => view_1
# normed_hidden_states => mul_2
# pow_1 => pow_1
# rsqrt => rsqrt
# variance => mean
triton_per_fused_add_mean_mul_pow_rsqrt_view_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp0 * tmp10
    tmp13 = tmp11 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwwg7j5y2aj6pvcu6bc3lbmhlxiaantzgjztad3su62kjrf4bhx.py
# Source Nodes: [float_1, full_like, gt, is_small, log, mul_3, mul_4, relative_buckets, relative_position, relative_position_1, relative_position_bucket, relative_position_if_large, relative_position_if_large_1, to_2, to_3, truediv, truediv_1, where], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.full_like, aten.gt, aten.log, aten.lt, aten.minimum, aten.mul, aten.sub, aten.where]
# float_1 => convert_element_type_1
# full_like => full_default_1
# gt => gt
# is_small => lt
# log => log
# mul_3 => mul_3
# mul_4 => mul_4
# relative_buckets => add_1
# relative_position => sub_1
# relative_position_1 => abs_1
# relative_position_bucket => add_3
# relative_position_if_large => add_2
# relative_position_if_large_1 => minimum
# to_2 => convert_element_type
# to_3 => convert_element_type_2
# truediv => div
# truediv_1 => div_1
# where => where
triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1], 16, tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp1
    tmp7 = tl.abs(tmp0)
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp7.to(tl.float32)
    tmp11 = 8.0
    tmp12 = tmp10 / tmp11
    tmp13 = tl.log(tmp12)
    tmp14 = 2.772588722239781
    tmp15 = tmp13 / tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17 + tmp8
    tmp19 = tl.full([1], 15, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp7, tmp20)
    tmp22 = tmp6 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllvgw2nxy2ki5xtv2ahwyub25oimxgwzrn7kpzk53xj2pwza7l6.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2 + ((-1)*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 > tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full([1, 1], 16, tl.int64)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6 + tmp2
        tmp8 = tl.abs(tmp1)
        tmp9 = tl.full([1, 1], 8, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp8.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = tl.log(tmp13)
        tmp15 = 2.772588722239781
        tmp16 = tmp14 / tmp15
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp18 + tmp9
        tmp20 = tl.full([1, 1], 15, tl.int64)
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp8, tmp21)
        tmp23 = tmp7 + tmp22
        tmp24 = tmp23 + 32
        tmp25 = tmp23 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp23)
        tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~rmask, "index out of bounds: 0 <= tmp26 < 32")
        tmp27 = tl.load(in_ptr1 + (x1 + (8*tmp26)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp0 + tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp64 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = r2 + ((-1)*x0)
        tmp34 = tl.full([1, 1], 0, tl.int64)
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp34
        tmp40 = tl.abs(tmp33)
        tmp41 = tl.full([1, 1], 8, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp40.to(tl.float32)
        tmp44 = 8.0
        tmp45 = tmp43 / tmp44
        tmp46 = tl.log(tmp45)
        tmp47 = 2.772588722239781
        tmp48 = tmp46 / tmp47
        tmp49 = tmp48 * tmp44
        tmp50 = tmp49.to(tl.int64)
        tmp51 = tmp50 + tmp41
        tmp52 = tl.full([1, 1], 15, tl.int64)
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = tl.where(tmp42, tmp40, tmp53)
        tmp55 = tmp39 + tmp54
        tmp56 = tmp55 + 32
        tmp57 = tmp55 < 0
        tmp58 = tl.where(tmp57, tmp56, tmp55)
        tl.device_assert(((0 <= tmp58) & (tmp58 < 32)) | ~rmask, "index out of bounds: 0 <= tmp58 < 32")
        tmp59 = tl.load(in_ptr1 + (x1 + (8*tmp58)), rmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp32 + tmp59
        tmp61 = tmp60 - tmp30
        tmp62 = tl.exp(tmp61)
        tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
        tmp65 = _tmp64 + tmp63
        _tmp64 = tl.where(rmask, tmp65, _tmp64)
    tmp64 = tl.sum(_tmp64, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp66 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp67 = r2 + ((-1)*x0)
        tmp68 = tl.full([1, 1], 0, tl.int64)
        tmp69 = tmp67 > tmp68
        tmp70 = tmp69.to(tl.int64)
        tmp71 = tl.full([1, 1], 16, tl.int64)
        tmp72 = tmp70 * tmp71
        tmp73 = tmp72 + tmp68
        tmp74 = tl.abs(tmp67)
        tmp75 = tl.full([1, 1], 8, tl.int64)
        tmp76 = tmp74 < tmp75
        tmp77 = tmp74.to(tl.float32)
        tmp78 = 8.0
        tmp79 = tmp77 / tmp78
        tmp80 = tl.log(tmp79)
        tmp81 = 2.772588722239781
        tmp82 = tmp80 / tmp81
        tmp83 = tmp82 * tmp78
        tmp84 = tmp83.to(tl.int64)
        tmp85 = tmp84 + tmp75
        tmp86 = tl.full([1, 1], 15, tl.int64)
        tmp87 = triton_helpers.minimum(tmp85, tmp86)
        tmp88 = tl.where(tmp76, tmp74, tmp87)
        tmp89 = tmp73 + tmp88
        tmp90 = tmp89 + 32
        tmp91 = tmp89 < 0
        tmp92 = tl.where(tmp91, tmp90, tmp89)
        tl.device_assert(((0 <= tmp92) & (tmp92 < 32)) | ~rmask, "index out of bounds: 0 <= tmp92 < 32")
        tmp93 = tl.load(in_ptr1 + (x1 + (8*tmp92)), rmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tmp66 + tmp93
        tmp95 = tmp94 - tmp30
        tmp96 = tl.exp(tmp95)
        tmp97 = tmp96 / tmp64
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp97, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifxsfoog3rarg2n33imurktfavxrxdtemvnoqi7whnim6457dln.py
# Source Nodes: [attn_output_1], Original ATen: [aten.view]
# attn_output_1 => view_19
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndsbcl7uxmlrzqk6dbx7lyjk2uwxjpi7b3edrxrsjlncgm2aq3e.py
# Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, hidden_states_7, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_5 => add_7
# forwarded_states => mul_6
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
# hidden_states_7 => view_21
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# variance_1 => mean_1
triton_per_fused_add_mean_mul_pow_rsqrt_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp14 = tmp2 * tmp12
    tmp15 = tmp13 * tmp14
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp2, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp12, xmask)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bp3tcc2s7pvqqr33nrqvwhs4b3ye7zgvn5p26o3lyar7ypw3lg.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.relu, aten.threshold_backward]
# hidden_states_8 => relu
triton_poi_fused_relu_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tl.store(in_out_ptr0 + (x0), tmp1, None)
    tl.store(out_ptr0 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblqmpx4ntlrbvn5x3m4plnxforjx62qrxpt2tcdw7qdx7gyijjg.py
# Source Nodes: [float_8, full_like_1, is_small_1, log_1, min_2, mul_34, relative_position, relative_position_3, relative_position_bucket_1, relative_position_if_large_2, relative_position_if_large_3, to_20, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
# float_8 => convert_element_type_5
# full_like_1 => full_default_4
# is_small_1 => lt_1
# log_1 => log_1
# min_2 => minimum_1
# mul_34 => mul_34
# relative_position => sub_1
# relative_position_3 => neg
# relative_position_bucket_1 => add_37
# relative_position_if_large_2 => add_36
# relative_position_if_large_3 => minimum_2
# to_20 => convert_element_type_6
# truediv_2 => div_8
# truediv_3 => div_9
# where_1 => where_1
# zeros_like => full_default_3
triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = (-1)*(tl.math.min(0, x0 + ((-1)*x1)))
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp0.to(tl.float32)
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tmp6 = tl.log(tmp5)
    tmp7 = 2.0794415416798357
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tmp10 + tmp1
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tl.where(tmp2, tmp0, tmp13)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvhhfrhguvvws3wydjg44lr4vlvjq5z7crvidlggmzenvqtq6mn.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
triton_red_fused__softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
        tmp2 = tl.full([1, 1], 16, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tmp1.to(tl.float32)
        tmp5 = 16.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.log(tmp6)
        tmp8 = 2.0794415416798357
        tmp9 = tmp7 / tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp2
        tmp13 = tl.full([1, 1], 31, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = tl.where(tmp3, tmp1, tmp14)
        tmp16 = tl.full([1, 1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 + 32
        tmp19 = tmp17 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp17)
        tl.device_assert((0 <= tmp20) & (tmp20 < 32), "index out of bounds: 0 <= tmp20 < 32")
        tmp21 = tl.load(in_ptr1 + (x1 + (8*tmp20)), None, eviction_policy='evict_last')
        tmp22 = r2
        tmp23 = x0
        tmp24 = tmp22 <= tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 1.0
        tmp27 = tmp26 - tmp25
        tmp28 = -3.4028234663852886e+38
        tmp29 = tmp27 * tmp28
        tmp30 = tmp21 + tmp29
        tmp31 = tmp0 + tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp35 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp36.to(tl.float32)
        tmp40 = 16.0
        tmp41 = tmp39 / tmp40
        tmp42 = tl.log(tmp41)
        tmp43 = 2.0794415416798357
        tmp44 = tmp42 / tmp43
        tmp45 = tmp44 * tmp40
        tmp46 = tmp45.to(tl.int64)
        tmp47 = tmp46 + tmp37
        tmp48 = tl.full([1, 1], 31, tl.int64)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.where(tmp38, tmp36, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp50 + tmp51
        tmp53 = tmp52 + 32
        tmp54 = tmp52 < 0
        tmp55 = tl.where(tmp54, tmp53, tmp52)
        tl.device_assert((0 <= tmp55) & (tmp55 < 32), "index out of bounds: 0 <= tmp55 < 32")
        tmp56 = tl.load(in_ptr1 + (x1 + (8*tmp55)), None, eviction_policy='evict_last')
        tmp57 = r2
        tmp58 = x0
        tmp59 = tmp57 <= tmp58
        tmp60 = tmp59.to(tl.float32)
        tmp61 = 1.0
        tmp62 = tmp61 - tmp60
        tmp63 = -3.4028234663852886e+38
        tmp64 = tmp62 * tmp63
        tmp65 = tmp56 + tmp64
        tmp66 = tmp35 + tmp65
        tmp67 = tmp66 - tmp33
        tmp68 = tl.exp(tmp67)
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
        tl.store(out_ptr1 + (r2 + (1024*x3)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp72 = tl.load(out_ptr1 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp74, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkzzpg7troy7cte5ipfguspxwfu3yggdygwbbrcma6wt6wytged.py
# Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# softmax_7 => amax_7, div_11, exp_7, sub_12, sum_8
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp11, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjouybsmt4orz5qdxjxrvsepbigudspvvgjx7men43oxafjakvf.py
# Source Nodes: [lm_logits, sequence_output_1], Original ATen: [aten.mul, aten.view]
# lm_logits => view_410
# sequence_output_1 => mul_71
triton_poi_fused_mul_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfhov5izyspnjkymbxg65cxw6vr3rh3ghtyfctlilvubz4rauuo.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_18, exp_18, log_2, sub_23, sub_24, sum_19
triton_red_fused__log_softmax_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 32128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (32128*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwoovhz7nnvohufi2gbofcezchm7rpenjx42wk3u4hu42yap5hat.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_7, div_22, full_default_7, ne, neg_1, sum_20, sum_21, where_3
triton_per_fused_nll_loss_forward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 32128
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 32128), "index out of bounds: 0 <= tmp12 < 32128")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (32128*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp7.to(tl.float32)
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (32128, 512), (512, 1))
    assert_size_stride(primals_34, (512, 512), (512, 1))
    assert_size_stride(primals_35, (512, 512), (512, 1))
    assert_size_stride(primals_36, (512, 512), (512, 1))
    assert_size_stride(primals_37, (32, 8), (8, 1))
    assert_size_stride(primals_38, (512, 512), (512, 1))
    assert_size_stride(primals_39, (2048, 512), (512, 1))
    assert_size_stride(primals_40, (512, 2048), (2048, 1))
    assert_size_stride(primals_41, (512, 512), (512, 1))
    assert_size_stride(primals_42, (512, 512), (512, 1))
    assert_size_stride(primals_43, (512, 512), (512, 1))
    assert_size_stride(primals_44, (512, 512), (512, 1))
    assert_size_stride(primals_45, (2048, 512), (512, 1))
    assert_size_stride(primals_46, (512, 2048), (2048, 1))
    assert_size_stride(primals_47, (512, 512), (512, 1))
    assert_size_stride(primals_48, (512, 512), (512, 1))
    assert_size_stride(primals_49, (512, 512), (512, 1))
    assert_size_stride(primals_50, (512, 512), (512, 1))
    assert_size_stride(primals_51, (2048, 512), (512, 1))
    assert_size_stride(primals_52, (512, 2048), (2048, 1))
    assert_size_stride(primals_53, (512, 512), (512, 1))
    assert_size_stride(primals_54, (512, 512), (512, 1))
    assert_size_stride(primals_55, (512, 512), (512, 1))
    assert_size_stride(primals_56, (512, 512), (512, 1))
    assert_size_stride(primals_57, (2048, 512), (512, 1))
    assert_size_stride(primals_58, (512, 2048), (2048, 1))
    assert_size_stride(primals_59, (512, 512), (512, 1))
    assert_size_stride(primals_60, (512, 512), (512, 1))
    assert_size_stride(primals_61, (512, 512), (512, 1))
    assert_size_stride(primals_62, (512, 512), (512, 1))
    assert_size_stride(primals_63, (2048, 512), (512, 1))
    assert_size_stride(primals_64, (512, 2048), (2048, 1))
    assert_size_stride(primals_65, (512, 512), (512, 1))
    assert_size_stride(primals_66, (512, 512), (512, 1))
    assert_size_stride(primals_67, (512, 512), (512, 1))
    assert_size_stride(primals_68, (512, 512), (512, 1))
    assert_size_stride(primals_69, (2048, 512), (512, 1))
    assert_size_stride(primals_70, (512, 2048), (2048, 1))
    assert_size_stride(primals_71, (512, 512), (512, 1))
    assert_size_stride(primals_72, (512, 512), (512, 1))
    assert_size_stride(primals_73, (512, 512), (512, 1))
    assert_size_stride(primals_74, (32, 8), (8, 1))
    assert_size_stride(primals_75, (512, 512), (512, 1))
    assert_size_stride(primals_76, (512, 512), (512, 1))
    assert_size_stride(primals_77, (512, 512), (512, 1))
    assert_size_stride(primals_78, (512, 512), (512, 1))
    assert_size_stride(primals_79, (512, 512), (512, 1))
    assert_size_stride(primals_80, (2048, 512), (512, 1))
    assert_size_stride(primals_81, (512, 2048), (2048, 1))
    assert_size_stride(primals_82, (512, 512), (512, 1))
    assert_size_stride(primals_83, (512, 512), (512, 1))
    assert_size_stride(primals_84, (512, 512), (512, 1))
    assert_size_stride(primals_85, (512, 512), (512, 1))
    assert_size_stride(primals_86, (512, 512), (512, 1))
    assert_size_stride(primals_87, (512, 512), (512, 1))
    assert_size_stride(primals_88, (512, 512), (512, 1))
    assert_size_stride(primals_89, (512, 512), (512, 1))
    assert_size_stride(primals_90, (2048, 512), (512, 1))
    assert_size_stride(primals_91, (512, 2048), (2048, 1))
    assert_size_stride(primals_92, (512, 512), (512, 1))
    assert_size_stride(primals_93, (512, 512), (512, 1))
    assert_size_stride(primals_94, (512, 512), (512, 1))
    assert_size_stride(primals_95, (512, 512), (512, 1))
    assert_size_stride(primals_96, (512, 512), (512, 1))
    assert_size_stride(primals_97, (512, 512), (512, 1))
    assert_size_stride(primals_98, (512, 512), (512, 1))
    assert_size_stride(primals_99, (512, 512), (512, 1))
    assert_size_stride(primals_100, (2048, 512), (512, 1))
    assert_size_stride(primals_101, (512, 2048), (2048, 1))
    assert_size_stride(primals_102, (512, 512), (512, 1))
    assert_size_stride(primals_103, (512, 512), (512, 1))
    assert_size_stride(primals_104, (512, 512), (512, 1))
    assert_size_stride(primals_105, (512, 512), (512, 1))
    assert_size_stride(primals_106, (512, 512), (512, 1))
    assert_size_stride(primals_107, (512, 512), (512, 1))
    assert_size_stride(primals_108, (512, 512), (512, 1))
    assert_size_stride(primals_109, (512, 512), (512, 1))
    assert_size_stride(primals_110, (2048, 512), (512, 1))
    assert_size_stride(primals_111, (512, 2048), (2048, 1))
    assert_size_stride(primals_112, (512, 512), (512, 1))
    assert_size_stride(primals_113, (512, 512), (512, 1))
    assert_size_stride(primals_114, (512, 512), (512, 1))
    assert_size_stride(primals_115, (512, 512), (512, 1))
    assert_size_stride(primals_116, (512, 512), (512, 1))
    assert_size_stride(primals_117, (512, 512), (512, 1))
    assert_size_stride(primals_118, (512, 512), (512, 1))
    assert_size_stride(primals_119, (512, 512), (512, 1))
    assert_size_stride(primals_120, (2048, 512), (512, 1))
    assert_size_stride(primals_121, (512, 2048), (2048, 1))
    assert_size_stride(primals_122, (512, 512), (512, 1))
    assert_size_stride(primals_123, (512, 512), (512, 1))
    assert_size_stride(primals_124, (512, 512), (512, 1))
    assert_size_stride(primals_125, (512, 512), (512, 1))
    assert_size_stride(primals_126, (512, 512), (512, 1))
    assert_size_stride(primals_127, (512, 512), (512, 1))
    assert_size_stride(primals_128, (512, 512), (512, 1))
    assert_size_stride(primals_129, (512, 512), (512, 1))
    assert_size_stride(primals_130, (2048, 512), (512, 1))
    assert_size_stride(primals_131, (512, 2048), (2048, 1))
    assert_size_stride(primals_132, (32128, 512), (512, 1))
    assert_size_stride(primals_133, (1, 1024), (1024, 1))
    assert_size_stride(primals_134, (1, 1024), (1024, 1))
    assert_size_stride(primals_135, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [inputs_embeds], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(primals_133, primals_33, buf0, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [hidden_states, inputs_embeds], Original ATen: [aten.embedding, aten.native_dropout]
        buf1 = aten.native_dropout(buf0, 0.1, True)
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        buf4 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf5 = reinterpret_tensor(buf4, (1, 1024, 1), (1024, 1, 1), 0); del buf4  # reuse
        buf6 = reinterpret_tensor(buf0, (1024, 512), (512, 1), 0); del buf0  # reuse
        # Source Nodes: [add, hidden_states_1, l__mod___encoder_block_0_layer_0_self_attention_q, normed_hidden_states, pow_1, rsqrt, variance], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_1.run(buf5, buf2, primals_1, buf6, 1024, 512, grid=grid(1024), stream=stream0)
        buf7 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_34, (512, 512), (1, 512), 0), out=buf7)
        buf8 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_35, (512, 512), (1, 512), 0), out=buf8)
        buf9 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_36, (512, 512), (1, 512), 0), out=buf9)
        buf10 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf8, (8, 64, 1024), (64, 1, 512), 0), out=buf10)
        buf11 = empty((1024, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_1, full_like, gt, is_small, log, mul_3, mul_4, relative_buckets, relative_position, relative_position_1, relative_position_bucket, relative_position_if_large, relative_position_if_large_1, to_2, to_3, truediv, truediv_1, where], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.full_like, aten.gt, aten.log, aten.lt, aten.minimum, aten.mul, aten.sub, aten.where]
        triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2.run(buf11, 1048576, grid=grid(1048576), stream=stream0)
        buf14 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf10, primals_37, buf14, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.native_dropout]
        buf15 = aten.native_dropout(buf14, 0.1, True)
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf9, (8, 1024, 64), (64, 512, 1), 0), out=buf18)
        buf19 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_1], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf18, buf19, 524288, grid=grid(524288), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (1024, 512), (512, 1), 0); del buf18  # reuse
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_38, (512, 512), (1, 512), 0), out=buf20)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf21 = aten.native_dropout(reinterpret_tensor(buf20, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = buf22; del buf22  # reuse
        buf25 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf25, (1, 1024, 1), (1024, 1, 1), 0); del buf25  # reuse
        buf27 = buf20; del buf20  # reuse
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, hidden_states_7, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf24, buf26, buf2, primals_2, buf27, 1024, 512, grid=grid(1024), stream=stream0)
        buf28 = empty((1024, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_39, (512, 2048), (1, 512), 0), out=buf28)
        buf29 = reinterpret_tensor(buf28, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf28  # reuse
        buf563 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf29, buf563, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_8, hidden_states_9], Original ATen: [aten.native_dropout, aten.relu]
        buf30 = aten.native_dropout(buf29, 0.1, True)
        buf31 = buf30[0]
        buf32 = buf30[1]
        del buf30
        buf33 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_40, (2048, 512), (1, 2048), 0), out=buf33)
        # Source Nodes: [l__mod___encoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf34 = aten.native_dropout(reinterpret_tensor(buf33, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf37 = buf35; del buf35  # reuse
        buf38 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf39 = reinterpret_tensor(buf38, (1, 1024, 1), (1024, 1, 1), 0); del buf38  # reuse
        buf40 = buf33; del buf33  # reuse
        # Source Nodes: [add_7, hidden_states_13, hidden_states_14, l__mod___encoder_block_1_layer_0_self_attention_q, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf37, buf39, buf24, primals_3, buf40, 1024, 512, grid=grid(1024), stream=stream0)
        buf41 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_41, (512, 512), (1, 512), 0), out=buf41)
        buf42 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_42, (512, 512), (1, 512), 0), out=buf42)
        buf43 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_43, (512, 512), (1, 512), 0), out=buf43)
        buf44 = buf10; del buf10  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf42, (8, 64, 1024), (64, 1, 512), 0), out=buf44)
        buf47 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf44, primals_37, buf47, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_3, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
        buf48 = aten.native_dropout(buf47, 0.1, True)
        buf49 = buf48[0]
        buf50 = buf48[1]
        del buf48
        buf51 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf43, (8, 1024, 64), (64, 512, 1), 0), out=buf51)
        buf52 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_3], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf51, buf52, 524288, grid=grid(524288), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (1024, 512), (512, 1), 0); del buf51  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_44, (512, 512), (1, 512), 0), out=buf53)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf54 = aten.native_dropout(reinterpret_tensor(buf53, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf57 = buf55; del buf55  # reuse
        buf58 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf59 = reinterpret_tensor(buf58, (1, 1024, 1), (1024, 1, 1), 0); del buf58  # reuse
        buf60 = buf53; del buf53  # reuse
        # Source Nodes: [add_9, forwarded_states_2, hidden_states_18, hidden_states_19, hidden_states_20, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf57, buf59, buf37, primals_4, buf60, 1024, 512, grid=grid(1024), stream=stream0)
        buf61 = reinterpret_tensor(buf29, (1024, 2048), (2048, 1), 0); del buf29  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_45, (512, 2048), (1, 512), 0), out=buf61)
        buf62 = reinterpret_tensor(buf61, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf61  # reuse
        buf562 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf62, buf562, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_21, hidden_states_22], Original ATen: [aten.native_dropout, aten.relu]
        buf63 = aten.native_dropout(buf62, 0.1, True)
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_46, (2048, 512), (1, 2048), 0), out=buf66)
        # Source Nodes: [l__mod___encoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf67 = aten.native_dropout(reinterpret_tensor(buf66, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = buf68; del buf68  # reuse
        buf71 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf72 = reinterpret_tensor(buf71, (1, 1024, 1), (1024, 1, 1), 0); del buf71  # reuse
        buf73 = buf66; del buf66  # reuse
        # Source Nodes: [add_11, hidden_states_26, hidden_states_27, l__mod___encoder_block_2_layer_0_self_attention_q, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf70, buf72, buf57, primals_5, buf73, 1024, 512, grid=grid(1024), stream=stream0)
        buf74 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf73, reinterpret_tensor(primals_47, (512, 512), (1, 512), 0), out=buf74)
        buf75 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf73, reinterpret_tensor(primals_48, (512, 512), (1, 512), 0), out=buf75)
        buf76 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf73, reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), out=buf76)
        buf77 = buf44; del buf44  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf75, (8, 64, 1024), (64, 1, 512), 0), out=buf77)
        buf80 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf77, primals_37, buf80, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_5, softmax_2], Original ATen: [aten._softmax, aten.native_dropout]
        buf81 = aten.native_dropout(buf80, 0.1, True)
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        buf84 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf76, (8, 1024, 64), (64, 512, 1), 0), out=buf84)
        buf85 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_5], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf84, buf85, 524288, grid=grid(524288), stream=stream0)
        buf86 = reinterpret_tensor(buf84, (1024, 512), (512, 1), 0); del buf84  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_50, (512, 512), (1, 512), 0), out=buf86)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf87 = aten.native_dropout(reinterpret_tensor(buf86, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf88 = buf87[0]
        buf89 = buf87[1]
        del buf87
        buf90 = buf88; del buf88  # reuse
        buf91 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf92 = reinterpret_tensor(buf91, (1, 1024, 1), (1024, 1, 1), 0); del buf91  # reuse
        buf93 = buf86; del buf86  # reuse
        # Source Nodes: [add_13, forwarded_states_4, hidden_states_31, hidden_states_32, hidden_states_33, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf90, buf92, buf70, primals_6, buf93, 1024, 512, grid=grid(1024), stream=stream0)
        buf94 = reinterpret_tensor(buf62, (1024, 2048), (2048, 1), 0); del buf62  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf93, reinterpret_tensor(primals_51, (512, 2048), (1, 512), 0), out=buf94)
        buf95 = reinterpret_tensor(buf94, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf94  # reuse
        buf561 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_34], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf95, buf561, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.native_dropout, aten.relu]
        buf96 = aten.native_dropout(buf95, 0.1, True)
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_52, (2048, 512), (1, 2048), 0), out=buf99)
        # Source Nodes: [l__mod___encoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf100 = aten.native_dropout(reinterpret_tensor(buf99, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf101 = buf100[0]
        buf102 = buf100[1]
        del buf100
        buf103 = buf101; del buf101  # reuse
        buf104 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf105 = reinterpret_tensor(buf104, (1, 1024, 1), (1024, 1, 1), 0); del buf104  # reuse
        buf106 = buf99; del buf99  # reuse
        # Source Nodes: [add_15, hidden_states_39, hidden_states_40, l__mod___encoder_block_3_layer_0_self_attention_q, normed_hidden_states_3, pow_7, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf103, buf105, buf90, primals_7, buf106, 1024, 512, grid=grid(1024), stream=stream0)
        buf107 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_53, (512, 512), (1, 512), 0), out=buf107)
        buf108 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_54, (512, 512), (1, 512), 0), out=buf108)
        buf109 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_55, (512, 512), (1, 512), 0), out=buf109)
        buf110 = buf77; del buf77  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf108, (8, 64, 1024), (64, 1, 512), 0), out=buf110)
        buf113 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf110, primals_37, buf113, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_7, softmax_3], Original ATen: [aten._softmax, aten.native_dropout]
        buf114 = aten.native_dropout(buf113, 0.1, True)
        buf115 = buf114[0]
        buf116 = buf114[1]
        del buf114
        buf117 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf109, (8, 1024, 64), (64, 512, 1), 0), out=buf117)
        buf118 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_7], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf117, buf118, 524288, grid=grid(524288), stream=stream0)
        buf119 = reinterpret_tensor(buf117, (1024, 512), (512, 1), 0); del buf117  # reuse
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf118, reinterpret_tensor(primals_56, (512, 512), (1, 512), 0), out=buf119)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf120 = aten.native_dropout(reinterpret_tensor(buf119, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf123 = buf121; del buf121  # reuse
        buf124 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf125 = reinterpret_tensor(buf124, (1, 1024, 1), (1024, 1, 1), 0); del buf124  # reuse
        buf126 = buf119; del buf119  # reuse
        # Source Nodes: [add_17, forwarded_states_6, hidden_states_44, hidden_states_45, hidden_states_46, pow_8, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf123, buf125, buf103, primals_8, buf126, 1024, 512, grid=grid(1024), stream=stream0)
        buf127 = reinterpret_tensor(buf95, (1024, 2048), (2048, 1), 0); del buf95  # reuse
        # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf126, reinterpret_tensor(primals_57, (512, 2048), (1, 512), 0), out=buf127)
        buf128 = reinterpret_tensor(buf127, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf127  # reuse
        buf560 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_47], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf128, buf560, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_47, hidden_states_48], Original ATen: [aten.native_dropout, aten.relu]
        buf129 = aten.native_dropout(buf128, 0.1, True)
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf132 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_58, (2048, 512), (1, 2048), 0), out=buf132)
        # Source Nodes: [l__mod___encoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf133 = aten.native_dropout(reinterpret_tensor(buf132, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf136 = buf134; del buf134  # reuse
        buf137 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf138 = reinterpret_tensor(buf137, (1, 1024, 1), (1024, 1, 1), 0); del buf137  # reuse
        buf139 = buf132; del buf132  # reuse
        # Source Nodes: [add_19, hidden_states_52, hidden_states_53, l__mod___encoder_block_4_layer_0_self_attention_q, normed_hidden_states_4, pow_9, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf136, buf138, buf123, primals_9, buf139, 1024, 512, grid=grid(1024), stream=stream0)
        buf140 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf139, reinterpret_tensor(primals_59, (512, 512), (1, 512), 0), out=buf140)
        buf141 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf139, reinterpret_tensor(primals_60, (512, 512), (1, 512), 0), out=buf141)
        buf142 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf139, reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), out=buf142)
        buf143 = buf110; del buf110  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf140, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf141, (8, 64, 1024), (64, 1, 512), 0), out=buf143)
        buf146 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf143, primals_37, buf146, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_9, softmax_4], Original ATen: [aten._softmax, aten.native_dropout]
        buf147 = aten.native_dropout(buf146, 0.1, True)
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf142, (8, 1024, 64), (64, 512, 1), 0), out=buf150)
        buf151 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_9], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf150, buf151, 524288, grid=grid(524288), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (1024, 512), (512, 1), 0); del buf150  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf151, reinterpret_tensor(primals_62, (512, 512), (1, 512), 0), out=buf152)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf153 = aten.native_dropout(reinterpret_tensor(buf152, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = buf154; del buf154  # reuse
        buf157 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf158 = reinterpret_tensor(buf157, (1, 1024, 1), (1024, 1, 1), 0); del buf157  # reuse
        buf159 = buf152; del buf152  # reuse
        # Source Nodes: [add_21, forwarded_states_8, hidden_states_57, hidden_states_58, hidden_states_59, pow_10, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf156, buf158, buf136, primals_10, buf159, 1024, 512, grid=grid(1024), stream=stream0)
        buf160 = reinterpret_tensor(buf128, (1024, 2048), (2048, 1), 0); del buf128  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
        extern_kernels.mm(buf159, reinterpret_tensor(primals_63, (512, 2048), (1, 512), 0), out=buf160)
        buf161 = reinterpret_tensor(buf160, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf160  # reuse
        buf559 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_60], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf161, buf559, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_60, hidden_states_61], Original ATen: [aten.native_dropout, aten.relu]
        buf162 = aten.native_dropout(buf161, 0.1, True)
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_64, (2048, 512), (1, 2048), 0), out=buf165)
        # Source Nodes: [l__mod___encoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf166 = aten.native_dropout(reinterpret_tensor(buf165, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf167 = buf166[0]
        buf168 = buf166[1]
        del buf166
        buf169 = buf167; del buf167  # reuse
        buf170 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf171 = reinterpret_tensor(buf170, (1, 1024, 1), (1024, 1, 1), 0); del buf170  # reuse
        buf172 = buf165; del buf165  # reuse
        # Source Nodes: [add_23, hidden_states_65, hidden_states_66, l__mod___encoder_block_5_layer_0_self_attention_q, normed_hidden_states_5, pow_11, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf169, buf171, buf156, primals_11, buf172, 1024, 512, grid=grid(1024), stream=stream0)
        buf173 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, reinterpret_tensor(primals_65, (512, 512), (1, 512), 0), out=buf173)
        buf174 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, reinterpret_tensor(primals_66, (512, 512), (1, 512), 0), out=buf174)
        buf175 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, reinterpret_tensor(primals_67, (512, 512), (1, 512), 0), out=buf175)
        buf176 = buf143; del buf143  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf174, (8, 64, 1024), (64, 1, 512), 0), out=buf176)
        buf179 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf176, primals_37, buf179, 8192, 1024, grid=grid(8192), stream=stream0)
        del primals_37
        # Source Nodes: [attn_weights_11, softmax_5], Original ATen: [aten._softmax, aten.native_dropout]
        buf180 = aten.native_dropout(buf179, 0.1, True)
        buf181 = buf180[0]
        buf182 = buf180[1]
        del buf180
        buf183 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf181, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf175, (8, 1024, 64), (64, 512, 1), 0), out=buf183)
        buf184 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_11], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf183, buf184, 524288, grid=grid(524288), stream=stream0)
        buf185 = reinterpret_tensor(buf183, (1024, 512), (512, 1), 0); del buf183  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, reinterpret_tensor(primals_68, (512, 512), (1, 512), 0), out=buf185)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf186 = aten.native_dropout(reinterpret_tensor(buf185, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = buf187; del buf187  # reuse
        buf190 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf191 = reinterpret_tensor(buf190, (1, 1024, 1), (1024, 1, 1), 0); del buf190  # reuse
        buf192 = buf185; del buf185  # reuse
        # Source Nodes: [add_25, forwarded_states_10, hidden_states_70, hidden_states_71, hidden_states_72, pow_12, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf189, buf191, buf169, primals_12, buf192, 1024, 512, grid=grid(1024), stream=stream0)
        buf193 = reinterpret_tensor(buf161, (1024, 2048), (2048, 1), 0); del buf161  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
        extern_kernels.mm(buf192, reinterpret_tensor(primals_69, (512, 2048), (1, 512), 0), out=buf193)
        buf194 = reinterpret_tensor(buf193, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf193  # reuse
        buf558 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf194, buf558, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.native_dropout, aten.relu]
        buf195 = aten.native_dropout(buf194, 0.1, True)
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_70, (2048, 512), (1, 2048), 0), out=buf198)
        # Source Nodes: [l__mod___encoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf199 = aten.native_dropout(reinterpret_tensor(buf198, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf200 = buf199[0]
        buf201 = buf199[1]
        del buf199
        buf202 = buf200; del buf200  # reuse
        buf203 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf204 = reinterpret_tensor(buf203, (1, 1024, 1), (1024, 1, 1), 0); del buf203  # reuse
        buf205 = reinterpret_tensor(buf198, (1, 1024, 512), (524288, 512, 1), 0); del buf198  # reuse
        # Source Nodes: [add_27, hidden_states_78, hidden_states_79, hidden_states_80, pow_13, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf202, buf204, buf189, primals_13, buf205, 1024, 512, grid=grid(1024), stream=stream0)
        # Source Nodes: [hidden_states_79, hidden_states_80, hidden_states_82], Original ATen: [aten.mul, aten.native_dropout]
        buf206 = aten.native_dropout(buf205, 0.1, True)
        buf207 = buf206[0]
        buf208 = buf206[1]
        del buf206
        buf209 = buf205; del buf205  # reuse
        # Source Nodes: [inputs_embeds_1], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_0.run(primals_135, primals_33, buf209, 524288, grid=grid(524288), stream=stream0)
        del primals_33
        # Source Nodes: [hidden_states_83, inputs_embeds_1], Original ATen: [aten.embedding, aten.native_dropout]
        buf210 = aten.native_dropout(buf209, 0.1, True)
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf213 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf214 = reinterpret_tensor(buf213, (1, 1024, 1), (1024, 1, 1), 0); del buf213  # reuse
        buf215 = reinterpret_tensor(buf209, (1024, 512), (512, 1), 0); del buf209  # reuse
        # Source Nodes: [add_28, hidden_states_84, l__mod___decoder_block_0_layer_0_self_attention_q, normed_hidden_states_6, pow_14, rsqrt_13, variance_13], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_1.run(buf214, buf211, primals_14, buf215, 1024, 512, grid=grid(1024), stream=stream0)
        buf216 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_71, (512, 512), (1, 512), 0), out=buf216)
        buf217 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_72, (512, 512), (1, 512), 0), out=buf217)
        buf218 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), out=buf218)
        buf219 = buf176; del buf176  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf217, (8, 64, 1024), (64, 1, 512), 0), out=buf219)
        buf220 = empty((1024, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_8, full_like_1, is_small_1, log_1, min_2, mul_34, relative_position, relative_position_3, relative_position_bucket_1, relative_position_if_large_2, relative_position_if_large_3, to_20, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
        triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_7.run(buf220, 1048576, grid=grid(1048576), stream=stream0)
        buf222 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf224 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_6], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf219, primals_74, buf222, buf224, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.native_dropout]
        buf225 = aten.native_dropout(buf224, 0.1, True)
        buf226 = buf225[0]
        buf227 = buf225[1]
        del buf225
        buf228 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf218, (8, 1024, 64), (64, 512, 1), 0), out=buf228)
        buf229 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_13], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf228, buf229, 524288, grid=grid(524288), stream=stream0)
        buf230 = reinterpret_tensor(buf228, (1024, 512), (512, 1), 0); del buf228  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, reinterpret_tensor(primals_75, (512, 512), (1, 512), 0), out=buf230)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf231 = aten.native_dropout(reinterpret_tensor(buf230, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf232 = buf231[0]
        buf233 = buf231[1]
        del buf231
        buf234 = buf232; del buf232  # reuse
        buf235 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf235, (1, 1024, 1), (1024, 1, 1), 0); del buf235  # reuse
        buf237 = buf230; del buf230  # reuse
        # Source Nodes: [add_33, hidden_states_88, hidden_states_89, l__mod___decoder_block_0_layer_1_enc_dec_attention_q, normed_hidden_states_7, pow_15, rsqrt_14, variance_14], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf234, buf236, buf211, primals_15, buf237, 1024, 512, grid=grid(1024), stream=stream0)
        buf238 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf237, reinterpret_tensor(primals_76, (512, 512), (1, 512), 0), out=buf238)
        buf239 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (1, 512), 0), out=buf239)
        buf240 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (1, 512), 0), out=buf240)
        buf241 = reinterpret_tensor(buf222, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf222  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf239, (8, 64, 1024), (64, 1, 512), 0), out=buf241)
        buf244 = reinterpret_tensor(buf219, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf219  # reuse
        # Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf241, buf244, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.native_dropout]
        buf245 = aten.native_dropout(buf244, 0.1, True)
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf240, (8, 1024, 64), (64, 512, 1), 0), out=buf248)
        buf249 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_15], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf248, buf249, 524288, grid=grid(524288), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (1024, 512), (512, 1), 0); del buf248  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf249, reinterpret_tensor(primals_79, (512, 512), (1, 512), 0), out=buf250)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf251 = aten.native_dropout(reinterpret_tensor(buf250, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf252 = buf251[0]
        buf253 = buf251[1]
        del buf251
        buf254 = buf252; del buf252  # reuse
        buf255 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf256 = reinterpret_tensor(buf255, (1, 1024, 1), (1024, 1, 1), 0); del buf255  # reuse
        buf257 = buf250; del buf250  # reuse
        # Source Nodes: [add_36, forwarded_states_12, hidden_states_92, hidden_states_93, hidden_states_94, pow_16, rsqrt_15, variance_15], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf254, buf256, buf234, primals_16, buf257, 1024, 512, grid=grid(1024), stream=stream0)
        buf258 = reinterpret_tensor(buf194, (1024, 2048), (2048, 1), 0); del buf194  # reuse
        # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
        extern_kernels.mm(buf257, reinterpret_tensor(primals_80, (512, 2048), (1, 512), 0), out=buf258)
        buf259 = reinterpret_tensor(buf258, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf258  # reuse
        buf557 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf259, buf557, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_95, hidden_states_96], Original ATen: [aten.native_dropout, aten.relu]
        buf260 = aten.native_dropout(buf259, 0.1, True)
        buf261 = buf260[0]
        buf262 = buf260[1]
        del buf260
        buf263 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_81, (2048, 512), (1, 2048), 0), out=buf263)
        # Source Nodes: [l__mod___decoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf264 = aten.native_dropout(reinterpret_tensor(buf263, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf265 = buf264[0]
        buf266 = buf264[1]
        del buf264
        buf267 = buf265; del buf265  # reuse
        buf268 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf269 = reinterpret_tensor(buf268, (1, 1024, 1), (1024, 1, 1), 0); del buf268  # reuse
        buf270 = buf263; del buf263  # reuse
        # Source Nodes: [add_38, hidden_states_100, hidden_states_101, l__mod___decoder_block_1_layer_0_self_attention_q, normed_hidden_states_8, pow_17, rsqrt_16, variance_16], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf267, buf269, buf254, primals_17, buf270, 1024, 512, grid=grid(1024), stream=stream0)
        buf271 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, reinterpret_tensor(primals_82, (512, 512), (1, 512), 0), out=buf271)
        buf272 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, reinterpret_tensor(primals_83, (512, 512), (1, 512), 0), out=buf272)
        buf273 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, reinterpret_tensor(primals_84, (512, 512), (1, 512), 0), out=buf273)
        buf274 = buf241; del buf241  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf272, (8, 64, 1024), (64, 1, 512), 0), out=buf274)
        buf276 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf278 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf274, primals_74, buf276, buf278, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_17, softmax_8], Original ATen: [aten._softmax, aten.native_dropout]
        buf279 = aten.native_dropout(buf278, 0.1, True)
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf282 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf273, (8, 1024, 64), (64, 512, 1), 0), out=buf282)
        buf283 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_17], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf282, buf283, 524288, grid=grid(524288), stream=stream0)
        buf284 = reinterpret_tensor(buf282, (1024, 512), (512, 1), 0); del buf282  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf283, reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), out=buf284)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf285 = aten.native_dropout(reinterpret_tensor(buf284, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf288 = buf286; del buf286  # reuse
        buf289 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf290 = reinterpret_tensor(buf289, (1, 1024, 1), (1024, 1, 1), 0); del buf289  # reuse
        buf291 = buf284; del buf284  # reuse
        # Source Nodes: [add_40, hidden_states_105, hidden_states_106, l__mod___decoder_block_1_layer_1_enc_dec_attention_q, normed_hidden_states_9, pow_18, rsqrt_17, variance_17], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf288, buf290, buf267, primals_18, buf291, 1024, 512, grid=grid(1024), stream=stream0)
        buf292 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf291, reinterpret_tensor(primals_86, (512, 512), (1, 512), 0), out=buf292)
        buf293 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (1, 512), 0), out=buf293)
        buf294 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (1, 512), 0), out=buf294)
        buf295 = reinterpret_tensor(buf276, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf276  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf293, (8, 64, 1024), (64, 1, 512), 0), out=buf295)
        buf298 = reinterpret_tensor(buf274, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf274  # reuse
        # Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf295, buf298, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_19, softmax_9], Original ATen: [aten._softmax, aten.native_dropout]
        buf299 = aten.native_dropout(buf298, 0.1, True)
        buf300 = buf299[0]
        buf301 = buf299[1]
        del buf299
        buf302 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf300, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf294, (8, 1024, 64), (64, 512, 1), 0), out=buf302)
        buf303 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_19], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf302, buf303, 524288, grid=grid(524288), stream=stream0)
        buf304 = reinterpret_tensor(buf302, (1024, 512), (512, 1), 0); del buf302  # reuse
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf303, reinterpret_tensor(primals_89, (512, 512), (1, 512), 0), out=buf304)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf305 = aten.native_dropout(reinterpret_tensor(buf304, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf306 = buf305[0]
        buf307 = buf305[1]
        del buf305
        buf308 = buf306; del buf306  # reuse
        buf309 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf310 = reinterpret_tensor(buf309, (1, 1024, 1), (1024, 1, 1), 0); del buf309  # reuse
        buf311 = buf304; del buf304  # reuse
        # Source Nodes: [add_42, forwarded_states_14, hidden_states_109, hidden_states_110, hidden_states_111, pow_19, rsqrt_18, variance_18], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf308, buf310, buf288, primals_19, buf311, 1024, 512, grid=grid(1024), stream=stream0)
        buf312 = reinterpret_tensor(buf259, (1024, 2048), (2048, 1), 0); del buf259  # reuse
        # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
        extern_kernels.mm(buf311, reinterpret_tensor(primals_90, (512, 2048), (1, 512), 0), out=buf312)
        buf313 = reinterpret_tensor(buf312, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf312  # reuse
        buf556 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_112], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf313, buf556, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_112, hidden_states_113], Original ATen: [aten.native_dropout, aten.relu]
        buf314 = aten.native_dropout(buf313, 0.1, True)
        buf315 = buf314[0]
        buf316 = buf314[1]
        del buf314
        buf317 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), out=buf317)
        # Source Nodes: [l__mod___decoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf318 = aten.native_dropout(reinterpret_tensor(buf317, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf321 = buf319; del buf319  # reuse
        buf322 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf323 = reinterpret_tensor(buf322, (1, 1024, 1), (1024, 1, 1), 0); del buf322  # reuse
        buf324 = buf317; del buf317  # reuse
        # Source Nodes: [add_44, hidden_states_117, hidden_states_118, l__mod___decoder_block_2_layer_0_self_attention_q, normed_hidden_states_10, pow_20, rsqrt_19, variance_19], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf321, buf323, buf308, primals_20, buf324, 1024, 512, grid=grid(1024), stream=stream0)
        buf325 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf324, reinterpret_tensor(primals_92, (512, 512), (1, 512), 0), out=buf325)
        buf326 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf324, reinterpret_tensor(primals_93, (512, 512), (1, 512), 0), out=buf326)
        buf327 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf324, reinterpret_tensor(primals_94, (512, 512), (1, 512), 0), out=buf327)
        buf328 = buf295; del buf295  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf326, (8, 64, 1024), (64, 1, 512), 0), out=buf328)
        buf330 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf332 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_10], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf328, primals_74, buf330, buf332, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_21, softmax_10], Original ATen: [aten._softmax, aten.native_dropout]
        buf333 = aten.native_dropout(buf332, 0.1, True)
        buf334 = buf333[0]
        buf335 = buf333[1]
        del buf333
        buf336 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf334, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf327, (8, 1024, 64), (64, 512, 1), 0), out=buf336)
        buf337 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_21], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf336, buf337, 524288, grid=grid(524288), stream=stream0)
        buf338 = reinterpret_tensor(buf336, (1024, 512), (512, 1), 0); del buf336  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf337, reinterpret_tensor(primals_95, (512, 512), (1, 512), 0), out=buf338)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf339 = aten.native_dropout(reinterpret_tensor(buf338, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf340 = buf339[0]
        buf341 = buf339[1]
        del buf339
        buf342 = buf340; del buf340  # reuse
        buf343 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf344 = reinterpret_tensor(buf343, (1, 1024, 1), (1024, 1, 1), 0); del buf343  # reuse
        buf345 = buf338; del buf338  # reuse
        # Source Nodes: [add_46, hidden_states_122, hidden_states_123, l__mod___decoder_block_2_layer_1_enc_dec_attention_q, normed_hidden_states_11, pow_21, rsqrt_20, variance_20], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf342, buf344, buf321, primals_21, buf345, 1024, 512, grid=grid(1024), stream=stream0)
        buf346 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf345, reinterpret_tensor(primals_96, (512, 512), (1, 512), 0), out=buf346)
        buf347 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), out=buf347)
        buf348 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (1, 512), 0), out=buf348)
        buf349 = reinterpret_tensor(buf330, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf330  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf347, (8, 64, 1024), (64, 1, 512), 0), out=buf349)
        buf352 = reinterpret_tensor(buf328, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf328  # reuse
        # Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf349, buf352, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_23, softmax_11], Original ATen: [aten._softmax, aten.native_dropout]
        buf353 = aten.native_dropout(buf352, 0.1, True)
        buf354 = buf353[0]
        buf355 = buf353[1]
        del buf353
        buf356 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf348, (8, 1024, 64), (64, 512, 1), 0), out=buf356)
        buf357 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_23], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf356, buf357, 524288, grid=grid(524288), stream=stream0)
        buf358 = reinterpret_tensor(buf356, (1024, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(primals_99, (512, 512), (1, 512), 0), out=buf358)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf359 = aten.native_dropout(reinterpret_tensor(buf358, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf360 = buf359[0]
        buf361 = buf359[1]
        del buf359
        buf362 = buf360; del buf360  # reuse
        buf363 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf364 = reinterpret_tensor(buf363, (1, 1024, 1), (1024, 1, 1), 0); del buf363  # reuse
        buf365 = buf358; del buf358  # reuse
        # Source Nodes: [add_48, forwarded_states_16, hidden_states_126, hidden_states_127, hidden_states_128, pow_22, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf362, buf364, buf342, primals_22, buf365, 1024, 512, grid=grid(1024), stream=stream0)
        buf366 = reinterpret_tensor(buf313, (1024, 2048), (2048, 1), 0); del buf313  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
        extern_kernels.mm(buf365, reinterpret_tensor(primals_100, (512, 2048), (1, 512), 0), out=buf366)
        buf367 = reinterpret_tensor(buf366, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf366  # reuse
        buf555 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_129], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf367, buf555, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.native_dropout, aten.relu]
        buf368 = aten.native_dropout(buf367, 0.1, True)
        buf369 = buf368[0]
        buf370 = buf368[1]
        del buf368
        buf371 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_101, (2048, 512), (1, 2048), 0), out=buf371)
        # Source Nodes: [l__mod___decoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf372 = aten.native_dropout(reinterpret_tensor(buf371, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = buf373; del buf373  # reuse
        buf376 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf377 = reinterpret_tensor(buf376, (1, 1024, 1), (1024, 1, 1), 0); del buf376  # reuse
        buf378 = buf371; del buf371  # reuse
        # Source Nodes: [add_50, hidden_states_134, hidden_states_135, l__mod___decoder_block_3_layer_0_self_attention_q, normed_hidden_states_12, pow_23, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf375, buf377, buf362, primals_23, buf378, 1024, 512, grid=grid(1024), stream=stream0)
        buf379 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf378, reinterpret_tensor(primals_102, (512, 512), (1, 512), 0), out=buf379)
        buf380 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf378, reinterpret_tensor(primals_103, (512, 512), (1, 512), 0), out=buf380)
        buf381 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf378, reinterpret_tensor(primals_104, (512, 512), (1, 512), 0), out=buf381)
        buf382 = buf349; del buf349  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf379, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf380, (8, 64, 1024), (64, 1, 512), 0), out=buf382)
        buf384 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf386 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf382, primals_74, buf384, buf386, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_25, softmax_12], Original ATen: [aten._softmax, aten.native_dropout]
        buf387 = aten.native_dropout(buf386, 0.1, True)
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf381, (8, 1024, 64), (64, 512, 1), 0), out=buf390)
        buf391 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_25], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf390, buf391, 524288, grid=grid(524288), stream=stream0)
        buf392 = reinterpret_tensor(buf390, (1024, 512), (512, 1), 0); del buf390  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf391, reinterpret_tensor(primals_105, (512, 512), (1, 512), 0), out=buf392)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf393 = aten.native_dropout(reinterpret_tensor(buf392, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf394 = buf393[0]
        buf395 = buf393[1]
        del buf393
        buf396 = buf394; del buf394  # reuse
        buf397 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf398 = reinterpret_tensor(buf397, (1, 1024, 1), (1024, 1, 1), 0); del buf397  # reuse
        buf399 = buf392; del buf392  # reuse
        # Source Nodes: [add_52, hidden_states_139, hidden_states_140, l__mod___decoder_block_3_layer_1_enc_dec_attention_q, normed_hidden_states_13, pow_24, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf396, buf398, buf375, primals_24, buf399, 1024, 512, grid=grid(1024), stream=stream0)
        buf400 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, reinterpret_tensor(primals_106, (512, 512), (1, 512), 0), out=buf400)
        buf401 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (1, 512), 0), out=buf401)
        buf402 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (1, 512), 0), out=buf402)
        buf403 = reinterpret_tensor(buf384, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf384  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf400, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf401, (8, 64, 1024), (64, 1, 512), 0), out=buf403)
        buf406 = reinterpret_tensor(buf382, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf382  # reuse
        # Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf403, buf406, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_27, softmax_13], Original ATen: [aten._softmax, aten.native_dropout]
        buf407 = aten.native_dropout(buf406, 0.1, True)
        buf408 = buf407[0]
        buf409 = buf407[1]
        del buf407
        buf410 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf402, (8, 1024, 64), (64, 512, 1), 0), out=buf410)
        buf411 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_27], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf410, buf411, 524288, grid=grid(524288), stream=stream0)
        buf412 = reinterpret_tensor(buf410, (1024, 512), (512, 1), 0); del buf410  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf411, reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), out=buf412)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf413 = aten.native_dropout(reinterpret_tensor(buf412, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf414 = buf413[0]
        buf415 = buf413[1]
        del buf413
        buf416 = buf414; del buf414  # reuse
        buf417 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf418 = reinterpret_tensor(buf417, (1, 1024, 1), (1024, 1, 1), 0); del buf417  # reuse
        buf419 = buf412; del buf412  # reuse
        # Source Nodes: [add_54, forwarded_states_18, hidden_states_143, hidden_states_144, hidden_states_145, pow_25, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf416, buf418, buf396, primals_25, buf419, 1024, 512, grid=grid(1024), stream=stream0)
        buf420 = reinterpret_tensor(buf367, (1024, 2048), (2048, 1), 0); del buf367  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
        extern_kernels.mm(buf419, reinterpret_tensor(primals_110, (512, 2048), (1, 512), 0), out=buf420)
        buf421 = reinterpret_tensor(buf420, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf420  # reuse
        buf554 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf421, buf554, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_146, hidden_states_147], Original ATen: [aten.native_dropout, aten.relu]
        buf422 = aten.native_dropout(buf421, 0.1, True)
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf425 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_111, (2048, 512), (1, 2048), 0), out=buf425)
        # Source Nodes: [l__mod___decoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf426 = aten.native_dropout(reinterpret_tensor(buf425, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf427 = buf426[0]
        buf428 = buf426[1]
        del buf426
        buf429 = buf427; del buf427  # reuse
        buf430 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf431 = reinterpret_tensor(buf430, (1, 1024, 1), (1024, 1, 1), 0); del buf430  # reuse
        buf432 = buf425; del buf425  # reuse
        # Source Nodes: [add_56, hidden_states_151, hidden_states_152, l__mod___decoder_block_4_layer_0_self_attention_q, normed_hidden_states_14, pow_26, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf429, buf431, buf416, primals_26, buf432, 1024, 512, grid=grid(1024), stream=stream0)
        buf433 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf432, reinterpret_tensor(primals_112, (512, 512), (1, 512), 0), out=buf433)
        buf434 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf432, reinterpret_tensor(primals_113, (512, 512), (1, 512), 0), out=buf434)
        buf435 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf432, reinterpret_tensor(primals_114, (512, 512), (1, 512), 0), out=buf435)
        buf436 = buf403; del buf403  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf433, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf434, (8, 64, 1024), (64, 1, 512), 0), out=buf436)
        buf438 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf440 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf436, primals_74, buf438, buf440, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_29, softmax_14], Original ATen: [aten._softmax, aten.native_dropout]
        buf441 = aten.native_dropout(buf440, 0.1, True)
        buf442 = buf441[0]
        buf443 = buf441[1]
        del buf441
        buf444 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf442, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf435, (8, 1024, 64), (64, 512, 1), 0), out=buf444)
        buf445 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_29], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf444, buf445, 524288, grid=grid(524288), stream=stream0)
        buf446 = reinterpret_tensor(buf444, (1024, 512), (512, 1), 0); del buf444  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf445, reinterpret_tensor(primals_115, (512, 512), (1, 512), 0), out=buf446)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf447 = aten.native_dropout(reinterpret_tensor(buf446, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = buf448; del buf448  # reuse
        buf451 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf452 = reinterpret_tensor(buf451, (1, 1024, 1), (1024, 1, 1), 0); del buf451  # reuse
        buf453 = buf446; del buf446  # reuse
        # Source Nodes: [add_58, hidden_states_156, hidden_states_157, l__mod___decoder_block_4_layer_1_enc_dec_attention_q, normed_hidden_states_15, pow_27, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf450, buf452, buf429, primals_27, buf453, 1024, 512, grid=grid(1024), stream=stream0)
        buf454 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf453, reinterpret_tensor(primals_116, (512, 512), (1, 512), 0), out=buf454)
        buf455 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (1, 512), 0), out=buf455)
        buf456 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (1, 512), 0), out=buf456)
        buf457 = reinterpret_tensor(buf438, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf438  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf454, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf455, (8, 64, 1024), (64, 1, 512), 0), out=buf457)
        buf460 = reinterpret_tensor(buf436, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf436  # reuse
        # Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf457, buf460, 8192, 1024, grid=grid(8192), stream=stream0)
        # Source Nodes: [attn_weights_31, softmax_15], Original ATen: [aten._softmax, aten.native_dropout]
        buf461 = aten.native_dropout(buf460, 0.1, True)
        buf462 = buf461[0]
        buf463 = buf461[1]
        del buf461
        buf464 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf456, (8, 1024, 64), (64, 512, 1), 0), out=buf464)
        buf465 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_31], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf464, buf465, 524288, grid=grid(524288), stream=stream0)
        buf466 = reinterpret_tensor(buf464, (1024, 512), (512, 1), 0); del buf464  # reuse
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf465, reinterpret_tensor(primals_119, (512, 512), (1, 512), 0), out=buf466)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf467 = aten.native_dropout(reinterpret_tensor(buf466, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = buf468; del buf468  # reuse
        buf471 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf472 = reinterpret_tensor(buf471, (1, 1024, 1), (1024, 1, 1), 0); del buf471  # reuse
        buf473 = buf466; del buf466  # reuse
        # Source Nodes: [add_60, forwarded_states_20, hidden_states_160, hidden_states_161, hidden_states_162, pow_28, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf470, buf472, buf450, primals_28, buf473, 1024, 512, grid=grid(1024), stream=stream0)
        buf474 = reinterpret_tensor(buf421, (1024, 2048), (2048, 1), 0); del buf421  # reuse
        # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
        extern_kernels.mm(buf473, reinterpret_tensor(primals_120, (512, 2048), (1, 512), 0), out=buf474)
        buf475 = reinterpret_tensor(buf474, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf474  # reuse
        buf553 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_163], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf475, buf553, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.native_dropout, aten.relu]
        buf476 = aten.native_dropout(buf475, 0.1, True)
        buf477 = buf476[0]
        buf478 = buf476[1]
        del buf476
        buf479 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_121, (2048, 512), (1, 2048), 0), out=buf479)
        # Source Nodes: [l__mod___decoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf480 = aten.native_dropout(reinterpret_tensor(buf479, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf481 = buf480[0]
        buf482 = buf480[1]
        del buf480
        buf483 = buf481; del buf481  # reuse
        buf484 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf485 = reinterpret_tensor(buf484, (1, 1024, 1), (1024, 1, 1), 0); del buf484  # reuse
        buf486 = buf479; del buf479  # reuse
        # Source Nodes: [add_62, hidden_states_168, hidden_states_169, l__mod___decoder_block_5_layer_0_self_attention_q, normed_hidden_states_16, pow_29, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf483, buf485, buf470, primals_29, buf486, 1024, 512, grid=grid(1024), stream=stream0)
        buf487 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf486, reinterpret_tensor(primals_122, (512, 512), (1, 512), 0), out=buf487)
        buf488 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf486, reinterpret_tensor(primals_123, (512, 512), (1, 512), 0), out=buf488)
        buf489 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf486, reinterpret_tensor(primals_124, (512, 512), (1, 512), 0), out=buf489)
        buf490 = buf457; del buf457  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf487, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf488, (8, 64, 1024), (64, 1, 512), 0), out=buf490)
        buf492 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf494 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_16], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf490, primals_74, buf492, buf494, 8192, 1024, grid=grid(8192), stream=stream0)
        del primals_74
        # Source Nodes: [attn_weights_33, softmax_16], Original ATen: [aten._softmax, aten.native_dropout]
        buf495 = aten.native_dropout(buf494, 0.1, True)
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf498 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf489, (8, 1024, 64), (64, 512, 1), 0), out=buf498)
        buf499 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_33], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf498, buf499, 524288, grid=grid(524288), stream=stream0)
        buf500 = reinterpret_tensor(buf498, (1024, 512), (512, 1), 0); del buf498  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf499, reinterpret_tensor(primals_125, (512, 512), (1, 512), 0), out=buf500)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf501 = aten.native_dropout(reinterpret_tensor(buf500, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf502 = buf501[0]
        buf503 = buf501[1]
        del buf501
        buf504 = buf502; del buf502  # reuse
        buf505 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf506 = reinterpret_tensor(buf505, (1, 1024, 1), (1024, 1, 1), 0); del buf505  # reuse
        buf507 = buf500; del buf500  # reuse
        # Source Nodes: [add_64, hidden_states_173, hidden_states_174, l__mod___decoder_block_5_layer_1_enc_dec_attention_q, normed_hidden_states_17, pow_30, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf504, buf506, buf483, primals_30, buf507, 1024, 512, grid=grid(1024), stream=stream0)
        buf508 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf507, reinterpret_tensor(primals_126, (512, 512), (1, 512), 0), out=buf508)
        buf509 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (1, 512), 0), out=buf509)
        buf510 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (1, 512), 0), out=buf510)
        buf511 = reinterpret_tensor(buf492, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf492  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf508, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf509, (8, 64, 1024), (64, 1, 512), 0), out=buf511)
        buf514 = reinterpret_tensor(buf490, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf490  # reuse
        # Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf511, buf514, 8192, 1024, grid=grid(8192), stream=stream0)
        del buf511
        # Source Nodes: [attn_weights_35, softmax_17], Original ATen: [aten._softmax, aten.native_dropout]
        buf515 = aten.native_dropout(buf514, 0.1, True)
        buf516 = buf515[0]
        buf517 = buf515[1]
        del buf515
        buf518 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf516, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf510, (8, 1024, 64), (64, 512, 1), 0), out=buf518)
        buf519 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_35], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf518, buf519, 524288, grid=grid(524288), stream=stream0)
        buf520 = reinterpret_tensor(buf518, (1024, 512), (512, 1), 0); del buf518  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(buf519, reinterpret_tensor(primals_129, (512, 512), (1, 512), 0), out=buf520)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf521 = aten.native_dropout(reinterpret_tensor(buf520, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf522 = buf521[0]
        buf523 = buf521[1]
        del buf521
        buf524 = buf522; del buf522  # reuse
        buf525 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf526 = reinterpret_tensor(buf525, (1, 1024, 1), (1024, 1, 1), 0); del buf525  # reuse
        buf527 = buf520; del buf520  # reuse
        # Source Nodes: [add_66, forwarded_states_22, hidden_states_177, hidden_states_178, hidden_states_179, pow_31, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf524, buf526, buf504, primals_31, buf527, 1024, 512, grid=grid(1024), stream=stream0)
        buf528 = reinterpret_tensor(buf475, (1024, 2048), (2048, 1), 0); del buf475  # reuse
        # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
        extern_kernels.mm(buf527, reinterpret_tensor(primals_130, (512, 2048), (1, 512), 0), out=buf528)
        buf529 = reinterpret_tensor(buf528, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf528  # reuse
        buf552 = empty((1, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_180], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_6.run(buf529, buf552, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [hidden_states_180, hidden_states_181], Original ATen: [aten.native_dropout, aten.relu]
        buf530 = aten.native_dropout(buf529, 0.1, True)
        del buf529
        buf531 = buf530[0]
        buf532 = buf530[1]
        del buf530
        buf533 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_131, (2048, 512), (1, 2048), 0), out=buf533)
        # Source Nodes: [l__mod___decoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf534 = aten.native_dropout(reinterpret_tensor(buf533, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
        buf535 = buf534[0]
        buf536 = buf534[1]
        del buf534
        buf537 = buf535; del buf535  # reuse
        buf538 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf539 = reinterpret_tensor(buf538, (1, 1024, 1), (1024, 1, 1), 0); del buf538  # reuse
        buf540 = reinterpret_tensor(buf533, (1, 1024, 512), (524288, 512, 1), 0); del buf533  # reuse
        # Source Nodes: [add_68, hidden_states_185, hidden_states_186, hidden_states_187, pow_32, rsqrt_31, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf537, buf539, buf524, primals_32, buf540, 1024, 512, grid=grid(1024), stream=stream0)
        # Source Nodes: [hidden_states_186, hidden_states_187, sequence_output], Original ATen: [aten.mul, aten.native_dropout]
        buf541 = aten.native_dropout(buf540, 0.1, True)
        del buf540
        buf542 = buf541[0]
        buf543 = buf541[1]
        del buf541
        buf544 = reinterpret_tensor(buf542, (1024, 512), (512, 1), 0); del buf542  # reuse
        # Source Nodes: [lm_logits, sequence_output_1], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf544, 524288, grid=grid(524288), stream=stream0)
        buf545 = empty((1024, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf544, reinterpret_tensor(primals_132, (512, 32128), (1, 512), 0), out=buf545)
        buf548 = empty((1024, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_11.run(buf545, buf548, 1024, 32128, grid=grid(1024), stream=stream0)
        buf551 = empty((), device='cuda', dtype=torch.float32)
        buf550 = empty((), device='cuda', dtype=torch.float32)
        buf564 = buf551; del buf551  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_12.run(buf564, primals_134, buf548, buf550, 1, 1024, grid=grid(1), stream=stream0)
        return (buf564, reinterpret_tensor(buf545, (1, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf217, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf218, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf239, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf240, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf272, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf273, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf293, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf294, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf326, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf327, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf347, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf348, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf380, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf381, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf401, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf402, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf434, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf435, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf455, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf456, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf488, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf489, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf509, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf510, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), buf207, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, primals_133, buf2, buf3, buf5, buf6, buf11, buf17, buf19, buf23, buf24, buf26, buf27, buf32, reinterpret_tensor(buf31, (1024, 2048), (2048, 1), 0), buf36, buf37, buf39, buf40, buf50, buf52, buf56, buf57, buf59, buf60, buf65, reinterpret_tensor(buf64, (1024, 2048), (2048, 1), 0), buf69, buf70, buf72, buf73, buf83, buf85, buf89, buf90, buf92, buf93, buf98, reinterpret_tensor(buf97, (1024, 2048), (2048, 1), 0), buf102, buf103, buf105, buf106, buf116, buf118, buf122, buf123, buf125, buf126, buf131, reinterpret_tensor(buf130, (1024, 2048), (2048, 1), 0), buf135, buf136, buf138, buf139, buf149, buf151, buf155, buf156, buf158, buf159, buf164, reinterpret_tensor(buf163, (1024, 2048), (2048, 1), 0), buf168, buf169, buf171, buf172, buf182, buf184, buf188, buf189, buf191, buf192, buf197, reinterpret_tensor(buf196, (1024, 2048), (2048, 1), 0), buf201, buf202, buf204, buf208, primals_135, buf211, buf212, buf214, buf215, buf220, buf227, buf229, buf233, buf234, buf236, buf237, reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), buf247, buf249, buf253, buf254, buf256, buf257, buf262, reinterpret_tensor(buf261, (1024, 2048), (2048, 1), 0), buf266, buf267, buf269, buf270, buf281, buf283, buf287, buf288, buf290, buf291, buf301, buf303, buf307, buf308, buf310, buf311, buf316, reinterpret_tensor(buf315, (1024, 2048), (2048, 1), 0), buf320, buf321, buf323, buf324, buf335, buf337, buf341, buf342, buf344, buf345, buf355, buf357, buf361, buf362, buf364, buf365, buf370, reinterpret_tensor(buf369, (1024, 2048), (2048, 1), 0), buf374, buf375, buf377, buf378, buf389, buf391, buf395, buf396, buf398, buf399, buf409, buf411, buf415, buf416, buf418, buf419, buf424, reinterpret_tensor(buf423, (1024, 2048), (2048, 1), 0), buf428, buf429, buf431, buf432, buf443, buf445, buf449, buf450, buf452, buf453, buf463, buf465, buf469, buf470, buf472, buf473, buf478, reinterpret_tensor(buf477, (1024, 2048), (2048, 1), 0), buf482, buf483, buf485, buf486, buf497, buf499, buf503, buf504, buf506, buf507, buf517, buf519, buf523, buf524, buf526, buf527, buf532, reinterpret_tensor(buf531, (1024, 2048), (2048, 1), 0), buf536, buf537, buf539, buf543, buf544, buf548, buf550, reinterpret_tensor(primals_132, (32128, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 2048), (2048, 1), 0), buf552, reinterpret_tensor(primals_130, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 512), (512, 1), 0), reinterpret_tensor(buf516, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf510, (8, 64, 1024), (64, 1, 512), 0), buf514, reinterpret_tensor(buf508, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf509, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 512), (512, 1), 0), reinterpret_tensor(primals_125, (512, 512), (512, 1), 0), reinterpret_tensor(buf496, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf489, (8, 64, 1024), (64, 1, 512), 0), buf494, reinterpret_tensor(buf487, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf488, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_124, (512, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 512), (512, 1), 0), reinterpret_tensor(primals_122, (512, 512), (512, 1), 0), reinterpret_tensor(primals_121, (512, 2048), (2048, 1), 0), buf553, reinterpret_tensor(primals_120, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 512), (512, 1), 0), reinterpret_tensor(buf462, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf456, (8, 64, 1024), (64, 1, 512), 0), buf460, reinterpret_tensor(buf454, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf455, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 512), (512, 1), 0), reinterpret_tensor(buf442, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf435, (8, 64, 1024), (64, 1, 512), 0), buf440, reinterpret_tensor(buf433, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf434, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_114, (512, 512), (512, 1), 0), reinterpret_tensor(primals_113, (512, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 512), (512, 1), 0), reinterpret_tensor(primals_111, (512, 2048), (2048, 1), 0), buf554, reinterpret_tensor(primals_110, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), reinterpret_tensor(buf408, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf402, (8, 64, 1024), (64, 1, 512), 0), buf406, reinterpret_tensor(buf400, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf401, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (512, 1), 0), reinterpret_tensor(primals_106, (512, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 512), (512, 1), 0), reinterpret_tensor(buf388, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf381, (8, 64, 1024), (64, 1, 512), 0), buf386, reinterpret_tensor(buf379, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf380, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_104, (512, 512), (512, 1), 0), reinterpret_tensor(primals_103, (512, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 512), (512, 1), 0), reinterpret_tensor(primals_101, (512, 2048), (2048, 1), 0), buf555, reinterpret_tensor(primals_100, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_99, (512, 512), (512, 1), 0), reinterpret_tensor(buf354, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf348, (8, 64, 1024), (64, 1, 512), 0), buf352, reinterpret_tensor(buf346, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf347, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 512), (512, 1), 0), reinterpret_tensor(primals_95, (512, 512), (512, 1), 0), reinterpret_tensor(buf334, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf327, (8, 64, 1024), (64, 1, 512), 0), buf332, reinterpret_tensor(buf325, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf326, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_94, (512, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 512), (512, 1), 0), reinterpret_tensor(primals_92, (512, 512), (512, 1), 0), reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), buf556, reinterpret_tensor(primals_90, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_89, (512, 512), (512, 1), 0), reinterpret_tensor(buf300, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf294, (8, 64, 1024), (64, 1, 512), 0), buf298, reinterpret_tensor(buf292, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf293, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), reinterpret_tensor(buf280, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf273, (8, 64, 1024), (64, 1, 512), 0), buf278, reinterpret_tensor(buf271, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf272, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_84, (512, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 512), (512, 1), 0), reinterpret_tensor(primals_82, (512, 512), (512, 1), 0), reinterpret_tensor(primals_81, (512, 2048), (2048, 1), 0), buf557, reinterpret_tensor(primals_80, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 512), (512, 1), 0), reinterpret_tensor(buf246, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf240, (8, 64, 1024), (64, 1, 512), 0), buf244, reinterpret_tensor(buf238, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf239, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 512), (512, 1), 0), reinterpret_tensor(primals_75, (512, 512), (512, 1), 0), reinterpret_tensor(buf226, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf218, (8, 64, 1024), (64, 1, 512), 0), buf224, reinterpret_tensor(buf216, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf217, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 512), (512, 1), 0), reinterpret_tensor(primals_71, (512, 512), (512, 1), 0), reinterpret_tensor(primals_70, (512, 2048), (2048, 1), 0), buf558, reinterpret_tensor(primals_69, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_68, (512, 512), (512, 1), 0), reinterpret_tensor(buf181, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf175, (8, 64, 1024), (64, 1, 512), 0), buf179, reinterpret_tensor(buf173, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf174, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_67, (512, 512), (512, 1), 0), reinterpret_tensor(primals_66, (512, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 512), (512, 1), 0), reinterpret_tensor(primals_64, (512, 2048), (2048, 1), 0), buf559, reinterpret_tensor(primals_63, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 512), (512, 1), 0), reinterpret_tensor(buf148, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf142, (8, 64, 1024), (64, 1, 512), 0), buf146, reinterpret_tensor(buf140, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf141, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), reinterpret_tensor(primals_60, (512, 512), (512, 1), 0), reinterpret_tensor(primals_59, (512, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 2048), (2048, 1), 0), buf560, reinterpret_tensor(primals_57, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_56, (512, 512), (512, 1), 0), reinterpret_tensor(buf115, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf109, (8, 64, 1024), (64, 1, 512), 0), buf113, reinterpret_tensor(buf107, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf108, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_55, (512, 512), (512, 1), 0), reinterpret_tensor(primals_54, (512, 512), (512, 1), 0), reinterpret_tensor(primals_53, (512, 512), (512, 1), 0), reinterpret_tensor(primals_52, (512, 2048), (2048, 1), 0), buf561, reinterpret_tensor(primals_51, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_50, (512, 512), (512, 1), 0), reinterpret_tensor(buf82, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf76, (8, 64, 1024), (64, 1, 512), 0), buf80, reinterpret_tensor(buf74, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf75, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 512), (512, 1), 0), reinterpret_tensor(primals_47, (512, 512), (512, 1), 0), reinterpret_tensor(primals_46, (512, 2048), (2048, 1), 0), buf562, reinterpret_tensor(primals_45, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_44, (512, 512), (512, 1), 0), reinterpret_tensor(buf49, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf43, (8, 64, 1024), (64, 1, 512), 0), buf47, reinterpret_tensor(buf41, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf42, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_43, (512, 512), (512, 1), 0), reinterpret_tensor(primals_42, (512, 512), (512, 1), 0), reinterpret_tensor(primals_41, (512, 512), (512, 1), 0), reinterpret_tensor(primals_40, (512, 2048), (2048, 1), 0), buf563, reinterpret_tensor(primals_39, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_38, (512, 512), (512, 1), 0), reinterpret_tensor(buf16, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf9, (8, 64, 1024), (64, 1, 512), 0), buf14, reinterpret_tensor(buf7, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf8, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_36, (512, 512), (512, 1), 0), reinterpret_tensor(primals_35, (512, 512), (512, 1), 0), reinterpret_tensor(primals_34, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_134 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_135 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5ForConditionalGeneration', benchmark_compiled_module)
