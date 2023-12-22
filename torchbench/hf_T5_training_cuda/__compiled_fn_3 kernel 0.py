
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5we73jkcblqjs6hd6bqcnhn74xghg6uhuht6w5zlsop576qq6y.py
# Source Nodes: [add, hidden_states_1, inputs_embeds, l__mod___model_encoder_block_0_layer_0_self_attention_q, normed_hidden_states, pow_1, rsqrt, variance], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add => add
# hidden_states_1 => mul_1
# inputs_embeds => embedding
# l__mod___model_encoder_block_0_layer_0_self_attention_q => view_1
# normed_hidden_states => mul_2
# pow_1 => pow_1
# rsqrt => rsqrt
# variance => mean
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_view_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, other=0.0)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp4 * tmp14
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp4, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgsm42b7q7hzwfmn6rjqi4urp2bnkbntlxigwyg53mcngdtflkm.py
# Source Nodes: [scores], Original ATen: [aten.clone]
# scores => clone_1
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 8
    x3 = (xindex // 524288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (524288*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf56vzsm4yf5hthmb2oteeqkr2s2cfr6456x2yhcu6epxjhzrw5a.py
# Source Nodes: [scores], Original ATen: [aten.clone]
# scores => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcbbmeamhrd23rmpmuk4xbz5ovqfsxszulktfhxpuecqlsul62v.py
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
triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5tk2nzgr3pfwlzy5uia4ogbm25orrqcn7qvms7kh3ozyoh2umi.py
# Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.clone]
# attn_weights_1 => clone_3
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_clone_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    _tmp32 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r3 + ((-1)*x0)
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
        tmp28 = 0.0
        tmp29 = tmp27 + tmp28
        tmp30 = tmp0 + tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = triton_helpers.maximum(_tmp32, tmp31)
        _tmp32 = tl.where(rmask, tmp33, _tmp32)
    tmp32 = triton_helpers.max2(_tmp32, 1)[:, None]
    _tmp68 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp34 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = r3 + ((-1)*x0)
        tmp36 = tl.full([1, 1], 0, tl.int64)
        tmp37 = tmp35 > tmp36
        tmp38 = tmp37.to(tl.int64)
        tmp39 = tl.full([1, 1], 16, tl.int64)
        tmp40 = tmp38 * tmp39
        tmp41 = tmp40 + tmp36
        tmp42 = tl.abs(tmp35)
        tmp43 = tl.full([1, 1], 8, tl.int64)
        tmp44 = tmp42 < tmp43
        tmp45 = tmp42.to(tl.float32)
        tmp46 = 8.0
        tmp47 = tmp45 / tmp46
        tmp48 = tl.log(tmp47)
        tmp49 = 2.772588722239781
        tmp50 = tmp48 / tmp49
        tmp51 = tmp50 * tmp46
        tmp52 = tmp51.to(tl.int64)
        tmp53 = tmp52 + tmp43
        tmp54 = tl.full([1, 1], 15, tl.int64)
        tmp55 = triton_helpers.minimum(tmp53, tmp54)
        tmp56 = tl.where(tmp44, tmp42, tmp55)
        tmp57 = tmp41 + tmp56
        tmp58 = tmp57 + 32
        tmp59 = tmp57 < 0
        tmp60 = tl.where(tmp59, tmp58, tmp57)
        tl.device_assert(((0 <= tmp60) & (tmp60 < 32)) | ~rmask, "index out of bounds: 0 <= tmp60 < 32")
        tmp61 = tl.load(in_ptr1 + (x1 + (8*tmp60)), rmask, eviction_policy='evict_last', other=0.0)
        tmp62 = 0.0
        tmp63 = tmp61 + tmp62
        tmp64 = tmp34 + tmp63
        tmp65 = tmp64 - tmp32
        tmp66 = tl.exp(tmp65)
        tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
        tmp69 = _tmp68 + tmp67
        _tmp68 = tl.where(rmask, tmp69, _tmp68)
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp70 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp71 = r3 + ((-1)*x0)
        tmp72 = tl.full([1, 1], 0, tl.int64)
        tmp73 = tmp71 > tmp72
        tmp74 = tmp73.to(tl.int64)
        tmp75 = tl.full([1, 1], 16, tl.int64)
        tmp76 = tmp74 * tmp75
        tmp77 = tmp76 + tmp72
        tmp78 = tl.abs(tmp71)
        tmp79 = tl.full([1, 1], 8, tl.int64)
        tmp80 = tmp78 < tmp79
        tmp81 = tmp78.to(tl.float32)
        tmp82 = 8.0
        tmp83 = tmp81 / tmp82
        tmp84 = tl.log(tmp83)
        tmp85 = 2.772588722239781
        tmp86 = tmp84 / tmp85
        tmp87 = tmp86 * tmp82
        tmp88 = tmp87.to(tl.int64)
        tmp89 = tmp88 + tmp79
        tmp90 = tl.full([1, 1], 15, tl.int64)
        tmp91 = triton_helpers.minimum(tmp89, tmp90)
        tmp92 = tl.where(tmp80, tmp78, tmp91)
        tmp93 = tmp77 + tmp92
        tmp94 = tmp93 + 32
        tmp95 = tmp93 < 0
        tmp96 = tl.where(tmp95, tmp94, tmp93)
        tl.device_assert(((0 <= tmp96) & (tmp96 < 32)) | ~rmask, "index out of bounds: 0 <= tmp96 < 32")
        tmp97 = tl.load(in_ptr1 + (x1 + (8*tmp96)), rmask, eviction_policy='evict_last', other=0.0)
        tmp98 = 0.0
        tmp99 = tmp97 + tmp98
        tmp100 = tmp70 + tmp99
        tmp101 = tmp100 - tmp32
        tmp102 = tl.exp(tmp101)
        tmp103 = tmp102 / tmp68
        tl.store(out_ptr2 + (r3 + (1024*x4)), tmp103, rmask)
        tl.store(out_ptr3 + (r3 + (1024*x4)), tmp103, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbe27pmby77fjwze5agnlhpab7ekhxsedzt2e2ccq2ffltpfu2s2.py
# Source Nodes: [attn_output_1], Original ATen: [aten.view]
# attn_output_1 => view_19
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 1024)) + (65536*(x0 // 64)) + (524288*(x1 // 1024)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozoxgydoi7b275rzmh5tykr5mzoulsxg22hwrvpaquk2wqbwvop.py
# Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, hidden_states_7, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_5 => add_7
# forwarded_states => mul_6
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
# hidden_states_7 => view_21
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# variance_1 => mean_1
triton_per_fused_add_mean_mul_pow_rsqrt_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp13 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp14 = tmp2 * tmp12
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuwtl7rwvxtefg4v7ie7aybwsguaawob75omdl7zhf7s7n4mc45.py
# Source Nodes: [forwarded_states_1, hidden_states_8], Original ATen: [aten.relu, aten.view]
# forwarded_states_1 => view_23
# hidden_states_8 => relu
triton_poi_fused_relu_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jlcxmm5rbqv25b5ycxr3un34jx4f7s7i6642stal6emqnuzcml.py
# Source Nodes: [add_7, hidden_states_13, hidden_states_14, hidden_states_5, l__mod___model_encoder_block_1_layer_0_self_attention_q, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_7 => add_9
# hidden_states_13 => add_8
# hidden_states_14 => mul_7
# hidden_states_5 => add_6
# l__mod___model_encoder_block_1_layer_0_self_attention_q => view_25
# normed_hidden_states_1 => mul_8
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# variance_2 => mean_2
triton_per_fused_add_mean_mul_pow_rsqrt_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp4 * tmp14
    tmp17 = tmp15 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwhu44qm5unf3spov6j3twsmesrntebhstqsv2sepyh6ffuxypn.py
# Source Nodes: [add_9, forwarded_states_2, hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_20, hidden_states_5, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_9 => add_12
# forwarded_states_2 => mul_10
# hidden_states_13 => add_8
# hidden_states_18 => add_11
# hidden_states_19 => mul_9
# hidden_states_20 => view_45
# hidden_states_5 => add_6
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# variance_3 => mean_3
triton_per_fused_add_mean_mul_pow_rsqrt_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = 512.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp6 * tmp16
    tmp19 = tmp17 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs54en5ng2yp2qyxynasoaksoyn3b6w6krgunadcxfqf37klrrz4.py
# Source Nodes: [add_11, hidden_states_13, hidden_states_18, hidden_states_26, hidden_states_27, hidden_states_5, l__mod___model_encoder_block_2_layer_0_self_attention_q, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_11 => add_14
# hidden_states_13 => add_8
# hidden_states_18 => add_11
# hidden_states_26 => add_13
# hidden_states_27 => mul_11
# hidden_states_5 => add_6
# l__mod___model_encoder_block_2_layer_0_self_attention_q => view_49
# normed_hidden_states_2 => mul_12
# pow_5 => pow_5
# rsqrt_4 => rsqrt_4
# variance_4 => mean_4
triton_per_fused_add_mean_mul_pow_rsqrt_view_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp8 * tmp18
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpujqxxmzvokrnmfntc2wdbpvhsmelae5sbknkua5ncxpt6u2n3x.py
# Source Nodes: [add_19, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_52, hidden_states_53, l__mod___model_encoder_block_4_layer_0_self_attention_q, normed_hidden_states_4, pow_9, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_19 => add_24
# hidden_states_31 => add_16
# hidden_states_39 => add_18
# hidden_states_44 => add_21
# hidden_states_52 => add_23
# hidden_states_53 => mul_19
# l__mod___model_encoder_block_4_layer_0_self_attention_q => view_97
# normed_hidden_states_4 => mul_20
# pow_9 => pow_9
# rsqrt_8 => rsqrt_8
# variance_8 => mean_8
triton_per_fused_add_mean_mul_pow_rsqrt_view_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp8 * tmp18
    tmp21 = tmp19 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/codt6wvtdfd6ows5dbrwn3tfx5jowwq6o5phkrv7pwb6hnksqiuv.py
# Source Nodes: [add_27, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_78, hidden_states_79, hidden_states_80, pow_13, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_27 => add_34
# hidden_states_57 => add_26
# hidden_states_65 => add_28
# hidden_states_70 => add_31
# hidden_states_78 => add_33
# hidden_states_79 => mul_27
# hidden_states_80 => mul_28
# pow_13 => pow_13
# rsqrt_12 => rsqrt_12
# variance_12 => mean_12
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp8 * tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcdbukypbpyf77yhyo2ltu6gctoshyd7lhyefp77g3f5witg3n4.py
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
triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/uf/cufzzpdfz2uebpmn7jt53z3r6wrfxyjmcwcyvctuoxrclx6opguw.py
# Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_weights_13 => clone_53
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
triton_red_fused__softmax_clone_detach_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_clone_detach_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.math.min(0, r3 + ((-1)*x0)))
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
        tmp22 = r3
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
        r3 = rindex
        tmp35 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = (-1)*(tl.math.min(0, r3 + ((-1)*x0)))
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
        tmp57 = r3
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
        tl.store(out_ptr1 + (r3 + (1024*x4)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp72 = tl.load(out_ptr1 + (r3 + (1024*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tl.store(out_ptr3 + (r3 + (1024*x4)), tmp74, rmask)
        tl.store(out_ptr4 + (r3 + (1024*x4)), tmp74, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgeebl3izcwlm5vpre2pjfelfvichinnkfpqz6rmozmec2gndx36.py
# Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_weights_15 => clone_59
# softmax_7 => amax_7, div_11, exp_7, sub_12, sum_8
triton_per_fused__softmax_clone_detach_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32768
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
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp11, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy54buw7tpjc3ofyuucuuceepkuc3mbzapjqufa6tsezcinwyhbu.py
# Source Nodes: [forwarded_states_21, hidden_states_163], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
# forwarded_states_21 => view_364
# hidden_states_163 => relu_10
triton_poi_fused_relu_threshold_backward_view_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_view_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tl.store(out_ptr0 + (x0), tmp1, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5r24d73ypcmefvxpnm5cou4ern7mxy7r35a3aa67qh6fynmuxnj.py
# Source Nodes: [add_68, hidden_states_177, hidden_states_185, hidden_states_186, hidden_states_187, lm_logits, pow_32, rsqrt_31, sequence_output_1, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_68 => add_87
# hidden_states_177 => add_84
# hidden_states_185 => add_86
# hidden_states_186 => mul_69
# hidden_states_187 => mul_70
# lm_logits => view_410
# pow_32 => pow_32
# rsqrt_31 => rsqrt_31
# sequence_output_1 => mul_71
# variance_31 => mean_31
triton_per_fused_add_mean_mul_pow_rsqrt_view_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel):
    xnumel = 4096
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp4 * tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = 0.04419417382415922
    tmp19 = tmp17 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceahwbeqzfviaewb7fxsago23qrqsdis5dcyun4p23ysffafgmfv.py
# Source Nodes: [hidden_states_146], Original ATen: [aten.relu, aten.threshold_backward]
# hidden_states_146 => relu_9
triton_poi_fused_relu_threshold_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134 = args
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
    assert_size_stride(primals_133, (4, 1024), (1024, 1))
    assert_size_stride(primals_134, (4, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf2 = reinterpret_tensor(buf1, (4, 1024, 1), (1024, 1, 1), 0); del buf1  # reuse
        buf3 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, hidden_states_1, inputs_embeds, l__mod___model_encoder_block_0_layer_0_self_attention_q, normed_hidden_states, pow_1, rsqrt, variance], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_view_0.run(buf2, primals_133, primals_33, primals_1, buf0, buf3, 4096, 512, grid=grid(4096), stream=stream0)
        buf4 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_34, (512, 512), (1, 512), 0), out=buf4)
        buf5 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_35, (512, 512), (1, 512), 0), out=buf5)
        buf6 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_36, (512, 512), (1, 512), 0), out=buf6)
        buf7 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf4, buf7, 2097152, grid=grid(2097152), stream=stream0)
        buf8 = reinterpret_tensor(buf4, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf4  # reuse
        # Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf5, buf8, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf9 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf8, (32, 64, 1024), (65536, 1024, 1), 0), out=buf9)
        buf10 = empty((1024, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_1, full_like, gt, is_small, log, mul_3, mul_4, relative_buckets, relative_position, relative_position_1, relative_position_bucket, relative_position_if_large, relative_position_if_large_1, to_2, to_3, truediv, truediv_1, where], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.full_like, aten.gt, aten.log, aten.lt, aten.minimum, aten.mul, aten.sub, aten.where]
        triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3.run(buf10, 1048576, grid=grid(1048576), stream=stream0)
        buf13 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf14 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf9, primals_37, buf13, buf14, 32768, 1024, grid=grid(32768), stream=stream0)
        buf15 = reinterpret_tensor(buf5, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf6, buf15, 2097152, grid=grid(2097152), stream=stream0)
        buf16 = reinterpret_tensor(buf6, (32, 1024, 64), (65536, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf15, (32, 1024, 64), (65536, 64, 1), 0), out=buf16)
        buf17 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_1], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf16, buf17, 2097152, grid=grid(2097152), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (4096, 512), (512, 1), 0); del buf16  # reuse
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(primals_38, (512, 512), (1, 512), 0), out=buf18)
        buf19 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf20 = reinterpret_tensor(buf19, (4, 1024, 1), (1024, 1, 1), 0); del buf19  # reuse
        buf21 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, hidden_states_7, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf20, buf0, buf18, primals_2, buf21, 4096, 512, grid=grid(4096), stream=stream0)
        buf22 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf21, reinterpret_tensor(primals_39, (512, 2048), (1, 512), 0), out=buf22)
        buf23 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_1, hidden_states_8], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf22, buf23, 8388608, grid=grid(8388608), stream=stream0)
        buf24 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, reinterpret_tensor(primals_40, (2048, 512), (1, 2048), 0), out=buf24)
        buf25 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 1024, 1), (1024, 1, 1), 0); del buf25  # reuse
        buf27 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, hidden_states_13, hidden_states_14, hidden_states_5, l__mod___model_encoder_block_1_layer_0_self_attention_q, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf26, buf0, buf18, buf24, primals_3, buf27, 4096, 512, grid=grid(4096), stream=stream0)
        buf28 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_41, (512, 512), (1, 512), 0), out=buf28)
        buf29 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_42, (512, 512), (1, 512), 0), out=buf29)
        buf30 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_43, (512, 512), (1, 512), 0), out=buf30)
        buf31 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf28, buf31, 2097152, grid=grid(2097152), stream=stream0)
        buf32 = reinterpret_tensor(buf28, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf28  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, buf32, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf33 = buf9; del buf9  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf32, (32, 64, 1024), (65536, 1024, 1), 0), out=buf33)
        buf36 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf37 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_3, softmax_1], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf33, primals_37, buf36, buf37, 32768, 1024, grid=grid(32768), stream=stream0)
        buf38 = reinterpret_tensor(buf29, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf30, buf38, 2097152, grid=grid(2097152), stream=stream0)
        buf39 = reinterpret_tensor(buf30, (32, 1024, 64), (65536, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf38, (32, 1024, 64), (65536, 64, 1), 0), out=buf39)
        buf40 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_3], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf39, buf40, 2097152, grid=grid(2097152), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (4096, 512), (512, 1), 0); del buf39  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_44, (512, 512), (1, 512), 0), out=buf41)
        buf42 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf43 = reinterpret_tensor(buf42, (4, 1024, 1), (1024, 1, 1), 0); del buf42  # reuse
        buf44 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, forwarded_states_2, hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_20, hidden_states_5, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf43, buf0, buf18, buf24, buf41, primals_4, buf44, 4096, 512, grid=grid(4096), stream=stream0)
        buf45 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf44, reinterpret_tensor(primals_45, (512, 2048), (1, 512), 0), out=buf45)
        buf46 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_3, hidden_states_21], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf45, buf46, 8388608, grid=grid(8388608), stream=stream0)
        buf47 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_46, (2048, 512), (1, 2048), 0), out=buf47)
        buf48 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf50 = reinterpret_tensor(buf49, (4, 1024, 1), (1024, 1, 1), 0); del buf49  # reuse
        buf51 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, hidden_states_13, hidden_states_18, hidden_states_26, hidden_states_27, hidden_states_5, l__mod___model_encoder_block_2_layer_0_self_attention_q, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_10.run(buf50, buf0, buf18, buf24, buf41, buf47, primals_5, buf48, buf51, 4096, 512, grid=grid(4096), stream=stream0)
        buf52 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_47, (512, 512), (1, 512), 0), out=buf52)
        buf53 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_48, (512, 512), (1, 512), 0), out=buf53)
        buf54 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), out=buf54)
        buf55 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf52, buf55, 2097152, grid=grid(2097152), stream=stream0)
        buf56 = reinterpret_tensor(buf52, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf52  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf53, buf56, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf57 = buf33; del buf33  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf56, (32, 64, 1024), (65536, 1024, 1), 0), out=buf57)
        buf60 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf61 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5, softmax_2], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf57, primals_37, buf60, buf61, 32768, 1024, grid=grid(32768), stream=stream0)
        buf62 = reinterpret_tensor(buf53, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf54, buf62, 2097152, grid=grid(2097152), stream=stream0)
        buf63 = reinterpret_tensor(buf54, (32, 1024, 64), (65536, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf62, (32, 1024, 64), (65536, 64, 1), 0), out=buf63)
        buf64 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_5], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf63, buf64, 2097152, grid=grid(2097152), stream=stream0)
        buf65 = reinterpret_tensor(buf63, (4096, 512), (512, 1), 0); del buf63  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf64, reinterpret_tensor(primals_50, (512, 512), (1, 512), 0), out=buf65)
        buf66 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf67 = reinterpret_tensor(buf66, (4, 1024, 1), (1024, 1, 1), 0); del buf66  # reuse
        buf68 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, forwarded_states_4, hidden_states_31, hidden_states_32, hidden_states_33, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf67, buf48, buf65, primals_6, buf68, 4096, 512, grid=grid(4096), stream=stream0)
        buf69 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf68, reinterpret_tensor(primals_51, (512, 2048), (1, 512), 0), out=buf69)
        buf70 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_5, hidden_states_34], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf69, buf70, 8388608, grid=grid(8388608), stream=stream0)
        buf71 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_52, (2048, 512), (1, 2048), 0), out=buf71)
        buf72 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf73 = reinterpret_tensor(buf72, (4, 1024, 1), (1024, 1, 1), 0); del buf72  # reuse
        buf74 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, hidden_states_31, hidden_states_39, hidden_states_40, l__mod___model_encoder_block_3_layer_0_self_attention_q, normed_hidden_states_3, pow_7, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf73, buf48, buf65, buf71, primals_7, buf74, 4096, 512, grid=grid(4096), stream=stream0)
        buf75 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_53, (512, 512), (1, 512), 0), out=buf75)
        buf76 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_54, (512, 512), (1, 512), 0), out=buf76)
        buf77 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_55, (512, 512), (1, 512), 0), out=buf77)
        buf78 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf75, buf78, 2097152, grid=grid(2097152), stream=stream0)
        buf79 = reinterpret_tensor(buf75, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf75  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf76, buf79, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf80 = buf57; del buf57  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf79, (32, 64, 1024), (65536, 1024, 1), 0), out=buf80)
        buf83 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf84 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7, softmax_3], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf80, primals_37, buf83, buf84, 32768, 1024, grid=grid(32768), stream=stream0)
        buf85 = reinterpret_tensor(buf76, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf77, buf85, 2097152, grid=grid(2097152), stream=stream0)
        buf86 = reinterpret_tensor(buf77, (32, 1024, 64), (65536, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf85, (32, 1024, 64), (65536, 64, 1), 0), out=buf86)
        buf87 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_7], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf86, buf87, 2097152, grid=grid(2097152), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (4096, 512), (512, 1), 0); del buf86  # reuse
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf87, reinterpret_tensor(primals_56, (512, 512), (1, 512), 0), out=buf88)
        buf89 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf90 = reinterpret_tensor(buf89, (4, 1024, 1), (1024, 1, 1), 0); del buf89  # reuse
        buf91 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, forwarded_states_6, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_45, hidden_states_46, pow_8, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf90, buf48, buf65, buf71, buf88, primals_8, buf91, 4096, 512, grid=grid(4096), stream=stream0)
        buf92 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf91, reinterpret_tensor(primals_57, (512, 2048), (1, 512), 0), out=buf92)
        buf93 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_7, hidden_states_47], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf92, buf93, 8388608, grid=grid(8388608), stream=stream0)
        buf94 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf93, reinterpret_tensor(primals_58, (2048, 512), (1, 2048), 0), out=buf94)
        buf95 = buf48; del buf48  # reuse
        buf96 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf97 = reinterpret_tensor(buf96, (4, 1024, 1), (1024, 1, 1), 0); del buf96  # reuse
        buf98 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_19, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_52, hidden_states_53, l__mod___model_encoder_block_4_layer_0_self_attention_q, normed_hidden_states_4, pow_9, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf95, buf97, buf65, buf71, buf88, buf94, primals_9, buf98, 4096, 512, grid=grid(4096), stream=stream0)
        buf99 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf98, reinterpret_tensor(primals_59, (512, 512), (1, 512), 0), out=buf99)
        buf100 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf98, reinterpret_tensor(primals_60, (512, 512), (1, 512), 0), out=buf100)
        buf101 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf98, reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), out=buf101)
        buf102 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf99, buf102, 2097152, grid=grid(2097152), stream=stream0)
        buf103 = reinterpret_tensor(buf99, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf99  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf100, buf103, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf104 = buf80; del buf80  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf103, (32, 64, 1024), (65536, 1024, 1), 0), out=buf104)
        buf107 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf108 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_9, softmax_4], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf104, primals_37, buf107, buf108, 32768, 1024, grid=grid(32768), stream=stream0)
        buf109 = reinterpret_tensor(buf100, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf101, buf109, 2097152, grid=grid(2097152), stream=stream0)
        buf110 = reinterpret_tensor(buf101, (32, 1024, 64), (65536, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf109, (32, 1024, 64), (65536, 64, 1), 0), out=buf110)
        buf111 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_9], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf110, buf111, 2097152, grid=grid(2097152), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (4096, 512), (512, 1), 0); del buf110  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf111, reinterpret_tensor(primals_62, (512, 512), (1, 512), 0), out=buf112)
        buf113 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf114 = reinterpret_tensor(buf113, (4, 1024, 1), (1024, 1, 1), 0); del buf113  # reuse
        buf115 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, forwarded_states_8, hidden_states_57, hidden_states_58, hidden_states_59, pow_10, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf114, buf95, buf112, primals_10, buf115, 4096, 512, grid=grid(4096), stream=stream0)
        buf116 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
        extern_kernels.mm(buf115, reinterpret_tensor(primals_63, (512, 2048), (1, 512), 0), out=buf116)
        buf117 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_9, hidden_states_60], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf116, buf117, 8388608, grid=grid(8388608), stream=stream0)
        buf118 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf117, reinterpret_tensor(primals_64, (2048, 512), (1, 2048), 0), out=buf118)
        buf119 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf120 = reinterpret_tensor(buf119, (4, 1024, 1), (1024, 1, 1), 0); del buf119  # reuse
        buf121 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, hidden_states_57, hidden_states_65, hidden_states_66, l__mod___model_encoder_block_5_layer_0_self_attention_q, normed_hidden_states_5, pow_11, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf120, buf95, buf112, buf118, primals_11, buf121, 4096, 512, grid=grid(4096), stream=stream0)
        buf122 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf121, reinterpret_tensor(primals_65, (512, 512), (1, 512), 0), out=buf122)
        buf123 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf121, reinterpret_tensor(primals_66, (512, 512), (1, 512), 0), out=buf123)
        buf124 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf121, reinterpret_tensor(primals_67, (512, 512), (1, 512), 0), out=buf124)
        buf125 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf122, buf125, 2097152, grid=grid(2097152), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf122  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf123, buf126, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf127 = buf104; del buf104  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf126, (32, 64, 1024), (65536, 1024, 1), 0), out=buf127)
        buf130 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf131 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_11, softmax_5], Original ATen: [aten._softmax, aten.clone]
        triton_red_fused__softmax_clone_4.run(buf127, primals_37, buf130, buf131, 32768, 1024, grid=grid(32768), stream=stream0)
        del primals_37
        buf132 = reinterpret_tensor(buf123, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf123  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf124, buf132, 2097152, grid=grid(2097152), stream=stream0)
        buf133 = reinterpret_tensor(buf124, (32, 1024, 64), (65536, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf132, (32, 1024, 64), (65536, 64, 1), 0), out=buf133)
        buf134 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_11], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf133, buf134, 2097152, grid=grid(2097152), stream=stream0)
        buf135 = reinterpret_tensor(buf133, (4096, 512), (512, 1), 0); del buf133  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf134, reinterpret_tensor(primals_68, (512, 512), (1, 512), 0), out=buf135)
        buf136 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf137 = reinterpret_tensor(buf136, (4, 1024, 1), (1024, 1, 1), 0); del buf136  # reuse
        buf138 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, forwarded_states_10, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_71, hidden_states_72, pow_12, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf137, buf95, buf112, buf118, buf135, primals_12, buf138, 4096, 512, grid=grid(4096), stream=stream0)
        buf139 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
        extern_kernels.mm(buf138, reinterpret_tensor(primals_69, (512, 2048), (1, 512), 0), out=buf139)
        buf140 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_11, hidden_states_73], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf139, buf140, 8388608, grid=grid(8388608), stream=stream0)
        buf141 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf140, reinterpret_tensor(primals_70, (2048, 512), (1, 2048), 0), out=buf141)
        buf142 = buf95; del buf95  # reuse
        buf143 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf144 = reinterpret_tensor(buf143, (4, 1024, 1), (1024, 1, 1), 0); del buf143  # reuse
        buf145 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_78, hidden_states_79, hidden_states_80, pow_13, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf142, buf144, buf112, buf118, buf135, buf141, primals_13, buf145, 4096, 512, grid=grid(4096), stream=stream0)
        buf146 = buf142; del buf142  # reuse
        buf147 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf148 = reinterpret_tensor(buf147, (4, 1024, 1), (1024, 1, 1), 0); del buf147  # reuse
        buf149 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, hidden_states_84, inputs_embeds_1, l__mod___model_decoder_block_0_layer_0_self_attention_q, normed_hidden_states_6, pow_14, rsqrt_13, variance_13], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_view_0.run(buf148, primals_134, primals_33, primals_14, buf146, buf149, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_33
        buf150 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf149, reinterpret_tensor(primals_71, (512, 512), (1, 512), 0), out=buf150)
        buf151 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf149, reinterpret_tensor(primals_72, (512, 512), (1, 512), 0), out=buf151)
        buf152 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf149, reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), out=buf152)
        buf153 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf150, buf153, 2097152, grid=grid(2097152), stream=stream0)
        buf154 = reinterpret_tensor(buf150, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf150  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf151, buf154, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf155 = buf127; del buf127  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf154, (32, 64, 1024), (65536, 1024, 1), 0), out=buf155)
        buf156 = empty((1024, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_8, full_like_1, is_small_1, log_1, min_2, mul_34, relative_position, relative_position_3, relative_position_bucket_1, relative_position_if_large_2, relative_position_if_large_3, to_20, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
        triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_13.run(buf156, 1048576, grid=grid(1048576), stream=stream0)
        buf158 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf160 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf407 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf155, primals_74, buf158, buf160, buf407, 32768, 1024, grid=grid(32768), stream=stream0)
        buf161 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf152, buf161, 2097152, grid=grid(2097152), stream=stream0)
        buf162 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf161, (32, 1024, 64), (65536, 64, 1), 0), out=buf162)
        buf163 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_13], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf162, buf163, 2097152, grid=grid(2097152), stream=stream0)
        buf164 = reinterpret_tensor(buf162, (4096, 512), (512, 1), 0); del buf162  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf163, reinterpret_tensor(primals_75, (512, 512), (1, 512), 0), out=buf164)
        buf165 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf165, (4, 1024, 1), (1024, 1, 1), 0); del buf165  # reuse
        buf167 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, hidden_states_88, hidden_states_89, l__mod___model_decoder_block_0_layer_1_enc_dec_attention_q, normed_hidden_states_7, pow_15, rsqrt_14, variance_14], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf166, buf146, buf164, primals_15, buf167, 4096, 512, grid=grid(4096), stream=stream0)
        buf168 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, reinterpret_tensor(primals_76, (512, 512), (1, 512), 0), out=buf168)
        buf169 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (1, 512), 0), out=buf169)
        buf170 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (1, 512), 0), out=buf170)
        buf171 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf168, buf171, 2097152, grid=grid(2097152), stream=stream0)
        buf172 = reinterpret_tensor(buf168, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf168  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf169, buf172, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf173 = reinterpret_tensor(buf158, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf158  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf172, (32, 64, 1024), (65536, 1024, 1), 0), out=buf173)
        buf176 = reinterpret_tensor(buf155, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf155  # reuse
        buf406 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf173, buf176, buf406, 32768, 1024, grid=grid(32768), stream=stream0)
        buf177 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf170, buf177, 2097152, grid=grid(2097152), stream=stream0)
        buf178 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf177, (32, 1024, 64), (65536, 64, 1), 0), out=buf178)
        buf179 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_15], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf178, buf179, 2097152, grid=grid(2097152), stream=stream0)
        buf180 = reinterpret_tensor(buf178, (4096, 512), (512, 1), 0); del buf178  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf179, reinterpret_tensor(primals_79, (512, 512), (1, 512), 0), out=buf180)
        buf181 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf182 = reinterpret_tensor(buf181, (4, 1024, 1), (1024, 1, 1), 0); del buf181  # reuse
        buf183 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, forwarded_states_12, hidden_states_88, hidden_states_92, hidden_states_93, hidden_states_94, pow_16, rsqrt_15, variance_15], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf182, buf146, buf164, buf180, primals_16, buf183, 4096, 512, grid=grid(4096), stream=stream0)
        buf184 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
        extern_kernels.mm(buf183, reinterpret_tensor(primals_80, (512, 2048), (1, 512), 0), out=buf184)
        buf185 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_13, hidden_states_95], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf184, buf185, 8388608, grid=grid(8388608), stream=stream0)
        buf186 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf185, reinterpret_tensor(primals_81, (2048, 512), (1, 2048), 0), out=buf186)
        buf187 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf188 = reinterpret_tensor(buf187, (4, 1024, 1), (1024, 1, 1), 0); del buf187  # reuse
        buf189 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_92, l__mod___model_decoder_block_1_layer_0_self_attention_q, normed_hidden_states_8, pow_17, rsqrt_16, variance_16], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf188, buf146, buf164, buf180, buf186, primals_17, buf189, 4096, 512, grid=grid(4096), stream=stream0)
        buf190 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, reinterpret_tensor(primals_82, (512, 512), (1, 512), 0), out=buf190)
        buf191 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, reinterpret_tensor(primals_83, (512, 512), (1, 512), 0), out=buf191)
        buf192 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, reinterpret_tensor(primals_84, (512, 512), (1, 512), 0), out=buf192)
        buf193 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf190, buf193, 2097152, grid=grid(2097152), stream=stream0)
        buf194 = reinterpret_tensor(buf190, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf190  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf191, buf194, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf195 = buf173; del buf173  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf194, (32, 64, 1024), (65536, 1024, 1), 0), out=buf195)
        buf197 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf199 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf404 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_17, softmax_8], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf195, primals_74, buf197, buf199, buf404, 32768, 1024, grid=grid(32768), stream=stream0)
        buf200 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf192, buf200, 2097152, grid=grid(2097152), stream=stream0)
        buf201 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf200, (32, 1024, 64), (65536, 64, 1), 0), out=buf201)
        buf202 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_17], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf201, buf202, 2097152, grid=grid(2097152), stream=stream0)
        buf203 = reinterpret_tensor(buf201, (4096, 512), (512, 1), 0); del buf201  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), out=buf203)
        buf204 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf206 = reinterpret_tensor(buf205, (4, 1024, 1), (1024, 1, 1), 0); del buf205  # reuse
        buf207 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, hidden_states_100, hidden_states_105, hidden_states_106, hidden_states_88, hidden_states_92, l__mod___model_decoder_block_1_layer_1_enc_dec_attention_q, normed_hidden_states_9, pow_18, rsqrt_17, variance_17], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_10.run(buf206, buf146, buf164, buf180, buf186, buf203, primals_18, buf204, buf207, 4096, 512, grid=grid(4096), stream=stream0)
        buf208 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf207, reinterpret_tensor(primals_86, (512, 512), (1, 512), 0), out=buf208)
        buf209 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (1, 512), 0), out=buf209)
        buf210 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (1, 512), 0), out=buf210)
        buf211 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf208, buf211, 2097152, grid=grid(2097152), stream=stream0)
        buf212 = reinterpret_tensor(buf208, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf208  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf209, buf212, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf213 = reinterpret_tensor(buf197, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf197  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf212, (32, 64, 1024), (65536, 1024, 1), 0), out=buf213)
        buf216 = reinterpret_tensor(buf195, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf195  # reuse
        buf403 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_19, softmax_9], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf213, buf216, buf403, 32768, 1024, grid=grid(32768), stream=stream0)
        buf217 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf210, buf217, 2097152, grid=grid(2097152), stream=stream0)
        buf218 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf217, (32, 1024, 64), (65536, 64, 1), 0), out=buf218)
        buf219 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_19], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf218, buf219, 2097152, grid=grid(2097152), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (4096, 512), (512, 1), 0); del buf218  # reuse
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf219, reinterpret_tensor(primals_89, (512, 512), (1, 512), 0), out=buf220)
        buf221 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf222 = reinterpret_tensor(buf221, (4, 1024, 1), (1024, 1, 1), 0); del buf221  # reuse
        buf223 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, forwarded_states_14, hidden_states_109, hidden_states_110, hidden_states_111, pow_19, rsqrt_18, variance_18], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf222, buf204, buf220, primals_19, buf223, 4096, 512, grid=grid(4096), stream=stream0)
        buf224 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
        extern_kernels.mm(buf223, reinterpret_tensor(primals_90, (512, 2048), (1, 512), 0), out=buf224)
        buf225 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_15, hidden_states_112], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf224, buf225, 8388608, grid=grid(8388608), stream=stream0)
        buf226 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf225, reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), out=buf226)
        buf227 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf228 = reinterpret_tensor(buf227, (4, 1024, 1), (1024, 1, 1), 0); del buf227  # reuse
        buf229 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, hidden_states_109, hidden_states_117, hidden_states_118, l__mod___model_decoder_block_2_layer_0_self_attention_q, normed_hidden_states_10, pow_20, rsqrt_19, variance_19], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf228, buf204, buf220, buf226, primals_20, buf229, 4096, 512, grid=grid(4096), stream=stream0)
        buf230 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, reinterpret_tensor(primals_92, (512, 512), (1, 512), 0), out=buf230)
        buf231 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, reinterpret_tensor(primals_93, (512, 512), (1, 512), 0), out=buf231)
        buf232 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, reinterpret_tensor(primals_94, (512, 512), (1, 512), 0), out=buf232)
        buf233 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, buf233, 2097152, grid=grid(2097152), stream=stream0)
        buf234 = reinterpret_tensor(buf230, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf230  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf231, buf234, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf235 = buf213; del buf213  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf234, (32, 64, 1024), (65536, 1024, 1), 0), out=buf235)
        buf237 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf239 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf401 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_21, softmax_10], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf235, primals_74, buf237, buf239, buf401, 32768, 1024, grid=grid(32768), stream=stream0)
        buf240 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf232, buf240, 2097152, grid=grid(2097152), stream=stream0)
        buf241 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf240, (32, 1024, 64), (65536, 64, 1), 0), out=buf241)
        buf242 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_21], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf241, buf242, 2097152, grid=grid(2097152), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (4096, 512), (512, 1), 0); del buf241  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf242, reinterpret_tensor(primals_95, (512, 512), (1, 512), 0), out=buf243)
        buf244 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf245 = reinterpret_tensor(buf244, (4, 1024, 1), (1024, 1, 1), 0); del buf244  # reuse
        buf246 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_123, l__mod___model_decoder_block_2_layer_1_enc_dec_attention_q, normed_hidden_states_11, pow_21, rsqrt_20, variance_20], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf245, buf204, buf220, buf226, buf243, primals_21, buf246, 4096, 512, grid=grid(4096), stream=stream0)
        buf247 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf246, reinterpret_tensor(primals_96, (512, 512), (1, 512), 0), out=buf247)
        buf248 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), out=buf248)
        buf249 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (1, 512), 0), out=buf249)
        buf250 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf247, buf250, 2097152, grid=grid(2097152), stream=stream0)
        buf251 = reinterpret_tensor(buf247, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf247  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf248, buf251, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf252 = reinterpret_tensor(buf237, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf237  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf251, (32, 64, 1024), (65536, 1024, 1), 0), out=buf252)
        buf255 = reinterpret_tensor(buf235, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf235  # reuse
        buf400 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_23, softmax_11], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf252, buf255, buf400, 32768, 1024, grid=grid(32768), stream=stream0)
        buf256 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf249, buf256, 2097152, grid=grid(2097152), stream=stream0)
        buf257 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf256, (32, 1024, 64), (65536, 64, 1), 0), out=buf257)
        buf258 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_23], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf257, buf258, 2097152, grid=grid(2097152), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (4096, 512), (512, 1), 0); del buf257  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf258, reinterpret_tensor(primals_99, (512, 512), (1, 512), 0), out=buf259)
        buf260 = buf204; del buf204  # reuse
        buf261 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf262 = reinterpret_tensor(buf261, (4, 1024, 1), (1024, 1, 1), 0); del buf261  # reuse
        buf263 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, forwarded_states_16, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_126, hidden_states_127, hidden_states_128, pow_22, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf260, buf262, buf220, buf226, buf243, buf259, primals_22, buf263, 4096, 512, grid=grid(4096), stream=stream0)
        buf264 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, reinterpret_tensor(primals_100, (512, 2048), (1, 512), 0), out=buf264)
        buf265 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_17, hidden_states_129], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf264, buf265, 8388608, grid=grid(8388608), stream=stream0)
        buf266 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf265, reinterpret_tensor(primals_101, (2048, 512), (1, 2048), 0), out=buf266)
        buf267 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf268 = reinterpret_tensor(buf267, (4, 1024, 1), (1024, 1, 1), 0); del buf267  # reuse
        buf269 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, hidden_states_134, hidden_states_135, l__mod___model_decoder_block_3_layer_0_self_attention_q, normed_hidden_states_12, pow_23, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf268, buf260, buf266, primals_23, buf269, 4096, 512, grid=grid(4096), stream=stream0)
        buf270 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf269, reinterpret_tensor(primals_102, (512, 512), (1, 512), 0), out=buf270)
        buf271 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf269, reinterpret_tensor(primals_103, (512, 512), (1, 512), 0), out=buf271)
        buf272 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf269, reinterpret_tensor(primals_104, (512, 512), (1, 512), 0), out=buf272)
        buf273 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf270, buf273, 2097152, grid=grid(2097152), stream=stream0)
        buf274 = reinterpret_tensor(buf270, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf270  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf271, buf274, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf275 = buf252; del buf252  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf273, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf274, (32, 64, 1024), (65536, 1024, 1), 0), out=buf275)
        buf277 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf279 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf398 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_25, softmax_12], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf275, primals_74, buf277, buf279, buf398, 32768, 1024, grid=grid(32768), stream=stream0)
        buf280 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf272, buf280, 2097152, grid=grid(2097152), stream=stream0)
        buf281 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf280, (32, 1024, 64), (65536, 64, 1), 0), out=buf281)
        buf282 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_25], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf281, buf282, 2097152, grid=grid(2097152), stream=stream0)
        buf283 = reinterpret_tensor(buf281, (4096, 512), (512, 1), 0); del buf281  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf282, reinterpret_tensor(primals_105, (512, 512), (1, 512), 0), out=buf283)
        buf284 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf285 = reinterpret_tensor(buf284, (4, 1024, 1), (1024, 1, 1), 0); del buf284  # reuse
        buf286 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_52, hidden_states_134, hidden_states_139, hidden_states_140, l__mod___model_decoder_block_3_layer_1_enc_dec_attention_q, normed_hidden_states_13, pow_24, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf285, buf260, buf266, buf283, primals_24, buf286, 4096, 512, grid=grid(4096), stream=stream0)
        buf287 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf286, reinterpret_tensor(primals_106, (512, 512), (1, 512), 0), out=buf287)
        buf288 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (1, 512), 0), out=buf288)
        buf289 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (1, 512), 0), out=buf289)
        buf290 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf287, buf290, 2097152, grid=grid(2097152), stream=stream0)
        buf291 = reinterpret_tensor(buf287, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf287  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf288, buf291, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf292 = reinterpret_tensor(buf277, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf277  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf291, (32, 64, 1024), (65536, 1024, 1), 0), out=buf292)
        buf295 = reinterpret_tensor(buf275, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf275  # reuse
        buf397 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_27, softmax_13], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf292, buf295, buf397, 32768, 1024, grid=grid(32768), stream=stream0)
        buf296 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf289, buf296, 2097152, grid=grid(2097152), stream=stream0)
        buf297 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf295, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf296, (32, 1024, 64), (65536, 64, 1), 0), out=buf297)
        buf298 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_27], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf297, buf298, 2097152, grid=grid(2097152), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (4096, 512), (512, 1), 0); del buf297  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf298, reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), out=buf299)
        buf300 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf301 = reinterpret_tensor(buf300, (4, 1024, 1), (1024, 1, 1), 0); del buf300  # reuse
        buf302 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_54, forwarded_states_18, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_144, hidden_states_145, pow_25, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf301, buf260, buf266, buf283, buf299, primals_25, buf302, 4096, 512, grid=grid(4096), stream=stream0)
        buf303 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
        extern_kernels.mm(buf302, reinterpret_tensor(primals_110, (512, 2048), (1, 512), 0), out=buf303)
        buf304 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_19, hidden_states_146], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf303, buf304, 8388608, grid=grid(8388608), stream=stream0)
        buf305 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf304, reinterpret_tensor(primals_111, (2048, 512), (1, 2048), 0), out=buf305)
        buf306 = buf260; del buf260  # reuse
        buf307 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf308 = reinterpret_tensor(buf307, (4, 1024, 1), (1024, 1, 1), 0); del buf307  # reuse
        buf309 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_56, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_151, hidden_states_152, l__mod___model_decoder_block_4_layer_0_self_attention_q, normed_hidden_states_14, pow_26, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf306, buf308, buf266, buf283, buf299, buf305, primals_26, buf309, 4096, 512, grid=grid(4096), stream=stream0)
        buf310 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, reinterpret_tensor(primals_112, (512, 512), (1, 512), 0), out=buf310)
        buf311 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, reinterpret_tensor(primals_113, (512, 512), (1, 512), 0), out=buf311)
        buf312 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, reinterpret_tensor(primals_114, (512, 512), (1, 512), 0), out=buf312)
        buf313 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf310, buf313, 2097152, grid=grid(2097152), stream=stream0)
        buf314 = reinterpret_tensor(buf310, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf310  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf311, buf314, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf315 = buf292; del buf292  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf313, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf314, (32, 64, 1024), (65536, 1024, 1), 0), out=buf315)
        buf317 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf319 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf395 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29, softmax_14], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf315, primals_74, buf317, buf319, buf395, 32768, 1024, grid=grid(32768), stream=stream0)
        buf320 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf312, buf320, 2097152, grid=grid(2097152), stream=stream0)
        buf321 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf320, (32, 1024, 64), (65536, 64, 1), 0), out=buf321)
        buf322 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_29], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf321, buf322, 2097152, grid=grid(2097152), stream=stream0)
        buf323 = reinterpret_tensor(buf321, (4096, 512), (512, 1), 0); del buf321  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf322, reinterpret_tensor(primals_115, (512, 512), (1, 512), 0), out=buf323)
        buf324 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf325 = reinterpret_tensor(buf324, (4, 1024, 1), (1024, 1, 1), 0); del buf324  # reuse
        buf326 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_58, hidden_states_156, hidden_states_157, l__mod___model_decoder_block_4_layer_1_enc_dec_attention_q, normed_hidden_states_15, pow_27, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf325, buf306, buf323, primals_27, buf326, 4096, 512, grid=grid(4096), stream=stream0)
        buf327 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf326, reinterpret_tensor(primals_116, (512, 512), (1, 512), 0), out=buf327)
        buf328 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (1, 512), 0), out=buf328)
        buf329 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (1, 512), 0), out=buf329)
        buf330 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf327, buf330, 2097152, grid=grid(2097152), stream=stream0)
        buf331 = reinterpret_tensor(buf327, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf327  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf328, buf331, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf332 = reinterpret_tensor(buf317, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf317  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf331, (32, 64, 1024), (65536, 1024, 1), 0), out=buf332)
        buf335 = reinterpret_tensor(buf315, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf315  # reuse
        buf394 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_31, softmax_15], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf332, buf335, buf394, 32768, 1024, grid=grid(32768), stream=stream0)
        buf336 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf329, buf336, 2097152, grid=grid(2097152), stream=stream0)
        buf337 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf336, (32, 1024, 64), (65536, 64, 1), 0), out=buf337)
        buf338 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_31], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf337, buf338, 2097152, grid=grid(2097152), stream=stream0)
        buf339 = reinterpret_tensor(buf337, (4096, 512), (512, 1), 0); del buf337  # reuse
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf338, reinterpret_tensor(primals_119, (512, 512), (1, 512), 0), out=buf339)
        buf340 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf341 = reinterpret_tensor(buf340, (4, 1024, 1), (1024, 1, 1), 0); del buf340  # reuse
        buf342 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, forwarded_states_20, hidden_states_156, hidden_states_160, hidden_states_161, hidden_states_162, pow_28, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_8.run(buf341, buf306, buf323, buf339, primals_28, buf342, 4096, 512, grid=grid(4096), stream=stream0)
        buf343 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
        extern_kernels.mm(buf342, reinterpret_tensor(primals_120, (512, 2048), (1, 512), 0), out=buf343)
        buf344 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        buf393 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [forwarded_states_21, hidden_states_163], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_16.run(buf343, buf344, buf393, 8388608, grid=grid(8388608), stream=stream0)
        buf345 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf344, reinterpret_tensor(primals_121, (2048, 512), (1, 2048), 0), out=buf345)
        buf346 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf347 = reinterpret_tensor(buf346, (4, 1024, 1), (1024, 1, 1), 0); del buf346  # reuse
        buf348 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_169, l__mod___model_decoder_block_5_layer_0_self_attention_q, normed_hidden_states_16, pow_29, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_9.run(buf347, buf306, buf323, buf339, buf345, primals_29, buf348, 4096, 512, grid=grid(4096), stream=stream0)
        buf349 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf348, reinterpret_tensor(primals_122, (512, 512), (1, 512), 0), out=buf349)
        buf350 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf348, reinterpret_tensor(primals_123, (512, 512), (1, 512), 0), out=buf350)
        buf351 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf348, reinterpret_tensor(primals_124, (512, 512), (1, 512), 0), out=buf351)
        buf352 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf349, buf352, 2097152, grid=grid(2097152), stream=stream0)
        buf353 = reinterpret_tensor(buf349, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf349  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf350, buf353, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf354 = buf332; del buf332  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf353, (32, 64, 1024), (65536, 1024, 1), 0), out=buf354)
        buf356 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf358 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf392 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_33, softmax_16], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_red_fused__softmax_clone_detach_14.run(buf354, primals_74, buf356, buf358, buf392, 32768, 1024, grid=grid(32768), stream=stream0)
        del primals_74
        buf359 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf351, buf359, 2097152, grid=grid(2097152), stream=stream0)
        buf360 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf359, (32, 1024, 64), (65536, 64, 1), 0), out=buf360)
        buf361 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_33], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf360, buf361, 2097152, grid=grid(2097152), stream=stream0)
        buf362 = reinterpret_tensor(buf360, (4096, 512), (512, 1), 0); del buf360  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf361, reinterpret_tensor(primals_125, (512, 512), (1, 512), 0), out=buf362)
        buf363 = buf306; del buf306  # reuse
        buf364 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf365 = reinterpret_tensor(buf364, (4, 1024, 1), (1024, 1, 1), 0); del buf364  # reuse
        buf366 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_64, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_173, hidden_states_174, l__mod___model_decoder_block_5_layer_1_enc_dec_attention_q, normed_hidden_states_17, pow_30, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf363, buf365, buf323, buf339, buf345, buf362, primals_30, buf366, 4096, 512, grid=grid(4096), stream=stream0)
        buf367 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf366, reinterpret_tensor(primals_126, (512, 512), (1, 512), 0), out=buf367)
        buf368 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (1, 512), 0), out=buf368)
        buf369 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (1, 512), 0), out=buf369)
        buf370 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf367, buf370, 2097152, grid=grid(2097152), stream=stream0)
        buf371 = reinterpret_tensor(buf367, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf367  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf368, buf371, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf372 = reinterpret_tensor(buf356, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf356  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf370, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf371, (32, 64, 1024), (65536, 1024, 1), 0), out=buf372)
        buf375 = reinterpret_tensor(buf354, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf354  # reuse
        buf391 = empty((4, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_35, softmax_17], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_15.run(buf372, buf375, buf391, 32768, 1024, grid=grid(32768), stream=stream0)
        del buf372
        buf376 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf369, buf376, 2097152, grid=grid(2097152), stream=stream0)
        buf377 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf376, (32, 1024, 64), (65536, 64, 1), 0), out=buf377)
        buf378 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_35], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf377, buf378, 2097152, grid=grid(2097152), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (4096, 512), (512, 1), 0); del buf377  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(buf378, reinterpret_tensor(primals_129, (512, 512), (1, 512), 0), out=buf379)
        buf380 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf381 = reinterpret_tensor(buf380, (4, 1024, 1), (1024, 1, 1), 0); del buf380  # reuse
        buf382 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, forwarded_states_22, hidden_states_177, hidden_states_178, hidden_states_179, pow_31, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_6.run(buf381, buf363, buf379, primals_31, buf382, 4096, 512, grid=grid(4096), stream=stream0)
        buf383 = buf343; del buf343  # reuse
        # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
        extern_kernels.mm(buf382, reinterpret_tensor(primals_130, (512, 2048), (1, 512), 0), out=buf383)
        buf384 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        buf390 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [forwarded_states_23, hidden_states_180], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_16.run(buf383, buf384, buf390, 8388608, grid=grid(8388608), stream=stream0)
        del buf383
        buf385 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf384, reinterpret_tensor(primals_131, (2048, 512), (1, 2048), 0), out=buf385)
        buf386 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cuda', dtype=torch.float32)
        buf387 = reinterpret_tensor(buf386, (4, 1024, 1), (1024, 1, 1), 0); del buf386  # reuse
        buf388 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, hidden_states_177, hidden_states_185, hidden_states_186, hidden_states_187, lm_logits, pow_32, rsqrt_31, sequence_output_1, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_17.run(buf387, buf363, buf379, buf385, primals_32, buf388, 4096, 512, grid=grid(4096), stream=stream0)
        del buf363
        buf389 = empty((4096, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf388, reinterpret_tensor(primals_132, (512, 32128), (1, 512), 0), out=buf389)
        buf396 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf303, buf396, 8388608, grid=grid(8388608), stream=stream0)
        del buf303
        buf399 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_129], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf264, buf399, 8388608, grid=grid(8388608), stream=stream0)
        del buf264
        buf402 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_112], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf224, buf402, 8388608, grid=grid(8388608), stream=stream0)
        del buf224
        buf405 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf184, buf405, 8388608, grid=grid(8388608), stream=stream0)
        del buf184
        buf408 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf139, buf408, 8388608, grid=grid(8388608), stream=stream0)
        del buf139
        buf409 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_60], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf116, buf409, 8388608, grid=grid(8388608), stream=stream0)
        del buf116
        buf410 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_47], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf92, buf410, 8388608, grid=grid(8388608), stream=stream0)
        del buf92
        buf411 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_34], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf69, buf411, 8388608, grid=grid(8388608), stream=stream0)
        del buf69
        buf412 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf45, buf412, 8388608, grid=grid(8388608), stream=stream0)
        del buf45
        buf413 = empty((4, 1024, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf22, buf413, 8388608, grid=grid(8388608), stream=stream0)
        return (reinterpret_tensor(buf389, (4, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf151, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf152, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf169, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf170, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf191, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf192, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf209, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf210, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf231, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf232, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf248, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf249, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf271, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf272, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf288, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf289, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf311, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf312, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf328, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf329, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf350, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf351, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf368, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf369, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), buf145, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_133, buf0, buf2, buf3, buf10, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf40, buf41, buf43, buf44, buf46, buf47, buf50, buf51, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf74, buf87, buf88, buf90, buf91, buf93, buf94, buf97, buf98, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf121, buf134, buf135, buf137, buf138, buf140, buf141, buf144, primals_134, buf146, buf148, buf149, buf156, buf163, buf164, buf166, buf167, reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), buf179, buf180, buf182, buf183, buf185, buf186, buf188, buf189, buf202, buf203, buf206, buf207, buf219, buf220, buf222, buf223, buf225, buf226, buf228, buf229, buf242, buf243, buf245, buf246, buf258, buf259, buf262, buf263, buf265, buf266, buf268, buf269, buf282, buf283, buf285, buf286, buf298, buf299, buf301, buf302, buf304, buf305, buf308, buf309, buf322, buf323, buf325, buf326, buf338, buf339, buf341, buf342, buf344, buf345, buf347, buf348, buf361, buf362, buf365, buf366, buf378, buf379, buf381, buf382, buf384, buf385, buf387, buf388, reinterpret_tensor(primals_132, (32128, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 2048), (2048, 1), 0), buf390, reinterpret_tensor(primals_130, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 512), (512, 1), 0), reinterpret_tensor(buf375, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf376, (32, 64, 1024), (65536, 1, 64), 0), buf391, reinterpret_tensor(buf370, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf371, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_128, (512, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 512), (512, 1), 0), reinterpret_tensor(primals_125, (512, 512), (512, 1), 0), reinterpret_tensor(buf358, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf359, (32, 64, 1024), (65536, 1, 64), 0), buf392, reinterpret_tensor(buf352, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf353, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_124, (512, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 512), (512, 1), 0), reinterpret_tensor(primals_122, (512, 512), (512, 1), 0), reinterpret_tensor(primals_121, (512, 2048), (2048, 1), 0), buf393, reinterpret_tensor(primals_120, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 512), (512, 1), 0), reinterpret_tensor(buf335, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf336, (32, 64, 1024), (65536, 1, 64), 0), buf394, reinterpret_tensor(buf330, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf331, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_118, (512, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 512), (512, 1), 0), reinterpret_tensor(buf319, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf320, (32, 64, 1024), (65536, 1, 64), 0), buf395, reinterpret_tensor(buf313, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf314, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_114, (512, 512), (512, 1), 0), reinterpret_tensor(primals_113, (512, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 512), (512, 1), 0), reinterpret_tensor(primals_111, (512, 2048), (2048, 1), 0), buf396, reinterpret_tensor(primals_110, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), reinterpret_tensor(buf295, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf296, (32, 64, 1024), (65536, 1, 64), 0), buf397, reinterpret_tensor(buf290, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf291, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_108, (512, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (512, 1), 0), reinterpret_tensor(primals_106, (512, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 512), (512, 1), 0), reinterpret_tensor(buf279, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf280, (32, 64, 1024), (65536, 1, 64), 0), buf398, reinterpret_tensor(buf273, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf274, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_104, (512, 512), (512, 1), 0), reinterpret_tensor(primals_103, (512, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 512), (512, 1), 0), reinterpret_tensor(primals_101, (512, 2048), (2048, 1), 0), buf399, reinterpret_tensor(primals_100, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_99, (512, 512), (512, 1), 0), reinterpret_tensor(buf255, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf256, (32, 64, 1024), (65536, 1, 64), 0), buf400, reinterpret_tensor(buf250, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf251, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_98, (512, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 512), (512, 1), 0), reinterpret_tensor(primals_95, (512, 512), (512, 1), 0), reinterpret_tensor(buf239, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf240, (32, 64, 1024), (65536, 1, 64), 0), buf401, reinterpret_tensor(buf233, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf234, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_94, (512, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 512), (512, 1), 0), reinterpret_tensor(primals_92, (512, 512), (512, 1), 0), reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), buf402, reinterpret_tensor(primals_90, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_89, (512, 512), (512, 1), 0), reinterpret_tensor(buf216, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf217, (32, 64, 1024), (65536, 1, 64), 0), buf403, reinterpret_tensor(buf211, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf212, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_88, (512, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), reinterpret_tensor(buf199, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf200, (32, 64, 1024), (65536, 1, 64), 0), buf404, reinterpret_tensor(buf193, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf194, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_84, (512, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 512), (512, 1), 0), reinterpret_tensor(primals_82, (512, 512), (512, 1), 0), reinterpret_tensor(primals_81, (512, 2048), (2048, 1), 0), buf405, reinterpret_tensor(primals_80, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 512), (512, 1), 0), reinterpret_tensor(buf176, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf177, (32, 64, 1024), (65536, 1, 64), 0), buf406, reinterpret_tensor(buf171, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf172, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_78, (512, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 512), (512, 1), 0), reinterpret_tensor(primals_75, (512, 512), (512, 1), 0), reinterpret_tensor(buf160, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf161, (32, 64, 1024), (65536, 1, 64), 0), buf407, reinterpret_tensor(buf153, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf154, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 512), (512, 1), 0), reinterpret_tensor(primals_71, (512, 512), (512, 1), 0), reinterpret_tensor(primals_70, (512, 2048), (2048, 1), 0), buf408, reinterpret_tensor(primals_69, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_68, (512, 512), (512, 1), 0), reinterpret_tensor(buf131, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf132, (32, 64, 1024), (65536, 1, 64), 0), buf130, reinterpret_tensor(buf125, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf126, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_67, (512, 512), (512, 1), 0), reinterpret_tensor(primals_66, (512, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 512), (512, 1), 0), reinterpret_tensor(primals_64, (512, 2048), (2048, 1), 0), buf409, reinterpret_tensor(primals_63, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 512), (512, 1), 0), reinterpret_tensor(buf108, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf109, (32, 64, 1024), (65536, 1, 64), 0), buf107, reinterpret_tensor(buf102, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf103, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), reinterpret_tensor(primals_60, (512, 512), (512, 1), 0), reinterpret_tensor(primals_59, (512, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 2048), (2048, 1), 0), buf410, reinterpret_tensor(primals_57, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_56, (512, 512), (512, 1), 0), reinterpret_tensor(buf84, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf85, (32, 64, 1024), (65536, 1, 64), 0), buf83, reinterpret_tensor(buf78, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf79, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_55, (512, 512), (512, 1), 0), reinterpret_tensor(primals_54, (512, 512), (512, 1), 0), reinterpret_tensor(primals_53, (512, 512), (512, 1), 0), reinterpret_tensor(primals_52, (512, 2048), (2048, 1), 0), buf411, reinterpret_tensor(primals_51, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_50, (512, 512), (512, 1), 0), reinterpret_tensor(buf61, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf62, (32, 64, 1024), (65536, 1, 64), 0), buf60, reinterpret_tensor(buf55, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf56, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 512), (512, 1), 0), reinterpret_tensor(primals_47, (512, 512), (512, 1), 0), reinterpret_tensor(primals_46, (512, 2048), (2048, 1), 0), buf412, reinterpret_tensor(primals_45, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_44, (512, 512), (512, 1), 0), reinterpret_tensor(buf37, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf38, (32, 64, 1024), (65536, 1, 64), 0), buf36, reinterpret_tensor(buf31, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf32, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_43, (512, 512), (512, 1), 0), reinterpret_tensor(primals_42, (512, 512), (512, 1), 0), reinterpret_tensor(primals_41, (512, 512), (512, 1), 0), reinterpret_tensor(primals_40, (512, 2048), (2048, 1), 0), buf413, reinterpret_tensor(primals_39, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_38, (512, 512), (512, 1), 0), reinterpret_tensor(buf14, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf15, (32, 64, 1024), (65536, 1, 64), 0), buf13, reinterpret_tensor(buf7, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf8, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_36, (512, 512), (512, 1), 0), reinterpret_tensor(primals_35, (512, 512), (512, 1), 0), reinterpret_tensor(primals_34, (512, 512), (512, 1), 0), )


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
    primals_133 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_134 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
