
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


# kernel path: /tmp/torchinductor_youkaichao/wk/cwk6kuwncxw4rpfedwxbccviyb64nxyk53nobaxllsozmf6gu3w4.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 250112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 250112), "index out of bounds: 0 <= tmp3 < 250112")
    tmp4 = tl.load(in_ptr1 + (x0 + (512*tmp3)), None)
    tl.store(out_ptr0 + (x2), tmp4, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kjaraqgmr5nikcaj4r2sksvl6gnmklddzn2aqtoeycg7gbeuus.py
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/ch/cchsbplkramysuth2tuncldq4igw2yertzyntoksgscqsienkdrn.py
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
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
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


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvcan2d273u4vnwpmnb242kljhqkahaj2v3wyuwoy7erkwflz3m.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
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
    tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp26 < 32")
    tmp27 = tl.load(in_ptr1 + (x1 + (6*tmp26)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp0 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, float("-inf"))
    tmp32 = triton_helpers.max2(tmp31, 1)[:, None]
    tmp33 = tmp28 - tmp32
    tmp34 = tl.exp(tmp33)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp34 / tmp38
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5kc54iby3o5jsmsx4gpsaikmvbws27mdzdastjvzxkg7roaaa5.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgr6sxtdtivpvh2gzatgxiz7brce52eqxokvf6fqwuk4c2lkx6gc.py
# Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_5 => add_7
# forwarded_states => mul_6
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
# l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0 => view_21
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/am/camlhjc7r5wftdhyg2bvoqdi4beoqz73hcjhcvidmkirqwaepra5.py
# Source Nodes: [add_6, add_7, hidden_gelu, hidden_states_7, mul_7, mul_8, mul_9, pow_3, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_6 => add_8
# add_7 => add_9
# hidden_gelu => mul_10
# hidden_states_7 => mul_11
# mul_7 => mul_7
# mul_8 => mul_8
# mul_9 => mul_9
# pow_3 => pow_3
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp14 = tl.load(in_ptr1 + (x0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6kjmoxsqymv6gcswuc5leowwjdzsprpzbwa5lxvynzyvnim5am.py
# Source Nodes: [float_10, full_like_1, is_small_1, log_1, min_2, mul_82, relative_position, relative_position_3, relative_position_bucket_1, relative_position_if_large_2, relative_position_if_large_3, to_24, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
# float_10 => convert_element_type_5
# full_like_1 => full_default_4
# is_small_1 => lt_1
# log_1 => log_1
# min_2 => minimum_1
# mul_82 => mul_82
# relative_position => sub_1
# relative_position_3 => neg
# relative_position_bucket_1 => add_63
# relative_position_if_large_2 => add_62
# relative_position_if_large_3 => minimum_2
# to_24 => convert_element_type_6
# truediv_2 => div_10
# truediv_3 => div_11
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
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
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


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcm6vivrw7uykofztpgzry375pkc6sfdvs4poezxgu6mngdlzfe.py
# Source Nodes: [softmax_8], Original ATen: [aten._softmax]
# softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
triton_per_fused__softmax_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
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
    tmp21 = tl.load(in_ptr1 + (x1 + (6*tmp20)), xmask, eviction_policy='evict_last')
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
    tmp34 = tl.where(rmask & xmask, tmp32, float("-inf"))
    tmp35 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp36 = tmp31 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp37 / tmp41
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp42, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyy72vmw7gwzf7ssm36pn4wkxgczahdfjz7kwvhpmflkbfryh2h.py
# Source Nodes: [softmax_9], Original ATen: [aten._softmax]
# softmax_9 => amax_9, div_13, exp_9, sub_14, sum_10
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr3qk56svreh5lkvqur6ifxiaah3cnbn7mbv52pkdyq3yr3hfh4.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 62528
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
        tmp0 = tl.load(in_ptr0 + (r1 + (62528*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyjnm67dhx2dmtib6ot5vphlixi2tlkq5pac3prb2rdw2tosrq4.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_per_fused__log_softmax_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_11', 'mutated_arg_names': []}
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
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrjfmveklj6dugr54yilroi4fiijrssidq24ndy5ixvawswby5j.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_29, sum_25
triton_red_fused__log_softmax_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 62528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (62528*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdraqxjvdhmygntny7mqaqlmmfbx3jy4ytnb2ebtzdkss62apmrn.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_29, sum_25
triton_per_fused__log_softmax_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zc/czcykcwkk3jxizs6x57tbnzxgg57466vb4qd7ycnxllwma3yozpg.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => log_2, sub_29, sub_30
triton_poi_fused__log_softmax_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32014336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 250112)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.log(tmp3)
    tmp5 = tmp2 - tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspr3ojgcw4ed7hho5cfgbuztdschulenhpa3slxi3e2ywr76a7e.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_7, div_28, full_default_7, ne, neg_1, sum_26, sum_27, where_3
triton_per_fused_nll_loss_forward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 250112
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 250112), "index out of bounds: 0 <= tmp12 < 250112")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (250112*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp7.to(tl.float32)
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193 = args
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
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (250112, 512), (512, 1))
    assert_size_stride(primals_44, (384, 512), (512, 1))
    assert_size_stride(primals_45, (384, 512), (512, 1))
    assert_size_stride(primals_46, (384, 512), (512, 1))
    assert_size_stride(primals_47, (32, 6), (6, 1))
    assert_size_stride(primals_48, (512, 384), (384, 1))
    assert_size_stride(primals_49, (1024, 512), (512, 1))
    assert_size_stride(primals_50, (1024, 512), (512, 1))
    assert_size_stride(primals_51, (512, 1024), (1024, 1))
    assert_size_stride(primals_52, (384, 512), (512, 1))
    assert_size_stride(primals_53, (384, 512), (512, 1))
    assert_size_stride(primals_54, (384, 512), (512, 1))
    assert_size_stride(primals_55, (512, 384), (384, 1))
    assert_size_stride(primals_56, (1024, 512), (512, 1))
    assert_size_stride(primals_57, (1024, 512), (512, 1))
    assert_size_stride(primals_58, (512, 1024), (1024, 1))
    assert_size_stride(primals_59, (384, 512), (512, 1))
    assert_size_stride(primals_60, (384, 512), (512, 1))
    assert_size_stride(primals_61, (384, 512), (512, 1))
    assert_size_stride(primals_62, (512, 384), (384, 1))
    assert_size_stride(primals_63, (1024, 512), (512, 1))
    assert_size_stride(primals_64, (1024, 512), (512, 1))
    assert_size_stride(primals_65, (512, 1024), (1024, 1))
    assert_size_stride(primals_66, (384, 512), (512, 1))
    assert_size_stride(primals_67, (384, 512), (512, 1))
    assert_size_stride(primals_68, (384, 512), (512, 1))
    assert_size_stride(primals_69, (512, 384), (384, 1))
    assert_size_stride(primals_70, (1024, 512), (512, 1))
    assert_size_stride(primals_71, (1024, 512), (512, 1))
    assert_size_stride(primals_72, (512, 1024), (1024, 1))
    assert_size_stride(primals_73, (384, 512), (512, 1))
    assert_size_stride(primals_74, (384, 512), (512, 1))
    assert_size_stride(primals_75, (384, 512), (512, 1))
    assert_size_stride(primals_76, (512, 384), (384, 1))
    assert_size_stride(primals_77, (1024, 512), (512, 1))
    assert_size_stride(primals_78, (1024, 512), (512, 1))
    assert_size_stride(primals_79, (512, 1024), (1024, 1))
    assert_size_stride(primals_80, (384, 512), (512, 1))
    assert_size_stride(primals_81, (384, 512), (512, 1))
    assert_size_stride(primals_82, (384, 512), (512, 1))
    assert_size_stride(primals_83, (512, 384), (384, 1))
    assert_size_stride(primals_84, (1024, 512), (512, 1))
    assert_size_stride(primals_85, (1024, 512), (512, 1))
    assert_size_stride(primals_86, (512, 1024), (1024, 1))
    assert_size_stride(primals_87, (384, 512), (512, 1))
    assert_size_stride(primals_88, (384, 512), (512, 1))
    assert_size_stride(primals_89, (384, 512), (512, 1))
    assert_size_stride(primals_90, (512, 384), (384, 1))
    assert_size_stride(primals_91, (1024, 512), (512, 1))
    assert_size_stride(primals_92, (1024, 512), (512, 1))
    assert_size_stride(primals_93, (512, 1024), (1024, 1))
    assert_size_stride(primals_94, (384, 512), (512, 1))
    assert_size_stride(primals_95, (384, 512), (512, 1))
    assert_size_stride(primals_96, (384, 512), (512, 1))
    assert_size_stride(primals_97, (512, 384), (384, 1))
    assert_size_stride(primals_98, (1024, 512), (512, 1))
    assert_size_stride(primals_99, (1024, 512), (512, 1))
    assert_size_stride(primals_100, (512, 1024), (1024, 1))
    assert_size_stride(primals_101, (384, 512), (512, 1))
    assert_size_stride(primals_102, (384, 512), (512, 1))
    assert_size_stride(primals_103, (384, 512), (512, 1))
    assert_size_stride(primals_104, (32, 6), (6, 1))
    assert_size_stride(primals_105, (512, 384), (384, 1))
    assert_size_stride(primals_106, (384, 512), (512, 1))
    assert_size_stride(primals_107, (384, 512), (512, 1))
    assert_size_stride(primals_108, (384, 512), (512, 1))
    assert_size_stride(primals_109, (512, 384), (384, 1))
    assert_size_stride(primals_110, (1024, 512), (512, 1))
    assert_size_stride(primals_111, (1024, 512), (512, 1))
    assert_size_stride(primals_112, (512, 1024), (1024, 1))
    assert_size_stride(primals_113, (384, 512), (512, 1))
    assert_size_stride(primals_114, (384, 512), (512, 1))
    assert_size_stride(primals_115, (384, 512), (512, 1))
    assert_size_stride(primals_116, (512, 384), (384, 1))
    assert_size_stride(primals_117, (384, 512), (512, 1))
    assert_size_stride(primals_118, (384, 512), (512, 1))
    assert_size_stride(primals_119, (384, 512), (512, 1))
    assert_size_stride(primals_120, (512, 384), (384, 1))
    assert_size_stride(primals_121, (1024, 512), (512, 1))
    assert_size_stride(primals_122, (1024, 512), (512, 1))
    assert_size_stride(primals_123, (512, 1024), (1024, 1))
    assert_size_stride(primals_124, (384, 512), (512, 1))
    assert_size_stride(primals_125, (384, 512), (512, 1))
    assert_size_stride(primals_126, (384, 512), (512, 1))
    assert_size_stride(primals_127, (512, 384), (384, 1))
    assert_size_stride(primals_128, (384, 512), (512, 1))
    assert_size_stride(primals_129, (384, 512), (512, 1))
    assert_size_stride(primals_130, (384, 512), (512, 1))
    assert_size_stride(primals_131, (512, 384), (384, 1))
    assert_size_stride(primals_132, (1024, 512), (512, 1))
    assert_size_stride(primals_133, (1024, 512), (512, 1))
    assert_size_stride(primals_134, (512, 1024), (1024, 1))
    assert_size_stride(primals_135, (384, 512), (512, 1))
    assert_size_stride(primals_136, (384, 512), (512, 1))
    assert_size_stride(primals_137, (384, 512), (512, 1))
    assert_size_stride(primals_138, (512, 384), (384, 1))
    assert_size_stride(primals_139, (384, 512), (512, 1))
    assert_size_stride(primals_140, (384, 512), (512, 1))
    assert_size_stride(primals_141, (384, 512), (512, 1))
    assert_size_stride(primals_142, (512, 384), (384, 1))
    assert_size_stride(primals_143, (1024, 512), (512, 1))
    assert_size_stride(primals_144, (1024, 512), (512, 1))
    assert_size_stride(primals_145, (512, 1024), (1024, 1))
    assert_size_stride(primals_146, (384, 512), (512, 1))
    assert_size_stride(primals_147, (384, 512), (512, 1))
    assert_size_stride(primals_148, (384, 512), (512, 1))
    assert_size_stride(primals_149, (512, 384), (384, 1))
    assert_size_stride(primals_150, (384, 512), (512, 1))
    assert_size_stride(primals_151, (384, 512), (512, 1))
    assert_size_stride(primals_152, (384, 512), (512, 1))
    assert_size_stride(primals_153, (512, 384), (384, 1))
    assert_size_stride(primals_154, (1024, 512), (512, 1))
    assert_size_stride(primals_155, (1024, 512), (512, 1))
    assert_size_stride(primals_156, (512, 1024), (1024, 1))
    assert_size_stride(primals_157, (384, 512), (512, 1))
    assert_size_stride(primals_158, (384, 512), (512, 1))
    assert_size_stride(primals_159, (384, 512), (512, 1))
    assert_size_stride(primals_160, (512, 384), (384, 1))
    assert_size_stride(primals_161, (384, 512), (512, 1))
    assert_size_stride(primals_162, (384, 512), (512, 1))
    assert_size_stride(primals_163, (384, 512), (512, 1))
    assert_size_stride(primals_164, (512, 384), (384, 1))
    assert_size_stride(primals_165, (1024, 512), (512, 1))
    assert_size_stride(primals_166, (1024, 512), (512, 1))
    assert_size_stride(primals_167, (512, 1024), (1024, 1))
    assert_size_stride(primals_168, (384, 512), (512, 1))
    assert_size_stride(primals_169, (384, 512), (512, 1))
    assert_size_stride(primals_170, (384, 512), (512, 1))
    assert_size_stride(primals_171, (512, 384), (384, 1))
    assert_size_stride(primals_172, (384, 512), (512, 1))
    assert_size_stride(primals_173, (384, 512), (512, 1))
    assert_size_stride(primals_174, (384, 512), (512, 1))
    assert_size_stride(primals_175, (512, 384), (384, 1))
    assert_size_stride(primals_176, (1024, 512), (512, 1))
    assert_size_stride(primals_177, (1024, 512), (512, 1))
    assert_size_stride(primals_178, (512, 1024), (1024, 1))
    assert_size_stride(primals_179, (384, 512), (512, 1))
    assert_size_stride(primals_180, (384, 512), (512, 1))
    assert_size_stride(primals_181, (384, 512), (512, 1))
    assert_size_stride(primals_182, (512, 384), (384, 1))
    assert_size_stride(primals_183, (384, 512), (512, 1))
    assert_size_stride(primals_184, (384, 512), (512, 1))
    assert_size_stride(primals_185, (384, 512), (512, 1))
    assert_size_stride(primals_186, (512, 384), (384, 1))
    assert_size_stride(primals_187, (1024, 512), (512, 1))
    assert_size_stride(primals_188, (1024, 512), (512, 1))
    assert_size_stride(primals_189, (512, 1024), (1024, 1))
    assert_size_stride(primals_190, (250112, 512), (512, 1))
    assert_size_stride(primals_191, (1, 128), (128, 1))
    assert_size_stride(primals_192, (1, 128), (128, 1))
    assert_size_stride(primals_193, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [inputs_embeds], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(primals_191, primals_43, buf0, 65536, grid=grid(65536), stream=stream0)
        # Source Nodes: [hidden_states, inputs_embeds], Original ATen: [aten.embedding, aten.native_dropout]
        buf1 = aten.native_dropout(buf0, 0.1, True)
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        buf4 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf5 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 1), 0); del buf4  # reuse
        buf6 = reinterpret_tensor(buf0, (128, 512), (512, 1), 0); del buf0  # reuse
        # Source Nodes: [add, hidden_states_1, l__mod___encoder_block_0_layer_0_self_attention_q, normed_hidden_states, pow_1, rsqrt, variance], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_1.run(buf5, buf2, primals_1, buf6, 128, 512, grid=grid(128), stream=stream0)
        buf7 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_44, (512, 384), (1, 512), 0), out=buf7)
        buf8 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_45, (512, 384), (1, 512), 0), out=buf8)
        buf9 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_46, (512, 384), (1, 512), 0), out=buf9)
        buf10 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf8, (6, 64, 128), (64, 1, 384), 0), out=buf10)
        buf11 = empty((128, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_1, full_like, gt, is_small, log, mul_3, mul_4, relative_buckets, relative_position, relative_position_1, relative_position_bucket, relative_position_if_large, relative_position_if_large_1, to_2, to_3, truediv, truediv_1, where], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.full_like, aten.gt, aten.log, aten.lt, aten.minimum, aten.mul, aten.sub, aten.where]
        triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2.run(buf11, 16384, grid=grid(16384), stream=stream0)
        buf14 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf10, primals_47, buf14, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.native_dropout]
        buf15 = aten.native_dropout(buf14, 0.1, True)
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf9, (6, 128, 64), (64, 384, 1), 0), out=buf18)
        buf19 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_1], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf18, buf19, 49152, grid=grid(49152), stream=stream0)
        buf20 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_48, (384, 512), (1, 384), 0), out=buf20)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf21 = aten.native_dropout(reinterpret_tensor(buf20, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = buf22; del buf22  # reuse
        buf25 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf25, (1, 128, 1), (128, 1, 1), 0); del buf25  # reuse
        buf27 = buf20; del buf20  # reuse
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf24, buf26, buf2, primals_2, buf27, 128, 512, grid=grid(128), stream=stream0)
        buf28 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_49, (512, 1024), (1, 512), 0), out=buf28)
        buf30 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_50, (512, 1024), (1, 512), 0), out=buf30)
        buf29 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf31 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7, hidden_gelu, hidden_states_7, mul_7, mul_8, mul_9, pow_3, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf28, buf30, buf29, buf31, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_7, hidden_gelu, hidden_states_7, hidden_states_8, mul_7], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf32 = aten.native_dropout(buf31, 0.1, True)
        buf33 = buf32[0]
        buf34 = buf32[1]
        del buf32
        buf35 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_51, (1024, 512), (1, 1024), 0), out=buf35)
        # Source Nodes: [l__mod___encoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf36 = aten.native_dropout(reinterpret_tensor(buf35, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = buf37; del buf37  # reuse
        buf40 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf41 = reinterpret_tensor(buf40, (1, 128, 1), (128, 1, 1), 0); del buf40  # reuse
        buf42 = buf35; del buf35  # reuse
        # Source Nodes: [add_9, hidden_states_12, hidden_states_13, l__mod___encoder_block_1_layer_0_self_attention_q, normed_hidden_states_1, pow_4, rsqrt_2, variance_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf39, buf41, buf24, primals_3, buf42, 128, 512, grid=grid(128), stream=stream0)
        buf43 = reinterpret_tensor(buf18, (128, 384), (384, 1), 0); del buf18  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_52, (512, 384), (1, 512), 0), out=buf43)
        buf44 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_53, (512, 384), (1, 512), 0), out=buf44)
        buf45 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_54, (512, 384), (1, 512), 0), out=buf45)
        buf46 = buf10; del buf10  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf44, (6, 64, 128), (64, 1, 384), 0), out=buf46)
        buf49 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf46, primals_47, buf49, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_3, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
        buf50 = aten.native_dropout(buf49, 0.1, True)
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf53 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf45, (6, 128, 64), (64, 384, 1), 0), out=buf53)
        buf54 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_3], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf53, buf54, 49152, grid=grid(49152), stream=stream0)
        buf55 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_55, (384, 512), (1, 384), 0), out=buf55)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf56 = aten.native_dropout(reinterpret_tensor(buf55, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = buf57; del buf57  # reuse
        buf60 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf61 = reinterpret_tensor(buf60, (1, 128, 1), (128, 1, 1), 0); del buf60  # reuse
        buf62 = buf55; del buf55  # reuse
        # Source Nodes: [add_11, forwarded_states_2, hidden_states_17, hidden_states_18, l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0, pow_5, rsqrt_3, variance_3], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf59, buf61, buf39, primals_4, buf62, 128, 512, grid=grid(128), stream=stream0)
        buf63 = reinterpret_tensor(buf31, (128, 1024), (1024, 1), 0); del buf31  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf62, reinterpret_tensor(primals_56, (512, 1024), (1, 512), 0), out=buf63)
        buf65 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf62, reinterpret_tensor(primals_57, (512, 1024), (1, 512), 0), out=buf65)
        buf64 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf66 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, add_13, hidden_gelu_1, hidden_states_19, mul_16, mul_17, mul_18, pow_6, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf63, buf65, buf64, buf66, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_13, hidden_gelu_1, hidden_states_19, hidden_states_20, mul_16], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf67 = aten.native_dropout(buf66, 0.1, True)
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_58, (1024, 512), (1, 1024), 0), out=buf70)
        # Source Nodes: [l__mod___encoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf71 = aten.native_dropout(reinterpret_tensor(buf70, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf72 = buf71[0]
        buf73 = buf71[1]
        del buf71
        buf74 = buf72; del buf72  # reuse
        buf75 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf76 = reinterpret_tensor(buf75, (1, 128, 1), (128, 1, 1), 0); del buf75  # reuse
        buf77 = buf70; del buf70  # reuse
        # Source Nodes: [add_15, hidden_states_24, hidden_states_25, l__mod___encoder_block_2_layer_0_self_attention_q, normed_hidden_states_2, pow_7, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf74, buf76, buf59, primals_5, buf77, 128, 512, grid=grid(128), stream=stream0)
        buf78 = reinterpret_tensor(buf53, (128, 384), (384, 1), 0); del buf53  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_59, (512, 384), (1, 512), 0), out=buf78)
        buf79 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_60, (512, 384), (1, 512), 0), out=buf79)
        buf80 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_61, (512, 384), (1, 512), 0), out=buf80)
        buf81 = buf46; del buf46  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf79, (6, 64, 128), (64, 1, 384), 0), out=buf81)
        buf84 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf81, primals_47, buf84, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_5, softmax_2], Original ATen: [aten._softmax, aten.native_dropout]
        buf85 = aten.native_dropout(buf84, 0.1, True)
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf80, (6, 128, 64), (64, 384, 1), 0), out=buf88)
        buf89 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_5], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf88, buf89, 49152, grid=grid(49152), stream=stream0)
        buf90 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf89, reinterpret_tensor(primals_62, (384, 512), (1, 384), 0), out=buf90)
        # Source Nodes: [l__mod___encoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf91 = aten.native_dropout(reinterpret_tensor(buf90, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf92; del buf92  # reuse
        buf95 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf96 = reinterpret_tensor(buf95, (1, 128, 1), (128, 1, 1), 0); del buf95  # reuse
        buf97 = buf90; del buf90  # reuse
        # Source Nodes: [add_17, forwarded_states_4, hidden_states_29, hidden_states_30, l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0, pow_8, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf94, buf96, buf74, primals_6, buf97, 128, 512, grid=grid(128), stream=stream0)
        buf98 = reinterpret_tensor(buf66, (128, 1024), (1024, 1), 0); del buf66  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf97, reinterpret_tensor(primals_63, (512, 1024), (1, 512), 0), out=buf98)
        buf100 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf97, reinterpret_tensor(primals_64, (512, 1024), (1, 512), 0), out=buf100)
        buf99 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf101 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, hidden_gelu_2, hidden_states_31, mul_25, mul_26, mul_27, pow_9, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf98, buf100, buf99, buf101, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_19, hidden_gelu_2, hidden_states_31, hidden_states_32, mul_25], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf102 = aten.native_dropout(buf101, 0.1, True)
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_65, (1024, 512), (1, 1024), 0), out=buf105)
        # Source Nodes: [l__mod___encoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf106 = aten.native_dropout(reinterpret_tensor(buf105, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf107 = buf106[0]
        buf108 = buf106[1]
        del buf106
        buf109 = buf107; del buf107  # reuse
        buf110 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf111 = reinterpret_tensor(buf110, (1, 128, 1), (128, 1, 1), 0); del buf110  # reuse
        buf112 = buf105; del buf105  # reuse
        # Source Nodes: [add_21, hidden_states_36, hidden_states_37, l__mod___encoder_block_3_layer_0_self_attention_q, normed_hidden_states_3, pow_10, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf109, buf111, buf94, primals_7, buf112, 128, 512, grid=grid(128), stream=stream0)
        buf113 = reinterpret_tensor(buf88, (128, 384), (384, 1), 0); del buf88  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_66, (512, 384), (1, 512), 0), out=buf113)
        buf114 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_67, (512, 384), (1, 512), 0), out=buf114)
        buf115 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_68, (512, 384), (1, 512), 0), out=buf115)
        buf116 = buf81; del buf81  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf113, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf114, (6, 64, 128), (64, 1, 384), 0), out=buf116)
        buf119 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf116, primals_47, buf119, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_7, softmax_3], Original ATen: [aten._softmax, aten.native_dropout]
        buf120 = aten.native_dropout(buf119, 0.1, True)
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf123 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf115, (6, 128, 64), (64, 384, 1), 0), out=buf123)
        buf124 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_7], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf123, buf124, 49152, grid=grid(49152), stream=stream0)
        buf125 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, reinterpret_tensor(primals_69, (384, 512), (1, 384), 0), out=buf125)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf126 = aten.native_dropout(reinterpret_tensor(buf125, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = buf127; del buf127  # reuse
        buf130 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf131 = reinterpret_tensor(buf130, (1, 128, 1), (128, 1, 1), 0); del buf130  # reuse
        buf132 = buf125; del buf125  # reuse
        # Source Nodes: [add_23, forwarded_states_6, hidden_states_41, hidden_states_42, l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0, pow_11, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf129, buf131, buf109, primals_8, buf132, 128, 512, grid=grid(128), stream=stream0)
        buf133 = reinterpret_tensor(buf101, (128, 1024), (1024, 1), 0); del buf101  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf132, reinterpret_tensor(primals_70, (512, 1024), (1, 512), 0), out=buf133)
        buf135 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf132, reinterpret_tensor(primals_71, (512, 1024), (1, 512), 0), out=buf135)
        buf134 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf136 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, add_25, hidden_gelu_3, hidden_states_43, mul_34, mul_35, mul_36, pow_12, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf133, buf135, buf134, buf136, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_25, hidden_gelu_3, hidden_states_43, hidden_states_44, mul_34], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf137 = aten.native_dropout(buf136, 0.1, True)
        buf138 = buf137[0]
        buf139 = buf137[1]
        del buf137
        buf140 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_72, (1024, 512), (1, 1024), 0), out=buf140)
        # Source Nodes: [l__mod___encoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf141 = aten.native_dropout(reinterpret_tensor(buf140, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = buf142; del buf142  # reuse
        buf145 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf146 = reinterpret_tensor(buf145, (1, 128, 1), (128, 1, 1), 0); del buf145  # reuse
        buf147 = buf140; del buf140  # reuse
        # Source Nodes: [add_27, hidden_states_48, hidden_states_49, l__mod___encoder_block_4_layer_0_self_attention_q, normed_hidden_states_4, pow_13, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf144, buf146, buf129, primals_9, buf147, 128, 512, grid=grid(128), stream=stream0)
        buf148 = reinterpret_tensor(buf123, (128, 384), (384, 1), 0); del buf123  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_73, (512, 384), (1, 512), 0), out=buf148)
        buf149 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_74, (512, 384), (1, 512), 0), out=buf149)
        buf150 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_75, (512, 384), (1, 512), 0), out=buf150)
        buf151 = buf116; del buf116  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf149, (6, 64, 128), (64, 1, 384), 0), out=buf151)
        buf154 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf151, primals_47, buf154, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_9, softmax_4], Original ATen: [aten._softmax, aten.native_dropout]
        buf155 = aten.native_dropout(buf154, 0.1, True)
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf150, (6, 128, 64), (64, 384, 1), 0), out=buf158)
        buf159 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_9], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf158, buf159, 49152, grid=grid(49152), stream=stream0)
        buf160 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf159, reinterpret_tensor(primals_76, (384, 512), (1, 384), 0), out=buf160)
        # Source Nodes: [l__mod___encoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf161 = aten.native_dropout(reinterpret_tensor(buf160, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf162; del buf162  # reuse
        buf165 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf165, (1, 128, 1), (128, 1, 1), 0); del buf165  # reuse
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [add_29, forwarded_states_8, hidden_states_53, hidden_states_54, l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0, pow_14, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf164, buf166, buf144, primals_10, buf167, 128, 512, grid=grid(128), stream=stream0)
        buf168 = reinterpret_tensor(buf136, (128, 1024), (1024, 1), 0); del buf136  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, reinterpret_tensor(primals_77, (512, 1024), (1, 512), 0), out=buf168)
        buf170 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, reinterpret_tensor(primals_78, (512, 1024), (1, 512), 0), out=buf170)
        buf169 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf171 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31, hidden_gelu_4, hidden_states_55, mul_43, mul_44, mul_45, pow_15, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf168, buf170, buf169, buf171, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_31, hidden_gelu_4, hidden_states_55, hidden_states_56, mul_43], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf172 = aten.native_dropout(buf171, 0.1, True)
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_79, (1024, 512), (1, 1024), 0), out=buf175)
        # Source Nodes: [l__mod___encoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf176 = aten.native_dropout(reinterpret_tensor(buf175, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf177 = buf176[0]
        buf178 = buf176[1]
        del buf176
        buf179 = buf177; del buf177  # reuse
        buf180 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf181 = reinterpret_tensor(buf180, (1, 128, 1), (128, 1, 1), 0); del buf180  # reuse
        buf182 = buf175; del buf175  # reuse
        # Source Nodes: [add_33, hidden_states_60, hidden_states_61, l__mod___encoder_block_5_layer_0_self_attention_q, normed_hidden_states_5, pow_16, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf179, buf181, buf164, primals_11, buf182, 128, 512, grid=grid(128), stream=stream0)
        buf183 = reinterpret_tensor(buf158, (128, 384), (384, 1), 0); del buf158  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf182, reinterpret_tensor(primals_80, (512, 384), (1, 512), 0), out=buf183)
        buf184 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf182, reinterpret_tensor(primals_81, (512, 384), (1, 512), 0), out=buf184)
        buf185 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf182, reinterpret_tensor(primals_82, (512, 384), (1, 512), 0), out=buf185)
        buf186 = buf151; del buf151  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf184, (6, 64, 128), (64, 1, 384), 0), out=buf186)
        buf189 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf186, primals_47, buf189, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_11, softmax_5], Original ATen: [aten._softmax, aten.native_dropout]
        buf190 = aten.native_dropout(buf189, 0.1, True)
        buf191 = buf190[0]
        buf192 = buf190[1]
        del buf190
        buf193 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf185, (6, 128, 64), (64, 384, 1), 0), out=buf193)
        buf194 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_11], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf193, buf194, 49152, grid=grid(49152), stream=stream0)
        buf195 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf194, reinterpret_tensor(primals_83, (384, 512), (1, 384), 0), out=buf195)
        # Source Nodes: [l__mod___encoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf196 = aten.native_dropout(reinterpret_tensor(buf195, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf197 = buf196[0]
        buf198 = buf196[1]
        del buf196
        buf199 = buf197; del buf197  # reuse
        buf200 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf201 = reinterpret_tensor(buf200, (1, 128, 1), (128, 1, 1), 0); del buf200  # reuse
        buf202 = buf195; del buf195  # reuse
        # Source Nodes: [add_35, forwarded_states_10, hidden_states_65, hidden_states_66, l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0, pow_17, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf199, buf201, buf179, primals_12, buf202, 128, 512, grid=grid(128), stream=stream0)
        buf203 = reinterpret_tensor(buf171, (128, 1024), (1024, 1), 0); del buf171  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_84, (512, 1024), (1, 512), 0), out=buf203)
        buf205 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_85, (512, 1024), (1, 512), 0), out=buf205)
        buf204 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf206 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, add_37, hidden_gelu_5, hidden_states_67, mul_52, mul_53, mul_54, pow_18, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf203, buf205, buf204, buf206, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_37, hidden_gelu_5, hidden_states_67, hidden_states_68, mul_52], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf207 = aten.native_dropout(buf206, 0.1, True)
        buf208 = buf207[0]
        buf209 = buf207[1]
        del buf207
        buf210 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_86, (1024, 512), (1, 1024), 0), out=buf210)
        # Source Nodes: [l__mod___encoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf211 = aten.native_dropout(reinterpret_tensor(buf210, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf212 = buf211[0]
        buf213 = buf211[1]
        del buf211
        buf214 = buf212; del buf212  # reuse
        buf215 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf216 = reinterpret_tensor(buf215, (1, 128, 1), (128, 1, 1), 0); del buf215  # reuse
        buf217 = buf210; del buf210  # reuse
        # Source Nodes: [add_39, hidden_states_72, hidden_states_73, l__mod___encoder_block_6_layer_0_self_attention_q, normed_hidden_states_6, pow_19, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf214, buf216, buf199, primals_13, buf217, 128, 512, grid=grid(128), stream=stream0)
        buf218 = reinterpret_tensor(buf193, (128, 384), (384, 1), 0); del buf193  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, reinterpret_tensor(primals_87, (512, 384), (1, 512), 0), out=buf218)
        buf219 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, reinterpret_tensor(primals_88, (512, 384), (1, 512), 0), out=buf219)
        buf220 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, reinterpret_tensor(primals_89, (512, 384), (1, 512), 0), out=buf220)
        buf221 = buf186; del buf186  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf218, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf219, (6, 64, 128), (64, 1, 384), 0), out=buf221)
        buf224 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_6], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf221, primals_47, buf224, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.native_dropout]
        buf225 = aten.native_dropout(buf224, 0.1, True)
        buf226 = buf225[0]
        buf227 = buf225[1]
        del buf225
        buf228 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf220, (6, 128, 64), (64, 384, 1), 0), out=buf228)
        buf229 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_13], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf228, buf229, 49152, grid=grid(49152), stream=stream0)
        buf230 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, reinterpret_tensor(primals_90, (384, 512), (1, 384), 0), out=buf230)
        # Source Nodes: [l__mod___encoder_block_6_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf231 = aten.native_dropout(reinterpret_tensor(buf230, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf232 = buf231[0]
        buf233 = buf231[1]
        del buf231
        buf234 = buf232; del buf232  # reuse
        buf235 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf235, (1, 128, 1), (128, 1, 1), 0); del buf235  # reuse
        buf237 = buf230; del buf230  # reuse
        # Source Nodes: [add_41, forwarded_states_12, hidden_states_77, hidden_states_78, l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0, pow_20, rsqrt_13, variance_13], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf234, buf236, buf214, primals_14, buf237, 128, 512, grid=grid(128), stream=stream0)
        buf238 = reinterpret_tensor(buf206, (128, 1024), (1024, 1), 0); del buf206  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf237, reinterpret_tensor(primals_91, (512, 1024), (1, 512), 0), out=buf238)
        buf240 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf237, reinterpret_tensor(primals_92, (512, 1024), (1, 512), 0), out=buf240)
        buf239 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf241 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43, hidden_gelu_6, hidden_states_79, mul_61, mul_62, mul_63, pow_21, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf238, buf240, buf239, buf241, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_43, hidden_gelu_6, hidden_states_79, hidden_states_80, mul_61], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf242 = aten.native_dropout(buf241, 0.1, True)
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        buf245 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_93, (1024, 512), (1, 1024), 0), out=buf245)
        # Source Nodes: [l__mod___encoder_block_6_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf246 = aten.native_dropout(reinterpret_tensor(buf245, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf247 = buf246[0]
        buf248 = buf246[1]
        del buf246
        buf249 = buf247; del buf247  # reuse
        buf250 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf251 = reinterpret_tensor(buf250, (1, 128, 1), (128, 1, 1), 0); del buf250  # reuse
        buf252 = buf245; del buf245  # reuse
        # Source Nodes: [add_45, hidden_states_84, hidden_states_85, l__mod___encoder_block_7_layer_0_self_attention_q, normed_hidden_states_7, pow_22, rsqrt_14, variance_14], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf249, buf251, buf234, primals_15, buf252, 128, 512, grid=grid(128), stream=stream0)
        buf253 = reinterpret_tensor(buf228, (128, 384), (384, 1), 0); del buf228  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf252, reinterpret_tensor(primals_94, (512, 384), (1, 512), 0), out=buf253)
        buf254 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf252, reinterpret_tensor(primals_95, (512, 384), (1, 512), 0), out=buf254)
        buf255 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf252, reinterpret_tensor(primals_96, (512, 384), (1, 512), 0), out=buf255)
        buf256 = buf221; del buf221  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf253, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf254, (6, 64, 128), (64, 1, 384), 0), out=buf256)
        buf259 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf256, primals_47, buf259, 768, 128, grid=grid(768), stream=stream0)
        del primals_47
        # Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.native_dropout]
        buf260 = aten.native_dropout(buf259, 0.1, True)
        buf261 = buf260[0]
        buf262 = buf260[1]
        del buf260
        buf263 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf255, (6, 128, 64), (64, 384, 1), 0), out=buf263)
        buf264 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_15], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf263, buf264, 49152, grid=grid(49152), stream=stream0)
        buf265 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf264, reinterpret_tensor(primals_97, (384, 512), (1, 384), 0), out=buf265)
        # Source Nodes: [l__mod___encoder_block_7_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf266 = aten.native_dropout(reinterpret_tensor(buf265, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = buf267; del buf267  # reuse
        buf270 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf271 = reinterpret_tensor(buf270, (1, 128, 1), (128, 1, 1), 0); del buf270  # reuse
        buf272 = buf265; del buf265  # reuse
        # Source Nodes: [add_47, forwarded_states_14, hidden_states_89, hidden_states_90, l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0, pow_23, rsqrt_15, variance_15], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf269, buf271, buf249, primals_16, buf272, 128, 512, grid=grid(128), stream=stream0)
        buf273 = reinterpret_tensor(buf241, (128, 1024), (1024, 1), 0); del buf241  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, reinterpret_tensor(primals_98, (512, 1024), (1, 512), 0), out=buf273)
        buf275 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, reinterpret_tensor(primals_99, (512, 1024), (1, 512), 0), out=buf275)
        buf274 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf276 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, add_49, hidden_gelu_7, hidden_states_91, mul_70, mul_71, mul_72, pow_24, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf273, buf275, buf274, buf276, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_49, hidden_gelu_7, hidden_states_91, hidden_states_92, mul_70], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf277 = aten.native_dropout(buf276, 0.1, True)
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_100, (1024, 512), (1, 1024), 0), out=buf280)
        # Source Nodes: [l__mod___encoder_block_7_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf281 = aten.native_dropout(reinterpret_tensor(buf280, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf282 = buf281[0]
        buf283 = buf281[1]
        del buf281
        buf284 = buf282; del buf282  # reuse
        buf285 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf286 = reinterpret_tensor(buf285, (1, 128, 1), (128, 1, 1), 0); del buf285  # reuse
        buf287 = reinterpret_tensor(buf280, (1, 128, 512), (65536, 512, 1), 0); del buf280  # reuse
        # Source Nodes: [add_51, hidden_states_96, hidden_states_97, hidden_states_98, pow_25, rsqrt_16, variance_16], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf284, buf286, buf269, primals_17, buf287, 128, 512, grid=grid(128), stream=stream0)
        # Source Nodes: [hidden_states_100, hidden_states_97, hidden_states_98], Original ATen: [aten.mul, aten.native_dropout]
        buf288 = aten.native_dropout(buf287, 0.1, True)
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf291 = buf287; del buf287  # reuse
        # Source Nodes: [inputs_embeds_1], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_0.run(primals_193, primals_43, buf291, 65536, grid=grid(65536), stream=stream0)
        del primals_43
        # Source Nodes: [hidden_states_101, inputs_embeds_1], Original ATen: [aten.embedding, aten.native_dropout]
        buf292 = aten.native_dropout(buf291, 0.1, True)
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        buf295 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf296 = reinterpret_tensor(buf295, (1, 128, 1), (128, 1, 1), 0); del buf295  # reuse
        buf297 = reinterpret_tensor(buf291, (128, 512), (512, 1), 0); del buf291  # reuse
        # Source Nodes: [add_52, hidden_states_102, l__mod___decoder_block_0_layer_0_self_attention_q, normed_hidden_states_8, pow_26, rsqrt_17, variance_17], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_1.run(buf296, buf293, primals_18, buf297, 128, 512, grid=grid(128), stream=stream0)
        buf298 = reinterpret_tensor(buf263, (128, 384), (384, 1), 0); del buf263  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, reinterpret_tensor(primals_101, (512, 384), (1, 512), 0), out=buf298)
        buf299 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, reinterpret_tensor(primals_102, (512, 384), (1, 512), 0), out=buf299)
        buf300 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, reinterpret_tensor(primals_103, (512, 384), (1, 512), 0), out=buf300)
        buf301 = buf256; del buf256  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf299, (6, 64, 128), (64, 1, 384), 0), out=buf301)
        buf302 = empty((128, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [float_10, full_like_1, is_small_1, log_1, min_2, mul_82, relative_position, relative_position_3, relative_position_bucket_1, relative_position_if_large_2, relative_position_if_large_3, to_24, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
        triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_7.run(buf302, 16384, grid=grid(16384), stream=stream0)
        buf306 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf301, primals_104, buf306, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_17, softmax_8], Original ATen: [aten._softmax, aten.native_dropout]
        buf307 = aten.native_dropout(buf306, 0.1, True)
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf308, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf300, (6, 128, 64), (64, 384, 1), 0), out=buf310)
        buf311 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_17], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf310, buf311, 49152, grid=grid(49152), stream=stream0)
        buf312 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf311, reinterpret_tensor(primals_105, (384, 512), (1, 384), 0), out=buf312)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf313 = aten.native_dropout(reinterpret_tensor(buf312, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf314; del buf314  # reuse
        buf317 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf318 = reinterpret_tensor(buf317, (1, 128, 1), (128, 1, 1), 0); del buf317  # reuse
        buf319 = buf312; del buf312  # reuse
        # Source Nodes: [add_57, hidden_states_106, hidden_states_107, l__mod___decoder_block_0_layer_1_enc_dec_attention_q, normed_hidden_states_9, pow_27, rsqrt_18, variance_18], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf316, buf318, buf293, primals_19, buf319, 128, 512, grid=grid(128), stream=stream0)
        buf320 = reinterpret_tensor(buf310, (128, 384), (384, 1), 0); del buf310  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf319, reinterpret_tensor(primals_106, (512, 384), (1, 512), 0), out=buf320)
        buf321 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 384), (1, 512), 0), out=buf321)
        buf322 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 384), (1, 512), 0), out=buf322)
        buf323 = buf301; del buf301  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf321, (6, 64, 128), (64, 1, 384), 0), out=buf323)
        buf326 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf323, buf326, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_19, softmax_9], Original ATen: [aten._softmax, aten.native_dropout]
        buf327 = aten.native_dropout(buf326, 0.1, True)
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf322, (6, 128, 64), (64, 384, 1), 0), out=buf330)
        buf331 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_19], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf330, buf331, 49152, grid=grid(49152), stream=stream0)
        buf332 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf331, reinterpret_tensor(primals_109, (384, 512), (1, 384), 0), out=buf332)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf333 = aten.native_dropout(reinterpret_tensor(buf332, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf334 = buf333[0]
        buf335 = buf333[1]
        del buf333
        buf336 = buf334; del buf334  # reuse
        buf337 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf338 = reinterpret_tensor(buf337, (1, 128, 1), (128, 1, 1), 0); del buf337  # reuse
        buf339 = buf332; del buf332  # reuse
        # Source Nodes: [add_60, forwarded_states_16, hidden_states_110, hidden_states_111, l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0, pow_28, rsqrt_19, variance_19], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf336, buf338, buf316, primals_20, buf339, 128, 512, grid=grid(128), stream=stream0)
        buf340 = reinterpret_tensor(buf276, (128, 1024), (1024, 1), 0); del buf276  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf339, reinterpret_tensor(primals_110, (512, 1024), (1, 512), 0), out=buf340)
        buf342 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf339, reinterpret_tensor(primals_111, (512, 1024), (1, 512), 0), out=buf342)
        buf341 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf343 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_61, add_62, hidden_gelu_8, hidden_states_112, mul_87, mul_88, mul_89, pow_29, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf340, buf342, buf341, buf343, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_62, hidden_gelu_8, hidden_states_112, hidden_states_113, mul_87], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf344 = aten.native_dropout(buf343, 0.1, True)
        buf345 = buf344[0]
        buf346 = buf344[1]
        del buf344
        buf347 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_112, (1024, 512), (1, 1024), 0), out=buf347)
        # Source Nodes: [l__mod___decoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf348 = aten.native_dropout(reinterpret_tensor(buf347, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf351 = buf349; del buf349  # reuse
        buf352 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf353 = reinterpret_tensor(buf352, (1, 128, 1), (128, 1, 1), 0); del buf352  # reuse
        buf354 = buf347; del buf347  # reuse
        # Source Nodes: [add_64, hidden_states_117, hidden_states_118, l__mod___decoder_block_1_layer_0_self_attention_q, normed_hidden_states_10, pow_30, rsqrt_20, variance_20], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf351, buf353, buf336, primals_21, buf354, 128, 512, grid=grid(128), stream=stream0)
        buf355 = reinterpret_tensor(buf330, (128, 384), (384, 1), 0); del buf330  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf354, reinterpret_tensor(primals_113, (512, 384), (1, 512), 0), out=buf355)
        buf356 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf354, reinterpret_tensor(primals_114, (512, 384), (1, 512), 0), out=buf356)
        buf357 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf354, reinterpret_tensor(primals_115, (512, 384), (1, 512), 0), out=buf357)
        buf358 = buf323; del buf323  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf356, (6, 64, 128), (64, 1, 384), 0), out=buf358)
        buf362 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf358, primals_104, buf362, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_21, softmax_10], Original ATen: [aten._softmax, aten.native_dropout]
        buf363 = aten.native_dropout(buf362, 0.1, True)
        buf364 = buf363[0]
        buf365 = buf363[1]
        del buf363
        buf366 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf357, (6, 128, 64), (64, 384, 1), 0), out=buf366)
        buf367 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_21], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf366, buf367, 49152, grid=grid(49152), stream=stream0)
        buf368 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf367, reinterpret_tensor(primals_116, (384, 512), (1, 384), 0), out=buf368)
        # Source Nodes: [l__mod___decoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf369 = aten.native_dropout(reinterpret_tensor(buf368, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf370 = buf369[0]
        buf371 = buf369[1]
        del buf369
        buf372 = buf370; del buf370  # reuse
        buf373 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf374 = reinterpret_tensor(buf373, (1, 128, 1), (128, 1, 1), 0); del buf373  # reuse
        buf375 = buf368; del buf368  # reuse
        # Source Nodes: [add_66, hidden_states_122, hidden_states_123, l__mod___decoder_block_1_layer_1_enc_dec_attention_q, normed_hidden_states_11, pow_31, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf372, buf374, buf351, primals_22, buf375, 128, 512, grid=grid(128), stream=stream0)
        buf376 = reinterpret_tensor(buf366, (128, 384), (384, 1), 0); del buf366  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf375, reinterpret_tensor(primals_117, (512, 384), (1, 512), 0), out=buf376)
        buf377 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 384), (1, 512), 0), out=buf377)
        buf378 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 384), (1, 512), 0), out=buf378)
        buf379 = buf358; del buf358  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf376, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf377, (6, 64, 128), (64, 1, 384), 0), out=buf379)
        buf382 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf379, buf382, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_23, softmax_11], Original ATen: [aten._softmax, aten.native_dropout]
        buf383 = aten.native_dropout(buf382, 0.1, True)
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf386 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf378, (6, 128, 64), (64, 384, 1), 0), out=buf386)
        buf387 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_23], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf386, buf387, 49152, grid=grid(49152), stream=stream0)
        buf388 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf387, reinterpret_tensor(primals_120, (384, 512), (1, 384), 0), out=buf388)
        # Source Nodes: [l__mod___decoder_block_1_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf389 = aten.native_dropout(reinterpret_tensor(buf388, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf392 = buf390; del buf390  # reuse
        buf393 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf394 = reinterpret_tensor(buf393, (1, 128, 1), (128, 1, 1), 0); del buf393  # reuse
        buf395 = buf388; del buf388  # reuse
        # Source Nodes: [add_68, forwarded_states_18, hidden_states_126, hidden_states_127, l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0, pow_32, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf392, buf394, buf372, primals_23, buf395, 128, 512, grid=grid(128), stream=stream0)
        buf396 = reinterpret_tensor(buf343, (128, 1024), (1024, 1), 0); del buf343  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf395, reinterpret_tensor(primals_121, (512, 1024), (1, 512), 0), out=buf396)
        buf398 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf395, reinterpret_tensor(primals_122, (512, 1024), (1, 512), 0), out=buf398)
        buf397 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf399 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_69, add_70, hidden_gelu_9, hidden_states_128, mul_100, mul_98, mul_99, pow_33, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf396, buf398, buf397, buf399, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_70, hidden_gelu_9, hidden_states_128, hidden_states_129, mul_98], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf400 = aten.native_dropout(buf399, 0.1, True)
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_123, (1024, 512), (1, 1024), 0), out=buf403)
        # Source Nodes: [l__mod___decoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf404 = aten.native_dropout(reinterpret_tensor(buf403, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf405 = buf404[0]
        buf406 = buf404[1]
        del buf404
        buf407 = buf405; del buf405  # reuse
        buf408 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf409 = reinterpret_tensor(buf408, (1, 128, 1), (128, 1, 1), 0); del buf408  # reuse
        buf410 = buf403; del buf403  # reuse
        # Source Nodes: [add_72, hidden_states_133, hidden_states_134, l__mod___decoder_block_2_layer_0_self_attention_q, normed_hidden_states_12, pow_34, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf407, buf409, buf392, primals_24, buf410, 128, 512, grid=grid(128), stream=stream0)
        buf411 = reinterpret_tensor(buf386, (128, 384), (384, 1), 0); del buf386  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf410, reinterpret_tensor(primals_124, (512, 384), (1, 512), 0), out=buf411)
        buf412 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf410, reinterpret_tensor(primals_125, (512, 384), (1, 512), 0), out=buf412)
        buf413 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf410, reinterpret_tensor(primals_126, (512, 384), (1, 512), 0), out=buf413)
        buf414 = buf379; del buf379  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf411, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf412, (6, 64, 128), (64, 1, 384), 0), out=buf414)
        buf418 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_12], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf414, primals_104, buf418, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_25, softmax_12], Original ATen: [aten._softmax, aten.native_dropout]
        buf419 = aten.native_dropout(buf418, 0.1, True)
        buf420 = buf419[0]
        buf421 = buf419[1]
        del buf419
        buf422 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf420, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf413, (6, 128, 64), (64, 384, 1), 0), out=buf422)
        buf423 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_25], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf422, buf423, 49152, grid=grid(49152), stream=stream0)
        buf424 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf423, reinterpret_tensor(primals_127, (384, 512), (1, 384), 0), out=buf424)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf425 = aten.native_dropout(reinterpret_tensor(buf424, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf426 = buf425[0]
        buf427 = buf425[1]
        del buf425
        buf428 = buf426; del buf426  # reuse
        buf429 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf430 = reinterpret_tensor(buf429, (1, 128, 1), (128, 1, 1), 0); del buf429  # reuse
        buf431 = buf424; del buf424  # reuse
        # Source Nodes: [add_74, hidden_states_138, hidden_states_139, l__mod___decoder_block_2_layer_1_enc_dec_attention_q, normed_hidden_states_13, pow_35, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf428, buf430, buf407, primals_25, buf431, 128, 512, grid=grid(128), stream=stream0)
        buf432 = reinterpret_tensor(buf422, (128, 384), (384, 1), 0); del buf422  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf431, reinterpret_tensor(primals_128, (512, 384), (1, 512), 0), out=buf432)
        buf433 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 384), (1, 512), 0), out=buf433)
        buf434 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_130, (512, 384), (1, 512), 0), out=buf434)
        buf435 = buf414; del buf414  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf433, (6, 64, 128), (64, 1, 384), 0), out=buf435)
        buf438 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf435, buf438, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_27, softmax_13], Original ATen: [aten._softmax, aten.native_dropout]
        buf439 = aten.native_dropout(buf438, 0.1, True)
        buf440 = buf439[0]
        buf441 = buf439[1]
        del buf439
        buf442 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf440, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf434, (6, 128, 64), (64, 384, 1), 0), out=buf442)
        buf443 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_27], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf442, buf443, 49152, grid=grid(49152), stream=stream0)
        buf444 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf443, reinterpret_tensor(primals_131, (384, 512), (1, 384), 0), out=buf444)
        # Source Nodes: [l__mod___decoder_block_2_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf445 = aten.native_dropout(reinterpret_tensor(buf444, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf446 = buf445[0]
        buf447 = buf445[1]
        del buf445
        buf448 = buf446; del buf446  # reuse
        buf449 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf450 = reinterpret_tensor(buf449, (1, 128, 1), (128, 1, 1), 0); del buf449  # reuse
        buf451 = buf444; del buf444  # reuse
        # Source Nodes: [add_76, forwarded_states_20, hidden_states_142, hidden_states_143, l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0, pow_36, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf448, buf450, buf428, primals_26, buf451, 128, 512, grid=grid(128), stream=stream0)
        buf452 = reinterpret_tensor(buf399, (128, 1024), (1024, 1), 0); del buf399  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf451, reinterpret_tensor(primals_132, (512, 1024), (1, 512), 0), out=buf452)
        buf454 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf451, reinterpret_tensor(primals_133, (512, 1024), (1, 512), 0), out=buf454)
        buf453 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf455 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_77, add_78, hidden_gelu_10, hidden_states_144, mul_109, mul_110, mul_111, pow_37, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf452, buf454, buf453, buf455, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_78, hidden_gelu_10, hidden_states_144, hidden_states_145, mul_109], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf456 = aten.native_dropout(buf455, 0.1, True)
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_134, (1024, 512), (1, 1024), 0), out=buf459)
        # Source Nodes: [l__mod___decoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf460 = aten.native_dropout(reinterpret_tensor(buf459, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf461 = buf460[0]
        buf462 = buf460[1]
        del buf460
        buf463 = buf461; del buf461  # reuse
        buf464 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf465 = reinterpret_tensor(buf464, (1, 128, 1), (128, 1, 1), 0); del buf464  # reuse
        buf466 = buf459; del buf459  # reuse
        # Source Nodes: [add_80, hidden_states_149, hidden_states_150, l__mod___decoder_block_3_layer_0_self_attention_q, normed_hidden_states_14, pow_38, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf463, buf465, buf448, primals_27, buf466, 128, 512, grid=grid(128), stream=stream0)
        buf467 = reinterpret_tensor(buf442, (128, 384), (384, 1), 0); del buf442  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf466, reinterpret_tensor(primals_135, (512, 384), (1, 512), 0), out=buf467)
        buf468 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf466, reinterpret_tensor(primals_136, (512, 384), (1, 512), 0), out=buf468)
        buf469 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf466, reinterpret_tensor(primals_137, (512, 384), (1, 512), 0), out=buf469)
        buf470 = buf435; del buf435  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf468, (6, 64, 128), (64, 1, 384), 0), out=buf470)
        buf474 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf470, primals_104, buf474, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_29, softmax_14], Original ATen: [aten._softmax, aten.native_dropout]
        buf475 = aten.native_dropout(buf474, 0.1, True)
        buf476 = buf475[0]
        buf477 = buf475[1]
        del buf475
        buf478 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf476, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf469, (6, 128, 64), (64, 384, 1), 0), out=buf478)
        buf479 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_29], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf478, buf479, 49152, grid=grid(49152), stream=stream0)
        buf480 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf479, reinterpret_tensor(primals_138, (384, 512), (1, 384), 0), out=buf480)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf481 = aten.native_dropout(reinterpret_tensor(buf480, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf482; del buf482  # reuse
        buf485 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf486 = reinterpret_tensor(buf485, (1, 128, 1), (128, 1, 1), 0); del buf485  # reuse
        buf487 = buf480; del buf480  # reuse
        # Source Nodes: [add_82, hidden_states_154, hidden_states_155, l__mod___decoder_block_3_layer_1_enc_dec_attention_q, normed_hidden_states_15, pow_39, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf484, buf486, buf463, primals_28, buf487, 128, 512, grid=grid(128), stream=stream0)
        buf488 = reinterpret_tensor(buf478, (128, 384), (384, 1), 0); del buf478  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf487, reinterpret_tensor(primals_139, (512, 384), (1, 512), 0), out=buf488)
        buf489 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_140, (512, 384), (1, 512), 0), out=buf489)
        buf490 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_141, (512, 384), (1, 512), 0), out=buf490)
        buf491 = buf470; del buf470  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf488, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf489, (6, 64, 128), (64, 1, 384), 0), out=buf491)
        buf494 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf491, buf494, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_31, softmax_15], Original ATen: [aten._softmax, aten.native_dropout]
        buf495 = aten.native_dropout(buf494, 0.1, True)
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf498 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf490, (6, 128, 64), (64, 384, 1), 0), out=buf498)
        buf499 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_31], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf498, buf499, 49152, grid=grid(49152), stream=stream0)
        buf500 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf499, reinterpret_tensor(primals_142, (384, 512), (1, 384), 0), out=buf500)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf501 = aten.native_dropout(reinterpret_tensor(buf500, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf502 = buf501[0]
        buf503 = buf501[1]
        del buf501
        buf504 = buf502; del buf502  # reuse
        buf505 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf506 = reinterpret_tensor(buf505, (1, 128, 1), (128, 1, 1), 0); del buf505  # reuse
        buf507 = buf500; del buf500  # reuse
        # Source Nodes: [add_84, forwarded_states_22, hidden_states_158, hidden_states_159, l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0, pow_40, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf504, buf506, buf484, primals_29, buf507, 128, 512, grid=grid(128), stream=stream0)
        buf508 = reinterpret_tensor(buf455, (128, 1024), (1024, 1), 0); del buf455  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf507, reinterpret_tensor(primals_143, (512, 1024), (1, 512), 0), out=buf508)
        buf510 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf507, reinterpret_tensor(primals_144, (512, 1024), (1, 512), 0), out=buf510)
        buf509 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf511 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_85, add_86, hidden_gelu_11, hidden_states_160, mul_120, mul_121, mul_122, pow_41, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf508, buf510, buf509, buf511, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_86, hidden_gelu_11, hidden_states_160, hidden_states_161, mul_120], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf512 = aten.native_dropout(buf511, 0.1, True)
        buf513 = buf512[0]
        buf514 = buf512[1]
        del buf512
        buf515 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_145, (1024, 512), (1, 1024), 0), out=buf515)
        # Source Nodes: [l__mod___decoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf516 = aten.native_dropout(reinterpret_tensor(buf515, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf517 = buf516[0]
        buf518 = buf516[1]
        del buf516
        buf519 = buf517; del buf517  # reuse
        buf520 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf521 = reinterpret_tensor(buf520, (1, 128, 1), (128, 1, 1), 0); del buf520  # reuse
        buf522 = buf515; del buf515  # reuse
        # Source Nodes: [add_88, hidden_states_165, hidden_states_166, l__mod___decoder_block_4_layer_0_self_attention_q, normed_hidden_states_16, pow_42, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf519, buf521, buf504, primals_30, buf522, 128, 512, grid=grid(128), stream=stream0)
        buf523 = reinterpret_tensor(buf498, (128, 384), (384, 1), 0); del buf498  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf522, reinterpret_tensor(primals_146, (512, 384), (1, 512), 0), out=buf523)
        buf524 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf522, reinterpret_tensor(primals_147, (512, 384), (1, 512), 0), out=buf524)
        buf525 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf522, reinterpret_tensor(primals_148, (512, 384), (1, 512), 0), out=buf525)
        buf526 = buf491; del buf491  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf523, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf524, (6, 64, 128), (64, 1, 384), 0), out=buf526)
        buf530 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_16], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf526, primals_104, buf530, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_33, softmax_16], Original ATen: [aten._softmax, aten.native_dropout]
        buf531 = aten.native_dropout(buf530, 0.1, True)
        buf532 = buf531[0]
        buf533 = buf531[1]
        del buf531
        buf534 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf525, (6, 128, 64), (64, 384, 1), 0), out=buf534)
        buf535 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_33], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf534, buf535, 49152, grid=grid(49152), stream=stream0)
        buf536 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf535, reinterpret_tensor(primals_149, (384, 512), (1, 384), 0), out=buf536)
        # Source Nodes: [l__mod___decoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf537 = aten.native_dropout(reinterpret_tensor(buf536, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf538 = buf537[0]
        buf539 = buf537[1]
        del buf537
        buf540 = buf538; del buf538  # reuse
        buf541 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf542 = reinterpret_tensor(buf541, (1, 128, 1), (128, 1, 1), 0); del buf541  # reuse
        buf543 = buf536; del buf536  # reuse
        # Source Nodes: [add_90, hidden_states_170, hidden_states_171, l__mod___decoder_block_4_layer_1_enc_dec_attention_q, normed_hidden_states_17, pow_43, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf540, buf542, buf519, primals_31, buf543, 128, 512, grid=grid(128), stream=stream0)
        buf544 = reinterpret_tensor(buf534, (128, 384), (384, 1), 0); del buf534  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf543, reinterpret_tensor(primals_150, (512, 384), (1, 512), 0), out=buf544)
        buf545 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_151, (512, 384), (1, 512), 0), out=buf545)
        buf546 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_152, (512, 384), (1, 512), 0), out=buf546)
        buf547 = buf526; del buf526  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf545, (6, 64, 128), (64, 1, 384), 0), out=buf547)
        buf550 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf547, buf550, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_35, softmax_17], Original ATen: [aten._softmax, aten.native_dropout]
        buf551 = aten.native_dropout(buf550, 0.1, True)
        buf552 = buf551[0]
        buf553 = buf551[1]
        del buf551
        buf554 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf546, (6, 128, 64), (64, 384, 1), 0), out=buf554)
        buf555 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_35], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf554, buf555, 49152, grid=grid(49152), stream=stream0)
        buf556 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(buf555, reinterpret_tensor(primals_153, (384, 512), (1, 384), 0), out=buf556)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf557 = aten.native_dropout(reinterpret_tensor(buf556, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf558 = buf557[0]
        buf559 = buf557[1]
        del buf557
        buf560 = buf558; del buf558  # reuse
        buf561 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf562 = reinterpret_tensor(buf561, (1, 128, 1), (128, 1, 1), 0); del buf561  # reuse
        buf563 = buf556; del buf556  # reuse
        # Source Nodes: [add_92, forwarded_states_24, hidden_states_174, hidden_states_175, l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0, pow_44, rsqrt_31, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf560, buf562, buf540, primals_32, buf563, 128, 512, grid=grid(128), stream=stream0)
        buf564 = reinterpret_tensor(buf511, (128, 1024), (1024, 1), 0); del buf511  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf563, reinterpret_tensor(primals_154, (512, 1024), (1, 512), 0), out=buf564)
        buf566 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf563, reinterpret_tensor(primals_155, (512, 1024), (1, 512), 0), out=buf566)
        buf565 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf567 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_93, add_94, hidden_gelu_12, hidden_states_176, mul_131, mul_132, mul_133, pow_45, tanh_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf564, buf566, buf565, buf567, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_94, hidden_gelu_12, hidden_states_176, hidden_states_177, mul_131], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf568 = aten.native_dropout(buf567, 0.1, True)
        buf569 = buf568[0]
        buf570 = buf568[1]
        del buf568
        buf571 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_156, (1024, 512), (1, 1024), 0), out=buf571)
        # Source Nodes: [l__mod___decoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf572 = aten.native_dropout(reinterpret_tensor(buf571, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf573 = buf572[0]
        buf574 = buf572[1]
        del buf572
        buf575 = buf573; del buf573  # reuse
        buf576 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf577 = reinterpret_tensor(buf576, (1, 128, 1), (128, 1, 1), 0); del buf576  # reuse
        buf578 = buf571; del buf571  # reuse
        # Source Nodes: [add_96, hidden_states_181, hidden_states_182, l__mod___decoder_block_5_layer_0_self_attention_q, normed_hidden_states_18, pow_46, rsqrt_32, variance_32], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf575, buf577, buf560, primals_33, buf578, 128, 512, grid=grid(128), stream=stream0)
        buf579 = reinterpret_tensor(buf554, (128, 384), (384, 1), 0); del buf554  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf578, reinterpret_tensor(primals_157, (512, 384), (1, 512), 0), out=buf579)
        buf580 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf578, reinterpret_tensor(primals_158, (512, 384), (1, 512), 0), out=buf580)
        buf581 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf578, reinterpret_tensor(primals_159, (512, 384), (1, 512), 0), out=buf581)
        buf582 = buf547; del buf547  # reuse
        # Source Nodes: [scores_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf579, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf580, (6, 64, 128), (64, 1, 384), 0), out=buf582)
        buf586 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_18], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf582, primals_104, buf586, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_37, softmax_18], Original ATen: [aten._softmax, aten.native_dropout]
        buf587 = aten.native_dropout(buf586, 0.1, True)
        buf588 = buf587[0]
        buf589 = buf587[1]
        del buf587
        buf590 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf581, (6, 128, 64), (64, 384, 1), 0), out=buf590)
        buf591 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_37], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf590, buf591, 49152, grid=grid(49152), stream=stream0)
        buf592 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_37], Original ATen: [aten.mm]
        extern_kernels.mm(buf591, reinterpret_tensor(primals_160, (384, 512), (1, 384), 0), out=buf592)
        # Source Nodes: [l__mod___decoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf593 = aten.native_dropout(reinterpret_tensor(buf592, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf594 = buf593[0]
        buf595 = buf593[1]
        del buf593
        buf596 = buf594; del buf594  # reuse
        buf597 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf598 = reinterpret_tensor(buf597, (1, 128, 1), (128, 1, 1), 0); del buf597  # reuse
        buf599 = buf592; del buf592  # reuse
        # Source Nodes: [add_98, hidden_states_186, hidden_states_187, l__mod___decoder_block_5_layer_1_enc_dec_attention_q, normed_hidden_states_19, pow_47, rsqrt_33, variance_33], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf596, buf598, buf575, primals_34, buf599, 128, 512, grid=grid(128), stream=stream0)
        buf600 = reinterpret_tensor(buf590, (128, 384), (384, 1), 0); del buf590  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf599, reinterpret_tensor(primals_161, (512, 384), (1, 512), 0), out=buf600)
        buf601 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_162, (512, 384), (1, 512), 0), out=buf601)
        buf602 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_163, (512, 384), (1, 512), 0), out=buf602)
        buf603 = buf582; del buf582  # reuse
        # Source Nodes: [scores_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf600, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf601, (6, 64, 128), (64, 1, 384), 0), out=buf603)
        buf606 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf603, buf606, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_39, softmax_19], Original ATen: [aten._softmax, aten.native_dropout]
        buf607 = aten.native_dropout(buf606, 0.1, True)
        buf608 = buf607[0]
        buf609 = buf607[1]
        del buf607
        buf610 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf608, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf602, (6, 128, 64), (64, 384, 1), 0), out=buf610)
        buf611 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_39], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf610, buf611, 49152, grid=grid(49152), stream=stream0)
        buf612 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_39], Original ATen: [aten.mm]
        extern_kernels.mm(buf611, reinterpret_tensor(primals_164, (384, 512), (1, 384), 0), out=buf612)
        # Source Nodes: [l__mod___decoder_block_5_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf613 = aten.native_dropout(reinterpret_tensor(buf612, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf614 = buf613[0]
        buf615 = buf613[1]
        del buf613
        buf616 = buf614; del buf614  # reuse
        buf617 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf618 = reinterpret_tensor(buf617, (1, 128, 1), (128, 1, 1), 0); del buf617  # reuse
        buf619 = buf612; del buf612  # reuse
        # Source Nodes: [add_100, forwarded_states_26, hidden_states_190, hidden_states_191, l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0, pow_48, rsqrt_34, variance_34], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf616, buf618, buf596, primals_35, buf619, 128, 512, grid=grid(128), stream=stream0)
        buf620 = reinterpret_tensor(buf567, (128, 1024), (1024, 1), 0); del buf567  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf619, reinterpret_tensor(primals_165, (512, 1024), (1, 512), 0), out=buf620)
        buf622 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf619, reinterpret_tensor(primals_166, (512, 1024), (1, 512), 0), out=buf622)
        buf621 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf623 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_101, add_102, hidden_gelu_13, hidden_states_192, mul_142, mul_143, mul_144, pow_49, tanh_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf620, buf622, buf621, buf623, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_102, hidden_gelu_13, hidden_states_192, hidden_states_193, mul_142], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf624 = aten.native_dropout(buf623, 0.1, True)
        buf625 = buf624[0]
        buf626 = buf624[1]
        del buf624
        buf627 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_167, (1024, 512), (1, 1024), 0), out=buf627)
        # Source Nodes: [l__mod___decoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf628 = aten.native_dropout(reinterpret_tensor(buf627, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf629 = buf628[0]
        buf630 = buf628[1]
        del buf628
        buf631 = buf629; del buf629  # reuse
        buf632 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf633 = reinterpret_tensor(buf632, (1, 128, 1), (128, 1, 1), 0); del buf632  # reuse
        buf634 = buf627; del buf627  # reuse
        # Source Nodes: [add_104, hidden_states_197, hidden_states_198, l__mod___decoder_block_6_layer_0_self_attention_q, normed_hidden_states_20, pow_50, rsqrt_35, variance_35], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf631, buf633, buf616, primals_36, buf634, 128, 512, grid=grid(128), stream=stream0)
        buf635 = reinterpret_tensor(buf610, (128, 384), (384, 1), 0); del buf610  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf634, reinterpret_tensor(primals_168, (512, 384), (1, 512), 0), out=buf635)
        buf636 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf634, reinterpret_tensor(primals_169, (512, 384), (1, 512), 0), out=buf636)
        buf637 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf634, reinterpret_tensor(primals_170, (512, 384), (1, 512), 0), out=buf637)
        buf638 = buf603; del buf603  # reuse
        # Source Nodes: [scores_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf635, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf636, (6, 64, 128), (64, 1, 384), 0), out=buf638)
        buf642 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_20], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf638, primals_104, buf642, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_41, softmax_20], Original ATen: [aten._softmax, aten.native_dropout]
        buf643 = aten.native_dropout(buf642, 0.1, True)
        buf644 = buf643[0]
        buf645 = buf643[1]
        del buf643
        buf646 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf644, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf637, (6, 128, 64), (64, 384, 1), 0), out=buf646)
        buf647 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_41], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf646, buf647, 49152, grid=grid(49152), stream=stream0)
        buf648 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf647, reinterpret_tensor(primals_171, (384, 512), (1, 384), 0), out=buf648)
        # Source Nodes: [l__mod___decoder_block_6_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf649 = aten.native_dropout(reinterpret_tensor(buf648, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf650 = buf649[0]
        buf651 = buf649[1]
        del buf649
        buf652 = buf650; del buf650  # reuse
        buf653 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf654 = reinterpret_tensor(buf653, (1, 128, 1), (128, 1, 1), 0); del buf653  # reuse
        buf655 = buf648; del buf648  # reuse
        # Source Nodes: [add_106, hidden_states_202, hidden_states_203, l__mod___decoder_block_6_layer_1_enc_dec_attention_q, normed_hidden_states_21, pow_51, rsqrt_36, variance_36], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf652, buf654, buf631, primals_37, buf655, 128, 512, grid=grid(128), stream=stream0)
        buf656 = reinterpret_tensor(buf646, (128, 384), (384, 1), 0); del buf646  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf655, reinterpret_tensor(primals_172, (512, 384), (1, 512), 0), out=buf656)
        buf657 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_173, (512, 384), (1, 512), 0), out=buf657)
        buf658 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_174, (512, 384), (1, 512), 0), out=buf658)
        buf659 = buf638; del buf638  # reuse
        # Source Nodes: [scores_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf656, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf657, (6, 64, 128), (64, 1, 384), 0), out=buf659)
        buf662 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf659, buf662, 768, 128, grid=grid(768), stream=stream0)
        # Source Nodes: [attn_weights_43, softmax_21], Original ATen: [aten._softmax, aten.native_dropout]
        buf663 = aten.native_dropout(buf662, 0.1, True)
        buf664 = buf663[0]
        buf665 = buf663[1]
        del buf663
        buf666 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf664, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf658, (6, 128, 64), (64, 384, 1), 0), out=buf666)
        buf667 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_43], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf666, buf667, 49152, grid=grid(49152), stream=stream0)
        buf668 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_43], Original ATen: [aten.mm]
        extern_kernels.mm(buf667, reinterpret_tensor(primals_175, (384, 512), (1, 384), 0), out=buf668)
        # Source Nodes: [l__mod___decoder_block_6_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf669 = aten.native_dropout(reinterpret_tensor(buf668, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf670 = buf669[0]
        buf671 = buf669[1]
        del buf669
        buf672 = buf670; del buf670  # reuse
        buf673 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf674 = reinterpret_tensor(buf673, (1, 128, 1), (128, 1, 1), 0); del buf673  # reuse
        buf675 = buf668; del buf668  # reuse
        # Source Nodes: [add_108, forwarded_states_28, hidden_states_206, hidden_states_207, l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0, pow_52, rsqrt_37, variance_37], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf672, buf674, buf652, primals_38, buf675, 128, 512, grid=grid(128), stream=stream0)
        buf676 = reinterpret_tensor(buf623, (128, 1024), (1024, 1), 0); del buf623  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf675, reinterpret_tensor(primals_176, (512, 1024), (1, 512), 0), out=buf676)
        buf678 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf675, reinterpret_tensor(primals_177, (512, 1024), (1, 512), 0), out=buf678)
        buf677 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf679 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_109, add_110, hidden_gelu_14, hidden_states_208, mul_153, mul_154, mul_155, pow_53, tanh_14], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf676, buf678, buf677, buf679, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_110, hidden_gelu_14, hidden_states_208, hidden_states_209, mul_153], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf680 = aten.native_dropout(buf679, 0.1, True)
        buf681 = buf680[0]
        buf682 = buf680[1]
        del buf680
        buf683 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf681, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_178, (1024, 512), (1, 1024), 0), out=buf683)
        # Source Nodes: [l__mod___decoder_block_6_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf684 = aten.native_dropout(reinterpret_tensor(buf683, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf685 = buf684[0]
        buf686 = buf684[1]
        del buf684
        buf687 = buf685; del buf685  # reuse
        buf688 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf689 = reinterpret_tensor(buf688, (1, 128, 1), (128, 1, 1), 0); del buf688  # reuse
        buf690 = buf683; del buf683  # reuse
        # Source Nodes: [add_112, hidden_states_213, hidden_states_214, l__mod___decoder_block_7_layer_0_self_attention_q, normed_hidden_states_22, pow_54, rsqrt_38, variance_38], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf687, buf689, buf672, primals_39, buf690, 128, 512, grid=grid(128), stream=stream0)
        buf691 = reinterpret_tensor(buf666, (128, 384), (384, 1), 0); del buf666  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf690, reinterpret_tensor(primals_179, (512, 384), (1, 512), 0), out=buf691)
        buf692 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(buf690, reinterpret_tensor(primals_180, (512, 384), (1, 512), 0), out=buf692)
        buf693 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf690, reinterpret_tensor(primals_181, (512, 384), (1, 512), 0), out=buf693)
        buf694 = buf659; del buf659  # reuse
        # Source Nodes: [scores_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf691, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf692, (6, 64, 128), (64, 1, 384), 0), out=buf694)
        buf698 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_22], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf694, primals_104, buf698, 768, 128, grid=grid(768), stream=stream0)
        del primals_104
        # Source Nodes: [attn_weights_45, softmax_22], Original ATen: [aten._softmax, aten.native_dropout]
        buf699 = aten.native_dropout(buf698, 0.1, True)
        buf700 = buf699[0]
        buf701 = buf699[1]
        del buf699
        buf702 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf700, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf693, (6, 128, 64), (64, 384, 1), 0), out=buf702)
        buf703 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_45], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf702, buf703, 49152, grid=grid(49152), stream=stream0)
        buf704 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_45], Original ATen: [aten.mm]
        extern_kernels.mm(buf703, reinterpret_tensor(primals_182, (384, 512), (1, 384), 0), out=buf704)
        # Source Nodes: [l__mod___decoder_block_7_layer_0_dropout], Original ATen: [aten.native_dropout]
        buf705 = aten.native_dropout(reinterpret_tensor(buf704, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf706 = buf705[0]
        buf707 = buf705[1]
        del buf705
        buf708 = buf706; del buf706  # reuse
        buf709 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf710 = reinterpret_tensor(buf709, (1, 128, 1), (128, 1, 1), 0); del buf709  # reuse
        buf711 = buf704; del buf704  # reuse
        # Source Nodes: [add_114, hidden_states_218, hidden_states_219, l__mod___decoder_block_7_layer_1_enc_dec_attention_q, normed_hidden_states_23, pow_55, rsqrt_39, variance_39], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf708, buf710, buf687, primals_40, buf711, 128, 512, grid=grid(128), stream=stream0)
        buf712 = reinterpret_tensor(buf702, (128, 384), (384, 1), 0); del buf702  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(buf711, reinterpret_tensor(primals_183, (512, 384), (1, 512), 0), out=buf712)
        buf713 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_184, (512, 384), (1, 512), 0), out=buf713)
        buf714 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 512), (512, 1), 0), reinterpret_tensor(primals_185, (512, 384), (1, 512), 0), out=buf714)
        buf715 = buf694; del buf694  # reuse
        # Source Nodes: [scores_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf712, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf713, (6, 64, 128), (64, 1, 384), 0), out=buf715)
        buf718 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf715, buf718, 768, 128, grid=grid(768), stream=stream0)
        del buf715
        # Source Nodes: [attn_weights_47, softmax_23], Original ATen: [aten._softmax, aten.native_dropout]
        buf719 = aten.native_dropout(buf718, 0.1, True)
        buf720 = buf719[0]
        buf721 = buf719[1]
        del buf719
        buf722 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf720, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf714, (6, 128, 64), (64, 384, 1), 0), out=buf722)
        buf723 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_47], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf722, buf723, 49152, grid=grid(49152), stream=stream0)
        del buf722
        buf724 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_47], Original ATen: [aten.mm]
        extern_kernels.mm(buf723, reinterpret_tensor(primals_186, (384, 512), (1, 384), 0), out=buf724)
        # Source Nodes: [l__mod___decoder_block_7_layer_1_dropout], Original ATen: [aten.native_dropout]
        buf725 = aten.native_dropout(reinterpret_tensor(buf724, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf726 = buf725[0]
        buf727 = buf725[1]
        del buf725
        buf728 = buf726; del buf726  # reuse
        buf729 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf730 = reinterpret_tensor(buf729, (1, 128, 1), (128, 1, 1), 0); del buf729  # reuse
        buf731 = buf724; del buf724  # reuse
        # Source Nodes: [add_116, forwarded_states_30, hidden_states_222, hidden_states_223, l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0, pow_56, rsqrt_40, variance_40], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf728, buf730, buf708, primals_41, buf731, 128, 512, grid=grid(128), stream=stream0)
        buf732 = reinterpret_tensor(buf679, (128, 1024), (1024, 1), 0); del buf679  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf731, reinterpret_tensor(primals_187, (512, 1024), (1, 512), 0), out=buf732)
        buf734 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf731, reinterpret_tensor(primals_188, (512, 1024), (1, 512), 0), out=buf734)
        buf733 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf735 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_117, add_118, hidden_gelu_15, hidden_states_224, mul_164, mul_165, mul_166, pow_57, tanh_15], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf732, buf734, buf733, buf735, 131072, grid=grid(131072), stream=stream0)
        # Source Nodes: [add_118, hidden_gelu_15, hidden_states_224, hidden_states_225, mul_164], Original ATen: [aten.add, aten.mul, aten.native_dropout]
        buf736 = aten.native_dropout(buf735, 0.1, True)
        del buf735
        buf737 = buf736[0]
        buf738 = buf736[1]
        del buf736
        buf739 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [forwarded_states_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_189, (1024, 512), (1, 1024), 0), out=buf739)
        # Source Nodes: [l__mod___decoder_block_7_layer__1__dropout], Original ATen: [aten.native_dropout]
        buf740 = aten.native_dropout(reinterpret_tensor(buf739, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
        buf741 = buf740[0]
        buf742 = buf740[1]
        del buf740
        buf743 = buf741; del buf741  # reuse
        buf744 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf745 = reinterpret_tensor(buf744, (1, 128, 1), (128, 1, 1), 0); del buf744  # reuse
        buf746 = reinterpret_tensor(buf739, (1, 128, 512), (65536, 512, 1), 0); del buf739  # reuse
        # Source Nodes: [add_120, hidden_states_229, hidden_states_230, hidden_states_231, pow_58, rsqrt_41, variance_41], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_5.run(buf743, buf745, buf728, primals_42, buf746, 128, 512, grid=grid(128), stream=stream0)
        # Source Nodes: [hidden_states_230, hidden_states_231, sequence_output], Original ATen: [aten.mul, aten.native_dropout]
        buf747 = aten.native_dropout(buf746, 0.1, True)
        del buf746
        buf748 = buf747[0]
        buf749 = buf747[1]
        del buf747
        buf750 = empty((128, 250112), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf748, (128, 512), (512, 1), 0), reinterpret_tensor(primals_190, (512, 250112), (1, 512), 0), out=buf750)
        buf751 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf750, buf751, 512, 62528, grid=grid(512), stream=stream0)
        buf752 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_11.run(buf751, buf752, 128, 4, grid=grid(128), stream=stream0)
        buf753 = buf751; del buf751  # reuse
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_12.run(buf750, buf752, buf753, 512, 62528, grid=grid(512), stream=stream0)
        buf754 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_13.run(buf753, buf754, 128, 4, grid=grid(128), stream=stream0)
        del buf753
        buf755 = empty((128, 250112), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_poi_fused__log_softmax_14.run(buf750, buf752, buf754, buf755, 32014336, grid=grid(32014336), stream=stream0)
        del buf752
        del buf754
        buf758 = empty((), device='cuda', dtype=torch.float32)
        buf757 = empty((), device='cuda', dtype=torch.float32)
        buf759 = buf758; del buf758  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_15.run(buf759, primals_192, buf755, buf757, 1, 128, grid=grid(1), stream=stream0)
        return (buf759, reinterpret_tensor(buf750, (1, 128, 250112), (32014336, 250112, 1), 0), reinterpret_tensor(buf299, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf300, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf321, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf322, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf356, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf357, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf377, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf378, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf412, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf413, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf433, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf434, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf468, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf469, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf489, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf490, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf524, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf525, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf545, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf546, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf580, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf581, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf601, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf602, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf636, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf637, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf657, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf658, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf692, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf693, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf713, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf714, (1, 6, 128, 64), (49152, 64, 384, 1), 0), buf289, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, primals_191, buf2, buf3, buf5, buf6, buf11, buf17, buf19, buf23, buf24, buf26, buf27, buf28, buf29, buf30, buf34, reinterpret_tensor(buf33, (128, 1024), (1024, 1), 0), buf38, buf39, buf41, buf42, buf52, buf54, buf58, buf59, buf61, buf62, buf63, buf64, buf65, buf69, reinterpret_tensor(buf68, (128, 1024), (1024, 1), 0), buf73, buf74, buf76, buf77, buf87, buf89, buf93, buf94, buf96, buf97, buf98, buf99, buf100, buf104, reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), buf108, buf109, buf111, buf112, buf122, buf124, buf128, buf129, buf131, buf132, buf133, buf134, buf135, buf139, reinterpret_tensor(buf138, (128, 1024), (1024, 1), 0), buf143, buf144, buf146, buf147, buf157, buf159, buf163, buf164, buf166, buf167, buf168, buf169, buf170, buf174, reinterpret_tensor(buf173, (128, 1024), (1024, 1), 0), buf178, buf179, buf181, buf182, buf192, buf194, buf198, buf199, buf201, buf202, buf203, buf204, buf205, buf209, reinterpret_tensor(buf208, (128, 1024), (1024, 1), 0), buf213, buf214, buf216, buf217, buf227, buf229, buf233, buf234, buf236, buf237, buf238, buf239, buf240, buf244, reinterpret_tensor(buf243, (128, 1024), (1024, 1), 0), buf248, buf249, buf251, buf252, buf262, buf264, buf268, buf269, buf271, buf272, buf273, buf274, buf275, buf279, reinterpret_tensor(buf278, (128, 1024), (1024, 1), 0), buf283, buf284, buf286, buf290, primals_193, buf293, buf294, buf296, buf297, buf302, buf309, buf311, buf315, buf316, buf318, buf319, reinterpret_tensor(buf289, (128, 512), (512, 1), 0), buf329, buf331, buf335, buf336, buf338, buf339, buf340, buf341, buf342, buf346, reinterpret_tensor(buf345, (128, 1024), (1024, 1), 0), buf350, buf351, buf353, buf354, buf365, buf367, buf371, buf372, buf374, buf375, buf385, buf387, buf391, buf392, buf394, buf395, buf396, buf397, buf398, buf402, reinterpret_tensor(buf401, (128, 1024), (1024, 1), 0), buf406, buf407, buf409, buf410, buf421, buf423, buf427, buf428, buf430, buf431, buf441, buf443, buf447, buf448, buf450, buf451, buf452, buf453, buf454, buf458, reinterpret_tensor(buf457, (128, 1024), (1024, 1), 0), buf462, buf463, buf465, buf466, buf477, buf479, buf483, buf484, buf486, buf487, buf497, buf499, buf503, buf504, buf506, buf507, buf508, buf509, buf510, buf514, reinterpret_tensor(buf513, (128, 1024), (1024, 1), 0), buf518, buf519, buf521, buf522, buf533, buf535, buf539, buf540, buf542, buf543, buf553, buf555, buf559, buf560, buf562, buf563, buf564, buf565, buf566, buf570, reinterpret_tensor(buf569, (128, 1024), (1024, 1), 0), buf574, buf575, buf577, buf578, buf589, buf591, buf595, buf596, buf598, buf599, buf609, buf611, buf615, buf616, buf618, buf619, buf620, buf621, buf622, buf626, reinterpret_tensor(buf625, (128, 1024), (1024, 1), 0), buf630, buf631, buf633, buf634, buf645, buf647, buf651, buf652, buf654, buf655, buf665, buf667, buf671, buf672, buf674, buf675, buf676, buf677, buf678, buf682, reinterpret_tensor(buf681, (128, 1024), (1024, 1), 0), buf686, buf687, buf689, buf690, buf701, buf703, buf707, buf708, buf710, buf711, buf721, buf723, buf727, buf728, buf730, buf731, buf732, buf733, buf734, buf738, reinterpret_tensor(buf737, (128, 1024), (1024, 1), 0), buf742, buf743, buf745, buf749, reinterpret_tensor(buf748, (128, 512), (512, 1), 0), buf755, buf757, reinterpret_tensor(primals_190, (250112, 512), (512, 1), 0), reinterpret_tensor(primals_189, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_188, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_187, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_186, (512, 384), (384, 1), 0), reinterpret_tensor(buf720, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf714, (6, 64, 128), (64, 1, 384), 0), buf718, reinterpret_tensor(buf712, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf713, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_185, (384, 512), (512, 1), 0), reinterpret_tensor(primals_184, (384, 512), (512, 1), 0), reinterpret_tensor(primals_183, (384, 512), (512, 1), 0), reinterpret_tensor(primals_182, (512, 384), (384, 1), 0), reinterpret_tensor(buf700, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf693, (6, 64, 128), (64, 1, 384), 0), buf698, reinterpret_tensor(buf691, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf692, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_181, (384, 512), (512, 1), 0), reinterpret_tensor(primals_180, (384, 512), (512, 1), 0), reinterpret_tensor(primals_179, (384, 512), (512, 1), 0), reinterpret_tensor(primals_178, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_177, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_176, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_175, (512, 384), (384, 1), 0), reinterpret_tensor(buf664, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf658, (6, 64, 128), (64, 1, 384), 0), buf662, reinterpret_tensor(buf656, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf657, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_174, (384, 512), (512, 1), 0), reinterpret_tensor(primals_173, (384, 512), (512, 1), 0), reinterpret_tensor(primals_172, (384, 512), (512, 1), 0), reinterpret_tensor(primals_171, (512, 384), (384, 1), 0), reinterpret_tensor(buf644, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf637, (6, 64, 128), (64, 1, 384), 0), buf642, reinterpret_tensor(buf635, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf636, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_170, (384, 512), (512, 1), 0), reinterpret_tensor(primals_169, (384, 512), (512, 1), 0), reinterpret_tensor(primals_168, (384, 512), (512, 1), 0), reinterpret_tensor(primals_167, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_166, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_165, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_164, (512, 384), (384, 1), 0), reinterpret_tensor(buf608, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf602, (6, 64, 128), (64, 1, 384), 0), buf606, reinterpret_tensor(buf600, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf601, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_163, (384, 512), (512, 1), 0), reinterpret_tensor(primals_162, (384, 512), (512, 1), 0), reinterpret_tensor(primals_161, (384, 512), (512, 1), 0), reinterpret_tensor(primals_160, (512, 384), (384, 1), 0), reinterpret_tensor(buf588, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf581, (6, 64, 128), (64, 1, 384), 0), buf586, reinterpret_tensor(buf579, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf580, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_159, (384, 512), (512, 1), 0), reinterpret_tensor(primals_158, (384, 512), (512, 1), 0), reinterpret_tensor(primals_157, (384, 512), (512, 1), 0), reinterpret_tensor(primals_156, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_155, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_154, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_153, (512, 384), (384, 1), 0), reinterpret_tensor(buf552, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf546, (6, 64, 128), (64, 1, 384), 0), buf550, reinterpret_tensor(buf544, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf545, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_152, (384, 512), (512, 1), 0), reinterpret_tensor(primals_151, (384, 512), (512, 1), 0), reinterpret_tensor(primals_150, (384, 512), (512, 1), 0), reinterpret_tensor(primals_149, (512, 384), (384, 1), 0), reinterpret_tensor(buf532, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf525, (6, 64, 128), (64, 1, 384), 0), buf530, reinterpret_tensor(buf523, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf524, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_148, (384, 512), (512, 1), 0), reinterpret_tensor(primals_147, (384, 512), (512, 1), 0), reinterpret_tensor(primals_146, (384, 512), (512, 1), 0), reinterpret_tensor(primals_145, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_144, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_143, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_142, (512, 384), (384, 1), 0), reinterpret_tensor(buf496, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf490, (6, 64, 128), (64, 1, 384), 0), buf494, reinterpret_tensor(buf488, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf489, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_141, (384, 512), (512, 1), 0), reinterpret_tensor(primals_140, (384, 512), (512, 1), 0), reinterpret_tensor(primals_139, (384, 512), (512, 1), 0), reinterpret_tensor(primals_138, (512, 384), (384, 1), 0), reinterpret_tensor(buf476, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf469, (6, 64, 128), (64, 1, 384), 0), buf474, reinterpret_tensor(buf467, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf468, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_137, (384, 512), (512, 1), 0), reinterpret_tensor(primals_136, (384, 512), (512, 1), 0), reinterpret_tensor(primals_135, (384, 512), (512, 1), 0), reinterpret_tensor(primals_134, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_133, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_132, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 384), (384, 1), 0), reinterpret_tensor(buf440, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf434, (6, 64, 128), (64, 1, 384), 0), buf438, reinterpret_tensor(buf432, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf433, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_130, (384, 512), (512, 1), 0), reinterpret_tensor(primals_129, (384, 512), (512, 1), 0), reinterpret_tensor(primals_128, (384, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 384), (384, 1), 0), reinterpret_tensor(buf420, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf413, (6, 64, 128), (64, 1, 384), 0), buf418, reinterpret_tensor(buf411, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf412, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_126, (384, 512), (512, 1), 0), reinterpret_tensor(primals_125, (384, 512), (512, 1), 0), reinterpret_tensor(primals_124, (384, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_122, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_121, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 384), (384, 1), 0), reinterpret_tensor(buf384, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf378, (6, 64, 128), (64, 1, 384), 0), buf382, reinterpret_tensor(buf376, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf377, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_119, (384, 512), (512, 1), 0), reinterpret_tensor(primals_118, (384, 512), (512, 1), 0), reinterpret_tensor(primals_117, (384, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 384), (384, 1), 0), reinterpret_tensor(buf364, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf357, (6, 64, 128), (64, 1, 384), 0), buf362, reinterpret_tensor(buf355, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf356, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_115, (384, 512), (512, 1), 0), reinterpret_tensor(primals_114, (384, 512), (512, 1), 0), reinterpret_tensor(primals_113, (384, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_111, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_110, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 384), (384, 1), 0), reinterpret_tensor(buf328, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf322, (6, 64, 128), (64, 1, 384), 0), buf326, reinterpret_tensor(buf320, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf321, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_108, (384, 512), (512, 1), 0), reinterpret_tensor(primals_107, (384, 512), (512, 1), 0), reinterpret_tensor(primals_106, (384, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 384), (384, 1), 0), reinterpret_tensor(buf308, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf300, (6, 64, 128), (64, 1, 384), 0), buf306, reinterpret_tensor(buf298, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf299, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_103, (384, 512), (512, 1), 0), reinterpret_tensor(primals_102, (384, 512), (512, 1), 0), reinterpret_tensor(primals_101, (384, 512), (512, 1), 0), reinterpret_tensor(primals_100, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_99, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_98, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 384), (384, 1), 0), reinterpret_tensor(buf261, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf255, (6, 64, 128), (64, 1, 384), 0), buf259, reinterpret_tensor(buf253, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf254, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_96, (384, 512), (512, 1), 0), reinterpret_tensor(primals_95, (384, 512), (512, 1), 0), reinterpret_tensor(primals_94, (384, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_92, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_91, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_90, (512, 384), (384, 1), 0), reinterpret_tensor(buf226, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf220, (6, 64, 128), (64, 1, 384), 0), buf224, reinterpret_tensor(buf218, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf219, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_89, (384, 512), (512, 1), 0), reinterpret_tensor(primals_88, (384, 512), (512, 1), 0), reinterpret_tensor(primals_87, (384, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_85, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_84, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 384), (384, 1), 0), reinterpret_tensor(buf191, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf185, (6, 64, 128), (64, 1, 384), 0), buf189, reinterpret_tensor(buf183, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf184, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_82, (384, 512), (512, 1), 0), reinterpret_tensor(primals_81, (384, 512), (512, 1), 0), reinterpret_tensor(primals_80, (384, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_78, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_77, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 384), (384, 1), 0), reinterpret_tensor(buf156, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf150, (6, 64, 128), (64, 1, 384), 0), buf154, reinterpret_tensor(buf148, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf149, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_75, (384, 512), (512, 1), 0), reinterpret_tensor(primals_74, (384, 512), (512, 1), 0), reinterpret_tensor(primals_73, (384, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_71, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_70, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_69, (512, 384), (384, 1), 0), reinterpret_tensor(buf121, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf115, (6, 64, 128), (64, 1, 384), 0), buf119, reinterpret_tensor(buf113, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf114, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_68, (384, 512), (512, 1), 0), reinterpret_tensor(primals_67, (384, 512), (512, 1), 0), reinterpret_tensor(primals_66, (384, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_64, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_63, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 384), (384, 1), 0), reinterpret_tensor(buf86, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf80, (6, 64, 128), (64, 1, 384), 0), buf84, reinterpret_tensor(buf78, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf79, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_61, (384, 512), (512, 1), 0), reinterpret_tensor(primals_60, (384, 512), (512, 1), 0), reinterpret_tensor(primals_59, (384, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_57, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_56, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_55, (512, 384), (384, 1), 0), reinterpret_tensor(buf51, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf45, (6, 64, 128), (64, 1, 384), 0), buf49, reinterpret_tensor(buf43, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf44, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_54, (384, 512), (512, 1), 0), reinterpret_tensor(primals_53, (384, 512), (512, 1), 0), reinterpret_tensor(primals_52, (384, 512), (512, 1), 0), reinterpret_tensor(primals_51, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_50, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_49, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 384), (384, 1), 0), reinterpret_tensor(buf16, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf9, (6, 64, 128), (64, 1, 384), 0), buf14, reinterpret_tensor(buf7, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf8, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_46, (384, 512), (512, 1), 0), reinterpret_tensor(primals_45, (384, 512), (512, 1), 0), reinterpret_tensor(primals_44, (384, 512), (512, 1), 0), )


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
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
