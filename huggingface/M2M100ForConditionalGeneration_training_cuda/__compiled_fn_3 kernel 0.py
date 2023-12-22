
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lxxcktaofbyxibvcliuk3lkog5pj2nxsyn5vc7qawagfsam3pn.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
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
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/so/csofobeuhlzcqgopoi7tcftbr3pezfms4gqlcraneoexsbfvr3ps.py
# Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states => add_2
# hidden_states_2 => add_3, add_4, mul_2, mul_3, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
# l__mod___model_encoder_layers_0_self_attn_q_proj => view_3
triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp23_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 128112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 128112")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 32.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tl.full([1, 1], 0, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.full([1, 1], 1, tl.int64)
        tmp12 = tmp0 != tmp11
        tmp13 = tmp12.to(tl.int32)
        tmp14 = tmp10 * tmp13
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15 + tmp11
        tmp17 = tmp16 + 1026
        tmp18 = tmp16 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp16)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp19 < 1026")
        tmp20 = tl.load(in_ptr3 + (r1 + (1024*tmp19)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp6 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_reduce(
            tmp22, tmp23_mean, tmp23_m2, tmp23_weight,
        )
        tmp23_mean = tl.where(rmask & xmask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(rmask & xmask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(rmask & xmask, tmp23_weight_next, tmp23_weight)
    tmp23_tmp, tmp24_tmp, tmp25_tmp = triton_helpers.welford(
        tmp23_mean, tmp23_m2, tmp23_weight, 1
    )
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp53 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp0 + 128112
        tmp27 = tmp0 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp0)
        tl.device_assert(((0 <= tmp28) & (tmp28 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp28 < 128112")
        tmp29 = tl.load(in_ptr1 + (r1 + (1024*tmp28)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = 32.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp7.to(tl.int32)
        tmp33 = tl.full([1, 1], 0, tl.int32)
        tmp34 = tmp32 + tmp33
        tmp35 = tl.full([1, 1], 1, tl.int64)
        tmp36 = tmp0 != tmp35
        tmp37 = tmp36.to(tl.int32)
        tmp38 = tmp34 * tmp37
        tmp39 = tmp38.to(tl.int64)
        tmp40 = tmp39 + tmp35
        tmp41 = tmp40 + 1026
        tmp42 = tmp40 < 0
        tmp43 = tl.where(tmp42, tmp41, tmp40)
        tl.device_assert(((0 <= tmp43) & (tmp43 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp43 < 1026")
        tmp44 = tl.load(in_ptr3 + (r1 + (1024*tmp43)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tmp31 + tmp44
        tmp46 = tmp45 - tmp23
        tmp47 = 1024.0
        tmp48 = tmp24 / tmp47
        tmp49 = 1e-05
        tmp50 = tmp48 + tmp49
        tmp51 = tl.math.rsqrt(tmp50)
        tmp52 = tmp46 * tmp51
        tmp54 = tmp52 * tmp53
        tmp56 = tmp54 + tmp55
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp52, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (1024*x0)), tmp56, rmask & xmask)
    tmp57 = 1024.0
    tmp58 = tmp24 / tmp57
    tmp59 = 1e-05
    tmp60 = tmp58 + tmp59
    tmp61 = tl.math.rsqrt(tmp60)
    tmp62 = tmp61 / tmp57
    tl.store(out_ptr4 + (x0), tmp62, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawncnrowv2arvtsgkifi7nkjx5mmzcpovrny32nyq6csudfdcfw.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_3
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzrr7ymfouxsj523h4wekgmgivujwxxjb25xyj2u7vi6q5mj2iy.py
# Source Nodes: [value_states], Original ATen: [aten.clone]
# value_states => clone_2
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkhsew3hcupijdel7iynzxh5vdoj47j5gi5shr3njpcfwxfkkwk.py
# Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
# attn_weights_1 => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp11, rmask)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4u2oxflvyq5hk46dgyvi5lnaxz2klnfiuhr6pvmzurwfs6wi5z.py
# Source Nodes: [hidden_states_3], Original ATen: [aten.view]
# hidden_states_3 => view_17
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgkhc63gv2gfwrzgjtc6623mcg3sxfoppebw7mluhpwjip3vobx.py
# Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_layers_0_fc1, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states => add_2
# hidden_states_6 => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
# l__mod___model_encoder_layers_0_fc1 => view_19
# residual_1 => add_5
triton_per_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 128112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 128112")
    tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, other=0.0)
    tmp5 = 32.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp0 != tmp11
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15 + tmp11
    tmp17 = tmp16 + 1026
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp19 < 1026")
    tmp20 = tl.load(in_ptr3 + (r1 + (1024*tmp19)), rmask & xmask, other=0.0)
    tmp21 = tmp6 + tmp20
    tmp24 = tmp22 + tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 1024, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 1024.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-05
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tmp47 / tmp43
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp52, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp53, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvfurwyzgemh4dtzd7se67qelavtntgxknb5khpgsb7nikdpgmn.py
# Source Nodes: [hidden_states_7, hidden_states_9], Original ATen: [aten.relu, aten.view]
# hidden_states_7 => relu
# hidden_states_9 => view_21
triton_poi_fused_relu_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu56x6imacuetchp7kc3im3m2tb7m242wd227m4ex46mme2maarx.py
# Source Nodes: [hidden_states_13, l__mod___model_encoder_layers_1_self_attn_q_proj, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_13 => add_10, add_9, mul_7, mul_8, rsqrt_2, sub_3, var_mean_2
# l__mod___model_encoder_layers_1_self_attn_q_proj => view_23
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/che6ephw3azmwtip55gocbvo37uwaunqtaoxpm5mcj3uf7okrhmm.py
# Source Nodes: [hidden_states_17, l__mod___model_encoder_layers_1_fc1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_17 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# l__mod___model_encoder_layers_1_fc1 => view_39
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyem7lhy65s37ozvqy3jtujknxphgegn2pkx7b6rrihzz4oznbv.py
# Source Nodes: [attn_probs_12, attn_weights_27], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_probs_12 => clone_101
# attn_weights_27 => amax_12, div_12, exp_12, sub_38, sum_13
triton_per_fused__softmax_clone_detach_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask, other=0.0)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp13 / tmp17
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp18, rmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celtcblimmy2pjskahdmadf35k3cmd4amfggwjfi2hcmnspg3tfk.py
# Source Nodes: [hidden_states_296, hidden_states_298], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
# hidden_states_296 => relu_22
# hidden_states_298 => view_663
triton_poi_fused_relu_threshold_backward_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_view_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5co35yqvjzejpj5cpvw44qwltwnthqtwo2u63uwatwi74ezdam.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_36
triton_red_fused__log_softmax_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32028
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32028*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7k2nch5ul5cnraipne4xdswtmnkcobmemuc4frjurmai4z3yss5.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_36
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
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3a573mllq34tidr6jh3e4icgipxt5wxyf5pbwe774ekxvf7fy7.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_36, sub_98, sum_37
triton_red_fused__log_softmax_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32028
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
        tmp0 = tl.load(in_ptr0 + (r2 + (32028*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvalhuigelrg2mwl3tos7fxklmqrrdbm7gnbxs5qzbrmqywriop.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_36, sub_98, sum_37
triton_per_fused__log_softmax_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqgehfn6w2hcs72arpy75jxombtrsayzdte7zzsfaebhore2re5.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => log, sub_98, sub_99
triton_poi_fused__log_softmax_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16398336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128112)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.log(tmp3)
    tmp5 = tmp2 - tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbyi673wpx7pagpcpt7kygfq2rcyiw7ijhk6yrzcniixjp7qfk2.py
# Source Nodes: [loss, masked_fill_], Original ATen: [aten.masked_fill, aten.nll_loss_forward]
# loss => convert_element_type_6, div_36, ne_2, neg, sum_38, sum_39, where_2
# masked_fill_ => full_default_1
triton_per_fused_masked_fill_nll_loss_forward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_masked_fill_nll_loss_forward_17', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp10 = tmp9 + 128112
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 128112), "index out of bounds: 0 <= tmp12 < 128112")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (128112*r0)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuuj4j7t34vrlfspsgsrsqv5jblqd6amzmmgov5yoazltjuwvag.py
# Source Nodes: [hidden_states_281], Original ATen: [aten.relu, aten.threshold_backward]
# hidden_states_281 => relu_21
triton_poi_fused_relu_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516 = args
    args.clear()
    assert_size_stride(primals_1, (128112, 1024), (1024, 1))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (1024, ), (1, ))
    assert_size_stride(primals_4, (1024, 1024), (1024, 1))
    assert_size_stride(primals_5, (1024, ), (1, ))
    assert_size_stride(primals_6, (1024, 1024), (1024, 1))
    assert_size_stride(primals_7, (1024, ), (1, ))
    assert_size_stride(primals_8, (1024, 1024), (1024, 1))
    assert_size_stride(primals_9, (1024, ), (1, ))
    assert_size_stride(primals_10, (1024, 1024), (1024, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (4096, 1024), (1024, 1))
    assert_size_stride(primals_15, (4096, ), (1, ))
    assert_size_stride(primals_16, (1024, 4096), (4096, 1))
    assert_size_stride(primals_17, (1024, ), (1, ))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, 1024), (1024, 1))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_22, (1024, 1024), (1024, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (1024, 1024), (1024, 1))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, 1024), (1024, 1))
    assert_size_stride(primals_27, (1024, ), (1, ))
    assert_size_stride(primals_28, (1024, ), (1, ))
    assert_size_stride(primals_29, (1024, ), (1, ))
    assert_size_stride(primals_30, (4096, 1024), (1024, 1))
    assert_size_stride(primals_31, (4096, ), (1, ))
    assert_size_stride(primals_32, (1024, 4096), (4096, 1))
    assert_size_stride(primals_33, (1024, ), (1, ))
    assert_size_stride(primals_34, (1024, ), (1, ))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, 1024), (1024, 1))
    assert_size_stride(primals_37, (1024, ), (1, ))
    assert_size_stride(primals_38, (1024, 1024), (1024, 1))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (1024, 1024), (1024, 1))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, 1024), (1024, 1))
    assert_size_stride(primals_43, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_46, (4096, 1024), (1024, 1))
    assert_size_stride(primals_47, (4096, ), (1, ))
    assert_size_stride(primals_48, (1024, 4096), (4096, 1))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, 1024), (1024, 1))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, 1024), (1024, 1))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1024, 1024), (1024, 1))
    assert_size_stride(primals_57, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, 1024), (1024, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, ), (1, ))
    assert_size_stride(primals_62, (4096, 1024), (1024, 1))
    assert_size_stride(primals_63, (4096, ), (1, ))
    assert_size_stride(primals_64, (1024, 4096), (4096, 1))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, 1024), (1024, 1))
    assert_size_stride(primals_69, (1024, ), (1, ))
    assert_size_stride(primals_70, (1024, 1024), (1024, 1))
    assert_size_stride(primals_71, (1024, ), (1, ))
    assert_size_stride(primals_72, (1024, 1024), (1024, 1))
    assert_size_stride(primals_73, (1024, ), (1, ))
    assert_size_stride(primals_74, (1024, 1024), (1024, 1))
    assert_size_stride(primals_75, (1024, ), (1, ))
    assert_size_stride(primals_76, (1024, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (4096, 1024), (1024, 1))
    assert_size_stride(primals_79, (4096, ), (1, ))
    assert_size_stride(primals_80, (1024, 4096), (4096, 1))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, ), (1, ))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, 1024), (1024, 1))
    assert_size_stride(primals_85, (1024, ), (1, ))
    assert_size_stride(primals_86, (1024, 1024), (1024, 1))
    assert_size_stride(primals_87, (1024, ), (1, ))
    assert_size_stride(primals_88, (1024, 1024), (1024, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1024, 1024), (1024, 1))
    assert_size_stride(primals_91, (1024, ), (1, ))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (4096, 1024), (1024, 1))
    assert_size_stride(primals_95, (4096, ), (1, ))
    assert_size_stride(primals_96, (1024, 4096), (4096, 1))
    assert_size_stride(primals_97, (1024, ), (1, ))
    assert_size_stride(primals_98, (1024, ), (1, ))
    assert_size_stride(primals_99, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, 1024), (1024, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, 1024), (1024, 1))
    assert_size_stride(primals_103, (1024, ), (1, ))
    assert_size_stride(primals_104, (1024, 1024), (1024, 1))
    assert_size_stride(primals_105, (1024, ), (1, ))
    assert_size_stride(primals_106, (1024, 1024), (1024, 1))
    assert_size_stride(primals_107, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (4096, 1024), (1024, 1))
    assert_size_stride(primals_111, (4096, ), (1, ))
    assert_size_stride(primals_112, (1024, 4096), (4096, 1))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_115, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, 1024), (1024, 1))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1024, 1024), (1024, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, 1024), (1024, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1024, 1024), (1024, 1))
    assert_size_stride(primals_123, (1024, ), (1, ))
    assert_size_stride(primals_124, (1024, ), (1, ))
    assert_size_stride(primals_125, (1024, ), (1, ))
    assert_size_stride(primals_126, (4096, 1024), (1024, 1))
    assert_size_stride(primals_127, (4096, ), (1, ))
    assert_size_stride(primals_128, (1024, 4096), (4096, 1))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, ), (1, ))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, 1024), (1024, 1))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, 1024), (1024, 1))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, 1024), (1024, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1024, 1024), (1024, 1))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (4096, 1024), (1024, 1))
    assert_size_stride(primals_143, (4096, ), (1, ))
    assert_size_stride(primals_144, (1024, 4096), (4096, 1))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, 1024), (1024, 1))
    assert_size_stride(primals_149, (1024, ), (1, ))
    assert_size_stride(primals_150, (1024, 1024), (1024, 1))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (1024, 1024), (1024, 1))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 1024), (1024, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_158, (4096, 1024), (1024, 1))
    assert_size_stride(primals_159, (4096, ), (1, ))
    assert_size_stride(primals_160, (1024, 4096), (4096, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, 1024), (1024, 1))
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (1024, 1024), (1024, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, 1024), (1024, 1))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, 1024), (1024, 1))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (1024, ), (1, ))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_174, (4096, 1024), (1024, 1))
    assert_size_stride(primals_175, (4096, ), (1, ))
    assert_size_stride(primals_176, (1024, 4096), (4096, 1))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_179, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, 1024), (1024, 1))
    assert_size_stride(primals_181, (1024, ), (1, ))
    assert_size_stride(primals_182, (1024, 1024), (1024, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, 1024), (1024, 1))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, 1024), (1024, 1))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (1024, ), (1, ))
    assert_size_stride(primals_189, (1024, ), (1, ))
    assert_size_stride(primals_190, (4096, 1024), (1024, 1))
    assert_size_stride(primals_191, (4096, ), (1, ))
    assert_size_stride(primals_192, (1024, 4096), (4096, 1))
    assert_size_stride(primals_193, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (128112, 1024), (1024, 1))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (1024, ), (1, ))
    assert_size_stride(primals_199, (1024, 1024), (1024, 1))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, 1024), (1024, 1))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_203, (1024, 1024), (1024, 1))
    assert_size_stride(primals_204, (1024, ), (1, ))
    assert_size_stride(primals_205, (1024, 1024), (1024, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, ), (1, ))
    assert_size_stride(primals_209, (1024, 1024), (1024, 1))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (1024, 1024), (1024, 1))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_213, (1024, 1024), (1024, 1))
    assert_size_stride(primals_214, (1024, ), (1, ))
    assert_size_stride(primals_215, (1024, 1024), (1024, 1))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (4096, 1024), (1024, 1))
    assert_size_stride(primals_220, (4096, ), (1, ))
    assert_size_stride(primals_221, (1024, 4096), (4096, 1))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_225, (1024, 1024), (1024, 1))
    assert_size_stride(primals_226, (1024, ), (1, ))
    assert_size_stride(primals_227, (1024, 1024), (1024, 1))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, 1024), (1024, 1))
    assert_size_stride(primals_230, (1024, ), (1, ))
    assert_size_stride(primals_231, (1024, 1024), (1024, 1))
    assert_size_stride(primals_232, (1024, ), (1, ))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, ), (1, ))
    assert_size_stride(primals_235, (1024, 1024), (1024, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_237, (1024, 1024), (1024, 1))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (1024, 1024), (1024, 1))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, 1024), (1024, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (4096, 1024), (1024, 1))
    assert_size_stride(primals_246, (4096, ), (1, ))
    assert_size_stride(primals_247, (1024, 4096), (4096, 1))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_251, (1024, 1024), (1024, 1))
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, 1024), (1024, 1))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_255, (1024, 1024), (1024, 1))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_257, (1024, 1024), (1024, 1))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_261, (1024, 1024), (1024, 1))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (1024, 1024), (1024, 1))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_265, (1024, 1024), (1024, 1))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_267, (1024, 1024), (1024, 1))
    assert_size_stride(primals_268, (1024, ), (1, ))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (4096, 1024), (1024, 1))
    assert_size_stride(primals_272, (4096, ), (1, ))
    assert_size_stride(primals_273, (1024, 4096), (4096, 1))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_277, (1024, 1024), (1024, 1))
    assert_size_stride(primals_278, (1024, ), (1, ))
    assert_size_stride(primals_279, (1024, 1024), (1024, 1))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (1024, 1024), (1024, 1))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (1024, 1024), (1024, 1))
    assert_size_stride(primals_284, (1024, ), (1, ))
    assert_size_stride(primals_285, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_287, (1024, 1024), (1024, 1))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, 1024), (1024, 1))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, 1024), (1024, 1))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_293, (1024, 1024), (1024, 1))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (4096, 1024), (1024, 1))
    assert_size_stride(primals_298, (4096, ), (1, ))
    assert_size_stride(primals_299, (1024, 4096), (4096, 1))
    assert_size_stride(primals_300, (1024, ), (1, ))
    assert_size_stride(primals_301, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, 1024), (1024, 1))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (1024, 1024), (1024, 1))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, 1024), (1024, 1))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_309, (1024, 1024), (1024, 1))
    assert_size_stride(primals_310, (1024, ), (1, ))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, 1024), (1024, 1))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, 1024), (1024, 1))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_317, (1024, 1024), (1024, 1))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, 1024), (1024, 1))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (1024, ), (1, ))
    assert_size_stride(primals_323, (4096, 1024), (1024, 1))
    assert_size_stride(primals_324, (4096, ), (1, ))
    assert_size_stride(primals_325, (1024, 4096), (4096, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, ), (1, ))
    assert_size_stride(primals_329, (1024, 1024), (1024, 1))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, 1024), (1024, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_333, (1024, 1024), (1024, 1))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_335, (1024, 1024), (1024, 1))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (1024, 1024), (1024, 1))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_341, (1024, 1024), (1024, 1))
    assert_size_stride(primals_342, (1024, ), (1, ))
    assert_size_stride(primals_343, (1024, 1024), (1024, 1))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_345, (1024, 1024), (1024, 1))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (1024, ), (1, ))
    assert_size_stride(primals_349, (4096, 1024), (1024, 1))
    assert_size_stride(primals_350, (4096, ), (1, ))
    assert_size_stride(primals_351, (1024, 4096), (4096, 1))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (1024, ), (1, ))
    assert_size_stride(primals_354, (1024, ), (1, ))
    assert_size_stride(primals_355, (1024, 1024), (1024, 1))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_357, (1024, 1024), (1024, 1))
    assert_size_stride(primals_358, (1024, ), (1, ))
    assert_size_stride(primals_359, (1024, 1024), (1024, 1))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, 1024), (1024, 1))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (1024, ), (1, ))
    assert_size_stride(primals_365, (1024, 1024), (1024, 1))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_367, (1024, 1024), (1024, 1))
    assert_size_stride(primals_368, (1024, ), (1, ))
    assert_size_stride(primals_369, (1024, 1024), (1024, 1))
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (1024, 1024), (1024, 1))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_374, (1024, ), (1, ))
    assert_size_stride(primals_375, (4096, 1024), (1024, 1))
    assert_size_stride(primals_376, (4096, ), (1, ))
    assert_size_stride(primals_377, (1024, 4096), (4096, 1))
    assert_size_stride(primals_378, (1024, ), (1, ))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (1024, ), (1, ))
    assert_size_stride(primals_381, (1024, 1024), (1024, 1))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_383, (1024, 1024), (1024, 1))
    assert_size_stride(primals_384, (1024, ), (1, ))
    assert_size_stride(primals_385, (1024, 1024), (1024, 1))
    assert_size_stride(primals_386, (1024, ), (1, ))
    assert_size_stride(primals_387, (1024, 1024), (1024, 1))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_389, (1024, ), (1, ))
    assert_size_stride(primals_390, (1024, ), (1, ))
    assert_size_stride(primals_391, (1024, 1024), (1024, 1))
    assert_size_stride(primals_392, (1024, ), (1, ))
    assert_size_stride(primals_393, (1024, 1024), (1024, 1))
    assert_size_stride(primals_394, (1024, ), (1, ))
    assert_size_stride(primals_395, (1024, 1024), (1024, 1))
    assert_size_stride(primals_396, (1024, ), (1, ))
    assert_size_stride(primals_397, (1024, 1024), (1024, 1))
    assert_size_stride(primals_398, (1024, ), (1, ))
    assert_size_stride(primals_399, (1024, ), (1, ))
    assert_size_stride(primals_400, (1024, ), (1, ))
    assert_size_stride(primals_401, (4096, 1024), (1024, 1))
    assert_size_stride(primals_402, (4096, ), (1, ))
    assert_size_stride(primals_403, (1024, 4096), (4096, 1))
    assert_size_stride(primals_404, (1024, ), (1, ))
    assert_size_stride(primals_405, (1024, ), (1, ))
    assert_size_stride(primals_406, (1024, ), (1, ))
    assert_size_stride(primals_407, (1024, 1024), (1024, 1))
    assert_size_stride(primals_408, (1024, ), (1, ))
    assert_size_stride(primals_409, (1024, 1024), (1024, 1))
    assert_size_stride(primals_410, (1024, ), (1, ))
    assert_size_stride(primals_411, (1024, 1024), (1024, 1))
    assert_size_stride(primals_412, (1024, ), (1, ))
    assert_size_stride(primals_413, (1024, 1024), (1024, 1))
    assert_size_stride(primals_414, (1024, ), (1, ))
    assert_size_stride(primals_415, (1024, ), (1, ))
    assert_size_stride(primals_416, (1024, ), (1, ))
    assert_size_stride(primals_417, (1024, 1024), (1024, 1))
    assert_size_stride(primals_418, (1024, ), (1, ))
    assert_size_stride(primals_419, (1024, 1024), (1024, 1))
    assert_size_stride(primals_420, (1024, ), (1, ))
    assert_size_stride(primals_421, (1024, 1024), (1024, 1))
    assert_size_stride(primals_422, (1024, ), (1, ))
    assert_size_stride(primals_423, (1024, 1024), (1024, 1))
    assert_size_stride(primals_424, (1024, ), (1, ))
    assert_size_stride(primals_425, (1024, ), (1, ))
    assert_size_stride(primals_426, (1024, ), (1, ))
    assert_size_stride(primals_427, (4096, 1024), (1024, 1))
    assert_size_stride(primals_428, (4096, ), (1, ))
    assert_size_stride(primals_429, (1024, 4096), (4096, 1))
    assert_size_stride(primals_430, (1024, ), (1, ))
    assert_size_stride(primals_431, (1024, ), (1, ))
    assert_size_stride(primals_432, (1024, ), (1, ))
    assert_size_stride(primals_433, (1024, 1024), (1024, 1))
    assert_size_stride(primals_434, (1024, ), (1, ))
    assert_size_stride(primals_435, (1024, 1024), (1024, 1))
    assert_size_stride(primals_436, (1024, ), (1, ))
    assert_size_stride(primals_437, (1024, 1024), (1024, 1))
    assert_size_stride(primals_438, (1024, ), (1, ))
    assert_size_stride(primals_439, (1024, 1024), (1024, 1))
    assert_size_stride(primals_440, (1024, ), (1, ))
    assert_size_stride(primals_441, (1024, ), (1, ))
    assert_size_stride(primals_442, (1024, ), (1, ))
    assert_size_stride(primals_443, (1024, 1024), (1024, 1))
    assert_size_stride(primals_444, (1024, ), (1, ))
    assert_size_stride(primals_445, (1024, 1024), (1024, 1))
    assert_size_stride(primals_446, (1024, ), (1, ))
    assert_size_stride(primals_447, (1024, 1024), (1024, 1))
    assert_size_stride(primals_448, (1024, ), (1, ))
    assert_size_stride(primals_449, (1024, 1024), (1024, 1))
    assert_size_stride(primals_450, (1024, ), (1, ))
    assert_size_stride(primals_451, (1024, ), (1, ))
    assert_size_stride(primals_452, (1024, ), (1, ))
    assert_size_stride(primals_453, (4096, 1024), (1024, 1))
    assert_size_stride(primals_454, (4096, ), (1, ))
    assert_size_stride(primals_455, (1024, 4096), (4096, 1))
    assert_size_stride(primals_456, (1024, ), (1, ))
    assert_size_stride(primals_457, (1024, ), (1, ))
    assert_size_stride(primals_458, (1024, ), (1, ))
    assert_size_stride(primals_459, (1024, 1024), (1024, 1))
    assert_size_stride(primals_460, (1024, ), (1, ))
    assert_size_stride(primals_461, (1024, 1024), (1024, 1))
    assert_size_stride(primals_462, (1024, ), (1, ))
    assert_size_stride(primals_463, (1024, 1024), (1024, 1))
    assert_size_stride(primals_464, (1024, ), (1, ))
    assert_size_stride(primals_465, (1024, 1024), (1024, 1))
    assert_size_stride(primals_466, (1024, ), (1, ))
    assert_size_stride(primals_467, (1024, ), (1, ))
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (1024, 1024), (1024, 1))
    assert_size_stride(primals_470, (1024, ), (1, ))
    assert_size_stride(primals_471, (1024, 1024), (1024, 1))
    assert_size_stride(primals_472, (1024, ), (1, ))
    assert_size_stride(primals_473, (1024, 1024), (1024, 1))
    assert_size_stride(primals_474, (1024, ), (1, ))
    assert_size_stride(primals_475, (1024, 1024), (1024, 1))
    assert_size_stride(primals_476, (1024, ), (1, ))
    assert_size_stride(primals_477, (1024, ), (1, ))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (4096, 1024), (1024, 1))
    assert_size_stride(primals_480, (4096, ), (1, ))
    assert_size_stride(primals_481, (1024, 4096), (4096, 1))
    assert_size_stride(primals_482, (1024, ), (1, ))
    assert_size_stride(primals_483, (1024, ), (1, ))
    assert_size_stride(primals_484, (1024, ), (1, ))
    assert_size_stride(primals_485, (1024, 1024), (1024, 1))
    assert_size_stride(primals_486, (1024, ), (1, ))
    assert_size_stride(primals_487, (1024, 1024), (1024, 1))
    assert_size_stride(primals_488, (1024, ), (1, ))
    assert_size_stride(primals_489, (1024, 1024), (1024, 1))
    assert_size_stride(primals_490, (1024, ), (1, ))
    assert_size_stride(primals_491, (1024, 1024), (1024, 1))
    assert_size_stride(primals_492, (1024, ), (1, ))
    assert_size_stride(primals_493, (1024, ), (1, ))
    assert_size_stride(primals_494, (1024, ), (1, ))
    assert_size_stride(primals_495, (1024, 1024), (1024, 1))
    assert_size_stride(primals_496, (1024, ), (1, ))
    assert_size_stride(primals_497, (1024, 1024), (1024, 1))
    assert_size_stride(primals_498, (1024, ), (1, ))
    assert_size_stride(primals_499, (1024, 1024), (1024, 1))
    assert_size_stride(primals_500, (1024, ), (1, ))
    assert_size_stride(primals_501, (1024, 1024), (1024, 1))
    assert_size_stride(primals_502, (1024, ), (1, ))
    assert_size_stride(primals_503, (1024, ), (1, ))
    assert_size_stride(primals_504, (1024, ), (1, ))
    assert_size_stride(primals_505, (4096, 1024), (1024, 1))
    assert_size_stride(primals_506, (4096, ), (1, ))
    assert_size_stride(primals_507, (1024, 4096), (4096, 1))
    assert_size_stride(primals_508, (1024, ), (1, ))
    assert_size_stride(primals_509, (1024, ), (1, ))
    assert_size_stride(primals_510, (1024, ), (1, ))
    assert_size_stride(primals_511, (128112, 1024), (1024, 1))
    assert_size_stride(primals_512, (1026, 1024), (1024, 1))
    assert_size_stride(primals_513, (1026, 1024), (1024, 1))
    assert_size_stride(primals_514, (1, 128), (128, 1))
    assert_size_stride(primals_515, (1, 128), (128, 1))
    assert_size_stride(primals_516, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(primals_516, buf0, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        buf2 = buf1
        del buf1
        buf6 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf7 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf992 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1.run(primals_516, primals_1, buf2, primals_512, primals_2, primals_3, buf6, buf7, buf992, 128, 1024, grid=grid(128), stream=stream0)
        del primals_3
        buf8 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, reinterpret_tensor(primals_4, (1024, 1024), (1, 1024), 0), out=buf8)
        buf9 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, reinterpret_tensor(primals_6, (1024, 1024), (1, 1024), 0), out=buf9)
        buf10 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, reinterpret_tensor(primals_8, (1024, 1024), (1, 1024), 0), out=buf10)
        buf11 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf8, primals_5, buf11, 131072, grid=grid(131072), stream=stream0)
        del primals_5
        buf12 = reinterpret_tensor(buf8, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf10, primals_9, buf12, 131072, grid=grid(131072), stream=stream0)
        del primals_9
        buf13 = reinterpret_tensor(buf10, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf9, primals_7, buf13, 131072, grid=grid(131072), stream=stream0)
        del primals_7
        buf14 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf13, (16, 64, 128), (8192, 1, 64), 0), out=buf14)
        buf15 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf16 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf17 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf14, buf15, buf16, buf17, 2048, 128, grid=grid(2048), stream=stream0)
        buf18 = reinterpret_tensor(buf9, (16, 128, 64), (8192, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf17, reinterpret_tensor(buf12, (16, 128, 64), (8192, 64, 1), 0), out=buf18)
        buf19 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf18, buf19, 131072, grid=grid(131072), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (128, 1024), (1024, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf19, reinterpret_tensor(primals_10, (1024, 1024), (1, 1024), 0), out=buf20)
        buf21 = reinterpret_tensor(buf20, (1, 128, 1024), (131072, 1024, 1), 0); del buf20  # reuse
        buf25 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf26 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf991 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_layers_0_fc1, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_6.run(buf21, primals_516, primals_1, buf2, primals_512, primals_11, primals_12, primals_13, buf25, buf26, buf991, 128, 1024, grid=grid(128), stream=stream0)
        del buf2
        del primals_1
        del primals_11
        del primals_13
        del primals_512
        buf27 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(primals_14, (1024, 4096), (1, 1024), 0), out=buf27)
        buf28 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7, hidden_states_9], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf27, primals_15, buf28, 524288, grid=grid(524288), stream=stream0)
        buf29 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf28, reinterpret_tensor(primals_16, (4096, 1024), (1, 4096), 0), out=buf29)
        buf33 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf34 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf989 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13, l__mod___model_encoder_layers_1_self_attn_q_proj, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf21, buf29, primals_17, primals_18, primals_19, buf33, buf34, buf989, 128, 1024, grid=grid(128), stream=stream0)
        del primals_19
        buf35 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_20, (1024, 1024), (1, 1024), 0), out=buf35)
        buf36 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_22, (1024, 1024), (1, 1024), 0), out=buf36)
        buf37 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_24, (1024, 1024), (1, 1024), 0), out=buf37)
        buf38 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf35, primals_21, buf38, 131072, grid=grid(131072), stream=stream0)
        del primals_21
        buf39 = reinterpret_tensor(buf35, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf37, primals_25, buf39, 131072, grid=grid(131072), stream=stream0)
        del primals_25
        buf40 = reinterpret_tensor(buf37, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf37  # reuse
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf36, primals_23, buf40, 131072, grid=grid(131072), stream=stream0)
        del primals_23
        buf41 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf40, (16, 64, 128), (8192, 1, 64), 0), out=buf41)
        buf42 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf43 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf44 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf41, buf42, buf43, buf44, 2048, 128, grid=grid(2048), stream=stream0)
        buf45 = reinterpret_tensor(buf36, (16, 128, 64), (8192, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf44, reinterpret_tensor(buf39, (16, 128, 64), (8192, 64, 1), 0), out=buf45)
        buf46 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf45, buf46, 131072, grid=grid(131072), stream=stream0)
        buf47 = reinterpret_tensor(buf45, (128, 1024), (1024, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf46, reinterpret_tensor(primals_26, (1024, 1024), (1, 1024), 0), out=buf47)
        buf48 = reinterpret_tensor(buf47, (1, 128, 1024), (131072, 1024, 1), 0); del buf47  # reuse
        buf52 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf53 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf988 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_17, l__mod___model_encoder_layers_1_fc1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf48, buf21, buf29, primals_17, primals_27, primals_28, primals_29, buf52, buf53, buf988, 128, 1024, grid=grid(128), stream=stream0)
        del primals_17
        del primals_27
        del primals_29
        buf54 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf53, reinterpret_tensor(primals_30, (1024, 4096), (1, 1024), 0), out=buf54)
        buf55 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, hidden_states_20], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf54, primals_31, buf55, 524288, grid=grid(524288), stream=stream0)
        buf56 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf55, reinterpret_tensor(primals_32, (4096, 1024), (1, 4096), 0), out=buf56)
        buf60 = buf21; del buf21  # reuse
        buf61 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf986 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, l__mod___model_encoder_layers_2_self_attn_q_proj, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf48, buf56, primals_33, primals_34, primals_35, buf60, buf61, buf986, 128, 1024, grid=grid(128), stream=stream0)
        del primals_35
        buf62 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_36, (1024, 1024), (1, 1024), 0), out=buf62)
        buf63 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_38, (1024, 1024), (1, 1024), 0), out=buf63)
        buf64 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_40, (1024, 1024), (1, 1024), 0), out=buf64)
        buf65 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf62, primals_37, buf65, 131072, grid=grid(131072), stream=stream0)
        del primals_37
        buf66 = reinterpret_tensor(buf62, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf62  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf64, primals_41, buf66, 131072, grid=grid(131072), stream=stream0)
        del primals_41
        buf67 = reinterpret_tensor(buf64, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf63, primals_39, buf67, 131072, grid=grid(131072), stream=stream0)
        del primals_39
        buf68 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf67, (16, 64, 128), (8192, 1, 64), 0), out=buf68)
        buf69 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf70 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf71 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf68, buf69, buf70, buf71, 2048, 128, grid=grid(2048), stream=stream0)
        buf72 = reinterpret_tensor(buf63, (16, 128, 64), (8192, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [attn_output_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf71, reinterpret_tensor(buf66, (16, 128, 64), (8192, 64, 1), 0), out=buf72)
        buf73 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf72, buf73, 131072, grid=grid(131072), stream=stream0)
        buf74 = reinterpret_tensor(buf72, (128, 1024), (1024, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_42, (1024, 1024), (1, 1024), 0), out=buf74)
        buf75 = reinterpret_tensor(buf74, (1, 128, 1024), (131072, 1024, 1), 0); del buf74  # reuse
        buf79 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf80 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf985 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_28, l__mod___model_encoder_layers_2_fc1, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf75, buf48, buf56, primals_33, primals_43, primals_44, primals_45, buf79, buf80, buf985, 128, 1024, grid=grid(128), stream=stream0)
        del primals_33
        del primals_43
        del primals_45
        buf81 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf80, reinterpret_tensor(primals_46, (1024, 4096), (1, 1024), 0), out=buf81)
        buf82 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_29, hidden_states_31], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf81, primals_47, buf82, 524288, grid=grid(524288), stream=stream0)
        buf83 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_48, (4096, 1024), (1, 4096), 0), out=buf83)
        buf87 = buf48; del buf48  # reuse
        buf88 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf983 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_35, l__mod___model_encoder_layers_3_self_attn_q_proj, residual_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf75, buf83, primals_49, primals_50, primals_51, buf87, buf88, buf983, 128, 1024, grid=grid(128), stream=stream0)
        del primals_51
        buf89 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf88, reinterpret_tensor(primals_52, (1024, 1024), (1, 1024), 0), out=buf89)
        buf90 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf88, reinterpret_tensor(primals_54, (1024, 1024), (1, 1024), 0), out=buf90)
        buf91 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf88, reinterpret_tensor(primals_56, (1024, 1024), (1, 1024), 0), out=buf91)
        buf92 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf89, primals_53, buf92, 131072, grid=grid(131072), stream=stream0)
        del primals_53
        buf93 = reinterpret_tensor(buf89, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf89  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf91, primals_57, buf93, 131072, grid=grid(131072), stream=stream0)
        del primals_57
        buf94 = reinterpret_tensor(buf91, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf91  # reuse
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf90, primals_55, buf94, 131072, grid=grid(131072), stream=stream0)
        del primals_55
        buf95 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf94, (16, 64, 128), (8192, 1, 64), 0), out=buf95)
        buf96 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf97 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf98 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf95, buf96, buf97, buf98, 2048, 128, grid=grid(2048), stream=stream0)
        buf99 = reinterpret_tensor(buf90, (16, 128, 64), (8192, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf98, reinterpret_tensor(buf93, (16, 128, 64), (8192, 64, 1), 0), out=buf99)
        buf100 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf99, buf100, 131072, grid=grid(131072), stream=stream0)
        buf101 = reinterpret_tensor(buf99, (128, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf100, reinterpret_tensor(primals_58, (1024, 1024), (1, 1024), 0), out=buf101)
        buf102 = reinterpret_tensor(buf101, (1, 128, 1024), (131072, 1024, 1), 0); del buf101  # reuse
        buf106 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf107 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf982 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39, l__mod___model_encoder_layers_3_fc1, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf102, buf75, buf83, primals_49, primals_59, primals_60, primals_61, buf106, buf107, buf982, 128, 1024, grid=grid(128), stream=stream0)
        del primals_49
        del primals_59
        del primals_61
        buf108 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf107, reinterpret_tensor(primals_62, (1024, 4096), (1, 1024), 0), out=buf108)
        buf109 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40, hidden_states_42], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf108, primals_63, buf109, 524288, grid=grid(524288), stream=stream0)
        buf110 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf109, reinterpret_tensor(primals_64, (4096, 1024), (1, 4096), 0), out=buf110)
        buf114 = buf75; del buf75  # reuse
        buf115 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf980 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_46, l__mod___model_encoder_layers_4_self_attn_q_proj, residual_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf102, buf110, primals_65, primals_66, primals_67, buf114, buf115, buf980, 128, 1024, grid=grid(128), stream=stream0)
        del primals_67
        buf116 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf115, reinterpret_tensor(primals_68, (1024, 1024), (1, 1024), 0), out=buf116)
        buf117 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf115, reinterpret_tensor(primals_70, (1024, 1024), (1, 1024), 0), out=buf117)
        buf118 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf115, reinterpret_tensor(primals_72, (1024, 1024), (1, 1024), 0), out=buf118)
        buf119 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf116, primals_69, buf119, 131072, grid=grid(131072), stream=stream0)
        del primals_69
        buf120 = reinterpret_tensor(buf116, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf118, primals_73, buf120, 131072, grid=grid(131072), stream=stream0)
        del primals_73
        buf121 = reinterpret_tensor(buf118, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf117, primals_71, buf121, 131072, grid=grid(131072), stream=stream0)
        del primals_71
        buf122 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf121, (16, 64, 128), (8192, 1, 64), 0), out=buf122)
        buf123 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf124 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf125 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf122, buf123, buf124, buf125, 2048, 128, grid=grid(2048), stream=stream0)
        buf126 = reinterpret_tensor(buf117, (16, 128, 64), (8192, 64, 1), 0); del buf117  # reuse
        # Source Nodes: [attn_output_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf125, reinterpret_tensor(buf120, (16, 128, 64), (8192, 64, 1), 0), out=buf126)
        buf127 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_47], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf126, buf127, 131072, grid=grid(131072), stream=stream0)
        buf128 = reinterpret_tensor(buf126, (128, 1024), (1024, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf127, reinterpret_tensor(primals_74, (1024, 1024), (1, 1024), 0), out=buf128)
        buf129 = reinterpret_tensor(buf128, (1, 128, 1024), (131072, 1024, 1), 0); del buf128  # reuse
        buf133 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf134 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf979 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, l__mod___model_encoder_layers_4_fc1, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf129, buf102, buf110, primals_65, primals_75, primals_76, primals_77, buf133, buf134, buf979, 128, 1024, grid=grid(128), stream=stream0)
        del primals_65
        del primals_75
        del primals_77
        buf135 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf134, reinterpret_tensor(primals_78, (1024, 4096), (1, 1024), 0), out=buf135)
        buf136 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf135, primals_79, buf136, 524288, grid=grid(524288), stream=stream0)
        buf137 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_80, (4096, 1024), (1, 4096), 0), out=buf137)
        buf141 = buf102; del buf102  # reuse
        buf142 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf977 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57, l__mod___model_encoder_layers_5_self_attn_q_proj, residual_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf129, buf137, primals_81, primals_82, primals_83, buf141, buf142, buf977, 128, 1024, grid=grid(128), stream=stream0)
        del primals_83
        buf143 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf142, reinterpret_tensor(primals_84, (1024, 1024), (1, 1024), 0), out=buf143)
        buf144 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf142, reinterpret_tensor(primals_86, (1024, 1024), (1, 1024), 0), out=buf144)
        buf145 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf142, reinterpret_tensor(primals_88, (1024, 1024), (1, 1024), 0), out=buf145)
        buf146 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf143, primals_85, buf146, 131072, grid=grid(131072), stream=stream0)
        del primals_85
        buf147 = reinterpret_tensor(buf143, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf145, primals_89, buf147, 131072, grid=grid(131072), stream=stream0)
        del primals_89
        buf148 = reinterpret_tensor(buf145, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf145  # reuse
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf144, primals_87, buf148, 131072, grid=grid(131072), stream=stream0)
        del primals_87
        buf149 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf148, (16, 64, 128), (8192, 1, 64), 0), out=buf149)
        buf150 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf151 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf152 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf149, buf150, buf151, buf152, 2048, 128, grid=grid(2048), stream=stream0)
        buf153 = reinterpret_tensor(buf144, (16, 128, 64), (8192, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf152, reinterpret_tensor(buf147, (16, 128, 64), (8192, 64, 1), 0), out=buf153)
        buf154 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf153, buf154, 131072, grid=grid(131072), stream=stream0)
        buf155 = reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf154, reinterpret_tensor(primals_90, (1024, 1024), (1, 1024), 0), out=buf155)
        buf156 = reinterpret_tensor(buf155, (1, 128, 1024), (131072, 1024, 1), 0); del buf155  # reuse
        buf160 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf161 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf976 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_61, l__mod___model_encoder_layers_5_fc1, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf156, buf129, buf137, primals_81, primals_91, primals_92, primals_93, buf160, buf161, buf976, 128, 1024, grid=grid(128), stream=stream0)
        del primals_81
        del primals_91
        del primals_93
        buf162 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf161, reinterpret_tensor(primals_94, (1024, 4096), (1, 1024), 0), out=buf162)
        buf163 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_62, hidden_states_64], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf162, primals_95, buf163, 524288, grid=grid(524288), stream=stream0)
        buf164 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf163, reinterpret_tensor(primals_96, (4096, 1024), (1, 4096), 0), out=buf164)
        buf168 = buf129; del buf129  # reuse
        buf169 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf974 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, l__mod___model_encoder_layers_6_self_attn_q_proj, residual_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf156, buf164, primals_97, primals_98, primals_99, buf168, buf169, buf974, 128, 1024, grid=grid(128), stream=stream0)
        del primals_99
        buf170 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_100, (1024, 1024), (1, 1024), 0), out=buf170)
        buf171 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_102, (1024, 1024), (1, 1024), 0), out=buf171)
        buf172 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_104, (1024, 1024), (1, 1024), 0), out=buf172)
        buf173 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf170, primals_101, buf173, 131072, grid=grid(131072), stream=stream0)
        del primals_101
        buf174 = reinterpret_tensor(buf170, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf172, primals_105, buf174, 131072, grid=grid(131072), stream=stream0)
        del primals_105
        buf175 = reinterpret_tensor(buf172, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf171, primals_103, buf175, 131072, grid=grid(131072), stream=stream0)
        del primals_103
        buf176 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf175, (16, 64, 128), (8192, 1, 64), 0), out=buf176)
        buf177 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf178 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf179 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf176, buf177, buf178, buf179, 2048, 128, grid=grid(2048), stream=stream0)
        buf180 = reinterpret_tensor(buf171, (16, 128, 64), (8192, 64, 1), 0); del buf171  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf179, reinterpret_tensor(buf174, (16, 128, 64), (8192, 64, 1), 0), out=buf180)
        buf181 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_69], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf180, buf181, 131072, grid=grid(131072), stream=stream0)
        buf182 = reinterpret_tensor(buf180, (128, 1024), (1024, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf181, reinterpret_tensor(primals_106, (1024, 1024), (1, 1024), 0), out=buf182)
        buf183 = reinterpret_tensor(buf182, (1, 128, 1024), (131072, 1024, 1), 0); del buf182  # reuse
        buf187 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf188 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf973 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72, l__mod___model_encoder_layers_6_fc1, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf183, buf156, buf164, primals_97, primals_107, primals_108, primals_109, buf187, buf188, buf973, 128, 1024, grid=grid(128), stream=stream0)
        del primals_107
        del primals_109
        del primals_97
        buf189 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf188, reinterpret_tensor(primals_110, (1024, 4096), (1, 1024), 0), out=buf189)
        buf190 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_73, hidden_states_75], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf189, primals_111, buf190, 524288, grid=grid(524288), stream=stream0)
        buf191 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf190, reinterpret_tensor(primals_112, (4096, 1024), (1, 4096), 0), out=buf191)
        buf195 = buf156; del buf156  # reuse
        buf196 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf971 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_79, l__mod___model_encoder_layers_7_self_attn_q_proj, residual_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf183, buf191, primals_113, primals_114, primals_115, buf195, buf196, buf971, 128, 1024, grid=grid(128), stream=stream0)
        del primals_115
        buf197 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf196, reinterpret_tensor(primals_116, (1024, 1024), (1, 1024), 0), out=buf197)
        buf198 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf196, reinterpret_tensor(primals_118, (1024, 1024), (1, 1024), 0), out=buf198)
        buf199 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf196, reinterpret_tensor(primals_120, (1024, 1024), (1, 1024), 0), out=buf199)
        buf200 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf197, primals_117, buf200, 131072, grid=grid(131072), stream=stream0)
        del primals_117
        buf201 = reinterpret_tensor(buf197, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf199, primals_121, buf201, 131072, grid=grid(131072), stream=stream0)
        del primals_121
        buf202 = reinterpret_tensor(buf199, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf199  # reuse
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf198, primals_119, buf202, 131072, grid=grid(131072), stream=stream0)
        del primals_119
        buf203 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf200, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf202, (16, 64, 128), (8192, 1, 64), 0), out=buf203)
        buf204 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf205 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf206 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf203, buf204, buf205, buf206, 2048, 128, grid=grid(2048), stream=stream0)
        buf207 = reinterpret_tensor(buf198, (16, 128, 64), (8192, 64, 1), 0); del buf198  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf206, reinterpret_tensor(buf201, (16, 128, 64), (8192, 64, 1), 0), out=buf207)
        buf208 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf207, buf208, 131072, grid=grid(131072), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (128, 1024), (1024, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf208, reinterpret_tensor(primals_122, (1024, 1024), (1, 1024), 0), out=buf209)
        buf210 = reinterpret_tensor(buf209, (1, 128, 1024), (131072, 1024, 1), 0); del buf209  # reuse
        buf214 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf215 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf970 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_83, l__mod___model_encoder_layers_7_fc1, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf210, buf183, buf191, primals_113, primals_123, primals_124, primals_125, buf214, buf215, buf970, 128, 1024, grid=grid(128), stream=stream0)
        del primals_113
        del primals_123
        del primals_125
        buf216 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf215, reinterpret_tensor(primals_126, (1024, 4096), (1, 1024), 0), out=buf216)
        buf217 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84, hidden_states_86], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf216, primals_127, buf217, 524288, grid=grid(524288), stream=stream0)
        buf218 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf217, reinterpret_tensor(primals_128, (4096, 1024), (1, 4096), 0), out=buf218)
        buf222 = buf183; del buf183  # reuse
        buf223 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf968 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90, l__mod___model_encoder_layers_8_self_attn_q_proj, residual_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf210, buf218, primals_129, primals_130, primals_131, buf222, buf223, buf968, 128, 1024, grid=grid(128), stream=stream0)
        del primals_131
        buf224 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf223, reinterpret_tensor(primals_132, (1024, 1024), (1, 1024), 0), out=buf224)
        buf225 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf223, reinterpret_tensor(primals_134, (1024, 1024), (1, 1024), 0), out=buf225)
        buf226 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf223, reinterpret_tensor(primals_136, (1024, 1024), (1, 1024), 0), out=buf226)
        buf227 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf224, primals_133, buf227, 131072, grid=grid(131072), stream=stream0)
        del primals_133
        buf228 = reinterpret_tensor(buf224, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf224  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf226, primals_137, buf228, 131072, grid=grid(131072), stream=stream0)
        del primals_137
        buf229 = reinterpret_tensor(buf226, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf226  # reuse
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf225, primals_135, buf229, 131072, grid=grid(131072), stream=stream0)
        del primals_135
        buf230 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf227, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf229, (16, 64, 128), (8192, 1, 64), 0), out=buf230)
        buf231 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf232 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf233 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf230, buf231, buf232, buf233, 2048, 128, grid=grid(2048), stream=stream0)
        buf234 = reinterpret_tensor(buf225, (16, 128, 64), (8192, 64, 1), 0); del buf225  # reuse
        # Source Nodes: [attn_output_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf233, reinterpret_tensor(buf228, (16, 128, 64), (8192, 64, 1), 0), out=buf234)
        buf235 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_91], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf234, buf235, 131072, grid=grid(131072), stream=stream0)
        buf236 = reinterpret_tensor(buf234, (128, 1024), (1024, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf235, reinterpret_tensor(primals_138, (1024, 1024), (1, 1024), 0), out=buf236)
        buf237 = reinterpret_tensor(buf236, (1, 128, 1024), (131072, 1024, 1), 0); del buf236  # reuse
        buf241 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf242 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf967 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_94, l__mod___model_encoder_layers_8_fc1, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf237, buf210, buf218, primals_129, primals_139, primals_140, primals_141, buf241, buf242, buf967, 128, 1024, grid=grid(128), stream=stream0)
        del primals_129
        del primals_139
        del primals_141
        buf243 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf242, reinterpret_tensor(primals_142, (1024, 4096), (1, 1024), 0), out=buf243)
        buf244 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, hidden_states_97], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf243, primals_143, buf244, 524288, grid=grid(524288), stream=stream0)
        buf245 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf244, reinterpret_tensor(primals_144, (4096, 1024), (1, 4096), 0), out=buf245)
        buf249 = buf210; del buf210  # reuse
        buf250 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf965 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_101, l__mod___model_encoder_layers_9_self_attn_q_proj, residual_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf237, buf245, primals_145, primals_146, primals_147, buf249, buf250, buf965, 128, 1024, grid=grid(128), stream=stream0)
        del primals_147
        buf251 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_148, (1024, 1024), (1, 1024), 0), out=buf251)
        buf252 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_150, (1024, 1024), (1, 1024), 0), out=buf252)
        buf253 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_152, (1024, 1024), (1, 1024), 0), out=buf253)
        buf254 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf251, primals_149, buf254, 131072, grid=grid(131072), stream=stream0)
        del primals_149
        buf255 = reinterpret_tensor(buf251, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf253, primals_153, buf255, 131072, grid=grid(131072), stream=stream0)
        del primals_153
        buf256 = reinterpret_tensor(buf253, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf252, primals_151, buf256, 131072, grid=grid(131072), stream=stream0)
        del primals_151
        buf257 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf254, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf256, (16, 64, 128), (8192, 1, 64), 0), out=buf257)
        buf258 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf259 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf260 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf257, buf258, buf259, buf260, 2048, 128, grid=grid(2048), stream=stream0)
        buf261 = reinterpret_tensor(buf252, (16, 128, 64), (8192, 64, 1), 0); del buf252  # reuse
        # Source Nodes: [attn_output_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf260, reinterpret_tensor(buf255, (16, 128, 64), (8192, 64, 1), 0), out=buf261)
        buf262 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf261, buf262, 131072, grid=grid(131072), stream=stream0)
        buf263 = reinterpret_tensor(buf261, (128, 1024), (1024, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf262, reinterpret_tensor(primals_154, (1024, 1024), (1, 1024), 0), out=buf263)
        buf264 = reinterpret_tensor(buf263, (1, 128, 1024), (131072, 1024, 1), 0); del buf263  # reuse
        buf268 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf269 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf964 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_105, l__mod___model_encoder_layers_9_fc1, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf264, buf237, buf245, primals_145, primals_155, primals_156, primals_157, buf268, buf269, buf964, 128, 1024, grid=grid(128), stream=stream0)
        del primals_145
        del primals_155
        del primals_157
        buf270 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf269, reinterpret_tensor(primals_158, (1024, 4096), (1, 1024), 0), out=buf270)
        buf271 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_106, hidden_states_108], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf270, primals_159, buf271, 524288, grid=grid(524288), stream=stream0)
        buf272 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_160, (4096, 1024), (1, 4096), 0), out=buf272)
        buf276 = buf237; del buf237  # reuse
        buf277 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf962 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_112, l__mod___model_encoder_layers_10_self_attn_q_proj, residual_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf264, buf272, primals_161, primals_162, primals_163, buf276, buf277, buf962, 128, 1024, grid=grid(128), stream=stream0)
        del primals_163
        buf278 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf277, reinterpret_tensor(primals_164, (1024, 1024), (1, 1024), 0), out=buf278)
        buf279 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf277, reinterpret_tensor(primals_166, (1024, 1024), (1, 1024), 0), out=buf279)
        buf280 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf277, reinterpret_tensor(primals_168, (1024, 1024), (1, 1024), 0), out=buf280)
        buf281 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf278, primals_165, buf281, 131072, grid=grid(131072), stream=stream0)
        del primals_165
        buf282 = reinterpret_tensor(buf278, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf280, primals_169, buf282, 131072, grid=grid(131072), stream=stream0)
        del primals_169
        buf283 = reinterpret_tensor(buf280, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf280  # reuse
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf279, primals_167, buf283, 131072, grid=grid(131072), stream=stream0)
        del primals_167
        buf284 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf283, (16, 64, 128), (8192, 1, 64), 0), out=buf284)
        buf285 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf286 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf287 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf284, buf285, buf286, buf287, 2048, 128, grid=grid(2048), stream=stream0)
        buf288 = reinterpret_tensor(buf279, (16, 128, 64), (8192, 64, 1), 0); del buf279  # reuse
        # Source Nodes: [attn_output_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf287, reinterpret_tensor(buf282, (16, 128, 64), (8192, 64, 1), 0), out=buf288)
        buf289 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_113], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf288, buf289, 131072, grid=grid(131072), stream=stream0)
        buf290 = reinterpret_tensor(buf288, (128, 1024), (1024, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf289, reinterpret_tensor(primals_170, (1024, 1024), (1, 1024), 0), out=buf290)
        buf291 = reinterpret_tensor(buf290, (1, 128, 1024), (131072, 1024, 1), 0); del buf290  # reuse
        buf295 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf296 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf961 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_116, l__mod___model_encoder_layers_10_fc1, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf291, buf264, buf272, primals_161, primals_171, primals_172, primals_173, buf295, buf296, buf961, 128, 1024, grid=grid(128), stream=stream0)
        del primals_161
        del primals_171
        del primals_173
        buf297 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf296, reinterpret_tensor(primals_174, (1024, 4096), (1, 1024), 0), out=buf297)
        buf298 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_117, hidden_states_119], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf297, primals_175, buf298, 524288, grid=grid(524288), stream=stream0)
        buf299 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf298, reinterpret_tensor(primals_176, (4096, 1024), (1, 4096), 0), out=buf299)
        buf303 = buf264; del buf264  # reuse
        buf304 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf959 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123, l__mod___model_encoder_layers_11_self_attn_q_proj, residual_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf291, buf299, primals_177, primals_178, primals_179, buf303, buf304, buf959, 128, 1024, grid=grid(128), stream=stream0)
        del primals_179
        buf305 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf304, reinterpret_tensor(primals_180, (1024, 1024), (1, 1024), 0), out=buf305)
        buf306 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf304, reinterpret_tensor(primals_182, (1024, 1024), (1, 1024), 0), out=buf306)
        buf307 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf304, reinterpret_tensor(primals_184, (1024, 1024), (1, 1024), 0), out=buf307)
        buf308 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf305, primals_181, buf308, 131072, grid=grid(131072), stream=stream0)
        del primals_181
        buf309 = reinterpret_tensor(buf305, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf305  # reuse
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf307, primals_185, buf309, 131072, grid=grid(131072), stream=stream0)
        del primals_185
        buf310 = reinterpret_tensor(buf307, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf307  # reuse
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf306, primals_183, buf310, 131072, grid=grid(131072), stream=stream0)
        del primals_183
        buf311 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf308, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf310, (16, 64, 128), (8192, 1, 64), 0), out=buf311)
        buf312 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf313 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf314 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf311, buf312, buf313, buf314, 2048, 128, grid=grid(2048), stream=stream0)
        buf315 = reinterpret_tensor(buf306, (16, 128, 64), (8192, 64, 1), 0); del buf306  # reuse
        # Source Nodes: [attn_output_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf314, reinterpret_tensor(buf309, (16, 128, 64), (8192, 64, 1), 0), out=buf315)
        buf316 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_124], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf315, buf316, 131072, grid=grid(131072), stream=stream0)
        buf317 = reinterpret_tensor(buf315, (128, 1024), (1024, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf316, reinterpret_tensor(primals_186, (1024, 1024), (1, 1024), 0), out=buf317)
        buf318 = reinterpret_tensor(buf317, (1, 128, 1024), (131072, 1024, 1), 0); del buf317  # reuse
        buf322 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf323 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf958 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_127, l__mod___model_encoder_layers_11_fc1, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf318, buf291, buf299, primals_177, primals_187, primals_188, primals_189, buf322, buf323, buf958, 128, 1024, grid=grid(128), stream=stream0)
        del primals_177
        del primals_187
        del primals_189
        buf324 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf323, reinterpret_tensor(primals_190, (1024, 4096), (1, 1024), 0), out=buf324)
        buf325 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_128, hidden_states_130], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf324, primals_191, buf325, 524288, grid=grid(524288), stream=stream0)
        buf326 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf325, reinterpret_tensor(primals_192, (4096, 1024), (1, 4096), 0), out=buf326)
        buf330 = buf291; del buf291  # reuse
        buf331 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf956 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_133, hidden_states_134], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf318, buf326, primals_193, primals_194, primals_195, buf330, buf331, buf956, 128, 1024, grid=grid(128), stream=stream0)
        del primals_193
        del primals_195
        buf332 = buf0; del buf0  # reuse
        # Source Nodes: [cumsum_1, mask_3, ne_1], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        triton_poi_fused__to_copy_cumsum_ne_0.run(primals_515, buf332, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [cumsum_1, mask_3, ne_1], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf333 = aten.cumsum(buf332, 1)
        del buf332
        buf334 = buf333
        del buf333
        buf338 = reinterpret_tensor(buf326, (1, 128, 1024), (131072, 1024, 1), 0); del buf326  # reuse
        buf339 = reinterpret_tensor(buf318, (128, 1024), (1024, 1), 0); del buf318  # reuse
        buf955 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_135, hidden_states_137, inputs_embeds_1, l__mod___model_decoder_embed_tokens, l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1.run(primals_515, primals_196, buf334, primals_513, primals_197, primals_198, buf338, buf339, buf955, 128, 1024, grid=grid(128), stream=stream0)
        del primals_198
        buf340 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf339, reinterpret_tensor(primals_199, (1024, 1024), (1, 1024), 0), out=buf340)
        buf341 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf339, reinterpret_tensor(primals_201, (1024, 1024), (1, 1024), 0), out=buf341)
        buf342 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf341, primals_202, buf342, 131072, grid=grid(131072), stream=stream0)
        del primals_202
        buf343 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf339, reinterpret_tensor(primals_203, (1024, 1024), (1, 1024), 0), out=buf343)
        buf344 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf343, primals_204, buf344, 131072, grid=grid(131072), stream=stream0)
        del primals_204
        buf345 = reinterpret_tensor(buf343, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf343  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf340, primals_200, buf345, 131072, grid=grid(131072), stream=stream0)
        del primals_200
        buf346 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf345, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf342, (16, 64, 128), (8192, 1, 64), 0), out=buf346)
        buf349 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf954 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_12, attn_weights_27], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf346, buf349, buf954, 2048, 128, grid=grid(2048), stream=stream0)
        buf350 = reinterpret_tensor(buf340, (16, 128, 64), (8192, 64, 1), 0); del buf340  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf349, reinterpret_tensor(buf344, (16, 128, 64), (8192, 64, 1), 0), out=buf350)
        buf351 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_138], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf350, buf351, 131072, grid=grid(131072), stream=stream0)
        buf352 = reinterpret_tensor(buf350, (128, 1024), (1024, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf351, reinterpret_tensor(primals_205, (1024, 1024), (1, 1024), 0), out=buf352)
        buf353 = reinterpret_tensor(buf352, (1, 128, 1024), (131072, 1024, 1), 0); del buf352  # reuse
        buf357 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf358 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf953 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_135, hidden_states_141, inputs_embeds_1, l__mod___model_decoder_embed_tokens, l__mod___model_decoder_layers_0_encoder_attn_q_proj, residual_25], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_6.run(buf353, primals_515, primals_196, buf334, primals_513, primals_206, primals_207, primals_208, buf357, buf358, buf953, 128, 1024, grid=grid(128), stream=stream0)
        del buf334
        del primals_196
        del primals_206
        del primals_208
        del primals_513
        buf359 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf358, reinterpret_tensor(primals_209, (1024, 1024), (1, 1024), 0), out=buf359)
        buf360 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_211, (1024, 1024), (1, 1024), 0), out=buf360)
        buf361 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf360, primals_212, buf361, 131072, grid=grid(131072), stream=stream0)
        del primals_212
        buf362 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_213, (1024, 1024), (1, 1024), 0), out=buf362)
        buf363 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf362, primals_214, buf363, 131072, grid=grid(131072), stream=stream0)
        del primals_214
        buf364 = reinterpret_tensor(buf362, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf362  # reuse
        # Source Nodes: [contiguous_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf359, primals_210, buf364, 131072, grid=grid(131072), stream=stream0)
        del primals_210
        buf365 = buf346; del buf346  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf361, (16, 64, 128), (8192, 1, 64), 0), out=buf365)
        buf366 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf367 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf368 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf365, buf366, buf367, buf368, 2048, 128, grid=grid(2048), stream=stream0)
        buf369 = reinterpret_tensor(buf359, (16, 128, 64), (8192, 64, 1), 0); del buf359  # reuse
        # Source Nodes: [attn_output_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf368, reinterpret_tensor(buf363, (16, 128, 64), (8192, 64, 1), 0), out=buf369)
        buf370 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_142], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf369, buf370, 131072, grid=grid(131072), stream=stream0)
        buf371 = reinterpret_tensor(buf369, (128, 1024), (1024, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf370, reinterpret_tensor(primals_215, (1024, 1024), (1, 1024), 0), out=buf371)
        buf375 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf376 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf952 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_145, l__mod___model_decoder_layers_0_fc1, residual_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf353, buf371, primals_216, primals_217, primals_218, buf375, buf376, buf952, 128, 1024, grid=grid(128), stream=stream0)
        del primals_218
        buf377 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf376, reinterpret_tensor(primals_219, (1024, 4096), (1, 1024), 0), out=buf377)
        buf378 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_146, hidden_states_148], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf377, primals_220, buf378, 524288, grid=grid(524288), stream=stream0)
        buf379 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf378, reinterpret_tensor(primals_221, (4096, 1024), (1, 4096), 0), out=buf379)
        buf380 = reinterpret_tensor(buf379, (1, 128, 1024), (131072, 1024, 1), 0); del buf379  # reuse
        buf384 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf385 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf950 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_152, l__mod___model_decoder_layers_1_self_attn_q_proj, residual_26, residual_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf380, buf353, buf371, primals_216, primals_222, primals_223, primals_224, buf384, buf385, buf950, 128, 1024, grid=grid(128), stream=stream0)
        del primals_216
        del primals_222
        del primals_224
        buf386 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf385, reinterpret_tensor(primals_225, (1024, 1024), (1, 1024), 0), out=buf386)
        buf387 = reinterpret_tensor(buf353, (128, 1024), (1024, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf385, reinterpret_tensor(primals_227, (1024, 1024), (1, 1024), 0), out=buf387)
        buf388 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf387, primals_228, buf388, 131072, grid=grid(131072), stream=stream0)
        del primals_228
        buf389 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf385, reinterpret_tensor(primals_229, (1024, 1024), (1, 1024), 0), out=buf389)
        buf390 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf389, primals_230, buf390, 131072, grid=grid(131072), stream=stream0)
        del primals_230
        buf391 = reinterpret_tensor(buf389, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf389  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf386, primals_226, buf391, 131072, grid=grid(131072), stream=stream0)
        del primals_226
        buf392 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf388, (16, 64, 128), (8192, 1, 64), 0), out=buf392)
        buf395 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf949 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_14, attn_weights_33], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf392, buf395, buf949, 2048, 128, grid=grid(2048), stream=stream0)
        buf396 = reinterpret_tensor(buf386, (16, 128, 64), (8192, 64, 1), 0); del buf386  # reuse
        # Source Nodes: [attn_output_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf395, reinterpret_tensor(buf390, (16, 128, 64), (8192, 64, 1), 0), out=buf396)
        buf397 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_153], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf396, buf397, 131072, grid=grid(131072), stream=stream0)
        buf398 = reinterpret_tensor(buf396, (128, 1024), (1024, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf397, reinterpret_tensor(primals_231, (1024, 1024), (1, 1024), 0), out=buf398)
        buf402 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf403 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf948 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, l__mod___model_decoder_layers_1_encoder_attn_q_proj, residual_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf380, buf398, primals_232, primals_233, primals_234, buf402, buf403, buf948, 128, 1024, grid=grid(128), stream=stream0)
        del primals_234
        buf404 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf403, reinterpret_tensor(primals_235, (1024, 1024), (1, 1024), 0), out=buf404)
        buf405 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_237, (1024, 1024), (1, 1024), 0), out=buf405)
        buf406 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf405, primals_238, buf406, 131072, grid=grid(131072), stream=stream0)
        del primals_238
        buf407 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_239, (1024, 1024), (1, 1024), 0), out=buf407)
        buf408 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf407, primals_240, buf408, 131072, grid=grid(131072), stream=stream0)
        del primals_240
        buf409 = reinterpret_tensor(buf407, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf407  # reuse
        # Source Nodes: [contiguous_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf404, primals_236, buf409, 131072, grid=grid(131072), stream=stream0)
        del primals_236
        buf410 = buf392; del buf392  # reuse
        # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf406, (16, 64, 128), (8192, 1, 64), 0), out=buf410)
        buf411 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf412 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf413 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf410, buf411, buf412, buf413, 2048, 128, grid=grid(2048), stream=stream0)
        buf414 = reinterpret_tensor(buf404, (16, 128, 64), (8192, 64, 1), 0); del buf404  # reuse
        # Source Nodes: [attn_output_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf413, reinterpret_tensor(buf408, (16, 128, 64), (8192, 64, 1), 0), out=buf414)
        buf415 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_157], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf414, buf415, 131072, grid=grid(131072), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (128, 1024), (1024, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf415, reinterpret_tensor(primals_241, (1024, 1024), (1, 1024), 0), out=buf416)
        buf417 = reinterpret_tensor(buf416, (1, 128, 1024), (131072, 1024, 1), 0); del buf416  # reuse
        buf421 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf422 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf947 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_160, l__mod___model_decoder_layers_1_fc1, residual_28, residual_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf417, buf380, buf398, primals_232, primals_242, primals_243, primals_244, buf421, buf422, buf947, 128, 1024, grid=grid(128), stream=stream0)
        del primals_232
        del primals_242
        del primals_244
        buf423 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf422, reinterpret_tensor(primals_245, (1024, 4096), (1, 1024), 0), out=buf423)
        buf424 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_161, hidden_states_163], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf423, primals_246, buf424, 524288, grid=grid(524288), stream=stream0)
        buf425 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf424, reinterpret_tensor(primals_247, (4096, 1024), (1, 4096), 0), out=buf425)
        buf429 = buf380; del buf380  # reuse
        buf430 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf945 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_167, l__mod___model_decoder_layers_2_self_attn_q_proj, residual_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf417, buf425, primals_248, primals_249, primals_250, buf429, buf430, buf945, 128, 1024, grid=grid(128), stream=stream0)
        del primals_250
        buf431 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf430, reinterpret_tensor(primals_251, (1024, 1024), (1, 1024), 0), out=buf431)
        buf432 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf430, reinterpret_tensor(primals_253, (1024, 1024), (1, 1024), 0), out=buf432)
        buf433 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf432, primals_254, buf433, 131072, grid=grid(131072), stream=stream0)
        del primals_254
        buf434 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf430, reinterpret_tensor(primals_255, (1024, 1024), (1, 1024), 0), out=buf434)
        buf435 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf434, primals_256, buf435, 131072, grid=grid(131072), stream=stream0)
        del primals_256
        buf436 = reinterpret_tensor(buf434, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf434  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf431, primals_252, buf436, 131072, grid=grid(131072), stream=stream0)
        del primals_252
        buf437 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf433, (16, 64, 128), (8192, 1, 64), 0), out=buf437)
        buf440 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf944 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_16, attn_weights_39], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf437, buf440, buf944, 2048, 128, grid=grid(2048), stream=stream0)
        buf441 = reinterpret_tensor(buf431, (16, 128, 64), (8192, 64, 1), 0); del buf431  # reuse
        # Source Nodes: [attn_output_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf440, reinterpret_tensor(buf435, (16, 128, 64), (8192, 64, 1), 0), out=buf441)
        buf442 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_168], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf441, buf442, 131072, grid=grid(131072), stream=stream0)
        buf443 = reinterpret_tensor(buf441, (128, 1024), (1024, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf442, reinterpret_tensor(primals_257, (1024, 1024), (1, 1024), 0), out=buf443)
        buf444 = reinterpret_tensor(buf443, (1, 128, 1024), (131072, 1024, 1), 0); del buf443  # reuse
        buf448 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf449 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf943 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171, l__mod___model_decoder_layers_2_encoder_attn_q_proj, residual_30, residual_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf444, buf417, buf425, primals_248, primals_258, primals_259, primals_260, buf448, buf449, buf943, 128, 1024, grid=grid(128), stream=stream0)
        del primals_248
        del primals_258
        del primals_260
        buf450 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf449, reinterpret_tensor(primals_261, (1024, 1024), (1, 1024), 0), out=buf450)
        buf451 = reinterpret_tensor(buf417, (128, 1024), (1024, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_263, (1024, 1024), (1, 1024), 0), out=buf451)
        buf452 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf451, primals_264, buf452, 131072, grid=grid(131072), stream=stream0)
        del primals_264
        buf453 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_265, (1024, 1024), (1, 1024), 0), out=buf453)
        buf454 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf453, primals_266, buf454, 131072, grid=grid(131072), stream=stream0)
        del primals_266
        buf455 = reinterpret_tensor(buf453, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf453  # reuse
        # Source Nodes: [contiguous_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf450, primals_262, buf455, 131072, grid=grid(131072), stream=stream0)
        del primals_262
        buf456 = buf437; del buf437  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf455, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf452, (16, 64, 128), (8192, 1, 64), 0), out=buf456)
        buf457 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf458 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf459 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_41], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf456, buf457, buf458, buf459, 2048, 128, grid=grid(2048), stream=stream0)
        buf460 = reinterpret_tensor(buf450, (16, 128, 64), (8192, 64, 1), 0); del buf450  # reuse
        # Source Nodes: [attn_output_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf459, reinterpret_tensor(buf454, (16, 128, 64), (8192, 64, 1), 0), out=buf460)
        buf461 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_172], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf460, buf461, 131072, grid=grid(131072), stream=stream0)
        buf462 = reinterpret_tensor(buf460, (128, 1024), (1024, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf461, reinterpret_tensor(primals_267, (1024, 1024), (1, 1024), 0), out=buf462)
        buf466 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf467 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf942 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_175, l__mod___model_decoder_layers_2_fc1, residual_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf444, buf462, primals_268, primals_269, primals_270, buf466, buf467, buf942, 128, 1024, grid=grid(128), stream=stream0)
        del primals_270
        buf468 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf467, reinterpret_tensor(primals_271, (1024, 4096), (1, 1024), 0), out=buf468)
        buf469 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_176, hidden_states_178], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf468, primals_272, buf469, 524288, grid=grid(524288), stream=stream0)
        buf470 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf469, reinterpret_tensor(primals_273, (4096, 1024), (1, 4096), 0), out=buf470)
        buf471 = reinterpret_tensor(buf470, (1, 128, 1024), (131072, 1024, 1), 0); del buf470  # reuse
        buf475 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf476 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf940 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_182, l__mod___model_decoder_layers_3_self_attn_q_proj, residual_32, residual_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf471, buf444, buf462, primals_268, primals_274, primals_275, primals_276, buf475, buf476, buf940, 128, 1024, grid=grid(128), stream=stream0)
        del primals_268
        del primals_274
        del primals_276
        buf477 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf476, reinterpret_tensor(primals_277, (1024, 1024), (1, 1024), 0), out=buf477)
        buf478 = reinterpret_tensor(buf444, (128, 1024), (1024, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf476, reinterpret_tensor(primals_279, (1024, 1024), (1, 1024), 0), out=buf478)
        buf479 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf478, primals_280, buf479, 131072, grid=grid(131072), stream=stream0)
        del primals_280
        buf480 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf476, reinterpret_tensor(primals_281, (1024, 1024), (1, 1024), 0), out=buf480)
        buf481 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf480, primals_282, buf481, 131072, grid=grid(131072), stream=stream0)
        del primals_282
        buf482 = reinterpret_tensor(buf480, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf480  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf477, primals_278, buf482, 131072, grid=grid(131072), stream=stream0)
        del primals_278
        buf483 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf482, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf479, (16, 64, 128), (8192, 1, 64), 0), out=buf483)
        buf486 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf939 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_18, attn_weights_45], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf483, buf486, buf939, 2048, 128, grid=grid(2048), stream=stream0)
        buf487 = reinterpret_tensor(buf477, (16, 128, 64), (8192, 64, 1), 0); del buf477  # reuse
        # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf486, reinterpret_tensor(buf481, (16, 128, 64), (8192, 64, 1), 0), out=buf487)
        buf488 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_183], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf487, buf488, 131072, grid=grid(131072), stream=stream0)
        buf489 = reinterpret_tensor(buf487, (128, 1024), (1024, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf488, reinterpret_tensor(primals_283, (1024, 1024), (1, 1024), 0), out=buf489)
        buf493 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf494 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf938 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_186, l__mod___model_decoder_layers_3_encoder_attn_q_proj, residual_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf471, buf489, primals_284, primals_285, primals_286, buf493, buf494, buf938, 128, 1024, grid=grid(128), stream=stream0)
        del primals_286
        buf495 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf494, reinterpret_tensor(primals_287, (1024, 1024), (1, 1024), 0), out=buf495)
        buf496 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_289, (1024, 1024), (1, 1024), 0), out=buf496)
        buf497 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf496, primals_290, buf497, 131072, grid=grid(131072), stream=stream0)
        del primals_290
        buf498 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_291, (1024, 1024), (1, 1024), 0), out=buf498)
        buf499 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf498, primals_292, buf499, 131072, grid=grid(131072), stream=stream0)
        del primals_292
        buf500 = reinterpret_tensor(buf498, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf498  # reuse
        # Source Nodes: [contiguous_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf495, primals_288, buf500, 131072, grid=grid(131072), stream=stream0)
        del primals_288
        buf501 = buf483; del buf483  # reuse
        # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf500, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf497, (16, 64, 128), (8192, 1, 64), 0), out=buf501)
        buf502 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf503 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf504 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf501, buf502, buf503, buf504, 2048, 128, grid=grid(2048), stream=stream0)
        buf505 = reinterpret_tensor(buf495, (16, 128, 64), (8192, 64, 1), 0); del buf495  # reuse
        # Source Nodes: [attn_output_95], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf504, reinterpret_tensor(buf499, (16, 128, 64), (8192, 64, 1), 0), out=buf505)
        buf506 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_187], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf505, buf506, 131072, grid=grid(131072), stream=stream0)
        buf507 = reinterpret_tensor(buf505, (128, 1024), (1024, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf506, reinterpret_tensor(primals_293, (1024, 1024), (1, 1024), 0), out=buf507)
        buf508 = reinterpret_tensor(buf507, (1, 128, 1024), (131072, 1024, 1), 0); del buf507  # reuse
        buf512 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf513 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf937 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_190, l__mod___model_decoder_layers_3_fc1, residual_34, residual_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf508, buf471, buf489, primals_284, primals_294, primals_295, primals_296, buf512, buf513, buf937, 128, 1024, grid=grid(128), stream=stream0)
        del primals_284
        del primals_294
        del primals_296
        buf514 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf513, reinterpret_tensor(primals_297, (1024, 4096), (1, 1024), 0), out=buf514)
        buf515 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_191, hidden_states_193], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf514, primals_298, buf515, 524288, grid=grid(524288), stream=stream0)
        buf516 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf515, reinterpret_tensor(primals_299, (4096, 1024), (1, 4096), 0), out=buf516)
        buf520 = buf471; del buf471  # reuse
        buf521 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf935 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_197, l__mod___model_decoder_layers_4_self_attn_q_proj, residual_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf508, buf516, primals_300, primals_301, primals_302, buf520, buf521, buf935, 128, 1024, grid=grid(128), stream=stream0)
        del primals_302
        buf522 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf521, reinterpret_tensor(primals_303, (1024, 1024), (1, 1024), 0), out=buf522)
        buf523 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf521, reinterpret_tensor(primals_305, (1024, 1024), (1, 1024), 0), out=buf523)
        buf524 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf523, primals_306, buf524, 131072, grid=grid(131072), stream=stream0)
        del primals_306
        buf525 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf521, reinterpret_tensor(primals_307, (1024, 1024), (1, 1024), 0), out=buf525)
        buf526 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf525, primals_308, buf526, 131072, grid=grid(131072), stream=stream0)
        del primals_308
        buf527 = reinterpret_tensor(buf525, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf525  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf522, primals_304, buf527, 131072, grid=grid(131072), stream=stream0)
        del primals_304
        buf528 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf527, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf524, (16, 64, 128), (8192, 1, 64), 0), out=buf528)
        buf531 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf934 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_20, attn_weights_51], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf528, buf531, buf934, 2048, 128, grid=grid(2048), stream=stream0)
        buf532 = reinterpret_tensor(buf522, (16, 128, 64), (8192, 64, 1), 0); del buf522  # reuse
        # Source Nodes: [attn_output_100], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf531, reinterpret_tensor(buf526, (16, 128, 64), (8192, 64, 1), 0), out=buf532)
        buf533 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_198], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf532, buf533, 131072, grid=grid(131072), stream=stream0)
        buf534 = reinterpret_tensor(buf532, (128, 1024), (1024, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf533, reinterpret_tensor(primals_309, (1024, 1024), (1, 1024), 0), out=buf534)
        buf535 = reinterpret_tensor(buf534, (1, 128, 1024), (131072, 1024, 1), 0); del buf534  # reuse
        buf539 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf540 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf933 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_201, l__mod___model_decoder_layers_4_encoder_attn_q_proj, residual_36, residual_37], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf535, buf508, buf516, primals_300, primals_310, primals_311, primals_312, buf539, buf540, buf933, 128, 1024, grid=grid(128), stream=stream0)
        del primals_300
        del primals_310
        del primals_312
        buf541 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf540, reinterpret_tensor(primals_313, (1024, 1024), (1, 1024), 0), out=buf541)
        buf542 = reinterpret_tensor(buf508, (128, 1024), (1024, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_315, (1024, 1024), (1, 1024), 0), out=buf542)
        buf543 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf542, primals_316, buf543, 131072, grid=grid(131072), stream=stream0)
        del primals_316
        buf544 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_317, (1024, 1024), (1, 1024), 0), out=buf544)
        buf545 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf544, primals_318, buf545, 131072, grid=grid(131072), stream=stream0)
        del primals_318
        buf546 = reinterpret_tensor(buf544, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf544  # reuse
        # Source Nodes: [contiguous_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf541, primals_314, buf546, 131072, grid=grid(131072), stream=stream0)
        del primals_314
        buf547 = buf528; del buf528  # reuse
        # Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf546, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf543, (16, 64, 128), (8192, 1, 64), 0), out=buf547)
        buf548 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf549 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf550 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_53], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf547, buf548, buf549, buf550, 2048, 128, grid=grid(2048), stream=stream0)
        buf551 = reinterpret_tensor(buf541, (16, 128, 64), (8192, 64, 1), 0); del buf541  # reuse
        # Source Nodes: [attn_output_105], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf550, reinterpret_tensor(buf545, (16, 128, 64), (8192, 64, 1), 0), out=buf551)
        buf552 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_202], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf551, buf552, 131072, grid=grid(131072), stream=stream0)
        buf553 = reinterpret_tensor(buf551, (128, 1024), (1024, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf552, reinterpret_tensor(primals_319, (1024, 1024), (1, 1024), 0), out=buf553)
        buf557 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf558 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf932 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_205, l__mod___model_decoder_layers_4_fc1, residual_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf535, buf553, primals_320, primals_321, primals_322, buf557, buf558, buf932, 128, 1024, grid=grid(128), stream=stream0)
        del primals_322
        buf559 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf558, reinterpret_tensor(primals_323, (1024, 4096), (1, 1024), 0), out=buf559)
        buf560 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_206, hidden_states_208], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf559, primals_324, buf560, 524288, grid=grid(524288), stream=stream0)
        buf561 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf560, reinterpret_tensor(primals_325, (4096, 1024), (1, 4096), 0), out=buf561)
        buf562 = reinterpret_tensor(buf561, (1, 128, 1024), (131072, 1024, 1), 0); del buf561  # reuse
        buf566 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf567 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf930 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_212, l__mod___model_decoder_layers_5_self_attn_q_proj, residual_38, residual_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf562, buf535, buf553, primals_320, primals_326, primals_327, primals_328, buf566, buf567, buf930, 128, 1024, grid=grid(128), stream=stream0)
        del primals_320
        del primals_326
        del primals_328
        buf568 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf567, reinterpret_tensor(primals_329, (1024, 1024), (1, 1024), 0), out=buf568)
        buf569 = reinterpret_tensor(buf535, (128, 1024), (1024, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf567, reinterpret_tensor(primals_331, (1024, 1024), (1, 1024), 0), out=buf569)
        buf570 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf569, primals_332, buf570, 131072, grid=grid(131072), stream=stream0)
        del primals_332
        buf571 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf567, reinterpret_tensor(primals_333, (1024, 1024), (1, 1024), 0), out=buf571)
        buf572 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf571, primals_334, buf572, 131072, grid=grid(131072), stream=stream0)
        del primals_334
        buf573 = reinterpret_tensor(buf571, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf571  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf568, primals_330, buf573, 131072, grid=grid(131072), stream=stream0)
        del primals_330
        buf574 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf573, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf570, (16, 64, 128), (8192, 1, 64), 0), out=buf574)
        buf577 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf929 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_22, attn_weights_57], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf574, buf577, buf929, 2048, 128, grid=grid(2048), stream=stream0)
        buf578 = reinterpret_tensor(buf568, (16, 128, 64), (8192, 64, 1), 0); del buf568  # reuse
        # Source Nodes: [attn_output_110], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf577, reinterpret_tensor(buf572, (16, 128, 64), (8192, 64, 1), 0), out=buf578)
        buf579 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_213], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf578, buf579, 131072, grid=grid(131072), stream=stream0)
        buf580 = reinterpret_tensor(buf578, (128, 1024), (1024, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf579, reinterpret_tensor(primals_335, (1024, 1024), (1, 1024), 0), out=buf580)
        buf584 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf585 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf928 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_216, l__mod___model_decoder_layers_5_encoder_attn_q_proj, residual_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf562, buf580, primals_336, primals_337, primals_338, buf584, buf585, buf928, 128, 1024, grid=grid(128), stream=stream0)
        del primals_338
        buf586 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf585, reinterpret_tensor(primals_339, (1024, 1024), (1, 1024), 0), out=buf586)
        buf587 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_341, (1024, 1024), (1, 1024), 0), out=buf587)
        buf588 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf587, primals_342, buf588, 131072, grid=grid(131072), stream=stream0)
        del primals_342
        buf589 = buf587; del buf587  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_343, (1024, 1024), (1, 1024), 0), out=buf589)
        buf590 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf589, primals_344, buf590, 131072, grid=grid(131072), stream=stream0)
        del primals_344
        buf591 = reinterpret_tensor(buf589, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf589  # reuse
        # Source Nodes: [contiguous_71], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf586, primals_340, buf591, 131072, grid=grid(131072), stream=stream0)
        del primals_340
        buf592 = buf574; del buf574  # reuse
        # Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf591, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf588, (16, 64, 128), (8192, 1, 64), 0), out=buf592)
        buf593 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf594 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf595 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_59], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf592, buf593, buf594, buf595, 2048, 128, grid=grid(2048), stream=stream0)
        buf596 = reinterpret_tensor(buf586, (16, 128, 64), (8192, 64, 1), 0); del buf586  # reuse
        # Source Nodes: [attn_output_115], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf595, reinterpret_tensor(buf590, (16, 128, 64), (8192, 64, 1), 0), out=buf596)
        buf597 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_217], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf596, buf597, 131072, grid=grid(131072), stream=stream0)
        buf598 = reinterpret_tensor(buf596, (128, 1024), (1024, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf597, reinterpret_tensor(primals_345, (1024, 1024), (1, 1024), 0), out=buf598)
        buf599 = reinterpret_tensor(buf598, (1, 128, 1024), (131072, 1024, 1), 0); del buf598  # reuse
        buf603 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf604 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf927 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_220, l__mod___model_decoder_layers_5_fc1, residual_40, residual_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf599, buf562, buf580, primals_336, primals_346, primals_347, primals_348, buf603, buf604, buf927, 128, 1024, grid=grid(128), stream=stream0)
        del primals_336
        del primals_346
        del primals_348
        buf605 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf604, reinterpret_tensor(primals_349, (1024, 4096), (1, 1024), 0), out=buf605)
        buf606 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_221, hidden_states_223], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf605, primals_350, buf606, 524288, grid=grid(524288), stream=stream0)
        buf607 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf606, reinterpret_tensor(primals_351, (4096, 1024), (1, 4096), 0), out=buf607)
        buf611 = buf562; del buf562  # reuse
        buf612 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf925 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_227, l__mod___model_decoder_layers_6_self_attn_q_proj, residual_42], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf599, buf607, primals_352, primals_353, primals_354, buf611, buf612, buf925, 128, 1024, grid=grid(128), stream=stream0)
        del primals_354
        buf613 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf612, reinterpret_tensor(primals_355, (1024, 1024), (1, 1024), 0), out=buf613)
        buf614 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf612, reinterpret_tensor(primals_357, (1024, 1024), (1, 1024), 0), out=buf614)
        buf615 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf614, primals_358, buf615, 131072, grid=grid(131072), stream=stream0)
        del primals_358
        buf616 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf612, reinterpret_tensor(primals_359, (1024, 1024), (1, 1024), 0), out=buf616)
        buf617 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf616, primals_360, buf617, 131072, grid=grid(131072), stream=stream0)
        del primals_360
        buf618 = reinterpret_tensor(buf616, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf616  # reuse
        # Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf613, primals_356, buf618, 131072, grid=grid(131072), stream=stream0)
        del primals_356
        buf619 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf618, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf615, (16, 64, 128), (8192, 1, 64), 0), out=buf619)
        buf622 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf924 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_24, attn_weights_63], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf619, buf622, buf924, 2048, 128, grid=grid(2048), stream=stream0)
        buf623 = reinterpret_tensor(buf613, (16, 128, 64), (8192, 64, 1), 0); del buf613  # reuse
        # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf622, reinterpret_tensor(buf617, (16, 128, 64), (8192, 64, 1), 0), out=buf623)
        buf624 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_228], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf623, buf624, 131072, grid=grid(131072), stream=stream0)
        buf625 = reinterpret_tensor(buf623, (128, 1024), (1024, 1), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf624, reinterpret_tensor(primals_361, (1024, 1024), (1, 1024), 0), out=buf625)
        buf626 = reinterpret_tensor(buf625, (1, 128, 1024), (131072, 1024, 1), 0); del buf625  # reuse
        buf630 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf631 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf923 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_231, l__mod___model_decoder_layers_6_encoder_attn_q_proj, residual_42, residual_43], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf626, buf599, buf607, primals_352, primals_362, primals_363, primals_364, buf630, buf631, buf923, 128, 1024, grid=grid(128), stream=stream0)
        del primals_352
        del primals_362
        del primals_364
        buf632 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf631, reinterpret_tensor(primals_365, (1024, 1024), (1, 1024), 0), out=buf632)
        buf633 = reinterpret_tensor(buf599, (128, 1024), (1024, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_367, (1024, 1024), (1, 1024), 0), out=buf633)
        buf634 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf633, primals_368, buf634, 131072, grid=grid(131072), stream=stream0)
        del primals_368
        buf635 = buf633; del buf633  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_369, (1024, 1024), (1, 1024), 0), out=buf635)
        buf636 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf635, primals_370, buf636, 131072, grid=grid(131072), stream=stream0)
        del primals_370
        buf637 = reinterpret_tensor(buf635, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf635  # reuse
        # Source Nodes: [contiguous_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf632, primals_366, buf637, 131072, grid=grid(131072), stream=stream0)
        del primals_366
        buf638 = buf619; del buf619  # reuse
        # Source Nodes: [attn_weights_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf637, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf634, (16, 64, 128), (8192, 1, 64), 0), out=buf638)
        buf639 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf640 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf641 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_65], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf638, buf639, buf640, buf641, 2048, 128, grid=grid(2048), stream=stream0)
        buf642 = reinterpret_tensor(buf632, (16, 128, 64), (8192, 64, 1), 0); del buf632  # reuse
        # Source Nodes: [attn_output_125], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf641, reinterpret_tensor(buf636, (16, 128, 64), (8192, 64, 1), 0), out=buf642)
        buf643 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_232], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf642, buf643, 131072, grid=grid(131072), stream=stream0)
        buf644 = reinterpret_tensor(buf642, (128, 1024), (1024, 1), 0); del buf642  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf643, reinterpret_tensor(primals_371, (1024, 1024), (1, 1024), 0), out=buf644)
        buf648 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf649 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf922 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_235, l__mod___model_decoder_layers_6_fc1, residual_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf626, buf644, primals_372, primals_373, primals_374, buf648, buf649, buf922, 128, 1024, grid=grid(128), stream=stream0)
        del primals_374
        buf650 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf649, reinterpret_tensor(primals_375, (1024, 4096), (1, 1024), 0), out=buf650)
        buf651 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_236, hidden_states_238], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf650, primals_376, buf651, 524288, grid=grid(524288), stream=stream0)
        buf652 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf651, reinterpret_tensor(primals_377, (4096, 1024), (1, 4096), 0), out=buf652)
        buf653 = reinterpret_tensor(buf652, (1, 128, 1024), (131072, 1024, 1), 0); del buf652  # reuse
        buf657 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf658 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf920 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_242, l__mod___model_decoder_layers_7_self_attn_q_proj, residual_44, residual_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf653, buf626, buf644, primals_372, primals_378, primals_379, primals_380, buf657, buf658, buf920, 128, 1024, grid=grid(128), stream=stream0)
        del primals_372
        del primals_378
        del primals_380
        buf659 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf658, reinterpret_tensor(primals_381, (1024, 1024), (1, 1024), 0), out=buf659)
        buf660 = reinterpret_tensor(buf626, (128, 1024), (1024, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf658, reinterpret_tensor(primals_383, (1024, 1024), (1, 1024), 0), out=buf660)
        buf661 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf660, primals_384, buf661, 131072, grid=grid(131072), stream=stream0)
        del primals_384
        buf662 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf658, reinterpret_tensor(primals_385, (1024, 1024), (1, 1024), 0), out=buf662)
        buf663 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf662, primals_386, buf663, 131072, grid=grid(131072), stream=stream0)
        del primals_386
        buf664 = reinterpret_tensor(buf662, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf662  # reuse
        # Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf659, primals_382, buf664, 131072, grid=grid(131072), stream=stream0)
        del primals_382
        buf665 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf664, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf661, (16, 64, 128), (8192, 1, 64), 0), out=buf665)
        buf668 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf919 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_26, attn_weights_69], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf665, buf668, buf919, 2048, 128, grid=grid(2048), stream=stream0)
        buf669 = reinterpret_tensor(buf659, (16, 128, 64), (8192, 64, 1), 0); del buf659  # reuse
        # Source Nodes: [attn_output_130], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf668, reinterpret_tensor(buf663, (16, 128, 64), (8192, 64, 1), 0), out=buf669)
        buf670 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_243], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf669, buf670, 131072, grid=grid(131072), stream=stream0)
        buf671 = reinterpret_tensor(buf669, (128, 1024), (1024, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf670, reinterpret_tensor(primals_387, (1024, 1024), (1, 1024), 0), out=buf671)
        buf675 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf676 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf918 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_246, l__mod___model_decoder_layers_7_encoder_attn_q_proj, residual_46], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf653, buf671, primals_388, primals_389, primals_390, buf675, buf676, buf918, 128, 1024, grid=grid(128), stream=stream0)
        del primals_390
        buf677 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf676, reinterpret_tensor(primals_391, (1024, 1024), (1, 1024), 0), out=buf677)
        buf678 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_393, (1024, 1024), (1, 1024), 0), out=buf678)
        buf679 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf678, primals_394, buf679, 131072, grid=grid(131072), stream=stream0)
        del primals_394
        buf680 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_395, (1024, 1024), (1, 1024), 0), out=buf680)
        buf681 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf680, primals_396, buf681, 131072, grid=grid(131072), stream=stream0)
        del primals_396
        buf682 = reinterpret_tensor(buf680, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf680  # reuse
        # Source Nodes: [contiguous_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf677, primals_392, buf682, 131072, grid=grid(131072), stream=stream0)
        del primals_392
        buf683 = buf665; del buf665  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf682, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf679, (16, 64, 128), (8192, 1, 64), 0), out=buf683)
        buf684 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf685 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf686 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_71], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf683, buf684, buf685, buf686, 2048, 128, grid=grid(2048), stream=stream0)
        buf687 = reinterpret_tensor(buf677, (16, 128, 64), (8192, 64, 1), 0); del buf677  # reuse
        # Source Nodes: [attn_output_135], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf686, reinterpret_tensor(buf681, (16, 128, 64), (8192, 64, 1), 0), out=buf687)
        buf688 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_247], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf687, buf688, 131072, grid=grid(131072), stream=stream0)
        buf689 = reinterpret_tensor(buf687, (128, 1024), (1024, 1), 0); del buf687  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf688, reinterpret_tensor(primals_397, (1024, 1024), (1, 1024), 0), out=buf689)
        buf690 = reinterpret_tensor(buf689, (1, 128, 1024), (131072, 1024, 1), 0); del buf689  # reuse
        buf694 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf695 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf917 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_250, l__mod___model_decoder_layers_7_fc1, residual_46, residual_47], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf690, buf653, buf671, primals_388, primals_398, primals_399, primals_400, buf694, buf695, buf917, 128, 1024, grid=grid(128), stream=stream0)
        del primals_388
        del primals_398
        del primals_400
        buf696 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf695, reinterpret_tensor(primals_401, (1024, 4096), (1, 1024), 0), out=buf696)
        buf697 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_251, hidden_states_253], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf696, primals_402, buf697, 524288, grid=grid(524288), stream=stream0)
        buf698 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf697, reinterpret_tensor(primals_403, (4096, 1024), (1, 4096), 0), out=buf698)
        buf702 = buf653; del buf653  # reuse
        buf703 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf915 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_257, l__mod___model_decoder_layers_8_self_attn_q_proj, residual_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf690, buf698, primals_404, primals_405, primals_406, buf702, buf703, buf915, 128, 1024, grid=grid(128), stream=stream0)
        del primals_406
        buf704 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf703, reinterpret_tensor(primals_407, (1024, 1024), (1, 1024), 0), out=buf704)
        buf705 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf703, reinterpret_tensor(primals_409, (1024, 1024), (1, 1024), 0), out=buf705)
        buf706 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf705, primals_410, buf706, 131072, grid=grid(131072), stream=stream0)
        del primals_410
        buf707 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf703, reinterpret_tensor(primals_411, (1024, 1024), (1, 1024), 0), out=buf707)
        buf708 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf707, primals_412, buf708, 131072, grid=grid(131072), stream=stream0)
        del primals_412
        buf709 = reinterpret_tensor(buf707, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf707  # reuse
        # Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf704, primals_408, buf709, 131072, grid=grid(131072), stream=stream0)
        del primals_408
        buf710 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf709, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf706, (16, 64, 128), (8192, 1, 64), 0), out=buf710)
        buf713 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf914 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_28, attn_weights_75], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf710, buf713, buf914, 2048, 128, grid=grid(2048), stream=stream0)
        buf714 = reinterpret_tensor(buf704, (16, 128, 64), (8192, 64, 1), 0); del buf704  # reuse
        # Source Nodes: [attn_output_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf713, reinterpret_tensor(buf708, (16, 128, 64), (8192, 64, 1), 0), out=buf714)
        buf715 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_258], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf714, buf715, 131072, grid=grid(131072), stream=stream0)
        buf716 = reinterpret_tensor(buf714, (128, 1024), (1024, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf715, reinterpret_tensor(primals_413, (1024, 1024), (1, 1024), 0), out=buf716)
        buf717 = reinterpret_tensor(buf716, (1, 128, 1024), (131072, 1024, 1), 0); del buf716  # reuse
        buf721 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf722 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf913 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_261, l__mod___model_decoder_layers_8_encoder_attn_q_proj, residual_48, residual_49], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf717, buf690, buf698, primals_404, primals_414, primals_415, primals_416, buf721, buf722, buf913, 128, 1024, grid=grid(128), stream=stream0)
        del primals_404
        del primals_414
        del primals_416
        buf723 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf722, reinterpret_tensor(primals_417, (1024, 1024), (1, 1024), 0), out=buf723)
        buf724 = reinterpret_tensor(buf690, (128, 1024), (1024, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_419, (1024, 1024), (1, 1024), 0), out=buf724)
        buf725 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf724, primals_420, buf725, 131072, grid=grid(131072), stream=stream0)
        del primals_420
        buf726 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_421, (1024, 1024), (1, 1024), 0), out=buf726)
        buf727 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf726, primals_422, buf727, 131072, grid=grid(131072), stream=stream0)
        del primals_422
        buf728 = reinterpret_tensor(buf726, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf726  # reuse
        # Source Nodes: [contiguous_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf723, primals_418, buf728, 131072, grid=grid(131072), stream=stream0)
        del primals_418
        buf729 = buf710; del buf710  # reuse
        # Source Nodes: [attn_weights_76], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf728, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf725, (16, 64, 128), (8192, 1, 64), 0), out=buf729)
        buf730 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf731 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf732 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_77], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf729, buf730, buf731, buf732, 2048, 128, grid=grid(2048), stream=stream0)
        buf733 = reinterpret_tensor(buf723, (16, 128, 64), (8192, 64, 1), 0); del buf723  # reuse
        # Source Nodes: [attn_output_145], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf732, reinterpret_tensor(buf727, (16, 128, 64), (8192, 64, 1), 0), out=buf733)
        buf734 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_262], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf733, buf734, 131072, grid=grid(131072), stream=stream0)
        buf735 = reinterpret_tensor(buf733, (128, 1024), (1024, 1), 0); del buf733  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf734, reinterpret_tensor(primals_423, (1024, 1024), (1, 1024), 0), out=buf735)
        buf739 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf740 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf912 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_265, l__mod___model_decoder_layers_8_fc1, residual_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf717, buf735, primals_424, primals_425, primals_426, buf739, buf740, buf912, 128, 1024, grid=grid(128), stream=stream0)
        del primals_426
        buf741 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf740, reinterpret_tensor(primals_427, (1024, 4096), (1, 1024), 0), out=buf741)
        buf742 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_266, hidden_states_268], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf741, primals_428, buf742, 524288, grid=grid(524288), stream=stream0)
        buf743 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf742, reinterpret_tensor(primals_429, (4096, 1024), (1, 4096), 0), out=buf743)
        buf744 = reinterpret_tensor(buf743, (1, 128, 1024), (131072, 1024, 1), 0); del buf743  # reuse
        buf748 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf749 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf910 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_272, l__mod___model_decoder_layers_9_self_attn_q_proj, residual_50, residual_51], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf744, buf717, buf735, primals_424, primals_430, primals_431, primals_432, buf748, buf749, buf910, 128, 1024, grid=grid(128), stream=stream0)
        del primals_424
        del primals_430
        del primals_432
        buf750 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf749, reinterpret_tensor(primals_433, (1024, 1024), (1, 1024), 0), out=buf750)
        buf751 = reinterpret_tensor(buf717, (128, 1024), (1024, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf749, reinterpret_tensor(primals_435, (1024, 1024), (1, 1024), 0), out=buf751)
        buf752 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf751, primals_436, buf752, 131072, grid=grid(131072), stream=stream0)
        del primals_436
        buf753 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf749, reinterpret_tensor(primals_437, (1024, 1024), (1, 1024), 0), out=buf753)
        buf754 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf753, primals_438, buf754, 131072, grid=grid(131072), stream=stream0)
        del primals_438
        buf755 = reinterpret_tensor(buf753, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf753  # reuse
        # Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf750, primals_434, buf755, 131072, grid=grid(131072), stream=stream0)
        del primals_434
        buf756 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf755, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf752, (16, 64, 128), (8192, 1, 64), 0), out=buf756)
        buf759 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf909 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_30, attn_weights_81], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf756, buf759, buf909, 2048, 128, grid=grid(2048), stream=stream0)
        buf760 = reinterpret_tensor(buf750, (16, 128, 64), (8192, 64, 1), 0); del buf750  # reuse
        # Source Nodes: [attn_output_150], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf759, reinterpret_tensor(buf754, (16, 128, 64), (8192, 64, 1), 0), out=buf760)
        buf761 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_273], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf760, buf761, 131072, grid=grid(131072), stream=stream0)
        buf762 = reinterpret_tensor(buf760, (128, 1024), (1024, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf761, reinterpret_tensor(primals_439, (1024, 1024), (1, 1024), 0), out=buf762)
        buf766 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf767 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf908 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_276, l__mod___model_decoder_layers_9_encoder_attn_q_proj, residual_52], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf744, buf762, primals_440, primals_441, primals_442, buf766, buf767, buf908, 128, 1024, grid=grid(128), stream=stream0)
        del primals_442
        buf768 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf767, reinterpret_tensor(primals_443, (1024, 1024), (1, 1024), 0), out=buf768)
        buf769 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_445, (1024, 1024), (1, 1024), 0), out=buf769)
        buf770 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf769, primals_446, buf770, 131072, grid=grid(131072), stream=stream0)
        del primals_446
        buf771 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_447, (1024, 1024), (1, 1024), 0), out=buf771)
        buf772 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf771, primals_448, buf772, 131072, grid=grid(131072), stream=stream0)
        del primals_448
        buf773 = reinterpret_tensor(buf771, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf771  # reuse
        # Source Nodes: [contiguous_95], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf768, primals_444, buf773, 131072, grid=grid(131072), stream=stream0)
        del primals_444
        buf774 = buf756; del buf756  # reuse
        # Source Nodes: [attn_weights_82], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf773, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf770, (16, 64, 128), (8192, 1, 64), 0), out=buf774)
        buf775 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf776 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf777 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_83], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf774, buf775, buf776, buf777, 2048, 128, grid=grid(2048), stream=stream0)
        buf778 = reinterpret_tensor(buf768, (16, 128, 64), (8192, 64, 1), 0); del buf768  # reuse
        # Source Nodes: [attn_output_155], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf777, reinterpret_tensor(buf772, (16, 128, 64), (8192, 64, 1), 0), out=buf778)
        buf779 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_277], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf778, buf779, 131072, grid=grid(131072), stream=stream0)
        buf780 = reinterpret_tensor(buf778, (128, 1024), (1024, 1), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf779, reinterpret_tensor(primals_449, (1024, 1024), (1, 1024), 0), out=buf780)
        buf781 = reinterpret_tensor(buf780, (1, 128, 1024), (131072, 1024, 1), 0); del buf780  # reuse
        buf785 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf786 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf907 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_280, l__mod___model_decoder_layers_9_fc1, residual_52, residual_53], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf781, buf744, buf762, primals_440, primals_450, primals_451, primals_452, buf785, buf786, buf907, 128, 1024, grid=grid(128), stream=stream0)
        del primals_440
        del primals_450
        del primals_452
        buf787 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf786, reinterpret_tensor(primals_453, (1024, 4096), (1, 1024), 0), out=buf787)
        buf788 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_281, hidden_states_283], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf787, primals_454, buf788, 524288, grid=grid(524288), stream=stream0)
        buf789 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf788, reinterpret_tensor(primals_455, (4096, 1024), (1, 4096), 0), out=buf789)
        buf793 = buf744; del buf744  # reuse
        buf794 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf905 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_287, l__mod___model_decoder_layers_10_self_attn_q_proj, residual_54], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf781, buf789, primals_456, primals_457, primals_458, buf793, buf794, buf905, 128, 1024, grid=grid(128), stream=stream0)
        del primals_458
        buf795 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf794, reinterpret_tensor(primals_459, (1024, 1024), (1, 1024), 0), out=buf795)
        buf796 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf794, reinterpret_tensor(primals_461, (1024, 1024), (1, 1024), 0), out=buf796)
        buf797 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf796, primals_462, buf797, 131072, grid=grid(131072), stream=stream0)
        del primals_462
        buf798 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf794, reinterpret_tensor(primals_463, (1024, 1024), (1, 1024), 0), out=buf798)
        buf799 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf798, primals_464, buf799, 131072, grid=grid(131072), stream=stream0)
        del primals_464
        buf800 = reinterpret_tensor(buf798, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf798  # reuse
        # Source Nodes: [contiguous_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf795, primals_460, buf800, 131072, grid=grid(131072), stream=stream0)
        del primals_460
        buf801 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf800, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf797, (16, 64, 128), (8192, 1, 64), 0), out=buf801)
        buf804 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf904 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_32, attn_weights_87], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf801, buf804, buf904, 2048, 128, grid=grid(2048), stream=stream0)
        buf805 = reinterpret_tensor(buf795, (16, 128, 64), (8192, 64, 1), 0); del buf795  # reuse
        # Source Nodes: [attn_output_160], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf804, reinterpret_tensor(buf799, (16, 128, 64), (8192, 64, 1), 0), out=buf805)
        buf806 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_288], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf805, buf806, 131072, grid=grid(131072), stream=stream0)
        buf807 = reinterpret_tensor(buf805, (128, 1024), (1024, 1), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf806, reinterpret_tensor(primals_465, (1024, 1024), (1, 1024), 0), out=buf807)
        buf808 = reinterpret_tensor(buf807, (1, 128, 1024), (131072, 1024, 1), 0); del buf807  # reuse
        buf812 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf813 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf903 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_291, l__mod___model_decoder_layers_10_encoder_attn_q_proj, residual_54, residual_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf808, buf781, buf789, primals_456, primals_466, primals_467, primals_468, buf812, buf813, buf903, 128, 1024, grid=grid(128), stream=stream0)
        del primals_456
        del primals_466
        del primals_468
        buf814 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf813, reinterpret_tensor(primals_469, (1024, 1024), (1, 1024), 0), out=buf814)
        buf815 = reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_471, (1024, 1024), (1, 1024), 0), out=buf815)
        buf816 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf815, primals_472, buf816, 131072, grid=grid(131072), stream=stream0)
        del primals_472
        buf817 = buf815; del buf815  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_473, (1024, 1024), (1, 1024), 0), out=buf817)
        buf818 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf817, primals_474, buf818, 131072, grid=grid(131072), stream=stream0)
        del primals_474
        buf819 = reinterpret_tensor(buf817, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf817  # reuse
        # Source Nodes: [contiguous_101], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf814, primals_470, buf819, 131072, grid=grid(131072), stream=stream0)
        del primals_470
        buf820 = buf801; del buf801  # reuse
        # Source Nodes: [attn_weights_88], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf819, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf816, (16, 64, 128), (8192, 1, 64), 0), out=buf820)
        buf821 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf822 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf823 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_89], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf820, buf821, buf822, buf823, 2048, 128, grid=grid(2048), stream=stream0)
        buf824 = reinterpret_tensor(buf814, (16, 128, 64), (8192, 64, 1), 0); del buf814  # reuse
        # Source Nodes: [attn_output_165], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf823, reinterpret_tensor(buf818, (16, 128, 64), (8192, 64, 1), 0), out=buf824)
        buf825 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_292], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf824, buf825, 131072, grid=grid(131072), stream=stream0)
        buf826 = reinterpret_tensor(buf824, (128, 1024), (1024, 1), 0); del buf824  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf825, reinterpret_tensor(primals_475, (1024, 1024), (1, 1024), 0), out=buf826)
        buf830 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf831 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf902 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_295, l__mod___model_decoder_layers_10_fc1, residual_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf808, buf826, primals_476, primals_477, primals_478, buf830, buf831, buf902, 128, 1024, grid=grid(128), stream=stream0)
        del primals_478
        buf832 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf831, reinterpret_tensor(primals_479, (1024, 4096), (1, 1024), 0), out=buf832)
        buf833 = empty((128, 4096), device='cuda', dtype=torch.float32)
        buf901 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_296, hidden_states_298], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_11.run(buf832, primals_480, buf833, buf901, 524288, grid=grid(524288), stream=stream0)
        del primals_480
        buf834 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf833, reinterpret_tensor(primals_481, (4096, 1024), (1, 4096), 0), out=buf834)
        buf835 = reinterpret_tensor(buf834, (1, 128, 1024), (131072, 1024, 1), 0); del buf834  # reuse
        buf839 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf840 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf900 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_302, l__mod___model_decoder_layers_11_self_attn_q_proj, residual_56, residual_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf835, buf808, buf826, primals_476, primals_482, primals_483, primals_484, buf839, buf840, buf900, 128, 1024, grid=grid(128), stream=stream0)
        del primals_476
        del primals_482
        del primals_484
        buf841 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf840, reinterpret_tensor(primals_485, (1024, 1024), (1, 1024), 0), out=buf841)
        buf842 = reinterpret_tensor(buf808, (128, 1024), (1024, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf840, reinterpret_tensor(primals_487, (1024, 1024), (1, 1024), 0), out=buf842)
        buf843 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf842, primals_488, buf843, 131072, grid=grid(131072), stream=stream0)
        del primals_488
        buf844 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf840, reinterpret_tensor(primals_489, (1024, 1024), (1, 1024), 0), out=buf844)
        buf845 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf844, primals_490, buf845, 131072, grid=grid(131072), stream=stream0)
        del primals_490
        buf846 = reinterpret_tensor(buf844, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf844  # reuse
        # Source Nodes: [contiguous_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf841, primals_486, buf846, 131072, grid=grid(131072), stream=stream0)
        del primals_486
        buf847 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf846, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf843, (16, 64, 128), (8192, 1, 64), 0), out=buf847)
        buf850 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        buf899 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_34, attn_weights_93], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_10.run(buf847, buf850, buf899, 2048, 128, grid=grid(2048), stream=stream0)
        buf851 = reinterpret_tensor(buf841, (16, 128, 64), (8192, 64, 1), 0); del buf841  # reuse
        # Source Nodes: [attn_output_170], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf850, reinterpret_tensor(buf845, (16, 128, 64), (8192, 64, 1), 0), out=buf851)
        buf852 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_303], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf851, buf852, 131072, grid=grid(131072), stream=stream0)
        buf853 = reinterpret_tensor(buf851, (128, 1024), (1024, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf852, reinterpret_tensor(primals_491, (1024, 1024), (1, 1024), 0), out=buf853)
        buf857 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf858 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf898 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_306, l__mod___model_decoder_layers_11_encoder_attn_q_proj, residual_58], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf835, buf853, primals_492, primals_493, primals_494, buf857, buf858, buf898, 128, 1024, grid=grid(128), stream=stream0)
        del primals_494
        buf859 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf858, reinterpret_tensor(primals_495, (1024, 1024), (1, 1024), 0), out=buf859)
        buf860 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_497, (1024, 1024), (1, 1024), 0), out=buf860)
        buf861 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf860, primals_498, buf861, 131072, grid=grid(131072), stream=stream0)
        del primals_498
        buf862 = buf860; del buf860  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_499, (1024, 1024), (1, 1024), 0), out=buf862)
        buf863 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf862, primals_500, buf863, 131072, grid=grid(131072), stream=stream0)
        del primals_500
        buf864 = reinterpret_tensor(buf862, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf862  # reuse
        # Source Nodes: [contiguous_107], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf859, primals_496, buf864, 131072, grid=grid(131072), stream=stream0)
        del primals_496
        buf865 = buf847; del buf847  # reuse
        # Source Nodes: [attn_weights_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf864, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf861, (16, 64, 128), (8192, 1, 64), 0), out=buf865)
        buf866 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf867 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf868 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_95], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf865, buf866, buf867, buf868, 2048, 128, grid=grid(2048), stream=stream0)
        buf869 = reinterpret_tensor(buf859, (16, 128, 64), (8192, 64, 1), 0); del buf859  # reuse
        # Source Nodes: [attn_output_175], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf868, reinterpret_tensor(buf863, (16, 128, 64), (8192, 64, 1), 0), out=buf869)
        buf870 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_307], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf869, buf870, 131072, grid=grid(131072), stream=stream0)
        buf871 = reinterpret_tensor(buf869, (128, 1024), (1024, 1), 0); del buf869  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf870, reinterpret_tensor(primals_501, (1024, 1024), (1, 1024), 0), out=buf871)
        buf872 = reinterpret_tensor(buf871, (1, 128, 1024), (131072, 1024, 1), 0); del buf871  # reuse
        buf876 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf877 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf897 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_310, l__mod___model_decoder_layers_11_fc1, residual_58, residual_59], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf872, buf835, buf853, primals_492, primals_502, primals_503, primals_504, buf876, buf877, buf897, 128, 1024, grid=grid(128), stream=stream0)
        del primals_492
        del primals_502
        del primals_504
        buf878 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf877, reinterpret_tensor(primals_505, (1024, 4096), (1, 1024), 0), out=buf878)
        buf879 = empty((128, 4096), device='cuda', dtype=torch.float32)
        buf896 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_311, hidden_states_313], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_11.run(buf878, primals_506, buf879, buf896, 524288, grid=grid(524288), stream=stream0)
        del buf878
        del primals_506
        buf880 = buf853; del buf853  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf879, reinterpret_tensor(primals_507, (4096, 1024), (1, 4096), 0), out=buf880)
        buf884 = buf835; del buf835  # reuse
        buf885 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf895 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_316, hidden_states_317, lm_logits], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf872, buf880, primals_508, primals_509, primals_510, buf884, buf885, buf895, 128, 1024, grid=grid(128), stream=stream0)
        del buf872
        del buf880
        del primals_508
        del primals_510
        buf886 = empty((128, 128112), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf885, reinterpret_tensor(primals_511, (1024, 128112), (1, 1024), 0), out=buf886)
        buf887 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_12.run(buf886, buf887, 512, 32028, grid=grid(512), stream=stream0)
        buf888 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_13.run(buf887, buf888, 128, 4, grid=grid(128), stream=stream0)
        buf889 = buf887; del buf887  # reuse
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_14.run(buf886, buf888, buf889, 512, 32028, grid=grid(512), stream=stream0)
        buf890 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_15.run(buf889, buf890, 128, 4, grid=grid(128), stream=stream0)
        del buf889
        buf891 = empty((128, 128112), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_poi_fused__log_softmax_16.run(buf886, buf888, buf890, buf891, 16398336, grid=grid(16398336), stream=stream0)
        del buf888
        del buf890
        buf894 = empty((), device='cuda', dtype=torch.float32)
        buf893 = empty((), device='cuda', dtype=torch.float32)
        buf993 = buf894; del buf894  # reuse
        # Source Nodes: [loss, masked_fill_], Original ATen: [aten.masked_fill, aten.nll_loss_forward]
        triton_per_fused_masked_fill_nll_loss_forward_17.run(buf993, primals_514, buf891, buf893, 1, 128, grid=grid(1), stream=stream0)
        buf906 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_281], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf787, primals_454, buf906, 524288, grid=grid(524288), stream=stream0)
        del buf787
        del primals_454
        buf911 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_266], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf741, primals_428, buf911, 524288, grid=grid(524288), stream=stream0)
        del buf741
        del primals_428
        buf916 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_251], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf696, primals_402, buf916, 524288, grid=grid(524288), stream=stream0)
        del buf696
        del primals_402
        buf921 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_236], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf650, primals_376, buf921, 524288, grid=grid(524288), stream=stream0)
        del buf650
        del primals_376
        buf926 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_221], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf605, primals_350, buf926, 524288, grid=grid(524288), stream=stream0)
        del buf605
        del primals_350
        buf931 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_206], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf559, primals_324, buf931, 524288, grid=grid(524288), stream=stream0)
        del buf559
        del primals_324
        buf936 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_191], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf514, primals_298, buf936, 524288, grid=grid(524288), stream=stream0)
        del buf514
        del primals_298
        buf941 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_176], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf468, primals_272, buf941, 524288, grid=grid(524288), stream=stream0)
        del buf468
        del primals_272
        buf946 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_161], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf423, primals_246, buf946, 524288, grid=grid(524288), stream=stream0)
        del buf423
        del primals_246
        buf951 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf377, primals_220, buf951, 524288, grid=grid(524288), stream=stream0)
        del buf377
        del primals_220
        buf957 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_128], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf324, primals_191, buf957, 524288, grid=grid(524288), stream=stream0)
        del buf324
        del primals_191
        buf960 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_117], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf297, primals_175, buf960, 524288, grid=grid(524288), stream=stream0)
        del buf297
        del primals_175
        buf963 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_106], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf270, primals_159, buf963, 524288, grid=grid(524288), stream=stream0)
        del buf270
        del primals_159
        buf966 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf243, primals_143, buf966, 524288, grid=grid(524288), stream=stream0)
        del buf243
        del primals_143
        buf969 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf216, primals_127, buf969, 524288, grid=grid(524288), stream=stream0)
        del buf216
        del primals_127
        buf972 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf189, primals_111, buf972, 524288, grid=grid(524288), stream=stream0)
        del buf189
        del primals_111
        buf975 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_62], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf162, primals_95, buf975, 524288, grid=grid(524288), stream=stream0)
        del buf162
        del primals_95
        buf978 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_51], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf135, primals_79, buf978, 524288, grid=grid(524288), stream=stream0)
        del buf135
        del primals_79
        buf981 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_40], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf108, primals_63, buf981, 524288, grid=grid(524288), stream=stream0)
        del buf108
        del primals_63
        buf984 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_29], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf81, primals_47, buf984, 524288, grid=grid(524288), stream=stream0)
        del buf81
        del primals_47
        buf987 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf54, primals_31, buf987, 524288, grid=grid(524288), stream=stream0)
        del buf54
        del primals_31
        buf990 = empty((1, 128, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_18.run(buf27, primals_15, buf990, 524288, grid=grid(524288), stream=stream0)
        del buf27
        del primals_15
        return (buf993, reinterpret_tensor(buf886, (1, 128, 128112), (16398336, 128112, 1), 0), buf342, buf344, buf361, buf363, buf388, buf390, buf406, buf408, buf433, buf435, buf452, buf454, buf479, buf481, buf497, buf499, buf524, buf526, buf543, buf545, buf570, buf572, buf588, buf590, buf615, buf617, buf634, buf636, buf661, buf663, buf679, buf681, buf706, buf708, buf725, buf727, buf752, buf754, buf770, buf772, buf797, buf799, buf816, buf818, buf843, buf845, buf861, buf863, buf331, primals_2, primals_12, primals_18, primals_28, primals_34, primals_44, primals_50, primals_60, primals_66, primals_76, primals_82, primals_92, primals_98, primals_108, primals_114, primals_124, primals_130, primals_140, primals_146, primals_156, primals_162, primals_172, primals_178, primals_188, primals_194, primals_197, primals_207, primals_217, primals_223, primals_233, primals_243, primals_249, primals_259, primals_269, primals_275, primals_285, primals_295, primals_301, primals_311, primals_321, primals_327, primals_337, primals_347, primals_353, primals_363, primals_373, primals_379, primals_389, primals_399, primals_405, primals_415, primals_425, primals_431, primals_441, primals_451, primals_457, primals_467, primals_477, primals_483, primals_493, primals_503, primals_509, primals_514, primals_516, buf6, buf7, buf14, buf15, buf16, buf19, buf25, buf26, buf28, buf33, buf34, buf41, buf42, buf43, buf46, buf52, buf53, buf55, buf60, buf61, buf68, buf69, buf70, buf73, buf79, buf80, buf82, buf87, buf88, buf95, buf96, buf97, buf100, buf106, buf107, buf109, buf114, buf115, buf122, buf123, buf124, buf127, buf133, buf134, buf136, buf141, buf142, buf149, buf150, buf151, buf154, buf160, buf161, buf163, buf168, buf169, buf176, buf177, buf178, buf181, buf187, buf188, buf190, buf195, buf196, buf203, buf204, buf205, buf208, buf214, buf215, buf217, buf222, buf223, buf230, buf231, buf232, buf235, buf241, buf242, buf244, buf249, buf250, buf257, buf258, buf259, buf262, buf268, buf269, buf271, buf276, buf277, buf284, buf285, buf286, buf289, buf295, buf296, buf298, buf303, buf304, buf311, buf312, buf313, buf316, buf322, buf323, buf325, buf330, primals_515, buf338, buf339, buf351, buf357, buf358, reinterpret_tensor(buf331, (128, 1024), (1024, 1), 0), buf365, buf366, buf367, buf370, buf375, buf376, buf378, buf384, buf385, buf397, buf402, buf403, buf410, buf411, buf412, buf415, buf421, buf422, buf424, buf429, buf430, buf442, buf448, buf449, buf456, buf457, buf458, buf461, buf466, buf467, buf469, buf475, buf476, buf488, buf493, buf494, buf501, buf502, buf503, buf506, buf512, buf513, buf515, buf520, buf521, buf533, buf539, buf540, buf547, buf548, buf549, buf552, buf557, buf558, buf560, buf566, buf567, buf579, buf584, buf585, buf592, buf593, buf594, buf597, buf603, buf604, buf606, buf611, buf612, buf624, buf630, buf631, buf638, buf639, buf640, buf643, buf648, buf649, buf651, buf657, buf658, buf670, buf675, buf676, buf683, buf684, buf685, buf688, buf694, buf695, buf697, buf702, buf703, buf715, buf721, buf722, buf729, buf730, buf731, buf734, buf739, buf740, buf742, buf748, buf749, buf761, buf766, buf767, buf774, buf775, buf776, buf779, buf785, buf786, buf788, buf793, buf794, buf806, buf812, buf813, buf820, buf821, buf822, buf825, buf830, buf831, buf833, buf839, buf840, buf852, buf857, buf858, buf865, buf866, buf867, buf870, buf876, buf877, buf879, buf884, buf885, buf891, buf893, reinterpret_tensor(primals_511, (128112, 1024), (1024, 1), 0), buf895, reinterpret_tensor(primals_507, (1024, 4096), (4096, 1), 0), buf896, reinterpret_tensor(primals_505, (4096, 1024), (1024, 1), 0), buf897, reinterpret_tensor(primals_501, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf868, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf863, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf864, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf861, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_499, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_497, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_495, (1024, 1024), (1024, 1), 0), buf898, reinterpret_tensor(primals_491, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf850, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf845, (16, 64, 128), (8192, 1, 64), 0), buf899, reinterpret_tensor(buf846, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf843, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_489, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_487, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_485, (1024, 1024), (1024, 1), 0), buf900, reinterpret_tensor(primals_481, (1024, 4096), (4096, 1), 0), buf901, reinterpret_tensor(primals_479, (4096, 1024), (1024, 1), 0), buf902, reinterpret_tensor(primals_475, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf823, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf818, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf819, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf816, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_473, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_471, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_469, (1024, 1024), (1024, 1), 0), buf903, reinterpret_tensor(primals_465, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf804, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf799, (16, 64, 128), (8192, 1, 64), 0), buf904, reinterpret_tensor(buf800, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf797, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_463, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_461, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_459, (1024, 1024), (1024, 1), 0), buf905, reinterpret_tensor(primals_455, (1024, 4096), (4096, 1), 0), buf906, reinterpret_tensor(primals_453, (4096, 1024), (1024, 1), 0), buf907, reinterpret_tensor(primals_449, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf777, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf772, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf773, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf770, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_447, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_445, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_443, (1024, 1024), (1024, 1), 0), buf908, reinterpret_tensor(primals_439, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf759, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf754, (16, 64, 128), (8192, 1, 64), 0), buf909, reinterpret_tensor(buf755, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf752, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_437, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_435, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_433, (1024, 1024), (1024, 1), 0), buf910, reinterpret_tensor(primals_429, (1024, 4096), (4096, 1), 0), buf911, reinterpret_tensor(primals_427, (4096, 1024), (1024, 1), 0), buf912, reinterpret_tensor(primals_423, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf732, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf727, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf728, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf725, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_421, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_419, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_417, (1024, 1024), (1024, 1), 0), buf913, reinterpret_tensor(primals_413, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf713, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf708, (16, 64, 128), (8192, 1, 64), 0), buf914, reinterpret_tensor(buf709, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf706, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_411, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_409, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_407, (1024, 1024), (1024, 1), 0), buf915, reinterpret_tensor(primals_403, (1024, 4096), (4096, 1), 0), buf916, reinterpret_tensor(primals_401, (4096, 1024), (1024, 1), 0), buf917, reinterpret_tensor(primals_397, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf686, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf681, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf682, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf679, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_395, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_393, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_391, (1024, 1024), (1024, 1), 0), buf918, reinterpret_tensor(primals_387, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf668, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf663, (16, 64, 128), (8192, 1, 64), 0), buf919, reinterpret_tensor(buf664, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf661, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_385, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_383, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_381, (1024, 1024), (1024, 1), 0), buf920, reinterpret_tensor(primals_377, (1024, 4096), (4096, 1), 0), buf921, reinterpret_tensor(primals_375, (4096, 1024), (1024, 1), 0), buf922, reinterpret_tensor(primals_371, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf641, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf636, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf637, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf634, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_369, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_367, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_365, (1024, 1024), (1024, 1), 0), buf923, reinterpret_tensor(primals_361, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf622, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf617, (16, 64, 128), (8192, 1, 64), 0), buf924, reinterpret_tensor(buf618, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf615, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_357, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_355, (1024, 1024), (1024, 1), 0), buf925, reinterpret_tensor(primals_351, (1024, 4096), (4096, 1), 0), buf926, reinterpret_tensor(primals_349, (4096, 1024), (1024, 1), 0), buf927, reinterpret_tensor(primals_345, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf595, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf590, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf591, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf588, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_343, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_341, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_339, (1024, 1024), (1024, 1), 0), buf928, reinterpret_tensor(primals_335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf577, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf572, (16, 64, 128), (8192, 1, 64), 0), buf929, reinterpret_tensor(buf573, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf570, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_333, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_331, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_329, (1024, 1024), (1024, 1), 0), buf930, reinterpret_tensor(primals_325, (1024, 4096), (4096, 1), 0), buf931, reinterpret_tensor(primals_323, (4096, 1024), (1024, 1), 0), buf932, reinterpret_tensor(primals_319, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf550, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf545, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf546, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf543, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_317, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_315, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_313, (1024, 1024), (1024, 1), 0), buf933, reinterpret_tensor(primals_309, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf531, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf526, (16, 64, 128), (8192, 1, 64), 0), buf934, reinterpret_tensor(buf527, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf524, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_307, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_305, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_303, (1024, 1024), (1024, 1), 0), buf935, reinterpret_tensor(primals_299, (1024, 4096), (4096, 1), 0), buf936, reinterpret_tensor(primals_297, (4096, 1024), (1024, 1), 0), buf937, reinterpret_tensor(primals_293, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf504, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf499, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf500, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf497, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_291, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_289, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_287, (1024, 1024), (1024, 1), 0), buf938, reinterpret_tensor(primals_283, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf486, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf481, (16, 64, 128), (8192, 1, 64), 0), buf939, reinterpret_tensor(buf482, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf479, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_281, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_279, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_277, (1024, 1024), (1024, 1), 0), buf940, reinterpret_tensor(primals_273, (1024, 4096), (4096, 1), 0), buf941, reinterpret_tensor(primals_271, (4096, 1024), (1024, 1), 0), buf942, reinterpret_tensor(primals_267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf459, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf454, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf455, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf452, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_265, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_263, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_261, (1024, 1024), (1024, 1), 0), buf943, reinterpret_tensor(primals_257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf440, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf435, (16, 64, 128), (8192, 1, 64), 0), buf944, reinterpret_tensor(buf436, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf433, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_255, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_253, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_251, (1024, 1024), (1024, 1), 0), buf945, reinterpret_tensor(primals_247, (1024, 4096), (4096, 1), 0), buf946, reinterpret_tensor(primals_245, (4096, 1024), (1024, 1), 0), buf947, reinterpret_tensor(primals_241, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf413, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf408, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf409, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf406, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_239, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_237, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_235, (1024, 1024), (1024, 1), 0), buf948, reinterpret_tensor(primals_231, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf395, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf390, (16, 64, 128), (8192, 1, 64), 0), buf949, reinterpret_tensor(buf391, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf388, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_229, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_225, (1024, 1024), (1024, 1), 0), buf950, reinterpret_tensor(primals_221, (1024, 4096), (4096, 1), 0), buf951, reinterpret_tensor(primals_219, (4096, 1024), (1024, 1), 0), buf952, reinterpret_tensor(primals_215, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf368, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf363, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf364, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf361, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_213, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_211, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_209, (1024, 1024), (1024, 1), 0), buf953, reinterpret_tensor(primals_205, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf349, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf344, (16, 64, 128), (8192, 1, 64), 0), buf954, reinterpret_tensor(buf345, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf342, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_203, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_201, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_199, (1024, 1024), (1024, 1), 0), buf955, buf956, reinterpret_tensor(primals_192, (1024, 4096), (4096, 1), 0), buf957, reinterpret_tensor(primals_190, (4096, 1024), (1024, 1), 0), buf958, reinterpret_tensor(primals_186, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf314, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf309, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf308, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf310, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_184, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_180, (1024, 1024), (1024, 1), 0), buf959, reinterpret_tensor(primals_176, (1024, 4096), (4096, 1), 0), buf960, reinterpret_tensor(primals_174, (4096, 1024), (1024, 1), 0), buf961, reinterpret_tensor(primals_170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf287, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf282, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf281, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf283, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_168, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_166, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_164, (1024, 1024), (1024, 1), 0), buf962, reinterpret_tensor(primals_160, (1024, 4096), (4096, 1), 0), buf963, reinterpret_tensor(primals_158, (4096, 1024), (1024, 1), 0), buf964, reinterpret_tensor(primals_154, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf260, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf255, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf254, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf256, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_152, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_150, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_148, (1024, 1024), (1024, 1), 0), buf965, reinterpret_tensor(primals_144, (1024, 4096), (4096, 1), 0), buf966, reinterpret_tensor(primals_142, (4096, 1024), (1024, 1), 0), buf967, reinterpret_tensor(primals_138, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf233, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf228, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf227, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf229, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_136, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_134, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_132, (1024, 1024), (1024, 1), 0), buf968, reinterpret_tensor(primals_128, (1024, 4096), (4096, 1), 0), buf969, reinterpret_tensor(primals_126, (4096, 1024), (1024, 1), 0), buf970, reinterpret_tensor(primals_122, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf206, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf201, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf200, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf202, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_120, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_118, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_116, (1024, 1024), (1024, 1), 0), buf971, reinterpret_tensor(primals_112, (1024, 4096), (4096, 1), 0), buf972, reinterpret_tensor(primals_110, (4096, 1024), (1024, 1), 0), buf973, reinterpret_tensor(primals_106, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf179, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf174, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf173, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf175, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_104, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_102, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_100, (1024, 1024), (1024, 1), 0), buf974, reinterpret_tensor(primals_96, (1024, 4096), (4096, 1), 0), buf975, reinterpret_tensor(primals_94, (4096, 1024), (1024, 1), 0), buf976, reinterpret_tensor(primals_90, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf152, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf147, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf146, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf148, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_88, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_86, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_84, (1024, 1024), (1024, 1), 0), buf977, reinterpret_tensor(primals_80, (1024, 4096), (4096, 1), 0), buf978, reinterpret_tensor(primals_78, (4096, 1024), (1024, 1), 0), buf979, reinterpret_tensor(primals_74, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf125, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf120, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf119, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf121, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_72, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_70, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_68, (1024, 1024), (1024, 1), 0), buf980, reinterpret_tensor(primals_64, (1024, 4096), (4096, 1), 0), buf981, reinterpret_tensor(primals_62, (4096, 1024), (1024, 1), 0), buf982, reinterpret_tensor(primals_58, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf98, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf93, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf92, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf94, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_56, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_54, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_52, (1024, 1024), (1024, 1), 0), buf983, reinterpret_tensor(primals_48, (1024, 4096), (4096, 1), 0), buf984, reinterpret_tensor(primals_46, (4096, 1024), (1024, 1), 0), buf985, reinterpret_tensor(primals_42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf71, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf66, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf65, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf67, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_40, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_38, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_36, (1024, 1024), (1024, 1), 0), buf986, reinterpret_tensor(primals_32, (1024, 4096), (4096, 1), 0), buf987, reinterpret_tensor(primals_30, (4096, 1024), (1024, 1), 0), buf988, reinterpret_tensor(primals_26, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf44, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf39, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf38, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf40, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_24, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_22, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_20, (1024, 1024), (1024, 1), 0), buf989, reinterpret_tensor(primals_16, (1024, 4096), (4096, 1), 0), buf990, reinterpret_tensor(primals_14, (4096, 1024), (1024, 1), 0), buf991, reinterpret_tensor(primals_10, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf17, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf12, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf11, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf13, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_8, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_6, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_4, (1024, 1024), (1024, 1), 0), buf992, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_515 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('M2M100ForConditionalGeneration', benchmark_compiled_module)
