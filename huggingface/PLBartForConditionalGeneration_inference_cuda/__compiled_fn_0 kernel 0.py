
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


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdh3ldmhxmv74gxlseoealhvaq2alhoanrq7tcak5iaacgckbpj.py
# Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# embed_pos => embedding_1
# hidden_states => add_1
# hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50005
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50005)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50005")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 27.712812921102035
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50005
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 50005)) | ~xmask, "index out of bounds: 0 <= tmp15 < 50005")
        tmp16 = tl.load(in_ptr1 + (r1 + (768*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 27.712812921102035
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 768.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcgfzkhuadsp6sfhtvns6tgbrnltb6okg4d34jrirummx5vk3ku.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm33qvhnojan7wn4iwmif5hbizhpp23mchvl3x3razoy6kgdxsh.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxmjcegg3pikhi3fmafxvox72y6xw4le3v36iw7e4oicjrp55qz.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_7
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcojxdl4yp73hk7crc4v4o3ano6im4htp3yccqycfjucwrbje6i.py
# Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_5 => add_4
# residual_1 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcmaxe5xphcludcm5yxhvly7xil3txwxhq5irs4zi4gmi4nriax.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# hidden_states_7 => add_7, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cozxqtzs3crrk2tbohf7uqesf6gqkqq6lamtfkedxrpzl5ecn5.py
# Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
# eq => eq
# masked_fill_ => full_default, where
# ne => ne
# sum_1 => sum_1
triton_per_fused_eq_masked_fill_ne_sum_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_masked_fill_ne_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
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
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfeviaeffvq23ykgujkew7qnqnbyjgjhytos3mxwb633ykmco4wj.py
# Source Nodes: [add_15, clone_1, eq, hidden_states_69, hidden_states_70, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.select_scatter, aten.slice_scatter]
# add_15 => add_47
# clone_1 => clone_1
# eq => eq
# hidden_states_69 => add_48
# hidden_states_70 => add_49, add_50, mul_52, mul_53, rsqrt_13, sub_20, var_mean_13
# inputs_embeds_1 => mul_51
# l__mod___model_decoder_embed_tokens => embedding_2
# masked_fill_ => full_default, where
# positions_2 => embedding_3
# setitem => copy, slice_scatter
# setitem_1 => copy_1, select_scatter
triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (1536 + r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp56 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 + 1024
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert((0 <= tmp9) & (tmp9 < 1024), "index out of bounds: 0 <= tmp9 < 1024")
    tmp10 = tl.load(in_ptr1 + (tmp9), None, eviction_policy='evict_last')
    tmp11 = tl.full([1], -100, tl.int64)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.where(tmp12, tmp5, tmp10)
    tmp14 = tmp0 >= tmp5
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + x0, [RBLOCK])), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 == tmp11
    tmp17 = tl.where(tmp16, tmp5, tmp15)
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp21 = tmp20 == tmp11
    tmp22 = tl.where(tmp21, tmp5, tmp20)
    tmp23 = tl.where(tmp14, tmp19, tmp22)
    tmp24 = tl.where(tmp2, tmp13, tmp23)
    tmp25 = tmp24 + 50005
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert((0 <= tmp27) & (tmp27 < 50005), "index out of bounds: 0 <= tmp27 < 50005")
    tmp28 = tl.load(in_ptr2 + (r1 + (768*tmp27)), rmask, other=0.0)
    tmp29 = 27.712812921102035
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = tl.full([1], 768, tl.int32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 / tmp41
    tmp43 = tmp33 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp49 = tmp32 - tmp42
    tmp50 = 768.0
    tmp51 = tmp48 / tmp50
    tmp52 = 1e-05
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp59, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c775xez3dq5yme7ots2ue2oyp2kqy3rfkrvo4e72flcbf7iw72vr.py
# Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
# attn_weights_15 => amax_6, div_6, exp_6, sub_21, sum_8
triton_per_fused__softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0.0)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp10, 0))
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp13 / tmp17
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37zkbefnpa7mc2j7olawgar6wjru7p47idf5hpghcfaxewpozlf.py
# Source Nodes: [attn_output_33], Original ATen: [aten.clone]
# attn_output_33 => clone_56
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgibb3dzmezggr76jeitxlrh6hax22v5r46x42mywxz7ajng34y4.py
# Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
# lm_logits_1 => add_117
# masked_lm_loss => amax_18, exp_18, sub_51, sum_20
triton_red_fused__log_softmax_add_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 50005
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (50005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r1 + (50005*x0)), tmp2, rmask & xmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (50005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 - tmp4
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp262uicc7cmbyt4g5srl7yvsqowdc5yqbzgwjfiu3k6kqcsjwzp.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_18, full_default_4, ne_2, ne_3, neg, sum_21, sum_22, where_3
triton_per_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
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
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 50005
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 50005), "index out of bounds: 0 <= tmp7 < 50005")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (50005*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 768), (768, 1))
    assert_size_stride(arg1_1, (1026, 768), (768, 1))
    assert_size_stride(arg2_1, (50005, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (3072, ), (1, ))
    assert_size_stride(arg65_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (50005, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (3072, 768), (768, 1))
    assert_size_stride(arg125_1, (3072, ), (1, ))
    assert_size_stride(arg126_1, (768, 3072), (3072, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, 768), (768, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, 768), (768, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (3072, 768), (768, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (768, 3072), (3072, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (768, 3072), (3072, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 768), (768, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, 768), (768, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, 768), (768, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, 768), (768, 1))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (3072, 768), (768, 1))
    assert_size_stride(arg203_1, (3072, ), (1, ))
    assert_size_stride(arg204_1, (768, 3072), (3072, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, 768), (768, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, 768), (768, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, 768), (768, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, 768), (768, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, 768), (768, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, 768), (768, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, 768), (768, 1))
    assert_size_stride(arg223_1, (768, ), (1, ))
    assert_size_stride(arg224_1, (768, 768), (768, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (3072, 768), (768, 1))
    assert_size_stride(arg229_1, (3072, ), (1, ))
    assert_size_stride(arg230_1, (768, 3072), (3072, 1))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, 768), (768, 1))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, 768), (768, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, 768), (768, 1))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, 768), (768, 1))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, 768), (768, 1))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, 768), (768, 1))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (768, 768), (768, 1))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, 768), (768, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (3072, 768), (768, 1))
    assert_size_stride(arg255_1, (3072, ), (1, ))
    assert_size_stride(arg256_1, (768, 3072), (3072, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (50005, 768), (768, 1))
    assert_size_stride(arg261_1, (1, 50005), (50005, 1))
    assert_size_stride(arg262_1, (1, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg263_1, arg2_1, arg0_1, arg3_1, arg4_1, buf3, 1024, 768, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg263_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf4 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf4)
        del arg5_1
        buf5 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), out=buf5)
        del arg7_1
        buf6 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf6)
        del arg9_1
        buf7 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg6_1, buf7, 786432, grid=grid(786432), stream=stream0)
        del arg6_1
        buf8 = reinterpret_tensor(buf4, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg8_1, buf8, 786432, grid=grid(786432), stream=stream0)
        del arg8_1
        buf9 = reinterpret_tensor(buf5, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg10_1, buf9, 786432, grid=grid(786432), stream=stream0)
        del arg10_1
        # Source Nodes: [], Original ATen: []
        buf10 = aten._scaled_dot_product_efficient_attention(buf7, buf8, buf9, None, True, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf11, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf15, 786432, grid=grid(786432), stream=stream0)
        buf16 = reinterpret_tensor(buf9, (1024, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (1024, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf16)
        del arg11_1
        buf20 = reinterpret_tensor(buf15, (1, 1024, 768), (786432, 768, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg12_1, arg13_1, arg14_1, buf20, 1024, 768, grid=grid(1024), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf21 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf21, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg16_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg16_1
        buf23 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), out=buf23)
        del arg17_1
        buf27 = reinterpret_tensor(buf16, (1, 1024, 768), (786432, 768, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg18_1, arg19_1, arg20_1, buf27, 1024, 768, grid=grid(1024), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf28)
        del arg21_1
        buf29 = reinterpret_tensor(buf20, (1024, 768), (768, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), out=buf29)
        del arg23_1
        buf30 = reinterpret_tensor(buf8, (1024, 768), (768, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), out=buf30)
        del arg25_1
        buf31 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg22_1, buf31, 786432, grid=grid(786432), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf28, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg24_1, buf32, 786432, grid=grid(786432), stream=stream0)
        del arg24_1
        buf33 = reinterpret_tensor(buf29, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg26_1, buf33, 786432, grid=grid(786432), stream=stream0)
        del arg26_1
        # Source Nodes: [], Original ATen: []
        buf34 = aten._scaled_dot_product_efficient_attention(buf31, buf32, buf33, None, True, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf35, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf39, 786432, grid=grid(786432), stream=stream0)
        buf40 = reinterpret_tensor(buf33, (1024, 768), (768, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf40)
        del arg27_1
        buf44 = reinterpret_tensor(buf39, (1, 1024, 768), (786432, 768, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_16, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg28_1, arg29_1, arg30_1, buf44, 1024, 768, grid=grid(1024), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf45 = reinterpret_tensor(buf22, (1024, 3072), (3072, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), out=buf45)
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg32_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg32_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), out=buf47)
        del arg33_1
        buf51 = buf27; del buf27  # reuse
        # Source Nodes: [hidden_states_22, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg34_1, arg35_1, arg36_1, buf51, 1024, 768, grid=grid(1024), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), out=buf52)
        del arg37_1
        buf53 = reinterpret_tensor(buf44, (1024, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), out=buf53)
        del arg39_1
        buf54 = reinterpret_tensor(buf32, (1024, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf54)
        del arg41_1
        buf55 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg38_1, buf55, 786432, grid=grid(786432), stream=stream0)
        del arg38_1
        buf56 = reinterpret_tensor(buf52, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg40_1, buf56, 786432, grid=grid(786432), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf53, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg42_1, buf57, 786432, grid=grid(786432), stream=stream0)
        del arg42_1
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, True, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf59, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf63, 786432, grid=grid(786432), stream=stream0)
        buf64 = reinterpret_tensor(buf57, (1024, 768), (768, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf64)
        del arg43_1
        buf68 = reinterpret_tensor(buf63, (1, 1024, 768), (786432, 768, 1), 0); del buf63  # reuse
        # Source Nodes: [hidden_states_27, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg44_1, arg45_1, arg46_1, buf68, 1024, 768, grid=grid(1024), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf69 = reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (1024, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf69)
        del arg47_1
        buf70 = reinterpret_tensor(buf69, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg48_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg48_1
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf71)
        del arg49_1
        buf75 = buf51; del buf51  # reuse
        # Source Nodes: [hidden_states_33, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg50_1, arg51_1, arg52_1, buf75, 1024, 768, grid=grid(1024), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), out=buf76)
        del arg53_1
        buf77 = reinterpret_tensor(buf68, (1024, 768), (768, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), out=buf77)
        del arg55_1
        buf78 = reinterpret_tensor(buf56, (1024, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf78)
        del arg57_1
        buf79 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg54_1, buf79, 786432, grid=grid(786432), stream=stream0)
        del arg54_1
        buf80 = reinterpret_tensor(buf76, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg56_1, buf80, 786432, grid=grid(786432), stream=stream0)
        del arg56_1
        buf81 = reinterpret_tensor(buf77, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg58_1, buf81, 786432, grid=grid(786432), stream=stream0)
        del arg58_1
        # Source Nodes: [], Original ATen: []
        buf82 = aten._scaled_dot_product_efficient_attention(buf79, buf80, buf81, None, True, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf83, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf87, 786432, grid=grid(786432), stream=stream0)
        buf88 = reinterpret_tensor(buf81, (1024, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_38, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg60_1, arg61_1, arg62_1, buf92, 1024, 768, grid=grid(1024), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf70, (1024, 3072), (3072, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1024, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg64_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg64_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg65_1
        buf99 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_44, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg66_1, arg67_1, arg68_1, buf99, 1024, 768, grid=grid(1024), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf100)
        del arg69_1
        buf101 = reinterpret_tensor(buf92, (1024, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), out=buf101)
        del arg71_1
        buf102 = reinterpret_tensor(buf80, (1024, 768), (768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf102)
        del arg73_1
        buf103 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg70_1, buf103, 786432, grid=grid(786432), stream=stream0)
        del arg70_1
        buf104 = reinterpret_tensor(buf100, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg72_1, buf104, 786432, grid=grid(786432), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf101, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg74_1, buf105, 786432, grid=grid(786432), stream=stream0)
        del arg74_1
        # Source Nodes: [], Original ATen: []
        buf106 = aten._scaled_dot_product_efficient_attention(buf103, buf104, buf105, None, True, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf107, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf111, 786432, grid=grid(786432), stream=stream0)
        buf112 = reinterpret_tensor(buf105, (1024, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), out=buf112)
        del arg75_1
        buf116 = reinterpret_tensor(buf111, (1, 1024, 768), (786432, 768, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_49, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg76_1, arg77_1, arg78_1, buf116, 1024, 768, grid=grid(1024), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf117 = reinterpret_tensor(buf94, (1024, 3072), (3072, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), out=buf117)
        del arg79_1
        buf118 = reinterpret_tensor(buf117, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf117  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg80_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg80_1
        buf119 = reinterpret_tensor(buf99, (1024, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), out=buf119)
        del arg81_1
        buf123 = reinterpret_tensor(buf112, (1, 1024, 768), (786432, 768, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_55, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg82_1, arg83_1, arg84_1, buf123, 1024, 768, grid=grid(1024), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), out=buf124)
        del arg85_1
        buf125 = reinterpret_tensor(buf116, (1024, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf125)
        del arg87_1
        buf126 = reinterpret_tensor(buf104, (1024, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf126)
        del arg89_1
        buf127 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg86_1, buf127, 786432, grid=grid(786432), stream=stream0)
        del arg86_1
        buf128 = reinterpret_tensor(buf124, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg88_1, buf128, 786432, grid=grid(786432), stream=stream0)
        del arg88_1
        buf129 = reinterpret_tensor(buf125, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg90_1, buf129, 786432, grid=grid(786432), stream=stream0)
        del arg90_1
        # Source Nodes: [], Original ATen: []
        buf130 = aten._scaled_dot_product_efficient_attention(buf127, buf128, buf129, None, True, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf131, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf135, 786432, grid=grid(786432), stream=stream0)
        buf136 = reinterpret_tensor(buf129, (1024, 768), (768, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1024, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf136)
        del arg91_1
        buf140 = reinterpret_tensor(buf135, (1, 1024, 768), (786432, 768, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_60, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg92_1, arg93_1, arg94_1, buf140, 1024, 768, grid=grid(1024), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf141 = reinterpret_tensor(buf118, (1024, 3072), (3072, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (1024, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), out=buf141)
        del arg95_1
        buf142 = reinterpret_tensor(buf141, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg96_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg96_1
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), out=buf143)
        del arg97_1
        buf171 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg98_1, arg99_1, arg100_1, buf171, 1024, 768, grid=grid(1024), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf147 = empty((1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
        triton_per_fused_eq_masked_fill_ne_sum_6.run(arg262_1, buf147, 1, 1024, grid=grid(1), stream=stream0)
        buf148 = reinterpret_tensor(buf143, (1, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
        buf152 = buf140; del buf140  # reuse
        # Source Nodes: [add_15, clone_1, eq, hidden_states_69, hidden_states_70, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.select_scatter, aten.slice_scatter]
        triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_7.run(buf147, arg262_1, arg101_1, arg1_1, arg102_1, arg103_1, buf148, buf152, 1024, 768, grid=grid(1024), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg1_1
        del buf147
        buf153 = reinterpret_tensor(buf148, (1024, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), out=buf153)
        del arg104_1
        buf154 = reinterpret_tensor(buf128, (1024, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), out=buf154)
        del arg106_1
        buf155 = buf127; del buf127  # reuse
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf154, arg107_1, buf155, 786432, grid=grid(786432), stream=stream0)
        del arg107_1
        buf156 = reinterpret_tensor(buf154, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf154  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf153, arg105_1, buf156, 786432, grid=grid(786432), stream=stream0)
        del arg105_1
        buf157 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf155, (12, 64, 1024), (65536, 1, 64), 0), out=buf157)
        buf162 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf157, buf162, 12288, 1024, grid=grid(12288), stream=stream0)
        buf160 = reinterpret_tensor(buf156, (1024, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf160)
        del arg108_1
        buf161 = reinterpret_tensor(buf153, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf160, arg109_1, buf161, 786432, grid=grid(786432), stream=stream0)
        del arg109_1
        buf163 = reinterpret_tensor(buf160, (12, 1024, 64), (65536, 64, 1), 0); del buf160  # reuse
        # Source Nodes: [attn_output_30, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf162, reinterpret_tensor(buf161, (12, 1024, 64), (65536, 64, 1), 0), out=buf163)
        buf164 = reinterpret_tensor(buf126, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf126  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf163, buf164, 786432, grid=grid(786432), stream=stream0)
        buf165 = reinterpret_tensor(buf163, (1024, 768), (768, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), out=buf165)
        del arg110_1
        buf169 = reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0); del buf164  # reuse
        # Source Nodes: [hidden_states_74, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf152, buf165, arg111_1, arg112_1, arg113_1, buf169, 1024, 768, grid=grid(1024), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        buf170 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), out=buf170)
        del arg114_1
        buf172 = reinterpret_tensor(buf152, (1024, 768), (768, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf172)
        del arg116_1
        buf173 = reinterpret_tensor(buf102, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf172, arg117_1, buf173, 786432, grid=grid(786432), stream=stream0)
        del arg117_1
        buf174 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf174)
        del arg118_1
        buf175 = reinterpret_tensor(buf78, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf174, arg119_1, buf175, 786432, grid=grid(786432), stream=stream0)
        del arg119_1
        buf176 = reinterpret_tensor(buf174, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf170, arg115_1, buf176, 786432, grid=grid(786432), stream=stream0)
        del arg115_1
        # Source Nodes: [], Original ATen: []
        buf177 = aten._scaled_dot_product_efficient_attention(buf176, reinterpret_tensor(buf173, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf175, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf178 = buf177[0]
        del buf177
        buf182 = reinterpret_tensor(buf178, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf182, 786432, grid=grid(786432), stream=stream0)
        buf183 = reinterpret_tensor(buf176, (1024, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), out=buf183)
        del arg120_1
        buf187 = reinterpret_tensor(buf182, (1, 1024, 768), (786432, 768, 1), 0); del buf182  # reuse
        # Source Nodes: [hidden_states_78, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf169, buf183, arg121_1, arg122_1, arg123_1, buf187, 1024, 768, grid=grid(1024), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        buf188 = reinterpret_tensor(buf142, (1024, 3072), (3072, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1024, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg124_1
        buf189 = reinterpret_tensor(buf188, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf188  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf189, arg125_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg125_1
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg126_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg126_1
        buf194 = buf169; del buf169  # reuse
        # Source Nodes: [hidden_states_84, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf187, buf190, arg127_1, arg128_1, arg129_1, buf194, 1024, 768, grid=grid(1024), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        buf195 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 768), (1, 768), 0), out=buf195)
        del arg130_1
        buf196 = reinterpret_tensor(buf187, (1024, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), out=buf196)
        del arg132_1
        buf197 = reinterpret_tensor(buf170, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf196, arg133_1, buf197, 786432, grid=grid(786432), stream=stream0)
        del arg133_1
        buf198 = reinterpret_tensor(buf196, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf195, arg131_1, buf198, 786432, grid=grid(786432), stream=stream0)
        del arg131_1
        buf199 = buf162; del buf162  # reuse
        # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf197, (12, 64, 1024), (65536, 1, 64), 0), out=buf199)
        buf204 = buf157; del buf157  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf199, buf204, 12288, 1024, grid=grid(12288), stream=stream0)
        buf202 = reinterpret_tensor(buf198, (1024, 768), (768, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), out=buf202)
        del arg134_1
        buf203 = reinterpret_tensor(buf195, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf195  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf202, arg135_1, buf203, 786432, grid=grid(786432), stream=stream0)
        del arg135_1
        buf205 = reinterpret_tensor(buf202, (12, 1024, 64), (65536, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [attn_output_40, attn_weights_21], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf204, reinterpret_tensor(buf203, (12, 1024, 64), (65536, 64, 1), 0), out=buf205)
        buf206 = reinterpret_tensor(buf54, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf205, buf206, 786432, grid=grid(786432), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (1024, 768), (768, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1024, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), out=buf207)
        del arg136_1
        buf211 = reinterpret_tensor(buf206, (1, 1024, 768), (786432, 768, 1), 0); del buf206  # reuse
        # Source Nodes: [hidden_states_89, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf194, buf207, arg137_1, arg138_1, arg139_1, buf211, 1024, 768, grid=grid(1024), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        buf212 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (1024, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf212)
        del arg140_1
        buf213 = reinterpret_tensor(buf194, (1024, 768), (768, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), out=buf213)
        del arg142_1
        buf214 = reinterpret_tensor(buf30, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf213, arg143_1, buf214, 786432, grid=grid(786432), stream=stream0)
        del arg143_1
        buf215 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 768), (1, 768), 0), out=buf215)
        del arg144_1
        buf216 = reinterpret_tensor(buf6, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf215, arg145_1, buf216, 786432, grid=grid(786432), stream=stream0)
        del arg145_1
        buf217 = reinterpret_tensor(buf215, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf212, arg141_1, buf217, 786432, grid=grid(786432), stream=stream0)
        del arg141_1
        # Source Nodes: [], Original ATen: []
        buf218 = aten._scaled_dot_product_efficient_attention(buf217, reinterpret_tensor(buf214, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf216, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf219 = buf218[0]
        del buf218
        buf223 = reinterpret_tensor(buf219, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf219  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf223, 786432, grid=grid(786432), stream=stream0)
        buf224 = reinterpret_tensor(buf217, (1024, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (1024, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), out=buf224)
        del arg146_1
        buf228 = reinterpret_tensor(buf223, (1, 1024, 768), (786432, 768, 1), 0); del buf223  # reuse
        # Source Nodes: [hidden_states_93, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf211, buf224, arg147_1, arg148_1, arg149_1, buf228, 1024, 768, grid=grid(1024), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        buf229 = reinterpret_tensor(buf189, (1024, 3072), (3072, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1024, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 3072), (1, 768), 0), out=buf229)
        del arg150_1
        buf230 = reinterpret_tensor(buf229, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf229  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf230, arg151_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg151_1
        buf231 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg152_1, (3072, 768), (1, 3072), 0), out=buf231)
        del arg152_1
        buf235 = buf211; del buf211  # reuse
        # Source Nodes: [hidden_states_99, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf228, buf231, arg153_1, arg154_1, arg155_1, buf235, 1024, 768, grid=grid(1024), stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        buf236 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), out=buf236)
        del arg156_1
        buf237 = reinterpret_tensor(buf228, (1024, 768), (768, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf237)
        del arg158_1
        buf238 = reinterpret_tensor(buf212, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf212  # reuse
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf237, arg159_1, buf238, 786432, grid=grid(786432), stream=stream0)
        del arg159_1
        buf239 = reinterpret_tensor(buf237, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf236, arg157_1, buf239, 786432, grid=grid(786432), stream=stream0)
        del arg157_1
        buf240 = buf204; del buf204  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf238, (12, 64, 1024), (65536, 1, 64), 0), out=buf240)
        buf245 = buf199; del buf199  # reuse
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf240, buf245, 12288, 1024, grid=grid(12288), stream=stream0)
        buf243 = reinterpret_tensor(buf239, (1024, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), out=buf243)
        del arg160_1
        buf244 = reinterpret_tensor(buf236, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf243, arg161_1, buf244, 786432, grid=grid(786432), stream=stream0)
        del arg161_1
        buf246 = reinterpret_tensor(buf243, (12, 1024, 64), (65536, 64, 1), 0); del buf243  # reuse
        # Source Nodes: [attn_output_50, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf245, reinterpret_tensor(buf244, (12, 1024, 64), (65536, 64, 1), 0), out=buf246)
        buf247 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf246, buf247, 786432, grid=grid(786432), stream=stream0)
        buf248 = reinterpret_tensor(buf246, (1024, 768), (768, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1024, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), out=buf248)
        del arg162_1
        buf252 = reinterpret_tensor(buf247, (1, 1024, 768), (786432, 768, 1), 0); del buf247  # reuse
        # Source Nodes: [hidden_states_104, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf235, buf248, arg163_1, arg164_1, arg165_1, buf252, 1024, 768, grid=grid(1024), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        buf253 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (1024, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), out=buf253)
        del arg166_1
        buf254 = reinterpret_tensor(buf235, (1024, 768), (768, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), out=buf254)
        del arg168_1
        buf255 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf254, arg169_1, buf255, 786432, grid=grid(786432), stream=stream0)
        del arg169_1
        buf256 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), out=buf256)
        del arg170_1
        buf257 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf256, arg171_1, buf257, 786432, grid=grid(786432), stream=stream0)
        del arg171_1
        buf258 = reinterpret_tensor(buf256, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf253, arg167_1, buf258, 786432, grid=grid(786432), stream=stream0)
        del arg167_1
        # Source Nodes: [], Original ATen: []
        buf259 = aten._scaled_dot_product_efficient_attention(buf258, reinterpret_tensor(buf255, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf257, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf260 = buf259[0]
        del buf259
        buf264 = reinterpret_tensor(buf260, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf264, 786432, grid=grid(786432), stream=stream0)
        buf265 = reinterpret_tensor(buf258, (1024, 768), (768, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (1024, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf265)
        del arg172_1
        buf269 = reinterpret_tensor(buf264, (1, 1024, 768), (786432, 768, 1), 0); del buf264  # reuse
        # Source Nodes: [hidden_states_108, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf252, buf265, arg173_1, arg174_1, arg175_1, buf269, 1024, 768, grid=grid(1024), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        buf270 = reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (1024, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 3072), (1, 768), 0), out=buf270)
        del arg176_1
        buf271 = reinterpret_tensor(buf270, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf270  # reuse
        # Source Nodes: [hidden_states_110], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf271, arg177_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg177_1
        buf272 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), out=buf272)
        del arg178_1
        buf276 = buf252; del buf252  # reuse
        # Source Nodes: [hidden_states_114, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf269, buf272, arg179_1, arg180_1, arg181_1, buf276, 1024, 768, grid=grid(1024), stream=stream0)
        del arg179_1
        del arg180_1
        del arg181_1
        buf277 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1024, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), out=buf277)
        del arg182_1
        buf278 = reinterpret_tensor(buf269, (1024, 768), (768, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1024, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), out=buf278)
        del arg184_1
        buf279 = reinterpret_tensor(buf253, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf278, arg185_1, buf279, 786432, grid=grid(786432), stream=stream0)
        del arg185_1
        buf280 = reinterpret_tensor(buf278, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf277, arg183_1, buf280, 786432, grid=grid(786432), stream=stream0)
        del arg183_1
        buf281 = buf245; del buf245  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf279, (12, 64, 1024), (65536, 1, 64), 0), out=buf281)
        buf286 = buf240; del buf240  # reuse
        # Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf281, buf286, 12288, 1024, grid=grid(12288), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (1024, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1024, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), out=buf284)
        del arg186_1
        buf285 = reinterpret_tensor(buf277, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf277  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf284, arg187_1, buf285, 786432, grid=grid(786432), stream=stream0)
        del arg187_1
        buf287 = reinterpret_tensor(buf284, (12, 1024, 64), (65536, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [attn_output_60, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf286, reinterpret_tensor(buf285, (12, 1024, 64), (65536, 64, 1), 0), out=buf287)
        buf288 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf287, buf288, 786432, grid=grid(786432), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (1024, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (1024, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), out=buf289)
        del arg188_1
        buf293 = reinterpret_tensor(buf288, (1, 1024, 768), (786432, 768, 1), 0); del buf288  # reuse
        # Source Nodes: [hidden_states_119, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf276, buf289, arg189_1, arg190_1, arg191_1, buf293, 1024, 768, grid=grid(1024), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        buf294 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (1024, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 768), (1, 768), 0), out=buf294)
        del arg192_1
        buf295 = reinterpret_tensor(buf276, (1024, 768), (768, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg194_1, (768, 768), (1, 768), 0), out=buf295)
        del arg194_1
        buf296 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf295, arg195_1, buf296, 786432, grid=grid(786432), stream=stream0)
        del arg195_1
        buf297 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 768), (1, 768), 0), out=buf297)
        del arg196_1
        buf298 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf297, arg197_1, buf298, 786432, grid=grid(786432), stream=stream0)
        del arg197_1
        buf299 = reinterpret_tensor(buf297, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf294, arg193_1, buf299, 786432, grid=grid(786432), stream=stream0)
        del arg193_1
        # Source Nodes: [], Original ATen: []
        buf300 = aten._scaled_dot_product_efficient_attention(buf299, reinterpret_tensor(buf296, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf298, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf301 = buf300[0]
        del buf300
        buf305 = reinterpret_tensor(buf301, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf301  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf305, 786432, grid=grid(786432), stream=stream0)
        buf306 = reinterpret_tensor(buf299, (1024, 768), (768, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf305, (1024, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 768), (1, 768), 0), out=buf306)
        del arg198_1
        buf310 = reinterpret_tensor(buf305, (1, 1024, 768), (786432, 768, 1), 0); del buf305  # reuse
        # Source Nodes: [hidden_states_123, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf293, buf306, arg199_1, arg200_1, arg201_1, buf310, 1024, 768, grid=grid(1024), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        buf311 = reinterpret_tensor(buf271, (1024, 3072), (3072, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1024, 768), (768, 1), 0), reinterpret_tensor(arg202_1, (768, 3072), (1, 768), 0), out=buf311)
        del arg202_1
        buf312 = reinterpret_tensor(buf311, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf311  # reuse
        # Source Nodes: [hidden_states_125], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf312, arg203_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg203_1
        buf313 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg204_1, (3072, 768), (1, 3072), 0), out=buf313)
        del arg204_1
        buf317 = buf293; del buf293  # reuse
        # Source Nodes: [hidden_states_129, residual_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf310, buf313, arg205_1, arg206_1, arg207_1, buf317, 1024, 768, grid=grid(1024), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        buf318 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1024, 768), (768, 1), 0), reinterpret_tensor(arg208_1, (768, 768), (1, 768), 0), out=buf318)
        del arg208_1
        buf319 = reinterpret_tensor(buf310, (1024, 768), (768, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1024, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 768), (1, 768), 0), out=buf319)
        del arg210_1
        buf320 = reinterpret_tensor(buf294, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf294  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf319, arg211_1, buf320, 786432, grid=grid(786432), stream=stream0)
        del arg211_1
        buf321 = reinterpret_tensor(buf319, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf319  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf318, arg209_1, buf321, 786432, grid=grid(786432), stream=stream0)
        del arg209_1
        buf322 = buf286; del buf286  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf320, (12, 64, 1024), (65536, 1, 64), 0), out=buf322)
        buf327 = buf281; del buf281  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf322, buf327, 12288, 1024, grid=grid(12288), stream=stream0)
        buf325 = reinterpret_tensor(buf321, (1024, 768), (768, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1024, 768), (768, 1), 0), reinterpret_tensor(arg212_1, (768, 768), (1, 768), 0), out=buf325)
        del arg212_1
        buf326 = reinterpret_tensor(buf318, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf318  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf325, arg213_1, buf326, 786432, grid=grid(786432), stream=stream0)
        del arg213_1
        buf328 = reinterpret_tensor(buf325, (12, 1024, 64), (65536, 64, 1), 0); del buf325  # reuse
        # Source Nodes: [attn_output_70, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf327, reinterpret_tensor(buf326, (12, 1024, 64), (65536, 64, 1), 0), out=buf328)
        buf329 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf328, buf329, 786432, grid=grid(786432), stream=stream0)
        buf330 = reinterpret_tensor(buf328, (1024, 768), (768, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (1024, 768), (768, 1), 0), reinterpret_tensor(arg214_1, (768, 768), (1, 768), 0), out=buf330)
        del arg214_1
        buf334 = reinterpret_tensor(buf329, (1, 1024, 768), (786432, 768, 1), 0); del buf329  # reuse
        # Source Nodes: [hidden_states_134, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf317, buf330, arg215_1, arg216_1, arg217_1, buf334, 1024, 768, grid=grid(1024), stream=stream0)
        del arg215_1
        del arg216_1
        del arg217_1
        buf335 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1024, 768), (768, 1), 0), reinterpret_tensor(arg218_1, (768, 768), (1, 768), 0), out=buf335)
        del arg218_1
        buf336 = reinterpret_tensor(buf317, (1024, 768), (768, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg220_1, (768, 768), (1, 768), 0), out=buf336)
        del arg220_1
        buf337 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf336, arg221_1, buf337, 786432, grid=grid(786432), stream=stream0)
        del arg221_1
        buf338 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 768), (1, 768), 0), out=buf338)
        del arg222_1
        buf339 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf338, arg223_1, buf339, 786432, grid=grid(786432), stream=stream0)
        del arg223_1
        buf340 = reinterpret_tensor(buf338, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf335, arg219_1, buf340, 786432, grid=grid(786432), stream=stream0)
        del arg219_1
        # Source Nodes: [], Original ATen: []
        buf341 = aten._scaled_dot_product_efficient_attention(buf340, reinterpret_tensor(buf337, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf339, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf342 = buf341[0]
        del buf341
        buf346 = reinterpret_tensor(buf342, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf346, 786432, grid=grid(786432), stream=stream0)
        buf347 = reinterpret_tensor(buf340, (1024, 768), (768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (1024, 768), (768, 1), 0), reinterpret_tensor(arg224_1, (768, 768), (1, 768), 0), out=buf347)
        del arg224_1
        buf351 = reinterpret_tensor(buf346, (1, 1024, 768), (786432, 768, 1), 0); del buf346  # reuse
        # Source Nodes: [hidden_states_138, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf334, buf347, arg225_1, arg226_1, arg227_1, buf351, 1024, 768, grid=grid(1024), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        buf352 = reinterpret_tensor(buf312, (1024, 3072), (3072, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (1024, 768), (768, 1), 0), reinterpret_tensor(arg228_1, (768, 3072), (1, 768), 0), out=buf352)
        del arg228_1
        buf353 = reinterpret_tensor(buf352, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf352  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf353, arg229_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg229_1
        buf354 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg230_1, (3072, 768), (1, 3072), 0), out=buf354)
        del arg230_1
        buf358 = buf334; del buf334  # reuse
        # Source Nodes: [hidden_states_144, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf351, buf354, arg231_1, arg232_1, arg233_1, buf358, 1024, 768, grid=grid(1024), stream=stream0)
        del arg231_1
        del arg232_1
        del arg233_1
        buf359 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1024, 768), (768, 1), 0), reinterpret_tensor(arg234_1, (768, 768), (1, 768), 0), out=buf359)
        del arg234_1
        buf360 = reinterpret_tensor(buf351, (1024, 768), (768, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1024, 768), (768, 1), 0), reinterpret_tensor(arg236_1, (768, 768), (1, 768), 0), out=buf360)
        del arg236_1
        buf361 = reinterpret_tensor(buf335, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf335  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf360, arg237_1, buf361, 786432, grid=grid(786432), stream=stream0)
        del arg237_1
        buf362 = reinterpret_tensor(buf360, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf360  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf359, arg235_1, buf362, 786432, grid=grid(786432), stream=stream0)
        del arg235_1
        buf363 = buf327; del buf327  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf361, (12, 64, 1024), (65536, 1, 64), 0), out=buf363)
        buf368 = buf322; del buf322  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf363, buf368, 12288, 1024, grid=grid(12288), stream=stream0)
        del buf363
        buf366 = reinterpret_tensor(buf362, (1024, 768), (768, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1024, 768), (768, 1), 0), reinterpret_tensor(arg238_1, (768, 768), (1, 768), 0), out=buf366)
        del arg238_1
        buf367 = reinterpret_tensor(buf359, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf359  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf366, arg239_1, buf367, 786432, grid=grid(786432), stream=stream0)
        del arg239_1
        buf369 = reinterpret_tensor(buf366, (12, 1024, 64), (65536, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [attn_output_80, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf368, reinterpret_tensor(buf367, (12, 1024, 64), (65536, 64, 1), 0), out=buf369)
        del buf368
        buf370 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf369, buf370, 786432, grid=grid(786432), stream=stream0)
        buf371 = reinterpret_tensor(buf369, (1024, 768), (768, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (1024, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 768), (1, 768), 0), out=buf371)
        del arg240_1
        buf375 = reinterpret_tensor(buf370, (1, 1024, 768), (786432, 768, 1), 0); del buf370  # reuse
        # Source Nodes: [hidden_states_149, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf358, buf371, arg241_1, arg242_1, arg243_1, buf375, 1024, 768, grid=grid(1024), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        buf376 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), reinterpret_tensor(arg244_1, (768, 768), (1, 768), 0), out=buf376)
        del arg244_1
        buf377 = reinterpret_tensor(buf358, (1024, 768), (768, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg246_1, (768, 768), (1, 768), 0), out=buf377)
        del arg246_1
        buf378 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf377, arg247_1, buf378, 786432, grid=grid(786432), stream=stream0)
        del arg247_1
        buf379 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 768), (1, 768), 0), out=buf379)
        del arg248_1
        buf380 = empty((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf379, arg249_1, buf380, 786432, grid=grid(786432), stream=stream0)
        del arg249_1
        buf381 = reinterpret_tensor(buf379, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf376, arg245_1, buf381, 786432, grid=grid(786432), stream=stream0)
        del arg245_1
        del buf376
        # Source Nodes: [], Original ATen: []
        buf382 = aten._scaled_dot_product_efficient_attention(buf381, reinterpret_tensor(buf378, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf380, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), None, True, scale=1.0)
        buf383 = buf382[0]
        del buf382
        buf387 = reinterpret_tensor(buf383, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf383  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf387, 786432, grid=grid(786432), stream=stream0)
        buf388 = reinterpret_tensor(buf381, (1024, 768), (768, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (1024, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 768), (1, 768), 0), out=buf388)
        del arg250_1
        buf392 = reinterpret_tensor(buf387, (1, 1024, 768), (786432, 768, 1), 0); del buf387  # reuse
        # Source Nodes: [hidden_states_153, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf375, buf388, arg251_1, arg252_1, arg253_1, buf392, 1024, 768, grid=grid(1024), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        buf393 = reinterpret_tensor(buf353, (1024, 3072), (3072, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (1024, 768), (768, 1), 0), reinterpret_tensor(arg254_1, (768, 3072), (1, 768), 0), out=buf393)
        del arg254_1
        buf394 = reinterpret_tensor(buf393, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf393  # reuse
        # Source Nodes: [hidden_states_155], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf394, arg255_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg255_1
        buf395 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg256_1, (3072, 768), (1, 3072), 0), out=buf395)
        del arg256_1
        del buf394
        buf399 = buf375; del buf375  # reuse
        # Source Nodes: [hidden_states_159, hidden_states_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf392, buf395, arg257_1, arg258_1, arg259_1, buf399, 1024, 768, grid=grid(1024), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del buf392
        del buf395
        buf400 = empty((1024, 50005), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (1024, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 50005), (1, 768), 0), out=buf400)
        del arg260_1
        del buf399
        buf401 = reinterpret_tensor(buf400, (1, 1024, 50005), (51205120, 50005, 1), 0); del buf400  # reuse
        buf402 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf403 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
        triton_red_fused__log_softmax_add_10.run(buf401, arg261_1, buf402, buf403, 1024, 50005, grid=grid(1024), stream=stream0)
        del arg261_1
        buf404 = empty((), device='cuda', dtype=torch.float32)
        buf406 = buf404; del buf404  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_11.run(buf406, arg262_1, buf401, buf402, buf403, 1, 1024, grid=grid(1), stream=stream0)
        del arg262_1
        return (buf406, buf401, buf155, buf161, buf173, buf175, buf197, buf203, buf214, buf216, buf238, buf244, buf255, buf257, buf279, buf285, buf296, buf298, buf320, buf326, buf337, buf339, buf361, buf367, buf378, buf380, buf171, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((50005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((50005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1, 50005), (50005, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PLBartForConditionalGeneration', benchmark_compiled_module)
