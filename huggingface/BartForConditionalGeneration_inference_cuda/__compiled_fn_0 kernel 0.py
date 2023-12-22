
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cghpx7275xpyln7w3xlimjx6gj7z6kqfzztaeh6vax5yeezxxotx.py
# Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# embed_pos => embedding_1
# hidden_states => add_1
# hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
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
    rnumel = 1024
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
        tmp7 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
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
        tmp19 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50265
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp15 < 50265")
        tmp16 = tl.load(in_ptr1 + (r1 + (1024*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 1024.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjtehi6rawy63zao4ubikl36xv2nl2akwjjkk4n7iox274fnzfp.py
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3ahz23apcnlfd5yqrcijazvlzqco45saoc7lywt2h3wjplicekf.py
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2zp6t2imrj2ozm4f4a5ostas6bgyl6qhloivn4mdb7d6kk44l2.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_6
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eu6wdnvxiydpqqrsbtdgvoduex3f4yce4uvfbhpvjfbrsgw625.py
# Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_5 => add_4
# residual_1 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
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
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvlwfokqqd73vj3zmaxeyv3xsy6jttgqmrj4vr5wgq5frop2xij.py
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
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/hy/chy7iv352mc6svwxtqiclgfcn4jyxppdliunermbxfnza6nc4wh2.py
# Source Nodes: [add_27, clone, eq, hidden_states_135, hidden_states_136, input_2, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.fill, aten.lift_fresh, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
# add_27 => add_89
# clone => clone
# eq => eq
# hidden_states_135 => add_90
# hidden_states_136 => add_91, add_92, mul_100, mul_101, rsqrt_25, sub_37, var_mean_25
# input_2 => full
# inputs_embeds_1 => mul_99
# l__mod___model_decoder_embed_tokens => embedding_2
# masked_fill_ => full_default_1, where
# positions_2 => embedding_3
# setitem => copy, slice_scatter
# setitem_1 => copy_1, full_default, select_scatter
triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_select_scatter_slice_scatter_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_select_scatter_slice_scatter_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 >= tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tl.full([1, 1], 0, tl.int64)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tl.full([1, 1], 2, tl.int64)
        tmp11 = tl.where(tmp2, tmp10, tmp9)
        tmp12 = tl.full([1, 1], -100, tl.int64)
        tmp13 = tmp11 == tmp12
        tmp14 = tl.where(tmp13, tmp3, tmp11)
        tmp15 = tmp14 + 50265
        tmp16 = tmp14 < 0
        tmp17 = tl.where(tmp16, tmp15, tmp14)
        tl.device_assert((0 <= tmp17) & (tmp17 < 50265), "index out of bounds: 0 <= tmp17 < 50265")
        tmp18 = tl.load(in_ptr1 + (r1 + (1024*tmp17)), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = x0
        tmp28 = tl.full([1, 1], 0, tl.int32)
        tmp29 = tmp27 == tmp28
        tmp30 = tl.full([1, 1], 1, tl.int64)
        tmp31 = tmp27 >= tmp30
        tmp32 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x0, [XBLOCK, RBLOCK])), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.full(tmp32.shape, 0, tmp32.dtype)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = tl.full([1, 1], 0, tl.int64)
        tmp36 = tl.where(tmp31, tmp34, tmp35)
        tmp37 = tl.full([1, 1], 2, tl.int64)
        tmp38 = tl.where(tmp29, tmp37, tmp36)
        tmp39 = tl.full([1, 1], -100, tl.int64)
        tmp40 = tmp38 == tmp39
        tmp41 = tl.where(tmp40, tmp30, tmp38)
        tmp42 = tmp41 + 50265
        tmp43 = tmp41 < 0
        tmp44 = tl.where(tmp43, tmp42, tmp41)
        tl.device_assert((0 <= tmp44) & (tmp44 < 50265), "index out of bounds: 0 <= tmp44 < 50265")
        tmp45 = tl.load(in_ptr1 + (r1 + (1024*tmp44)), rmask, eviction_policy='evict_first', other=0.0)
        tmp46 = 1.0
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tmp50 = tmp49 - tmp24
        tmp51 = 1024.0
        tmp52 = tmp25 / tmp51
        tmp53 = 1e-05
        tmp54 = tmp52 + tmp53
        tmp55 = tl.math.rsqrt(tmp54)
        tmp56 = tmp50 * tmp55
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp60, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywpekihgqka6wji5ygazgekexihqbolzxyzzh3fdqb5cnmfseaw.py
# Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
# attn_weights_27 => amax_12, div_12, exp_12, sub_38, sum_13
triton_per_fused__softmax_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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


# kernel path: /tmp/torchinductor_youkaichao/if/cif7o64t3b2oktohctcjgp6s5uoym7uhqk6uq7ans4vncxjxvpxt.py
# Source Nodes: [attn_output_63], Original ATen: [aten.clone]
# attn_output_63 => clone_103
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfofj5nxfgb7egq4kjqmr2ennji3ca7cgrfic2dicvjxx3zzo6j.py
# Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
# lm_logits_1 => add_225
# masked_lm_loss => amax_36, exp_36, sub_98, sum_37
triton_red_fused__log_softmax_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 50265
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
        tmp0 = tl.load(in_out_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r1 + (50265*x0)), tmp2, rmask & xmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 - tmp4
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capiom43odqw2vtqswiadcpxmdtcv7qaamxeg5hl4jmth3dz3a6b.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_36, full_default_5, ne_1, ne_2, neg, sum_38, sum_39, where_3
triton_per_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = tmp4 + 50265
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 50265), "index out of bounds: 0 <= tmp7 < 50265")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (50265*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg96_1, (4096, ), (1, ))
    assert_size_stride(arg97_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg112_1, (4096, ), (1, ))
    assert_size_stride(arg113_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg128_1, (4096, ), (1, ))
    assert_size_stride(arg129_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg144_1, (4096, ), (1, ))
    assert_size_stride(arg145_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg160_1, (4096, ), (1, ))
    assert_size_stride(arg161_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg176_1, (4096, ), (1, ))
    assert_size_stride(arg177_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg192_1, (4096, ), (1, ))
    assert_size_stride(arg193_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg221_1, (4096, ), (1, ))
    assert_size_stride(arg222_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg247_1, (4096, ), (1, ))
    assert_size_stride(arg248_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg273_1, (4096, ), (1, ))
    assert_size_stride(arg274_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (1024, ), (1, ))
    assert_size_stride(arg278_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg299_1, (4096, ), (1, ))
    assert_size_stride(arg300_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg325_1, (4096, ), (1, ))
    assert_size_stride(arg326_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg351_1, (4096, ), (1, ))
    assert_size_stride(arg352_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg377_1, (4096, ), (1, ))
    assert_size_stride(arg378_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg389_1, (1024, ), (1, ))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg397_1, (1024, ), (1, ))
    assert_size_stride(arg398_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg399_1, (1024, ), (1, ))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg403_1, (4096, ), (1, ))
    assert_size_stride(arg404_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (1024, ), (1, ))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg409_1, (1024, ), (1, ))
    assert_size_stride(arg410_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg411_1, (1024, ), (1, ))
    assert_size_stride(arg412_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg415_1, (1024, ), (1, ))
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (1024, ), (1, ))
    assert_size_stride(arg427_1, (1024, ), (1, ))
    assert_size_stride(arg428_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg429_1, (4096, ), (1, ))
    assert_size_stride(arg430_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (1024, ), (1, ))
    assert_size_stride(arg434_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg435_1, (1024, ), (1, ))
    assert_size_stride(arg436_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg439_1, (1024, ), (1, ))
    assert_size_stride(arg440_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (1024, ), (1, ))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (1024, ), (1, ))
    assert_size_stride(arg454_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg455_1, (4096, ), (1, ))
    assert_size_stride(arg456_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg457_1, (1024, ), (1, ))
    assert_size_stride(arg458_1, (1024, ), (1, ))
    assert_size_stride(arg459_1, (1024, ), (1, ))
    assert_size_stride(arg460_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg463_1, (1024, ), (1, ))
    assert_size_stride(arg464_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg471_1, (1024, ), (1, ))
    assert_size_stride(arg472_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg475_1, (1024, ), (1, ))
    assert_size_stride(arg476_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (1024, ), (1, ))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg481_1, (4096, ), (1, ))
    assert_size_stride(arg482_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (1024, ), (1, ))
    assert_size_stride(arg485_1, (1024, ), (1, ))
    assert_size_stride(arg486_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg499_1, (1024, ), (1, ))
    assert_size_stride(arg500_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg501_1, (1024, ), (1, ))
    assert_size_stride(arg502_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg507_1, (4096, ), (1, ))
    assert_size_stride(arg508_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg509_1, (1024, ), (1, ))
    assert_size_stride(arg510_1, (1024, ), (1, ))
    assert_size_stride(arg511_1, (1024, ), (1, ))
    assert_size_stride(arg512_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg513_1, (1, 50265), (50265, 1))
    assert_size_stride(arg514_1, (1, 1024), (1024, 1))
    assert_size_stride(arg515_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg515_1, arg2_1, arg0_1, arg3_1, arg4_1, buf3, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg515_1
        buf4 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg5_1
        buf5 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg7_1
        buf6 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf6)
        del arg9_1
        buf7 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg6_1, buf7, 1048576, grid=grid(1048576), stream=stream0)
        del arg6_1
        buf8 = reinterpret_tensor(buf4, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg8_1, buf8, 1048576, grid=grid(1048576), stream=stream0)
        del arg8_1
        buf9 = reinterpret_tensor(buf5, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg10_1, buf9, 1048576, grid=grid(1048576), stream=stream0)
        del arg10_1
        del buf6
        # Source Nodes: [], Original ATen: []
        buf10 = aten._scaled_dot_product_efficient_attention(buf7, buf8, buf9, None, True, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf11, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf15, 1048576, grid=grid(1048576), stream=stream0)
        buf16 = reinterpret_tensor(buf9, (1024, 1024), (1024, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg11_1
        buf20 = reinterpret_tensor(buf15, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg12_1, arg13_1, arg14_1, buf20, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf21 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg15_1, (1024, 4096), (1, 1024), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf21, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg16_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg16_1
        buf23 = reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 1024), (1, 4096), 0), out=buf23)
        del arg17_1
        buf27 = reinterpret_tensor(buf16, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg18_1, arg19_1, arg20_1, buf27, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), out=buf28)
        del arg21_1
        buf29 = reinterpret_tensor(buf20, (1024, 1024), (1024, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg23_1
        buf30 = reinterpret_tensor(buf8, (1024, 1024), (1024, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg25_1
        buf31 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg22_1, buf31, 1048576, grid=grid(1048576), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf28, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg24_1, buf32, 1048576, grid=grid(1048576), stream=stream0)
        del arg24_1
        buf33 = reinterpret_tensor(buf29, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg26_1, buf33, 1048576, grid=grid(1048576), stream=stream0)
        del arg26_1
        del buf30
        # Source Nodes: [], Original ATen: []
        buf34 = aten._scaled_dot_product_efficient_attention(buf31, buf32, buf33, None, True, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf35, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf39, 1048576, grid=grid(1048576), stream=stream0)
        buf40 = reinterpret_tensor(buf33, (1024, 1024), (1024, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), out=buf40)
        del arg27_1
        buf44 = reinterpret_tensor(buf39, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_16, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg28_1, arg29_1, arg30_1, buf44, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf45 = reinterpret_tensor(buf22, (1024, 4096), (4096, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg31_1, (1024, 4096), (1, 1024), 0), out=buf45)
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg32_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg32_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg33_1, (4096, 1024), (1, 4096), 0), out=buf47)
        del arg33_1
        buf51 = buf27; del buf27  # reuse
        # Source Nodes: [hidden_states_22, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg34_1, arg35_1, arg36_1, buf51, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), out=buf52)
        del arg37_1
        buf53 = reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf53)
        del arg39_1
        buf54 = reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg41_1
        buf55 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg38_1, buf55, 1048576, grid=grid(1048576), stream=stream0)
        del arg38_1
        buf56 = reinterpret_tensor(buf52, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg40_1, buf56, 1048576, grid=grid(1048576), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf53, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg42_1, buf57, 1048576, grid=grid(1048576), stream=stream0)
        del arg42_1
        del buf54
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, True, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf59, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf63, 1048576, grid=grid(1048576), stream=stream0)
        buf64 = reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), out=buf64)
        del arg43_1
        buf68 = reinterpret_tensor(buf63, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf63  # reuse
        # Source Nodes: [hidden_states_27, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg44_1, arg45_1, arg46_1, buf68, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf69 = reinterpret_tensor(buf46, (1024, 4096), (4096, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg47_1, (1024, 4096), (1, 1024), 0), out=buf69)
        del arg47_1
        buf70 = reinterpret_tensor(buf69, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg48_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg48_1
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg49_1, (4096, 1024), (1, 4096), 0), out=buf71)
        del arg49_1
        buf75 = buf51; del buf51  # reuse
        # Source Nodes: [hidden_states_33, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg50_1, arg51_1, arg52_1, buf75, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), out=buf76)
        del arg53_1
        buf77 = reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf77)
        del arg55_1
        buf78 = reinterpret_tensor(buf56, (1024, 1024), (1024, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf78)
        del arg57_1
        buf79 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg54_1, buf79, 1048576, grid=grid(1048576), stream=stream0)
        del arg54_1
        buf80 = reinterpret_tensor(buf76, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg56_1, buf80, 1048576, grid=grid(1048576), stream=stream0)
        del arg56_1
        buf81 = reinterpret_tensor(buf77, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg58_1, buf81, 1048576, grid=grid(1048576), stream=stream0)
        del arg58_1
        del buf78
        # Source Nodes: [], Original ATen: []
        buf82 = aten._scaled_dot_product_efficient_attention(buf79, buf80, buf81, None, True, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf83, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf87, 1048576, grid=grid(1048576), stream=stream0)
        buf88 = reinterpret_tensor(buf81, (1024, 1024), (1024, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf87, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_38, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg60_1, arg61_1, arg62_1, buf92, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf70, (1024, 4096), (4096, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg63_1, (1024, 4096), (1, 1024), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg64_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg64_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg65_1, (4096, 1024), (1, 4096), 0), out=buf95)
        del arg65_1
        buf99 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_44, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg66_1, arg67_1, arg68_1, buf99, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), out=buf100)
        del arg69_1
        buf101 = reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf101)
        del arg71_1
        buf102 = reinterpret_tensor(buf80, (1024, 1024), (1024, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf102)
        del arg73_1
        buf103 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg70_1, buf103, 1048576, grid=grid(1048576), stream=stream0)
        del arg70_1
        buf104 = reinterpret_tensor(buf100, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg72_1, buf104, 1048576, grid=grid(1048576), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf101, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg74_1, buf105, 1048576, grid=grid(1048576), stream=stream0)
        del arg74_1
        del buf102
        # Source Nodes: [], Original ATen: []
        buf106 = aten._scaled_dot_product_efficient_attention(buf103, buf104, buf105, None, True, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf107, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf111, 1048576, grid=grid(1048576), stream=stream0)
        buf112 = reinterpret_tensor(buf105, (1024, 1024), (1024, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), out=buf112)
        del arg75_1
        buf116 = reinterpret_tensor(buf111, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_49, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg76_1, arg77_1, arg78_1, buf116, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf117 = reinterpret_tensor(buf94, (1024, 4096), (4096, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg79_1, (1024, 4096), (1, 1024), 0), out=buf117)
        del arg79_1
        buf118 = reinterpret_tensor(buf117, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf117  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg80_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg80_1
        buf119 = reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg81_1, (4096, 1024), (1, 4096), 0), out=buf119)
        del arg81_1
        buf123 = reinterpret_tensor(buf112, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_55, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg82_1, arg83_1, arg84_1, buf123, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), out=buf124)
        del arg85_1
        buf125 = reinterpret_tensor(buf116, (1024, 1024), (1024, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf125)
        del arg87_1
        buf126 = reinterpret_tensor(buf104, (1024, 1024), (1024, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf126)
        del arg89_1
        buf127 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg86_1, buf127, 1048576, grid=grid(1048576), stream=stream0)
        del arg86_1
        buf128 = reinterpret_tensor(buf124, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg88_1, buf128, 1048576, grid=grid(1048576), stream=stream0)
        del arg88_1
        buf129 = reinterpret_tensor(buf125, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg90_1, buf129, 1048576, grid=grid(1048576), stream=stream0)
        del arg90_1
        del buf126
        # Source Nodes: [], Original ATen: []
        buf130 = aten._scaled_dot_product_efficient_attention(buf127, buf128, buf129, None, True, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf131, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf135, 1048576, grid=grid(1048576), stream=stream0)
        buf136 = reinterpret_tensor(buf129, (1024, 1024), (1024, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg91_1
        buf140 = reinterpret_tensor(buf135, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_60, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg92_1, arg93_1, arg94_1, buf140, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf141 = reinterpret_tensor(buf118, (1024, 4096), (4096, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg95_1, (1024, 4096), (1, 1024), 0), out=buf141)
        del arg95_1
        buf142 = reinterpret_tensor(buf141, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg96_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg96_1
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 1024), (1, 4096), 0), out=buf143)
        del arg97_1
        buf147 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg98_1, arg99_1, arg100_1, buf147, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf148 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), out=buf148)
        del arg101_1
        buf149 = reinterpret_tensor(buf140, (1024, 1024), (1024, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf149)
        del arg103_1
        buf150 = reinterpret_tensor(buf128, (1024, 1024), (1024, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf150)
        del arg105_1
        buf151 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf148, arg102_1, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del arg102_1
        buf152 = reinterpret_tensor(buf148, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf149, arg104_1, buf152, 1048576, grid=grid(1048576), stream=stream0)
        del arg104_1
        buf153 = reinterpret_tensor(buf149, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf150, arg106_1, buf153, 1048576, grid=grid(1048576), stream=stream0)
        del arg106_1
        del buf150
        # Source Nodes: [], Original ATen: []
        buf154 = aten._scaled_dot_product_efficient_attention(buf151, buf152, buf153, None, True, scale=1.0)
        buf155 = buf154[0]
        del buf154
        buf159 = reinterpret_tensor(buf155, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf159, 1048576, grid=grid(1048576), stream=stream0)
        buf160 = reinterpret_tensor(buf153, (1024, 1024), (1024, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), out=buf160)
        del arg107_1
        buf164 = reinterpret_tensor(buf159, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf159  # reuse
        # Source Nodes: [hidden_states_71, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf147, buf160, arg108_1, arg109_1, arg110_1, buf164, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        buf165 = reinterpret_tensor(buf142, (1024, 4096), (4096, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 4096), (1, 1024), 0), out=buf165)
        del arg111_1
        buf166 = reinterpret_tensor(buf165, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf166, arg112_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg112_1
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 1024), (1, 4096), 0), out=buf167)
        del arg113_1
        buf171 = buf147; del buf147  # reuse
        # Source Nodes: [hidden_states_77, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf164, buf167, arg114_1, arg115_1, arg116_1, buf171, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf172 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), out=buf172)
        del arg117_1
        buf173 = reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf173)
        del arg119_1
        buf174 = reinterpret_tensor(buf152, (1024, 1024), (1024, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf174)
        del arg121_1
        buf175 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf172, arg118_1, buf175, 1048576, grid=grid(1048576), stream=stream0)
        del arg118_1
        buf176 = reinterpret_tensor(buf172, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf173, arg120_1, buf176, 1048576, grid=grid(1048576), stream=stream0)
        del arg120_1
        buf177 = reinterpret_tensor(buf173, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf174, arg122_1, buf177, 1048576, grid=grid(1048576), stream=stream0)
        del arg122_1
        del buf174
        # Source Nodes: [], Original ATen: []
        buf178 = aten._scaled_dot_product_efficient_attention(buf175, buf176, buf177, None, True, scale=1.0)
        buf179 = buf178[0]
        del buf178
        buf183 = reinterpret_tensor(buf179, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf183, 1048576, grid=grid(1048576), stream=stream0)
        buf184 = reinterpret_tensor(buf177, (1024, 1024), (1024, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg123_1
        buf188 = reinterpret_tensor(buf183, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf183  # reuse
        # Source Nodes: [hidden_states_82, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf171, buf184, arg124_1, arg125_1, arg126_1, buf188, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf189 = reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg127_1, (1024, 4096), (1, 1024), 0), out=buf189)
        del arg127_1
        buf190 = reinterpret_tensor(buf189, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf189  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf190, arg128_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg128_1
        buf191 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 1024), (1, 4096), 0), out=buf191)
        del arg129_1
        buf195 = buf171; del buf171  # reuse
        # Source Nodes: [hidden_states_88, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf188, buf191, arg130_1, arg131_1, arg132_1, buf195, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        buf196 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), out=buf196)
        del arg133_1
        buf197 = reinterpret_tensor(buf188, (1024, 1024), (1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf197)
        del arg135_1
        buf198 = reinterpret_tensor(buf176, (1024, 1024), (1024, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf198)
        del arg137_1
        buf199 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf196, arg134_1, buf199, 1048576, grid=grid(1048576), stream=stream0)
        del arg134_1
        buf200 = reinterpret_tensor(buf196, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf197, arg136_1, buf200, 1048576, grid=grid(1048576), stream=stream0)
        del arg136_1
        buf201 = reinterpret_tensor(buf197, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf198, arg138_1, buf201, 1048576, grid=grid(1048576), stream=stream0)
        del arg138_1
        del buf198
        # Source Nodes: [], Original ATen: []
        buf202 = aten._scaled_dot_product_efficient_attention(buf199, buf200, buf201, None, True, scale=1.0)
        buf203 = buf202[0]
        del buf202
        buf207 = reinterpret_tensor(buf203, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf207, 1048576, grid=grid(1048576), stream=stream0)
        buf208 = reinterpret_tensor(buf201, (1024, 1024), (1024, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg139_1
        buf212 = reinterpret_tensor(buf207, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf207  # reuse
        # Source Nodes: [hidden_states_93, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf195, buf208, arg140_1, arg141_1, arg142_1, buf212, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        buf213 = reinterpret_tensor(buf190, (1024, 4096), (4096, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 4096), (1, 1024), 0), out=buf213)
        del arg143_1
        buf214 = reinterpret_tensor(buf213, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf213  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf214, arg144_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg144_1
        buf215 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg145_1, (4096, 1024), (1, 4096), 0), out=buf215)
        del arg145_1
        buf219 = buf195; del buf195  # reuse
        # Source Nodes: [hidden_states_99, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf212, buf215, arg146_1, arg147_1, arg148_1, buf219, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        buf220 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf220)
        del arg149_1
        buf221 = reinterpret_tensor(buf212, (1024, 1024), (1024, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf221)
        del arg151_1
        buf222 = reinterpret_tensor(buf200, (1024, 1024), (1024, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf222)
        del arg153_1
        buf223 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf220, arg150_1, buf223, 1048576, grid=grid(1048576), stream=stream0)
        del arg150_1
        buf224 = reinterpret_tensor(buf220, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf221, arg152_1, buf224, 1048576, grid=grid(1048576), stream=stream0)
        del arg152_1
        buf225 = reinterpret_tensor(buf221, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf222, arg154_1, buf225, 1048576, grid=grid(1048576), stream=stream0)
        del arg154_1
        del buf222
        # Source Nodes: [], Original ATen: []
        buf226 = aten._scaled_dot_product_efficient_attention(buf223, buf224, buf225, None, True, scale=1.0)
        buf227 = buf226[0]
        del buf226
        buf231 = reinterpret_tensor(buf227, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf227  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf231, 1048576, grid=grid(1048576), stream=stream0)
        buf232 = reinterpret_tensor(buf225, (1024, 1024), (1024, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), out=buf232)
        del arg155_1
        buf236 = reinterpret_tensor(buf231, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf231  # reuse
        # Source Nodes: [hidden_states_104, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf219, buf232, arg156_1, arg157_1, arg158_1, buf236, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg156_1
        del arg157_1
        del arg158_1
        buf237 = reinterpret_tensor(buf214, (1024, 4096), (4096, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 4096), (1, 1024), 0), out=buf237)
        del arg159_1
        buf238 = reinterpret_tensor(buf237, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf237  # reuse
        # Source Nodes: [hidden_states_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf238, arg160_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg160_1
        buf239 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg161_1, (4096, 1024), (1, 4096), 0), out=buf239)
        del arg161_1
        buf243 = buf219; del buf219  # reuse
        # Source Nodes: [hidden_states_110, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf236, buf239, arg162_1, arg163_1, arg164_1, buf243, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        buf244 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), out=buf244)
        del arg165_1
        buf245 = reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg167_1
        buf246 = reinterpret_tensor(buf224, (1024, 1024), (1024, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf246)
        del arg169_1
        buf247 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf244, arg166_1, buf247, 1048576, grid=grid(1048576), stream=stream0)
        del arg166_1
        buf248 = reinterpret_tensor(buf244, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf245, arg168_1, buf248, 1048576, grid=grid(1048576), stream=stream0)
        del arg168_1
        buf249 = reinterpret_tensor(buf245, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf246, arg170_1, buf249, 1048576, grid=grid(1048576), stream=stream0)
        del arg170_1
        del buf246
        # Source Nodes: [], Original ATen: []
        buf250 = aten._scaled_dot_product_efficient_attention(buf247, buf248, buf249, None, True, scale=1.0)
        buf251 = buf250[0]
        del buf250
        buf255 = reinterpret_tensor(buf251, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf255, 1048576, grid=grid(1048576), stream=stream0)
        buf256 = reinterpret_tensor(buf249, (1024, 1024), (1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), out=buf256)
        del arg171_1
        buf260 = reinterpret_tensor(buf255, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf255  # reuse
        # Source Nodes: [hidden_states_115, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf243, buf256, arg172_1, arg173_1, arg174_1, buf260, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        buf261 = reinterpret_tensor(buf238, (1024, 4096), (4096, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg175_1, (1024, 4096), (1, 1024), 0), out=buf261)
        del arg175_1
        buf262 = reinterpret_tensor(buf261, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf261  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf262, arg176_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg176_1
        buf263 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 1024), (1, 4096), 0), out=buf263)
        del arg177_1
        buf267 = buf243; del buf243  # reuse
        # Source Nodes: [hidden_states_121, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf260, buf263, arg178_1, arg179_1, arg180_1, buf267, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        buf268 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), out=buf268)
        del arg181_1
        buf269 = reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf269)
        del arg183_1
        buf270 = reinterpret_tensor(buf248, (1024, 1024), (1024, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg185_1
        buf271 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf268, arg182_1, buf271, 1048576, grid=grid(1048576), stream=stream0)
        del arg182_1
        buf272 = reinterpret_tensor(buf268, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf269, arg184_1, buf272, 1048576, grid=grid(1048576), stream=stream0)
        del arg184_1
        buf273 = reinterpret_tensor(buf269, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf270, arg186_1, buf273, 1048576, grid=grid(1048576), stream=stream0)
        del arg186_1
        # Source Nodes: [], Original ATen: []
        buf274 = aten._scaled_dot_product_efficient_attention(buf271, buf272, buf273, None, True, scale=1.0)
        buf275 = buf274[0]
        del buf274
        buf279 = reinterpret_tensor(buf275, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf275  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf279, 1048576, grid=grid(1048576), stream=stream0)
        buf280 = reinterpret_tensor(buf273, (1024, 1024), (1024, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg187_1
        buf284 = reinterpret_tensor(buf279, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf279  # reuse
        # Source Nodes: [hidden_states_126, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf267, buf280, arg188_1, arg189_1, arg190_1, buf284, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        buf285 = reinterpret_tensor(buf262, (1024, 4096), (4096, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg191_1, (1024, 4096), (1, 1024), 0), out=buf285)
        del arg191_1
        buf286 = reinterpret_tensor(buf285, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf285  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf286, arg192_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg192_1
        buf287 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg193_1, (4096, 1024), (1, 4096), 0), out=buf287)
        del arg193_1
        buf313 = buf267; del buf267  # reuse
        # Source Nodes: [hidden_states_132, hidden_states_134], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf284, buf287, arg194_1, arg195_1, arg196_1, buf313, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        buf294 = reinterpret_tensor(buf287, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf287  # reuse
        # Source Nodes: [add_27, clone, eq, hidden_states_135, hidden_states_136, input_2, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.fill, aten.lift_fresh, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_select_scatter_slice_scatter_6.run(arg514_1, arg197_1, arg1_1, arg198_1, arg199_1, buf294, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg1_1
        buf295 = reinterpret_tensor(buf284, (1024, 1024), (1024, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg200_1, (1024, 1024), (1, 1024), 0), out=buf295)
        del arg200_1
        buf296 = reinterpret_tensor(buf272, (1024, 1024), (1024, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg202_1, (1024, 1024), (1, 1024), 0), out=buf296)
        del arg202_1
        buf297 = buf271; del buf271  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf295, arg201_1, buf297, 1048576, grid=grid(1048576), stream=stream0)
        del arg201_1
        buf298 = reinterpret_tensor(buf295, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf295  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf296, arg203_1, buf298, 1048576, grid=grid(1048576), stream=stream0)
        del arg203_1
        buf299 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf298, (16, 64, 1024), (65536, 1, 64), 0), out=buf299)
        buf303 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf299, buf303, 16384, 1024, grid=grid(16384), stream=stream0)
        buf302 = reinterpret_tensor(buf298, (1024, 1024), (1024, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg204_1, (1024, 1024), (1, 1024), 0), out=buf302)
        del arg204_1
        buf304 = buf297; del buf297  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf302, arg205_1, buf304, 1048576, grid=grid(1048576), stream=stream0)
        del arg205_1
        buf305 = reinterpret_tensor(buf302, (16, 1024, 64), (65536, 64, 1), 0); del buf302  # reuse
        # Source Nodes: [attn_output_60, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf303, reinterpret_tensor(buf304, (16, 1024, 64), (65536, 64, 1), 0), out=buf305)
        buf306 = reinterpret_tensor(buf304, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf304  # reuse
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf305, buf306, 1048576, grid=grid(1048576), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (1024, 1024), (1024, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg206_1, (1024, 1024), (1, 1024), 0), out=buf307)
        del arg206_1
        buf311 = reinterpret_tensor(buf306, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf306  # reuse
        # Source Nodes: [hidden_states_140, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf294, buf307, arg207_1, arg208_1, arg209_1, buf311, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        buf312 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg210_1, (1024, 1024), (1, 1024), 0), out=buf312)
        del arg210_1
        buf314 = reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg212_1, (1024, 1024), (1, 1024), 0), out=buf314)
        del arg212_1
        buf315 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg214_1, (1024, 1024), (1, 1024), 0), out=buf315)
        del arg214_1
        buf316 = reinterpret_tensor(buf270, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf312, arg211_1, buf316, 1048576, grid=grid(1048576), stream=stream0)
        del arg211_1
        buf317 = reinterpret_tensor(buf312, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf314, arg213_1, buf317, 1048576, grid=grid(1048576), stream=stream0)
        del arg213_1
        buf318 = reinterpret_tensor(buf314, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf315, arg215_1, buf318, 1048576, grid=grid(1048576), stream=stream0)
        del arg215_1
        del buf315
        # Source Nodes: [], Original ATen: []
        buf319 = aten._scaled_dot_product_efficient_attention(buf316, buf317, buf318, None, True, scale=1.0)
        buf320 = buf319[0]
        del buf319
        buf324 = reinterpret_tensor(buf320, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf324, 1048576, grid=grid(1048576), stream=stream0)
        buf325 = reinterpret_tensor(buf318, (1024, 1024), (1024, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg216_1, (1024, 1024), (1, 1024), 0), out=buf325)
        del arg216_1
        buf329 = reinterpret_tensor(buf324, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf324  # reuse
        # Source Nodes: [hidden_states_144, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf311, buf325, arg217_1, arg218_1, arg219_1, buf329, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        buf330 = reinterpret_tensor(buf286, (1024, 4096), (4096, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg220_1, (1024, 4096), (1, 1024), 0), out=buf330)
        del arg220_1
        buf331 = reinterpret_tensor(buf330, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf330  # reuse
        # Source Nodes: [hidden_states_146], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf331, arg221_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg221_1
        buf332 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg222_1, (4096, 1024), (1, 4096), 0), out=buf332)
        del arg222_1
        buf336 = buf311; del buf311  # reuse
        # Source Nodes: [hidden_states_150, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf329, buf332, arg223_1, arg224_1, arg225_1, buf336, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg223_1
        del arg224_1
        del arg225_1
        buf337 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg226_1, (1024, 1024), (1, 1024), 0), out=buf337)
        del arg226_1
        buf338 = reinterpret_tensor(buf329, (1024, 1024), (1024, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg228_1, (1024, 1024), (1, 1024), 0), out=buf338)
        del arg228_1
        buf339 = buf317; del buf317  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf337, arg227_1, buf339, 1048576, grid=grid(1048576), stream=stream0)
        del arg227_1
        buf340 = reinterpret_tensor(buf337, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf337  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf338, arg229_1, buf340, 1048576, grid=grid(1048576), stream=stream0)
        del arg229_1
        buf341 = buf303; del buf303  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf339, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf340, (16, 64, 1024), (65536, 1, 64), 0), out=buf341)
        buf345 = buf299; del buf299  # reuse
        # Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf341, buf345, 16384, 1024, grid=grid(16384), stream=stream0)
        buf344 = reinterpret_tensor(buf340, (1024, 1024), (1024, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg230_1, (1024, 1024), (1, 1024), 0), out=buf344)
        del arg230_1
        buf346 = buf339; del buf339  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf344, arg231_1, buf346, 1048576, grid=grid(1048576), stream=stream0)
        del arg231_1
        buf347 = reinterpret_tensor(buf344, (16, 1024, 64), (65536, 64, 1), 0); del buf344  # reuse
        # Source Nodes: [attn_output_70, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf345, reinterpret_tensor(buf346, (16, 1024, 64), (65536, 64, 1), 0), out=buf347)
        buf348 = reinterpret_tensor(buf346, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf346  # reuse
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf347, buf348, 1048576, grid=grid(1048576), stream=stream0)
        buf349 = reinterpret_tensor(buf347, (1024, 1024), (1024, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg232_1, (1024, 1024), (1, 1024), 0), out=buf349)
        del arg232_1
        buf353 = reinterpret_tensor(buf348, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf348  # reuse
        # Source Nodes: [hidden_states_155, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf336, buf349, arg233_1, arg234_1, arg235_1, buf353, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg233_1
        del arg234_1
        del arg235_1
        buf354 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg236_1, (1024, 1024), (1, 1024), 0), out=buf354)
        del arg236_1
        buf355 = reinterpret_tensor(buf336, (1024, 1024), (1024, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg238_1, (1024, 1024), (1, 1024), 0), out=buf355)
        del arg238_1
        buf356 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg240_1, (1024, 1024), (1, 1024), 0), out=buf356)
        del arg240_1
        buf357 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf354, arg237_1, buf357, 1048576, grid=grid(1048576), stream=stream0)
        del arg237_1
        buf358 = reinterpret_tensor(buf354, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf355, arg239_1, buf358, 1048576, grid=grid(1048576), stream=stream0)
        del arg239_1
        buf359 = reinterpret_tensor(buf355, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf356, arg241_1, buf359, 1048576, grid=grid(1048576), stream=stream0)
        del arg241_1
        del buf356
        # Source Nodes: [], Original ATen: []
        buf360 = aten._scaled_dot_product_efficient_attention(buf357, buf358, buf359, None, True, scale=1.0)
        buf361 = buf360[0]
        del buf360
        buf365 = reinterpret_tensor(buf361, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf361  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf365, 1048576, grid=grid(1048576), stream=stream0)
        buf366 = reinterpret_tensor(buf359, (1024, 1024), (1024, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg242_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg242_1
        buf370 = reinterpret_tensor(buf365, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf365  # reuse
        # Source Nodes: [hidden_states_159, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf353, buf366, arg243_1, arg244_1, arg245_1, buf370, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg243_1
        del arg244_1
        del arg245_1
        buf371 = reinterpret_tensor(buf331, (1024, 4096), (4096, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg246_1, (1024, 4096), (1, 1024), 0), out=buf371)
        del arg246_1
        buf372 = reinterpret_tensor(buf371, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf371  # reuse
        # Source Nodes: [hidden_states_161], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf372, arg247_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg247_1
        buf373 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg248_1, (4096, 1024), (1, 4096), 0), out=buf373)
        del arg248_1
        buf377 = buf353; del buf353  # reuse
        # Source Nodes: [hidden_states_165, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf370, buf373, arg249_1, arg250_1, arg251_1, buf377, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg249_1
        del arg250_1
        del arg251_1
        buf378 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg252_1, (1024, 1024), (1, 1024), 0), out=buf378)
        del arg252_1
        buf379 = reinterpret_tensor(buf370, (1024, 1024), (1024, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg254_1, (1024, 1024), (1, 1024), 0), out=buf379)
        del arg254_1
        buf380 = buf358; del buf358  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf378, arg253_1, buf380, 1048576, grid=grid(1048576), stream=stream0)
        del arg253_1
        buf381 = reinterpret_tensor(buf378, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf378  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf379, arg255_1, buf381, 1048576, grid=grid(1048576), stream=stream0)
        del arg255_1
        buf382 = buf345; del buf345  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf381, (16, 64, 1024), (65536, 1, 64), 0), out=buf382)
        buf386 = buf341; del buf341  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf382, buf386, 16384, 1024, grid=grid(16384), stream=stream0)
        buf385 = reinterpret_tensor(buf381, (1024, 1024), (1024, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg256_1, (1024, 1024), (1, 1024), 0), out=buf385)
        del arg256_1
        buf387 = buf380; del buf380  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf385, arg257_1, buf387, 1048576, grid=grid(1048576), stream=stream0)
        del arg257_1
        buf388 = reinterpret_tensor(buf385, (16, 1024, 64), (65536, 64, 1), 0); del buf385  # reuse
        # Source Nodes: [attn_output_80, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf386, reinterpret_tensor(buf387, (16, 1024, 64), (65536, 64, 1), 0), out=buf388)
        buf389 = reinterpret_tensor(buf387, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf387  # reuse
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf388, buf389, 1048576, grid=grid(1048576), stream=stream0)
        buf390 = reinterpret_tensor(buf388, (1024, 1024), (1024, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg258_1, (1024, 1024), (1, 1024), 0), out=buf390)
        del arg258_1
        buf394 = reinterpret_tensor(buf389, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf389  # reuse
        # Source Nodes: [hidden_states_170, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf377, buf390, arg259_1, arg260_1, arg261_1, buf394, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        buf395 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg262_1, (1024, 1024), (1, 1024), 0), out=buf395)
        del arg262_1
        buf396 = reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg264_1, (1024, 1024), (1, 1024), 0), out=buf396)
        del arg264_1
        buf397 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg266_1, (1024, 1024), (1, 1024), 0), out=buf397)
        del arg266_1
        buf398 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf395, arg263_1, buf398, 1048576, grid=grid(1048576), stream=stream0)
        del arg263_1
        buf399 = reinterpret_tensor(buf395, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf396, arg265_1, buf399, 1048576, grid=grid(1048576), stream=stream0)
        del arg265_1
        buf400 = reinterpret_tensor(buf396, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf397, arg267_1, buf400, 1048576, grid=grid(1048576), stream=stream0)
        del arg267_1
        del buf397
        # Source Nodes: [], Original ATen: []
        buf401 = aten._scaled_dot_product_efficient_attention(buf398, buf399, buf400, None, True, scale=1.0)
        buf402 = buf401[0]
        del buf401
        buf406 = reinterpret_tensor(buf402, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf402  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf406, 1048576, grid=grid(1048576), stream=stream0)
        buf407 = reinterpret_tensor(buf400, (1024, 1024), (1024, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg268_1, (1024, 1024), (1, 1024), 0), out=buf407)
        del arg268_1
        buf411 = reinterpret_tensor(buf406, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf406  # reuse
        # Source Nodes: [hidden_states_174, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf394, buf407, arg269_1, arg270_1, arg271_1, buf411, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        buf412 = reinterpret_tensor(buf372, (1024, 4096), (4096, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg272_1, (1024, 4096), (1, 1024), 0), out=buf412)
        del arg272_1
        buf413 = reinterpret_tensor(buf412, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf412  # reuse
        # Source Nodes: [hidden_states_176], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf413, arg273_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg273_1
        buf414 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf413, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg274_1, (4096, 1024), (1, 4096), 0), out=buf414)
        del arg274_1
        buf418 = buf394; del buf394  # reuse
        # Source Nodes: [hidden_states_180, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf411, buf414, arg275_1, arg276_1, arg277_1, buf418, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg275_1
        del arg276_1
        del arg277_1
        buf419 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg278_1, (1024, 1024), (1, 1024), 0), out=buf419)
        del arg278_1
        buf420 = reinterpret_tensor(buf411, (1024, 1024), (1024, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg280_1, (1024, 1024), (1, 1024), 0), out=buf420)
        del arg280_1
        buf421 = buf399; del buf399  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf419, arg279_1, buf421, 1048576, grid=grid(1048576), stream=stream0)
        del arg279_1
        buf422 = reinterpret_tensor(buf419, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf419  # reuse
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf420, arg281_1, buf422, 1048576, grid=grid(1048576), stream=stream0)
        del arg281_1
        buf423 = buf386; del buf386  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf422, (16, 64, 1024), (65536, 1, 64), 0), out=buf423)
        buf427 = buf382; del buf382  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf423, buf427, 16384, 1024, grid=grid(16384), stream=stream0)
        buf426 = reinterpret_tensor(buf422, (1024, 1024), (1024, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg282_1, (1024, 1024), (1, 1024), 0), out=buf426)
        del arg282_1
        buf428 = buf421; del buf421  # reuse
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf426, arg283_1, buf428, 1048576, grid=grid(1048576), stream=stream0)
        del arg283_1
        buf429 = reinterpret_tensor(buf426, (16, 1024, 64), (65536, 64, 1), 0); del buf426  # reuse
        # Source Nodes: [attn_output_90, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf427, reinterpret_tensor(buf428, (16, 1024, 64), (65536, 64, 1), 0), out=buf429)
        buf430 = reinterpret_tensor(buf428, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf428  # reuse
        # Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf429, buf430, 1048576, grid=grid(1048576), stream=stream0)
        buf431 = reinterpret_tensor(buf429, (1024, 1024), (1024, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg284_1, (1024, 1024), (1, 1024), 0), out=buf431)
        del arg284_1
        buf435 = reinterpret_tensor(buf430, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf430  # reuse
        # Source Nodes: [hidden_states_185, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf418, buf431, arg285_1, arg286_1, arg287_1, buf435, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg285_1
        del arg286_1
        del arg287_1
        buf436 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg288_1, (1024, 1024), (1, 1024), 0), out=buf436)
        del arg288_1
        buf437 = reinterpret_tensor(buf418, (1024, 1024), (1024, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg290_1, (1024, 1024), (1, 1024), 0), out=buf437)
        del arg290_1
        buf438 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg292_1, (1024, 1024), (1, 1024), 0), out=buf438)
        del arg292_1
        buf439 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf436, arg289_1, buf439, 1048576, grid=grid(1048576), stream=stream0)
        del arg289_1
        buf440 = reinterpret_tensor(buf436, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf437, arg291_1, buf440, 1048576, grid=grid(1048576), stream=stream0)
        del arg291_1
        buf441 = reinterpret_tensor(buf437, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf438, arg293_1, buf441, 1048576, grid=grid(1048576), stream=stream0)
        del arg293_1
        del buf438
        # Source Nodes: [], Original ATen: []
        buf442 = aten._scaled_dot_product_efficient_attention(buf439, buf440, buf441, None, True, scale=1.0)
        buf443 = buf442[0]
        del buf442
        buf447 = reinterpret_tensor(buf443, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf443  # reuse
        # Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf447, 1048576, grid=grid(1048576), stream=stream0)
        buf448 = reinterpret_tensor(buf441, (1024, 1024), (1024, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf447, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg294_1, (1024, 1024), (1, 1024), 0), out=buf448)
        del arg294_1
        buf452 = reinterpret_tensor(buf447, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf447  # reuse
        # Source Nodes: [hidden_states_189, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf435, buf448, arg295_1, arg296_1, arg297_1, buf452, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg295_1
        del arg296_1
        del arg297_1
        buf453 = reinterpret_tensor(buf413, (1024, 4096), (4096, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg298_1, (1024, 4096), (1, 1024), 0), out=buf453)
        del arg298_1
        buf454 = reinterpret_tensor(buf453, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf453  # reuse
        # Source Nodes: [hidden_states_191], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf454, arg299_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg299_1
        buf455 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf454, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg300_1, (4096, 1024), (1, 4096), 0), out=buf455)
        del arg300_1
        buf459 = buf435; del buf435  # reuse
        # Source Nodes: [hidden_states_195, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf452, buf455, arg301_1, arg302_1, arg303_1, buf459, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        buf460 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg304_1, (1024, 1024), (1, 1024), 0), out=buf460)
        del arg304_1
        buf461 = reinterpret_tensor(buf452, (1024, 1024), (1024, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg306_1, (1024, 1024), (1, 1024), 0), out=buf461)
        del arg306_1
        buf462 = buf440; del buf440  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf460, arg305_1, buf462, 1048576, grid=grid(1048576), stream=stream0)
        del arg305_1
        buf463 = reinterpret_tensor(buf460, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf460  # reuse
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf461, arg307_1, buf463, 1048576, grid=grid(1048576), stream=stream0)
        del arg307_1
        buf464 = buf427; del buf427  # reuse
        # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf463, (16, 64, 1024), (65536, 1, 64), 0), out=buf464)
        buf468 = buf423; del buf423  # reuse
        # Source Nodes: [attn_weights_51], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf464, buf468, 16384, 1024, grid=grid(16384), stream=stream0)
        buf467 = reinterpret_tensor(buf463, (1024, 1024), (1024, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg308_1, (1024, 1024), (1, 1024), 0), out=buf467)
        del arg308_1
        buf469 = buf462; del buf462  # reuse
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf467, arg309_1, buf469, 1048576, grid=grid(1048576), stream=stream0)
        del arg309_1
        buf470 = reinterpret_tensor(buf467, (16, 1024, 64), (65536, 64, 1), 0); del buf467  # reuse
        # Source Nodes: [attn_output_100, attn_weights_51], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf468, reinterpret_tensor(buf469, (16, 1024, 64), (65536, 64, 1), 0), out=buf470)
        buf471 = reinterpret_tensor(buf469, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf469  # reuse
        # Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf470, buf471, 1048576, grid=grid(1048576), stream=stream0)
        buf472 = reinterpret_tensor(buf470, (1024, 1024), (1024, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg310_1, (1024, 1024), (1, 1024), 0), out=buf472)
        del arg310_1
        buf476 = reinterpret_tensor(buf471, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf471  # reuse
        # Source Nodes: [hidden_states_200, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf459, buf472, arg311_1, arg312_1, arg313_1, buf476, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg311_1
        del arg312_1
        del arg313_1
        buf477 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg314_1, (1024, 1024), (1, 1024), 0), out=buf477)
        del arg314_1
        buf478 = reinterpret_tensor(buf459, (1024, 1024), (1024, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg316_1, (1024, 1024), (1, 1024), 0), out=buf478)
        del arg316_1
        buf479 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg318_1, (1024, 1024), (1, 1024), 0), out=buf479)
        del arg318_1
        buf480 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf477, arg315_1, buf480, 1048576, grid=grid(1048576), stream=stream0)
        del arg315_1
        buf481 = reinterpret_tensor(buf477, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf478, arg317_1, buf481, 1048576, grid=grid(1048576), stream=stream0)
        del arg317_1
        buf482 = reinterpret_tensor(buf478, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf479, arg319_1, buf482, 1048576, grid=grid(1048576), stream=stream0)
        del arg319_1
        del buf479
        # Source Nodes: [], Original ATen: []
        buf483 = aten._scaled_dot_product_efficient_attention(buf480, buf481, buf482, None, True, scale=1.0)
        buf484 = buf483[0]
        del buf483
        buf488 = reinterpret_tensor(buf484, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf484  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf488, 1048576, grid=grid(1048576), stream=stream0)
        buf489 = reinterpret_tensor(buf482, (1024, 1024), (1024, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg320_1, (1024, 1024), (1, 1024), 0), out=buf489)
        del arg320_1
        buf493 = reinterpret_tensor(buf488, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf488  # reuse
        # Source Nodes: [hidden_states_204, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf476, buf489, arg321_1, arg322_1, arg323_1, buf493, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg321_1
        del arg322_1
        del arg323_1
        buf494 = reinterpret_tensor(buf454, (1024, 4096), (4096, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf493, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg324_1, (1024, 4096), (1, 1024), 0), out=buf494)
        del arg324_1
        buf495 = reinterpret_tensor(buf494, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf494  # reuse
        # Source Nodes: [hidden_states_206], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf495, arg325_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg325_1
        buf496 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg326_1, (4096, 1024), (1, 4096), 0), out=buf496)
        del arg326_1
        buf500 = buf476; del buf476  # reuse
        # Source Nodes: [hidden_states_210, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf493, buf496, arg327_1, arg328_1, arg329_1, buf500, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        buf501 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg330_1, (1024, 1024), (1, 1024), 0), out=buf501)
        del arg330_1
        buf502 = reinterpret_tensor(buf493, (1024, 1024), (1024, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg332_1, (1024, 1024), (1, 1024), 0), out=buf502)
        del arg332_1
        buf503 = buf481; del buf481  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf501, arg331_1, buf503, 1048576, grid=grid(1048576), stream=stream0)
        del arg331_1
        buf504 = reinterpret_tensor(buf501, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf501  # reuse
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf502, arg333_1, buf504, 1048576, grid=grid(1048576), stream=stream0)
        del arg333_1
        buf505 = buf468; del buf468  # reuse
        # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf503, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf504, (16, 64, 1024), (65536, 1, 64), 0), out=buf505)
        buf509 = buf464; del buf464  # reuse
        # Source Nodes: [attn_weights_57], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf505, buf509, 16384, 1024, grid=grid(16384), stream=stream0)
        buf508 = reinterpret_tensor(buf504, (1024, 1024), (1024, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg334_1, (1024, 1024), (1, 1024), 0), out=buf508)
        del arg334_1
        buf510 = buf503; del buf503  # reuse
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf508, arg335_1, buf510, 1048576, grid=grid(1048576), stream=stream0)
        del arg335_1
        buf511 = reinterpret_tensor(buf508, (16, 1024, 64), (65536, 64, 1), 0); del buf508  # reuse
        # Source Nodes: [attn_output_110, attn_weights_57], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf509, reinterpret_tensor(buf510, (16, 1024, 64), (65536, 64, 1), 0), out=buf511)
        buf512 = reinterpret_tensor(buf510, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf510  # reuse
        # Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf511, buf512, 1048576, grid=grid(1048576), stream=stream0)
        buf513 = reinterpret_tensor(buf511, (1024, 1024), (1024, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg336_1, (1024, 1024), (1, 1024), 0), out=buf513)
        del arg336_1
        buf517 = reinterpret_tensor(buf512, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf512  # reuse
        # Source Nodes: [hidden_states_215, residual_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf500, buf513, arg337_1, arg338_1, arg339_1, buf517, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        buf518 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg340_1, (1024, 1024), (1, 1024), 0), out=buf518)
        del arg340_1
        buf519 = reinterpret_tensor(buf500, (1024, 1024), (1024, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg342_1, (1024, 1024), (1, 1024), 0), out=buf519)
        del arg342_1
        buf520 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg344_1, (1024, 1024), (1, 1024), 0), out=buf520)
        del arg344_1
        buf521 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf518, arg341_1, buf521, 1048576, grid=grid(1048576), stream=stream0)
        del arg341_1
        buf522 = reinterpret_tensor(buf518, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf519, arg343_1, buf522, 1048576, grid=grid(1048576), stream=stream0)
        del arg343_1
        buf523 = reinterpret_tensor(buf519, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf520, arg345_1, buf523, 1048576, grid=grid(1048576), stream=stream0)
        del arg345_1
        del buf520
        # Source Nodes: [], Original ATen: []
        buf524 = aten._scaled_dot_product_efficient_attention(buf521, buf522, buf523, None, True, scale=1.0)
        buf525 = buf524[0]
        del buf524
        buf529 = reinterpret_tensor(buf525, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf525  # reuse
        # Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf529, 1048576, grid=grid(1048576), stream=stream0)
        buf530 = reinterpret_tensor(buf523, (1024, 1024), (1024, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg346_1, (1024, 1024), (1, 1024), 0), out=buf530)
        del arg346_1
        buf534 = reinterpret_tensor(buf529, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf529  # reuse
        # Source Nodes: [hidden_states_219, residual_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf517, buf530, arg347_1, arg348_1, arg349_1, buf534, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        buf535 = reinterpret_tensor(buf495, (1024, 4096), (4096, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf534, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg350_1, (1024, 4096), (1, 1024), 0), out=buf535)
        del arg350_1
        buf536 = reinterpret_tensor(buf535, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf535  # reuse
        # Source Nodes: [hidden_states_221], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf536, arg351_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg351_1
        buf537 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf536, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg352_1, (4096, 1024), (1, 4096), 0), out=buf537)
        del arg352_1
        buf541 = buf517; del buf517  # reuse
        # Source Nodes: [hidden_states_225, residual_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf534, buf537, arg353_1, arg354_1, arg355_1, buf541, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg353_1
        del arg354_1
        del arg355_1
        buf542 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg356_1, (1024, 1024), (1, 1024), 0), out=buf542)
        del arg356_1
        buf543 = reinterpret_tensor(buf534, (1024, 1024), (1024, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg358_1, (1024, 1024), (1, 1024), 0), out=buf543)
        del arg358_1
        buf544 = buf522; del buf522  # reuse
        # Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf542, arg357_1, buf544, 1048576, grid=grid(1048576), stream=stream0)
        del arg357_1
        buf545 = reinterpret_tensor(buf542, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf542  # reuse
        # Source Nodes: [key_states_48], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf543, arg359_1, buf545, 1048576, grid=grid(1048576), stream=stream0)
        del arg359_1
        buf546 = buf509; del buf509  # reuse
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf545, (16, 64, 1024), (65536, 1, 64), 0), out=buf546)
        buf550 = buf505; del buf505  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf546, buf550, 16384, 1024, grid=grid(16384), stream=stream0)
        buf549 = reinterpret_tensor(buf545, (1024, 1024), (1024, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg360_1, (1024, 1024), (1, 1024), 0), out=buf549)
        del arg360_1
        buf551 = buf544; del buf544  # reuse
        # Source Nodes: [value_states_48], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf549, arg361_1, buf551, 1048576, grid=grid(1048576), stream=stream0)
        del arg361_1
        buf552 = reinterpret_tensor(buf549, (16, 1024, 64), (65536, 64, 1), 0); del buf549  # reuse
        # Source Nodes: [attn_output_120, attn_weights_63], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf550, reinterpret_tensor(buf551, (16, 1024, 64), (65536, 64, 1), 0), out=buf552)
        buf553 = reinterpret_tensor(buf551, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf551  # reuse
        # Source Nodes: [attn_output_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf552, buf553, 1048576, grid=grid(1048576), stream=stream0)
        buf554 = reinterpret_tensor(buf552, (1024, 1024), (1024, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg362_1, (1024, 1024), (1, 1024), 0), out=buf554)
        del arg362_1
        buf558 = reinterpret_tensor(buf553, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf553  # reuse
        # Source Nodes: [hidden_states_230, residual_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf541, buf554, arg363_1, arg364_1, arg365_1, buf558, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg363_1
        del arg364_1
        del arg365_1
        buf559 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf558, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg366_1, (1024, 1024), (1, 1024), 0), out=buf559)
        del arg366_1
        buf560 = reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg368_1, (1024, 1024), (1, 1024), 0), out=buf560)
        del arg368_1
        buf561 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg370_1, (1024, 1024), (1, 1024), 0), out=buf561)
        del arg370_1
        buf562 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf559, arg367_1, buf562, 1048576, grid=grid(1048576), stream=stream0)
        del arg367_1
        buf563 = reinterpret_tensor(buf559, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf560, arg369_1, buf563, 1048576, grid=grid(1048576), stream=stream0)
        del arg369_1
        buf564 = reinterpret_tensor(buf560, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf561, arg371_1, buf564, 1048576, grid=grid(1048576), stream=stream0)
        del arg371_1
        del buf561
        # Source Nodes: [], Original ATen: []
        buf565 = aten._scaled_dot_product_efficient_attention(buf562, buf563, buf564, None, True, scale=1.0)
        buf566 = buf565[0]
        del buf565
        buf570 = reinterpret_tensor(buf566, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf566  # reuse
        # Source Nodes: [attn_output_128], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf570, 1048576, grid=grid(1048576), stream=stream0)
        buf571 = reinterpret_tensor(buf564, (1024, 1024), (1024, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg372_1, (1024, 1024), (1, 1024), 0), out=buf571)
        del arg372_1
        buf575 = reinterpret_tensor(buf570, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf570  # reuse
        # Source Nodes: [hidden_states_234, residual_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf558, buf571, arg373_1, arg374_1, arg375_1, buf575, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg373_1
        del arg374_1
        del arg375_1
        buf576 = reinterpret_tensor(buf536, (1024, 4096), (4096, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg376_1, (1024, 4096), (1, 1024), 0), out=buf576)
        del arg376_1
        buf577 = reinterpret_tensor(buf576, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf576  # reuse
        # Source Nodes: [hidden_states_236], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf577, arg377_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg377_1
        buf578 = buf571; del buf571  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf577, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg378_1, (4096, 1024), (1, 4096), 0), out=buf578)
        del arg378_1
        buf582 = buf558; del buf558  # reuse
        # Source Nodes: [hidden_states_240, residual_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf575, buf578, arg379_1, arg380_1, arg381_1, buf582, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg379_1
        del arg380_1
        del arg381_1
        buf583 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg382_1, (1024, 1024), (1, 1024), 0), out=buf583)
        del arg382_1
        buf584 = reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg384_1, (1024, 1024), (1, 1024), 0), out=buf584)
        del arg384_1
        buf585 = buf563; del buf563  # reuse
        # Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf583, arg383_1, buf585, 1048576, grid=grid(1048576), stream=stream0)
        del arg383_1
        buf586 = reinterpret_tensor(buf583, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf583  # reuse
        # Source Nodes: [key_states_52], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf584, arg385_1, buf586, 1048576, grid=grid(1048576), stream=stream0)
        del arg385_1
        buf587 = buf550; del buf550  # reuse
        # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf586, (16, 64, 1024), (65536, 1, 64), 0), out=buf587)
        buf591 = buf546; del buf546  # reuse
        # Source Nodes: [attn_weights_69], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf587, buf591, 16384, 1024, grid=grid(16384), stream=stream0)
        buf590 = reinterpret_tensor(buf586, (1024, 1024), (1024, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg386_1, (1024, 1024), (1, 1024), 0), out=buf590)
        del arg386_1
        buf592 = buf585; del buf585  # reuse
        # Source Nodes: [value_states_52], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf590, arg387_1, buf592, 1048576, grid=grid(1048576), stream=stream0)
        del arg387_1
        buf593 = reinterpret_tensor(buf590, (16, 1024, 64), (65536, 64, 1), 0); del buf590  # reuse
        # Source Nodes: [attn_output_130, attn_weights_69], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf591, reinterpret_tensor(buf592, (16, 1024, 64), (65536, 64, 1), 0), out=buf593)
        buf594 = reinterpret_tensor(buf592, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf592  # reuse
        # Source Nodes: [attn_output_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf593, buf594, 1048576, grid=grid(1048576), stream=stream0)
        buf595 = reinterpret_tensor(buf593, (1024, 1024), (1024, 1), 0); del buf593  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf594, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg388_1, (1024, 1024), (1, 1024), 0), out=buf595)
        del arg388_1
        buf599 = reinterpret_tensor(buf594, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf594  # reuse
        # Source Nodes: [hidden_states_245, residual_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf582, buf595, arg389_1, arg390_1, arg391_1, buf599, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg389_1
        del arg390_1
        del arg391_1
        buf600 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg392_1, (1024, 1024), (1, 1024), 0), out=buf600)
        del arg392_1
        buf601 = reinterpret_tensor(buf582, (1024, 1024), (1024, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg394_1, (1024, 1024), (1, 1024), 0), out=buf601)
        del arg394_1
        buf602 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg396_1, (1024, 1024), (1, 1024), 0), out=buf602)
        del arg396_1
        buf603 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf600, arg393_1, buf603, 1048576, grid=grid(1048576), stream=stream0)
        del arg393_1
        buf604 = reinterpret_tensor(buf600, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf601, arg395_1, buf604, 1048576, grid=grid(1048576), stream=stream0)
        del arg395_1
        buf605 = reinterpret_tensor(buf601, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf602, arg397_1, buf605, 1048576, grid=grid(1048576), stream=stream0)
        del arg397_1
        del buf602
        # Source Nodes: [], Original ATen: []
        buf606 = aten._scaled_dot_product_efficient_attention(buf603, buf604, buf605, None, True, scale=1.0)
        buf607 = buf606[0]
        del buf606
        buf611 = reinterpret_tensor(buf607, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf607  # reuse
        # Source Nodes: [attn_output_138], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf611, 1048576, grid=grid(1048576), stream=stream0)
        buf612 = reinterpret_tensor(buf605, (1024, 1024), (1024, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg398_1, (1024, 1024), (1, 1024), 0), out=buf612)
        del arg398_1
        buf616 = reinterpret_tensor(buf611, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf611  # reuse
        # Source Nodes: [hidden_states_249, residual_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf599, buf612, arg399_1, arg400_1, arg401_1, buf616, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg399_1
        del arg400_1
        del arg401_1
        buf617 = reinterpret_tensor(buf577, (1024, 4096), (4096, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf616, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg402_1, (1024, 4096), (1, 1024), 0), out=buf617)
        del arg402_1
        buf618 = reinterpret_tensor(buf617, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf617  # reuse
        # Source Nodes: [hidden_states_251], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf618, arg403_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg403_1
        buf619 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf618, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg404_1, (4096, 1024), (1, 4096), 0), out=buf619)
        del arg404_1
        buf623 = buf599; del buf599  # reuse
        # Source Nodes: [hidden_states_255, residual_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf616, buf619, arg405_1, arg406_1, arg407_1, buf623, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg405_1
        del arg406_1
        del arg407_1
        buf624 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg408_1, (1024, 1024), (1, 1024), 0), out=buf624)
        del arg408_1
        buf625 = reinterpret_tensor(buf616, (1024, 1024), (1024, 1), 0); del buf616  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg410_1, (1024, 1024), (1, 1024), 0), out=buf625)
        del arg410_1
        buf626 = buf604; del buf604  # reuse
        # Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf624, arg409_1, buf626, 1048576, grid=grid(1048576), stream=stream0)
        del arg409_1
        buf627 = reinterpret_tensor(buf624, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf624  # reuse
        # Source Nodes: [key_states_56], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf625, arg411_1, buf627, 1048576, grid=grid(1048576), stream=stream0)
        del arg411_1
        buf628 = buf591; del buf591  # reuse
        # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf626, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf627, (16, 64, 1024), (65536, 1, 64), 0), out=buf628)
        buf632 = buf587; del buf587  # reuse
        # Source Nodes: [attn_weights_75], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf628, buf632, 16384, 1024, grid=grid(16384), stream=stream0)
        buf631 = reinterpret_tensor(buf627, (1024, 1024), (1024, 1), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg412_1, (1024, 1024), (1, 1024), 0), out=buf631)
        del arg412_1
        buf633 = buf626; del buf626  # reuse
        # Source Nodes: [value_states_56], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf631, arg413_1, buf633, 1048576, grid=grid(1048576), stream=stream0)
        del arg413_1
        buf634 = reinterpret_tensor(buf631, (16, 1024, 64), (65536, 64, 1), 0); del buf631  # reuse
        # Source Nodes: [attn_output_140, attn_weights_75], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf632, reinterpret_tensor(buf633, (16, 1024, 64), (65536, 64, 1), 0), out=buf634)
        buf635 = reinterpret_tensor(buf633, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf633  # reuse
        # Source Nodes: [attn_output_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf634, buf635, 1048576, grid=grid(1048576), stream=stream0)
        buf636 = reinterpret_tensor(buf634, (1024, 1024), (1024, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf635, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg414_1, (1024, 1024), (1, 1024), 0), out=buf636)
        del arg414_1
        buf640 = reinterpret_tensor(buf635, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf635  # reuse
        # Source Nodes: [hidden_states_260, residual_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf623, buf636, arg415_1, arg416_1, arg417_1, buf640, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg415_1
        del arg416_1
        del arg417_1
        buf641 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg418_1, (1024, 1024), (1, 1024), 0), out=buf641)
        del arg418_1
        buf642 = reinterpret_tensor(buf623, (1024, 1024), (1024, 1), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg420_1, (1024, 1024), (1, 1024), 0), out=buf642)
        del arg420_1
        buf643 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg422_1, (1024, 1024), (1, 1024), 0), out=buf643)
        del arg422_1
        buf644 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf641, arg419_1, buf644, 1048576, grid=grid(1048576), stream=stream0)
        del arg419_1
        buf645 = reinterpret_tensor(buf641, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf641  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf642, arg421_1, buf645, 1048576, grid=grid(1048576), stream=stream0)
        del arg421_1
        buf646 = reinterpret_tensor(buf642, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf642  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf643, arg423_1, buf646, 1048576, grid=grid(1048576), stream=stream0)
        del arg423_1
        del buf643
        # Source Nodes: [], Original ATen: []
        buf647 = aten._scaled_dot_product_efficient_attention(buf644, buf645, buf646, None, True, scale=1.0)
        buf648 = buf647[0]
        del buf647
        buf652 = reinterpret_tensor(buf648, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf648  # reuse
        # Source Nodes: [attn_output_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf652, 1048576, grid=grid(1048576), stream=stream0)
        buf653 = reinterpret_tensor(buf646, (1024, 1024), (1024, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf652, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg424_1, (1024, 1024), (1, 1024), 0), out=buf653)
        del arg424_1
        buf657 = reinterpret_tensor(buf652, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf652  # reuse
        # Source Nodes: [hidden_states_264, residual_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf640, buf653, arg425_1, arg426_1, arg427_1, buf657, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg425_1
        del arg426_1
        del arg427_1
        buf658 = reinterpret_tensor(buf618, (1024, 4096), (4096, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf657, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg428_1, (1024, 4096), (1, 1024), 0), out=buf658)
        del arg428_1
        buf659 = reinterpret_tensor(buf658, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf658  # reuse
        # Source Nodes: [hidden_states_266], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf659, arg429_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg429_1
        buf660 = buf653; del buf653  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf659, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg430_1, (4096, 1024), (1, 4096), 0), out=buf660)
        del arg430_1
        buf664 = buf640; del buf640  # reuse
        # Source Nodes: [hidden_states_270, residual_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf657, buf660, arg431_1, arg432_1, arg433_1, buf664, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg431_1
        del arg432_1
        del arg433_1
        buf665 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg434_1, (1024, 1024), (1, 1024), 0), out=buf665)
        del arg434_1
        buf666 = reinterpret_tensor(buf657, (1024, 1024), (1024, 1), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg436_1, (1024, 1024), (1, 1024), 0), out=buf666)
        del arg436_1
        buf667 = buf645; del buf645  # reuse
        # Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf665, arg435_1, buf667, 1048576, grid=grid(1048576), stream=stream0)
        del arg435_1
        buf668 = reinterpret_tensor(buf665, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf665  # reuse
        # Source Nodes: [key_states_60], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf666, arg437_1, buf668, 1048576, grid=grid(1048576), stream=stream0)
        del arg437_1
        buf669 = buf632; del buf632  # reuse
        # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf667, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf668, (16, 64, 1024), (65536, 1, 64), 0), out=buf669)
        buf673 = buf628; del buf628  # reuse
        # Source Nodes: [attn_weights_81], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf669, buf673, 16384, 1024, grid=grid(16384), stream=stream0)
        buf672 = reinterpret_tensor(buf668, (1024, 1024), (1024, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg438_1, (1024, 1024), (1, 1024), 0), out=buf672)
        del arg438_1
        buf674 = buf667; del buf667  # reuse
        # Source Nodes: [value_states_60], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf672, arg439_1, buf674, 1048576, grid=grid(1048576), stream=stream0)
        del arg439_1
        buf675 = reinterpret_tensor(buf672, (16, 1024, 64), (65536, 64, 1), 0); del buf672  # reuse
        # Source Nodes: [attn_output_150, attn_weights_81], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf673, reinterpret_tensor(buf674, (16, 1024, 64), (65536, 64, 1), 0), out=buf675)
        buf676 = reinterpret_tensor(buf674, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf674  # reuse
        # Source Nodes: [attn_output_153], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf675, buf676, 1048576, grid=grid(1048576), stream=stream0)
        buf677 = reinterpret_tensor(buf675, (1024, 1024), (1024, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf676, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg440_1, (1024, 1024), (1, 1024), 0), out=buf677)
        del arg440_1
        buf681 = reinterpret_tensor(buf676, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf676  # reuse
        # Source Nodes: [hidden_states_275, residual_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf664, buf677, arg441_1, arg442_1, arg443_1, buf681, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg441_1
        del arg442_1
        del arg443_1
        buf682 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf681, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg444_1, (1024, 1024), (1, 1024), 0), out=buf682)
        del arg444_1
        buf683 = reinterpret_tensor(buf664, (1024, 1024), (1024, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg446_1, (1024, 1024), (1, 1024), 0), out=buf683)
        del arg446_1
        buf684 = buf666; del buf666  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg448_1, (1024, 1024), (1, 1024), 0), out=buf684)
        del arg448_1
        buf685 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf682, arg445_1, buf685, 1048576, grid=grid(1048576), stream=stream0)
        del arg445_1
        buf686 = reinterpret_tensor(buf682, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf682  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf683, arg447_1, buf686, 1048576, grid=grid(1048576), stream=stream0)
        del arg447_1
        buf687 = reinterpret_tensor(buf683, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf684, arg449_1, buf687, 1048576, grid=grid(1048576), stream=stream0)
        del arg449_1
        del buf684
        # Source Nodes: [], Original ATen: []
        buf688 = aten._scaled_dot_product_efficient_attention(buf685, buf686, buf687, None, True, scale=1.0)
        buf689 = buf688[0]
        del buf688
        buf693 = reinterpret_tensor(buf689, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf689  # reuse
        # Source Nodes: [attn_output_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf693, 1048576, grid=grid(1048576), stream=stream0)
        buf694 = reinterpret_tensor(buf687, (1024, 1024), (1024, 1), 0); del buf687  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf693, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg450_1, (1024, 1024), (1, 1024), 0), out=buf694)
        del arg450_1
        buf698 = reinterpret_tensor(buf693, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf693  # reuse
        # Source Nodes: [hidden_states_279, residual_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf681, buf694, arg451_1, arg452_1, arg453_1, buf698, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg451_1
        del arg452_1
        del arg453_1
        buf699 = reinterpret_tensor(buf659, (1024, 4096), (4096, 1), 0); del buf659  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf698, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg454_1, (1024, 4096), (1, 1024), 0), out=buf699)
        del arg454_1
        buf700 = reinterpret_tensor(buf699, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf699  # reuse
        # Source Nodes: [hidden_states_281], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf700, arg455_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg455_1
        buf701 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf700, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg456_1, (4096, 1024), (1, 4096), 0), out=buf701)
        del arg456_1
        buf705 = buf681; del buf681  # reuse
        # Source Nodes: [hidden_states_285, residual_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf698, buf701, arg457_1, arg458_1, arg459_1, buf705, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        buf706 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg460_1, (1024, 1024), (1, 1024), 0), out=buf706)
        del arg460_1
        buf707 = reinterpret_tensor(buf698, (1024, 1024), (1024, 1), 0); del buf698  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg462_1, (1024, 1024), (1, 1024), 0), out=buf707)
        del arg462_1
        buf708 = buf686; del buf686  # reuse
        # Source Nodes: [contiguous_98], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf706, arg461_1, buf708, 1048576, grid=grid(1048576), stream=stream0)
        del arg461_1
        buf709 = reinterpret_tensor(buf706, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf706  # reuse
        # Source Nodes: [key_states_64], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf707, arg463_1, buf709, 1048576, grid=grid(1048576), stream=stream0)
        del arg463_1
        buf710 = buf673; del buf673  # reuse
        # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf708, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf709, (16, 64, 1024), (65536, 1, 64), 0), out=buf710)
        buf714 = buf669; del buf669  # reuse
        # Source Nodes: [attn_weights_87], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf710, buf714, 16384, 1024, grid=grid(16384), stream=stream0)
        buf713 = reinterpret_tensor(buf709, (1024, 1024), (1024, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg464_1, (1024, 1024), (1, 1024), 0), out=buf713)
        del arg464_1
        buf715 = buf708; del buf708  # reuse
        # Source Nodes: [value_states_64], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf713, arg465_1, buf715, 1048576, grid=grid(1048576), stream=stream0)
        del arg465_1
        buf716 = reinterpret_tensor(buf713, (16, 1024, 64), (65536, 64, 1), 0); del buf713  # reuse
        # Source Nodes: [attn_output_160, attn_weights_87], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf714, reinterpret_tensor(buf715, (16, 1024, 64), (65536, 64, 1), 0), out=buf716)
        buf717 = reinterpret_tensor(buf715, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf715  # reuse
        # Source Nodes: [attn_output_163], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf716, buf717, 1048576, grid=grid(1048576), stream=stream0)
        buf718 = reinterpret_tensor(buf716, (1024, 1024), (1024, 1), 0); del buf716  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf717, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg466_1, (1024, 1024), (1, 1024), 0), out=buf718)
        del arg466_1
        buf722 = reinterpret_tensor(buf717, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf717  # reuse
        # Source Nodes: [hidden_states_290, residual_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf705, buf718, arg467_1, arg468_1, arg469_1, buf722, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        buf723 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf722, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg470_1, (1024, 1024), (1, 1024), 0), out=buf723)
        del arg470_1
        buf724 = reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg472_1, (1024, 1024), (1, 1024), 0), out=buf724)
        del arg472_1
        buf725 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg474_1, (1024, 1024), (1, 1024), 0), out=buf725)
        del arg474_1
        buf726 = buf685; del buf685  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf723, arg471_1, buf726, 1048576, grid=grid(1048576), stream=stream0)
        del arg471_1
        buf727 = reinterpret_tensor(buf723, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf724, arg473_1, buf727, 1048576, grid=grid(1048576), stream=stream0)
        del arg473_1
        buf728 = reinterpret_tensor(buf724, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf725, arg475_1, buf728, 1048576, grid=grid(1048576), stream=stream0)
        del arg475_1
        del buf725
        # Source Nodes: [], Original ATen: []
        buf729 = aten._scaled_dot_product_efficient_attention(buf726, buf727, buf728, None, True, scale=1.0)
        buf730 = buf729[0]
        del buf729
        buf734 = reinterpret_tensor(buf730, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf730  # reuse
        # Source Nodes: [attn_output_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf734, 1048576, grid=grid(1048576), stream=stream0)
        buf735 = reinterpret_tensor(buf728, (1024, 1024), (1024, 1), 0); del buf728  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf734, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg476_1, (1024, 1024), (1, 1024), 0), out=buf735)
        del arg476_1
        buf739 = reinterpret_tensor(buf734, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf734  # reuse
        # Source Nodes: [hidden_states_294, residual_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf722, buf735, arg477_1, arg478_1, arg479_1, buf739, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        buf740 = reinterpret_tensor(buf700, (1024, 4096), (4096, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf739, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg480_1, (1024, 4096), (1, 1024), 0), out=buf740)
        del arg480_1
        buf741 = reinterpret_tensor(buf740, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf740  # reuse
        # Source Nodes: [hidden_states_296], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf741, arg481_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg481_1
        buf742 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf741, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg482_1, (4096, 1024), (1, 4096), 0), out=buf742)
        del arg482_1
        buf746 = buf722; del buf722  # reuse
        # Source Nodes: [hidden_states_300, residual_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf739, buf742, arg483_1, arg484_1, arg485_1, buf746, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg483_1
        del arg484_1
        del arg485_1
        buf747 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg486_1, (1024, 1024), (1, 1024), 0), out=buf747)
        del arg486_1
        buf748 = reinterpret_tensor(buf739, (1024, 1024), (1024, 1), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg488_1, (1024, 1024), (1, 1024), 0), out=buf748)
        del arg488_1
        buf749 = buf727; del buf727  # reuse
        # Source Nodes: [contiguous_104], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf747, arg487_1, buf749, 1048576, grid=grid(1048576), stream=stream0)
        del arg487_1
        buf750 = reinterpret_tensor(buf747, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf747  # reuse
        # Source Nodes: [key_states_68], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf748, arg489_1, buf750, 1048576, grid=grid(1048576), stream=stream0)
        del arg489_1
        buf751 = buf714; del buf714  # reuse
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf749, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf750, (16, 64, 1024), (65536, 1, 64), 0), out=buf751)
        buf755 = buf710; del buf710  # reuse
        # Source Nodes: [attn_weights_93], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf751, buf755, 16384, 1024, grid=grid(16384), stream=stream0)
        del buf751
        buf754 = reinterpret_tensor(buf750, (1024, 1024), (1024, 1), 0); del buf750  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg490_1, (1024, 1024), (1, 1024), 0), out=buf754)
        del arg490_1
        buf756 = buf749; del buf749  # reuse
        # Source Nodes: [value_states_68], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf754, arg491_1, buf756, 1048576, grid=grid(1048576), stream=stream0)
        del arg491_1
        buf757 = reinterpret_tensor(buf754, (16, 1024, 64), (65536, 64, 1), 0); del buf754  # reuse
        # Source Nodes: [attn_output_170, attn_weights_93], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf755, reinterpret_tensor(buf756, (16, 1024, 64), (65536, 64, 1), 0), out=buf757)
        del buf755
        buf758 = reinterpret_tensor(buf756, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf756  # reuse
        # Source Nodes: [attn_output_173], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf757, buf758, 1048576, grid=grid(1048576), stream=stream0)
        buf759 = reinterpret_tensor(buf757, (1024, 1024), (1024, 1), 0); del buf757  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf758, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg492_1, (1024, 1024), (1, 1024), 0), out=buf759)
        del arg492_1
        buf763 = reinterpret_tensor(buf758, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf758  # reuse
        # Source Nodes: [hidden_states_305, residual_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf746, buf759, arg493_1, arg494_1, arg495_1, buf763, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg493_1
        del arg494_1
        del arg495_1
        buf764 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf763, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg496_1, (1024, 1024), (1, 1024), 0), out=buf764)
        del arg496_1
        buf765 = reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg498_1, (1024, 1024), (1, 1024), 0), out=buf765)
        del arg498_1
        buf766 = buf748; del buf748  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg500_1, (1024, 1024), (1, 1024), 0), out=buf766)
        del arg500_1
        buf767 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf764, arg497_1, buf767, 1048576, grid=grid(1048576), stream=stream0)
        del arg497_1
        buf768 = reinterpret_tensor(buf764, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf765, arg499_1, buf768, 1048576, grid=grid(1048576), stream=stream0)
        del arg499_1
        buf769 = reinterpret_tensor(buf765, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf766, arg501_1, buf769, 1048576, grid=grid(1048576), stream=stream0)
        del arg501_1
        del buf766
        # Source Nodes: [], Original ATen: []
        buf770 = aten._scaled_dot_product_efficient_attention(buf767, buf768, buf769, None, True, scale=1.0)
        del buf767
        del buf768
        buf771 = buf770[0]
        del buf770
        buf775 = reinterpret_tensor(buf771, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf771  # reuse
        # Source Nodes: [attn_output_178], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf775, 1048576, grid=grid(1048576), stream=stream0)
        buf776 = reinterpret_tensor(buf769, (1024, 1024), (1024, 1), 0); del buf769  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg502_1, (1024, 1024), (1, 1024), 0), out=buf776)
        del arg502_1
        buf780 = reinterpret_tensor(buf775, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf775  # reuse
        # Source Nodes: [hidden_states_309, residual_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf763, buf776, arg503_1, arg504_1, arg505_1, buf780, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg503_1
        del arg504_1
        del arg505_1
        buf781 = reinterpret_tensor(buf741, (1024, 4096), (4096, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf780, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg506_1, (1024, 4096), (1, 1024), 0), out=buf781)
        del arg506_1
        buf782 = reinterpret_tensor(buf781, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf781  # reuse
        # Source Nodes: [hidden_states_311], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf782, arg507_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg507_1
        buf783 = buf776; del buf776  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf782, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg508_1, (4096, 1024), (1, 4096), 0), out=buf783)
        del arg508_1
        del buf782
        buf787 = buf763; del buf763  # reuse
        # Source Nodes: [hidden_states_315, hidden_states_317], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf780, buf783, arg509_1, arg510_1, arg511_1, buf787, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg509_1
        del arg510_1
        del arg511_1
        del buf780
        del buf783
        buf788 = empty((1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg512_1, (1024, 50265), (1, 1024), 0), out=buf788)
        del arg512_1
        del buf787
        buf789 = reinterpret_tensor(buf788, (1, 1024, 50265), (51471360, 50265, 1), 0); del buf788  # reuse
        buf790 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf791 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
        triton_red_fused__log_softmax_add_9.run(buf789, arg513_1, buf790, buf791, 1024, 50265, grid=grid(1024), stream=stream0)
        del arg513_1
        buf792 = empty((), device='cuda', dtype=torch.float32)
        buf794 = buf792; del buf792  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_10.run(buf794, arg514_1, buf789, buf790, buf791, 1, 1024, grid=grid(1), stream=stream0)
        del arg514_1
        return (buf794, buf789, buf313, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
