
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gawpagtbyrtw4yorr7olzgu3fma4zoufyznmzouke32c4xfvce.py
# Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# hidden_states => add_2
# hidden_states_2 => add_3, add_4, mul_1, mul_2, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_embed_tokens => embedding
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp1 = tmp0 + 256008
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 256008)) | ~xmask, "index out of bounds: 0 <= tmp3 < 256008")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 32.0
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
        tmp13 = tmp0 + 256008
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 256008)) | ~xmask, "index out of bounds: 0 <= tmp15 < 256008")
        tmp16 = tl.load(in_ptr1 + (r1 + (1024*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 32.0
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


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gax6334qzekxhy2aesjywl5msnqvansqebtlt2tfynzbrw2w26.py
# Source Nodes: [key_states], Original ATen: [aten.clone]
# key_states => clone_1
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrsxupe5636mx2n57deyzxtqulbvmqqeyejithuljlywvma7twd.py
# Source Nodes: [attn_weights_4], Original ATen: [aten._softmax]
# attn_weights_4 => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp8 = triton_helpers.maximum(tmp7, tmp5)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cyw777ynzwzk6hzpxbdvdlt4vcsdef2ixeuhwfe7fvqzbt673ryl.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_5
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c732mhizl4vhe4j3h2o27zq6qji5favoyxq7abtygxjpougztjff.py
# Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_embed_tokens, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# hidden_states => add_2
# hidden_states_6 => add_7, add_8, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => mul
# l__mod___model_embed_tokens => embedding
# residual_1 => add_6
triton_per_fused_add_embedding_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 256008
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 256008)) | ~xmask, "index out of bounds: 0 <= tmp3 < 256008")
    tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, other=0.0)
    tmp5 = 32.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 1024, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 1024.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuklwt2vbojntdr7hn45tjmo3uj2ojdqrhywalfy2327p7eyfy4.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# hidden_states_7 => add_9, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvyak5lvhztyi6qxzi6c4h2rnt2wgyfiiq3fgknnycajyaj2ayv.py
# Source Nodes: [hidden_states_13, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_13 => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_10
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpxgo7fq4hpn2mpakkzwubpehizqhy4wzucbu5wmst6fqgw4xzl.py
# Source Nodes: [hidden_states_17, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_17 => add_15, add_16, mul_12, mul_13, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_10
# residual_3 => add_14
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2h7fqossci33qc6umag3eg3oaf5dummfnyzudkhuy2qw47c3es.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_red_fused__log_softmax_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64002
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
        tmp0 = tl.load(in_ptr0 + (r1 + (64002*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyx4cc7g4mversmsnni4ldg2ur42kntrvw2ssviiuaywzb35nfhj.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_per_fused__log_softmax_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rc/crcv3v4gy6o4uzlwgw5giy53gqszzt3k6cwbuvyp6vg3imj4jva3.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_73, sum_25
triton_red_fused__log_softmax_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64002
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
        tmp0 = tl.load(in_ptr0 + (r2 + (64002*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ga/cgannrqm2o7yybovckmqvymqutdxopcvuvj4p6qsh3nifecsbsd2.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_73, sum_25
triton_per_fused__log_softmax_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ad/cadzr4zfyo7knbcua3f6djkevdcdt3yll5eu7iz7hvsmwhc3w5rq.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_24, full_default_28, ne_1, ne_2, neg, sum_26, sum_27, where_2
triton_per_fused_nll_loss_forward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp19 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp0 = r0
    tmp1 = tl.full([1, 1], 127, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 127, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0, [XBLOCK, RBLOCK])), rmask & tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp12 = tl.full([1, 1], -100, tl.int64)
    tmp13 = tmp11 != tmp12
    tmp14 = tl.where(tmp13, tmp11, tmp8)
    tmp15 = tmp14 + 256008
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tl.device_assert((0 <= tmp17) & (tmp17 < 256008), "index out of bounds: 0 <= tmp17 < 256008")
    tmp18 = tl.load(in_ptr1 + (tmp17 + (256008*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp22 = tl.log(tmp21)
    tmp23 = tmp20 - tmp22
    tmp24 = -tmp23
    tmp25 = 0.0
    tmp26 = tl.where(tmp13, tmp24, tmp25)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp13.to(tl.int64)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp30 / tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256008, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1024, ), (1, ))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg3_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (1024, ), (1, ))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg30_1, (4096, ), (1, ))
    assert_size_stride(arg31_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg46_1, (4096, ), (1, ))
    assert_size_stride(arg47_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg62_1, (4096, ), (1, ))
    assert_size_stride(arg63_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg78_1, (4096, ), (1, ))
    assert_size_stride(arg79_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg94_1, (4096, ), (1, ))
    assert_size_stride(arg95_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg110_1, (4096, ), (1, ))
    assert_size_stride(arg111_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg126_1, (4096, ), (1, ))
    assert_size_stride(arg127_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg142_1, (4096, ), (1, ))
    assert_size_stride(arg143_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg158_1, (4096, ), (1, ))
    assert_size_stride(arg159_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg174_1, (4096, ), (1, ))
    assert_size_stride(arg175_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg190_1, (4096, ), (1, ))
    assert_size_stride(arg191_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg206_1, (4096, ), (1, ))
    assert_size_stride(arg207_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg222_1, (4096, ), (1, ))
    assert_size_stride(arg223_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg238_1, (4096, ), (1, ))
    assert_size_stride(arg239_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg254_1, (4096, ), (1, ))
    assert_size_stride(arg255_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg270_1, (4096, ), (1, ))
    assert_size_stride(arg271_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg286_1, (4096, ), (1, ))
    assert_size_stride(arg287_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (1024, ), (1, ))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg302_1, (4096, ), (1, ))
    assert_size_stride(arg303_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg318_1, (4096, ), (1, ))
    assert_size_stride(arg319_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg334_1, (4096, ), (1, ))
    assert_size_stride(arg335_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg350_1, (4096, ), (1, ))
    assert_size_stride(arg351_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg366_1, (4096, ), (1, ))
    assert_size_stride(arg367_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg382_1, (4096, ), (1, ))
    assert_size_stride(arg383_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (256008, 1024), (1024, 1))
    assert_size_stride(arg388_1, (2050, 1024), (1024, 1))
    assert_size_stride(arg389_1, (1, 128), (128, 1))
    assert_size_stride(arg390_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg389_1, arg0_1, arg388_1, arg1_1, arg2_1, buf3, 128, 1024, grid=grid(128), stream=stream0)
        del arg1_1
        del arg2_1
        buf4 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg3_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg3_1
        buf5 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg5_1
        buf6 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, arg6_1, buf6, 131072, grid=grid(131072), stream=stream0)
        del arg6_1
        buf7 = reinterpret_tensor(buf5, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, arg4_1, buf7, 131072, grid=grid(131072), stream=stream0)
        del arg4_1
        buf8 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf6, (16, 64, 128), (8192, 1, 64), 0), out=buf8)
        buf13 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf8, buf13, 2048, 128, grid=grid(2048), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (128, 1024), (1024, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf11)
        del arg7_1
        buf12 = reinterpret_tensor(buf3, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg8_1, buf12, 131072, grid=grid(131072), stream=stream0)
        del arg8_1
        buf14 = reinterpret_tensor(buf11, (16, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output, attn_weights_4], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (16, 128, 64), (8192, 64, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf4, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 131072, grid=grid(131072), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (128, 1024), (1024, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg9_1
        buf17 = reinterpret_tensor(buf16, (1, 128, 1024), (131072, 1024, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (1, 128, 1024), (131072, 1024, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_embed_tokens, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_embedding_mul_native_layer_norm_5.run(buf17, arg389_1, arg0_1, arg388_1, arg10_1, arg11_1, arg12_1, buf21, 128, 1024, grid=grid(128), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg388_1
        del arg389_1
        buf22 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 4096), (1, 1024), 0), out=buf22)
        del arg13_1
        buf23 = reinterpret_tensor(buf22, (1, 128, 4096), (524288, 4096, 1), 0); del buf22  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf23, arg14_1, 524288, grid=grid(524288), stream=stream0)
        del arg14_1
        buf24 = reinterpret_tensor(buf21, (128, 1024), (1024, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg15_1, (4096, 1024), (1, 4096), 0), out=buf24)
        del arg15_1
        buf28 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf17, buf24, arg16_1, arg17_1, arg18_1, buf28, 128, 1024, grid=grid(128), stream=stream0)
        del arg17_1
        del arg18_1
        buf29 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg19_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg19_1
        buf30 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg21_1
        buf31 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf30, arg22_1, buf31, 131072, grid=grid(131072), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf30, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, arg20_1, buf32, 131072, grid=grid(131072), stream=stream0)
        del arg20_1
        buf33 = buf13; del buf13  # reuse
        # Source Nodes: [attn_weights_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf31, (16, 64, 128), (8192, 1, 64), 0), out=buf33)
        buf38 = buf8; del buf8  # reuse
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf33, buf38, 2048, 128, grid=grid(2048), stream=stream0)
        buf36 = reinterpret_tensor(buf32, (128, 1024), (1024, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf36)
        del arg23_1
        buf37 = reinterpret_tensor(buf28, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, arg24_1, buf37, 131072, grid=grid(131072), stream=stream0)
        del arg24_1
        buf39 = reinterpret_tensor(buf36, (16, 128, 64), (8192, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [attn_output_5, attn_weights_9], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf38, reinterpret_tensor(buf37, (16, 128, 64), (8192, 64, 1), 0), out=buf39)
        buf40 = reinterpret_tensor(buf29, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf39, buf40, 131072, grid=grid(131072), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (128, 1024), (1024, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf41)
        del arg25_1
        buf42 = reinterpret_tensor(buf41, (1, 128, 1024), (131072, 1024, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (1, 128, 1024), (131072, 1024, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_17, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf42, buf17, buf24, arg16_1, arg26_1, arg27_1, arg28_1, buf46, 128, 1024, grid=grid(128), stream=stream0)
        del arg16_1
        del arg26_1
        del arg27_1
        del arg28_1
        buf47 = reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 4096), (1, 1024), 0), out=buf47)
        del arg29_1
        buf48 = reinterpret_tensor(buf47, (1, 128, 4096), (524288, 4096, 1), 0); del buf47  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf48, arg30_1, 524288, grid=grid(524288), stream=stream0)
        del arg30_1
        buf49 = reinterpret_tensor(buf46, (128, 1024), (1024, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg31_1, (4096, 1024), (1, 4096), 0), out=buf49)
        del arg31_1
        buf53 = reinterpret_tensor(buf24, (1, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [hidden_states_24, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf42, buf49, arg32_1, arg33_1, arg34_1, buf53, 128, 1024, grid=grid(128), stream=stream0)
        del arg33_1
        del arg34_1
        buf54 = reinterpret_tensor(buf17, (128, 1024), (1024, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg35_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg35_1
        buf55 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg37_1
        buf56 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf55, arg38_1, buf56, 131072, grid=grid(131072), stream=stream0)
        del arg38_1
        buf57 = reinterpret_tensor(buf55, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf54, arg36_1, buf57, 131072, grid=grid(131072), stream=stream0)
        del arg36_1
        buf58 = buf38; del buf38  # reuse
        # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf56, (16, 64, 128), (8192, 1, 64), 0), out=buf58)
        buf63 = buf33; del buf33  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf58, buf63, 2048, 128, grid=grid(2048), stream=stream0)
        buf61 = reinterpret_tensor(buf57, (128, 1024), (1024, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf61)
        del arg39_1
        buf62 = reinterpret_tensor(buf53, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf61, arg40_1, buf62, 131072, grid=grid(131072), stream=stream0)
        del arg40_1
        buf64 = reinterpret_tensor(buf61, (16, 128, 64), (8192, 64, 1), 0); del buf61  # reuse
        # Source Nodes: [attn_output_10, attn_weights_14], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf63, reinterpret_tensor(buf62, (16, 128, 64), (8192, 64, 1), 0), out=buf64)
        buf65 = reinterpret_tensor(buf54, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf64, buf65, 131072, grid=grid(131072), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (128, 1024), (1024, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf66)
        del arg41_1
        buf67 = reinterpret_tensor(buf66, (1, 128, 1024), (131072, 1024, 1), 0); del buf66  # reuse
        buf71 = reinterpret_tensor(buf65, (1, 128, 1024), (131072, 1024, 1), 0); del buf65  # reuse
        # Source Nodes: [hidden_states_28, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf67, buf42, buf49, arg32_1, arg42_1, arg43_1, arg44_1, buf71, 128, 1024, grid=grid(128), stream=stream0)
        del arg32_1
        del arg42_1
        del arg43_1
        del arg44_1
        buf72 = reinterpret_tensor(buf48, (128, 4096), (4096, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 4096), (1, 1024), 0), out=buf72)
        del arg45_1
        buf73 = reinterpret_tensor(buf72, (1, 128, 4096), (524288, 4096, 1), 0); del buf72  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf73, arg46_1, 524288, grid=grid(524288), stream=stream0)
        del arg46_1
        buf74 = reinterpret_tensor(buf71, (128, 1024), (1024, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg47_1, (4096, 1024), (1, 4096), 0), out=buf74)
        del arg47_1
        buf78 = reinterpret_tensor(buf49, (1, 128, 1024), (131072, 1024, 1), 0); del buf49  # reuse
        # Source Nodes: [hidden_states_35, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf67, buf74, arg48_1, arg49_1, arg50_1, buf78, 128, 1024, grid=grid(128), stream=stream0)
        del arg49_1
        del arg50_1
        buf79 = reinterpret_tensor(buf42, (128, 1024), (1024, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 1024), (1, 1024), 0), out=buf79)
        del arg51_1
        buf80 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), out=buf80)
        del arg53_1
        buf81 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf80, arg54_1, buf81, 131072, grid=grid(131072), stream=stream0)
        del arg54_1
        buf82 = reinterpret_tensor(buf80, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf79, arg52_1, buf82, 131072, grid=grid(131072), stream=stream0)
        del arg52_1
        buf83 = buf63; del buf63  # reuse
        # Source Nodes: [attn_weights_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf81, (16, 64, 128), (8192, 1, 64), 0), out=buf83)
        buf88 = buf58; del buf58  # reuse
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf83, buf88, 2048, 128, grid=grid(2048), stream=stream0)
        buf86 = reinterpret_tensor(buf82, (128, 1024), (1024, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf86)
        del arg55_1
        buf87 = reinterpret_tensor(buf78, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf86, arg56_1, buf87, 131072, grid=grid(131072), stream=stream0)
        del arg56_1
        buf89 = reinterpret_tensor(buf86, (16, 128, 64), (8192, 64, 1), 0); del buf86  # reuse
        # Source Nodes: [attn_output_15, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf88, reinterpret_tensor(buf87, (16, 128, 64), (8192, 64, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf79, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf79  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf90, 131072, grid=grid(131072), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (128, 1024), (1024, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf91)
        del arg57_1
        buf92 = reinterpret_tensor(buf91, (1, 128, 1024), (131072, 1024, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (1, 128, 1024), (131072, 1024, 1), 0); del buf90  # reuse
        # Source Nodes: [hidden_states_39, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf92, buf67, buf74, arg48_1, arg58_1, arg59_1, arg60_1, buf96, 128, 1024, grid=grid(128), stream=stream0)
        del arg48_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf97 = reinterpret_tensor(buf73, (128, 4096), (4096, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 4096), (1, 1024), 0), out=buf97)
        del arg61_1
        buf98 = reinterpret_tensor(buf97, (1, 128, 4096), (524288, 4096, 1), 0); del buf97  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf98, arg62_1, 524288, grid=grid(524288), stream=stream0)
        del arg62_1
        buf99 = reinterpret_tensor(buf96, (128, 1024), (1024, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg63_1, (4096, 1024), (1, 4096), 0), out=buf99)
        del arg63_1
        buf103 = reinterpret_tensor(buf74, (1, 128, 1024), (131072, 1024, 1), 0); del buf74  # reuse
        # Source Nodes: [hidden_states_46, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf92, buf99, arg64_1, arg65_1, arg66_1, buf103, 128, 1024, grid=grid(128), stream=stream0)
        del arg65_1
        del arg66_1
        buf104 = reinterpret_tensor(buf67, (128, 1024), (1024, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg67_1, (1024, 1024), (1, 1024), 0), out=buf104)
        del arg67_1
        buf105 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), out=buf105)
        del arg69_1
        buf106 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf105, arg70_1, buf106, 131072, grid=grid(131072), stream=stream0)
        del arg70_1
        buf107 = reinterpret_tensor(buf105, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf105  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, arg68_1, buf107, 131072, grid=grid(131072), stream=stream0)
        del arg68_1
        buf108 = buf88; del buf88  # reuse
        # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf106, (16, 64, 128), (8192, 1, 64), 0), out=buf108)
        buf113 = buf83; del buf83  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf108, buf113, 2048, 128, grid=grid(2048), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (128, 1024), (1024, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf111)
        del arg71_1
        buf112 = reinterpret_tensor(buf103, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, arg72_1, buf112, 131072, grid=grid(131072), stream=stream0)
        del arg72_1
        buf114 = reinterpret_tensor(buf111, (16, 128, 64), (8192, 64, 1), 0); del buf111  # reuse
        # Source Nodes: [attn_output_20, attn_weights_24], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf113, reinterpret_tensor(buf112, (16, 128, 64), (8192, 64, 1), 0), out=buf114)
        buf115 = reinterpret_tensor(buf104, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf114, buf115, 131072, grid=grid(131072), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (128, 1024), (1024, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf116)
        del arg73_1
        buf117 = reinterpret_tensor(buf116, (1, 128, 1024), (131072, 1024, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf115, (1, 128, 1024), (131072, 1024, 1), 0); del buf115  # reuse
        # Source Nodes: [hidden_states_50, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf117, buf92, buf99, arg64_1, arg74_1, arg75_1, arg76_1, buf121, 128, 1024, grid=grid(128), stream=stream0)
        del arg64_1
        del arg74_1
        del arg75_1
        del arg76_1
        buf122 = reinterpret_tensor(buf98, (128, 4096), (4096, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 4096), (1, 1024), 0), out=buf122)
        del arg77_1
        buf123 = reinterpret_tensor(buf122, (1, 128, 4096), (524288, 4096, 1), 0); del buf122  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf123, arg78_1, 524288, grid=grid(524288), stream=stream0)
        del arg78_1
        buf124 = reinterpret_tensor(buf121, (128, 1024), (1024, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg79_1, (4096, 1024), (1, 4096), 0), out=buf124)
        del arg79_1
        buf128 = reinterpret_tensor(buf99, (1, 128, 1024), (131072, 1024, 1), 0); del buf99  # reuse
        # Source Nodes: [hidden_states_57, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf117, buf124, arg80_1, arg81_1, arg82_1, buf128, 128, 1024, grid=grid(128), stream=stream0)
        del arg81_1
        del arg82_1
        buf129 = reinterpret_tensor(buf92, (128, 1024), (1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg83_1, (1024, 1024), (1, 1024), 0), out=buf129)
        del arg83_1
        buf130 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), out=buf130)
        del arg85_1
        buf131 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf130, arg86_1, buf131, 131072, grid=grid(131072), stream=stream0)
        del arg86_1
        buf132 = reinterpret_tensor(buf130, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf130  # reuse
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, arg84_1, buf132, 131072, grid=grid(131072), stream=stream0)
        del arg84_1
        buf133 = buf113; del buf113  # reuse
        # Source Nodes: [attn_weights_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf131, (16, 64, 128), (8192, 1, 64), 0), out=buf133)
        buf138 = buf108; del buf108  # reuse
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf133, buf138, 2048, 128, grid=grid(2048), stream=stream0)
        buf136 = reinterpret_tensor(buf132, (128, 1024), (1024, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg87_1
        buf137 = reinterpret_tensor(buf128, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf128  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, arg88_1, buf137, 131072, grid=grid(131072), stream=stream0)
        del arg88_1
        buf139 = reinterpret_tensor(buf136, (16, 128, 64), (8192, 64, 1), 0); del buf136  # reuse
        # Source Nodes: [attn_output_25, attn_weights_29], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf138, reinterpret_tensor(buf137, (16, 128, 64), (8192, 64, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf129, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf139, buf140, 131072, grid=grid(131072), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (128, 1024), (1024, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf141)
        del arg89_1
        buf142 = reinterpret_tensor(buf141, (1, 128, 1024), (131072, 1024, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf140, (1, 128, 1024), (131072, 1024, 1), 0); del buf140  # reuse
        # Source Nodes: [hidden_states_61, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf142, buf117, buf124, arg80_1, arg90_1, arg91_1, arg92_1, buf146, 128, 1024, grid=grid(128), stream=stream0)
        del arg80_1
        del arg90_1
        del arg91_1
        del arg92_1
        buf147 = reinterpret_tensor(buf123, (128, 4096), (4096, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 4096), (1, 1024), 0), out=buf147)
        del arg93_1
        buf148 = reinterpret_tensor(buf147, (1, 128, 4096), (524288, 4096, 1), 0); del buf147  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf148, arg94_1, 524288, grid=grid(524288), stream=stream0)
        del arg94_1
        buf149 = reinterpret_tensor(buf146, (128, 1024), (1024, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg95_1, (4096, 1024), (1, 4096), 0), out=buf149)
        del arg95_1
        buf153 = reinterpret_tensor(buf124, (1, 128, 1024), (131072, 1024, 1), 0); del buf124  # reuse
        # Source Nodes: [hidden_states_68, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf142, buf149, arg96_1, arg97_1, arg98_1, buf153, 128, 1024, grid=grid(128), stream=stream0)
        del arg97_1
        del arg98_1
        buf154 = reinterpret_tensor(buf117, (128, 1024), (1024, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 1024), (1, 1024), 0), out=buf154)
        del arg99_1
        buf155 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), out=buf155)
        del arg101_1
        buf156 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg102_1, buf156, 131072, grid=grid(131072), stream=stream0)
        del arg102_1
        buf157 = reinterpret_tensor(buf155, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf154, arg100_1, buf157, 131072, grid=grid(131072), stream=stream0)
        del arg100_1
        buf158 = buf138; del buf138  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf156, (16, 64, 128), (8192, 1, 64), 0), out=buf158)
        buf163 = buf133; del buf133  # reuse
        # Source Nodes: [attn_weights_34], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf158, buf163, 2048, 128, grid=grid(2048), stream=stream0)
        buf161 = reinterpret_tensor(buf157, (128, 1024), (1024, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf161)
        del arg103_1
        buf162 = reinterpret_tensor(buf153, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf161, arg104_1, buf162, 131072, grid=grid(131072), stream=stream0)
        del arg104_1
        buf164 = reinterpret_tensor(buf161, (16, 128, 64), (8192, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [attn_output_30, attn_weights_34], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf163, reinterpret_tensor(buf162, (16, 128, 64), (8192, 64, 1), 0), out=buf164)
        buf165 = reinterpret_tensor(buf154, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf154  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf164, buf165, 131072, grid=grid(131072), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (128, 1024), (1024, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf166)
        del arg105_1
        buf167 = reinterpret_tensor(buf166, (1, 128, 1024), (131072, 1024, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf165, (1, 128, 1024), (131072, 1024, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_72, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf167, buf142, buf149, arg96_1, arg106_1, arg107_1, arg108_1, buf171, 128, 1024, grid=grid(128), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg96_1
        buf172 = reinterpret_tensor(buf148, (128, 4096), (4096, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 4096), (1, 1024), 0), out=buf172)
        del arg109_1
        buf173 = reinterpret_tensor(buf172, (1, 128, 4096), (524288, 4096, 1), 0); del buf172  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf173, arg110_1, 524288, grid=grid(524288), stream=stream0)
        del arg110_1
        buf174 = reinterpret_tensor(buf171, (128, 1024), (1024, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg111_1, (4096, 1024), (1, 4096), 0), out=buf174)
        del arg111_1
        buf178 = reinterpret_tensor(buf149, (1, 128, 1024), (131072, 1024, 1), 0); del buf149  # reuse
        # Source Nodes: [hidden_states_79, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf167, buf174, arg112_1, arg113_1, arg114_1, buf178, 128, 1024, grid=grid(128), stream=stream0)
        del arg113_1
        del arg114_1
        buf179 = reinterpret_tensor(buf142, (128, 1024), (1024, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg115_1, (1024, 1024), (1, 1024), 0), out=buf179)
        del arg115_1
        buf180 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), out=buf180)
        del arg117_1
        buf181 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, arg118_1, buf181, 131072, grid=grid(131072), stream=stream0)
        del arg118_1
        buf182 = reinterpret_tensor(buf180, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf180  # reuse
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, arg116_1, buf182, 131072, grid=grid(131072), stream=stream0)
        del arg116_1
        buf183 = buf163; del buf163  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf181, (16, 64, 128), (8192, 1, 64), 0), out=buf183)
        buf188 = buf158; del buf158  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf183, buf188, 2048, 128, grid=grid(2048), stream=stream0)
        buf186 = reinterpret_tensor(buf182, (128, 1024), (1024, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf186)
        del arg119_1
        buf187 = reinterpret_tensor(buf178, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, arg120_1, buf187, 131072, grid=grid(131072), stream=stream0)
        del arg120_1
        buf189 = reinterpret_tensor(buf186, (16, 128, 64), (8192, 64, 1), 0); del buf186  # reuse
        # Source Nodes: [attn_output_35, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf188, reinterpret_tensor(buf187, (16, 128, 64), (8192, 64, 1), 0), out=buf189)
        buf190 = reinterpret_tensor(buf179, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf189, buf190, 131072, grid=grid(131072), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (128, 1024), (1024, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf191)
        del arg121_1
        buf192 = reinterpret_tensor(buf191, (1, 128, 1024), (131072, 1024, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf190, (1, 128, 1024), (131072, 1024, 1), 0); del buf190  # reuse
        # Source Nodes: [hidden_states_83, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf192, buf167, buf174, arg112_1, arg122_1, arg123_1, arg124_1, buf196, 128, 1024, grid=grid(128), stream=stream0)
        del arg112_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf197 = reinterpret_tensor(buf173, (128, 4096), (4096, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 4096), (1, 1024), 0), out=buf197)
        del arg125_1
        buf198 = reinterpret_tensor(buf197, (1, 128, 4096), (524288, 4096, 1), 0); del buf197  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf198, arg126_1, 524288, grid=grid(524288), stream=stream0)
        del arg126_1
        buf199 = reinterpret_tensor(buf196, (128, 1024), (1024, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg127_1, (4096, 1024), (1, 4096), 0), out=buf199)
        del arg127_1
        buf203 = reinterpret_tensor(buf174, (1, 128, 1024), (131072, 1024, 1), 0); del buf174  # reuse
        # Source Nodes: [hidden_states_90, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf192, buf199, arg128_1, arg129_1, arg130_1, buf203, 128, 1024, grid=grid(128), stream=stream0)
        del arg129_1
        del arg130_1
        buf204 = reinterpret_tensor(buf167, (128, 1024), (1024, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 1024), (1, 1024), 0), out=buf204)
        del arg131_1
        buf205 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), out=buf205)
        del arg133_1
        buf206 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf205, arg134_1, buf206, 131072, grid=grid(131072), stream=stream0)
        del arg134_1
        buf207 = reinterpret_tensor(buf205, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf204, arg132_1, buf207, 131072, grid=grid(131072), stream=stream0)
        del arg132_1
        buf208 = buf188; del buf188  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf206, (16, 64, 128), (8192, 1, 64), 0), out=buf208)
        buf213 = buf183; del buf183  # reuse
        # Source Nodes: [attn_weights_44], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf208, buf213, 2048, 128, grid=grid(2048), stream=stream0)
        buf211 = reinterpret_tensor(buf207, (128, 1024), (1024, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf211)
        del arg135_1
        buf212 = reinterpret_tensor(buf203, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, arg136_1, buf212, 131072, grid=grid(131072), stream=stream0)
        del arg136_1
        buf214 = reinterpret_tensor(buf211, (16, 128, 64), (8192, 64, 1), 0); del buf211  # reuse
        # Source Nodes: [attn_output_40, attn_weights_44], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf213, reinterpret_tensor(buf212, (16, 128, 64), (8192, 64, 1), 0), out=buf214)
        buf215 = reinterpret_tensor(buf204, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf204  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf214, buf215, 131072, grid=grid(131072), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (128, 1024), (1024, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf216)
        del arg137_1
        buf217 = reinterpret_tensor(buf216, (1, 128, 1024), (131072, 1024, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf215, (1, 128, 1024), (131072, 1024, 1), 0); del buf215  # reuse
        # Source Nodes: [hidden_states_94, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf217, buf192, buf199, arg128_1, arg138_1, arg139_1, arg140_1, buf221, 128, 1024, grid=grid(128), stream=stream0)
        del arg128_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf222 = reinterpret_tensor(buf198, (128, 4096), (4096, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 4096), (1, 1024), 0), out=buf222)
        del arg141_1
        buf223 = reinterpret_tensor(buf222, (1, 128, 4096), (524288, 4096, 1), 0); del buf222  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf223, arg142_1, 524288, grid=grid(524288), stream=stream0)
        del arg142_1
        buf224 = reinterpret_tensor(buf221, (128, 1024), (1024, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg143_1, (4096, 1024), (1, 4096), 0), out=buf224)
        del arg143_1
        buf228 = reinterpret_tensor(buf199, (1, 128, 1024), (131072, 1024, 1), 0); del buf199  # reuse
        # Source Nodes: [hidden_states_101, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf217, buf224, arg144_1, arg145_1, arg146_1, buf228, 128, 1024, grid=grid(128), stream=stream0)
        del arg145_1
        del arg146_1
        buf229 = reinterpret_tensor(buf192, (128, 1024), (1024, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg147_1, (1024, 1024), (1, 1024), 0), out=buf229)
        del arg147_1
        buf230 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf230)
        del arg149_1
        buf231 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, arg150_1, buf231, 131072, grid=grid(131072), stream=stream0)
        del arg150_1
        buf232 = reinterpret_tensor(buf230, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf230  # reuse
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf229, arg148_1, buf232, 131072, grid=grid(131072), stream=stream0)
        del arg148_1
        buf233 = buf213; del buf213  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf231, (16, 64, 128), (8192, 1, 64), 0), out=buf233)
        buf238 = buf208; del buf208  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf233, buf238, 2048, 128, grid=grid(2048), stream=stream0)
        buf236 = reinterpret_tensor(buf232, (128, 1024), (1024, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf236)
        del arg151_1
        buf237 = reinterpret_tensor(buf228, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf236, arg152_1, buf237, 131072, grid=grid(131072), stream=stream0)
        del arg152_1
        buf239 = reinterpret_tensor(buf236, (16, 128, 64), (8192, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [attn_output_45, attn_weights_49], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf238, reinterpret_tensor(buf237, (16, 128, 64), (8192, 64, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf229, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf239, buf240, 131072, grid=grid(131072), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (128, 1024), (1024, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf241)
        del arg153_1
        buf242 = reinterpret_tensor(buf241, (1, 128, 1024), (131072, 1024, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf240, (1, 128, 1024), (131072, 1024, 1), 0); del buf240  # reuse
        # Source Nodes: [hidden_states_105, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf242, buf217, buf224, arg144_1, arg154_1, arg155_1, arg156_1, buf246, 128, 1024, grid=grid(128), stream=stream0)
        del arg144_1
        del arg154_1
        del arg155_1
        del arg156_1
        buf247 = reinterpret_tensor(buf223, (128, 4096), (4096, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 4096), (1, 1024), 0), out=buf247)
        del arg157_1
        buf248 = reinterpret_tensor(buf247, (1, 128, 4096), (524288, 4096, 1), 0); del buf247  # reuse
        # Source Nodes: [hidden_states_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf248, arg158_1, 524288, grid=grid(524288), stream=stream0)
        del arg158_1
        buf249 = reinterpret_tensor(buf246, (128, 1024), (1024, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg159_1, (4096, 1024), (1, 4096), 0), out=buf249)
        del arg159_1
        buf253 = reinterpret_tensor(buf224, (1, 128, 1024), (131072, 1024, 1), 0); del buf224  # reuse
        # Source Nodes: [hidden_states_112, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf242, buf249, arg160_1, arg161_1, arg162_1, buf253, 128, 1024, grid=grid(128), stream=stream0)
        del arg161_1
        del arg162_1
        buf254 = reinterpret_tensor(buf217, (128, 1024), (1024, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg163_1, (1024, 1024), (1, 1024), 0), out=buf254)
        del arg163_1
        buf255 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), out=buf255)
        del arg165_1
        buf256 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf255, arg166_1, buf256, 131072, grid=grid(131072), stream=stream0)
        del arg166_1
        buf257 = reinterpret_tensor(buf255, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf255  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf254, arg164_1, buf257, 131072, grid=grid(131072), stream=stream0)
        del arg164_1
        buf258 = buf238; del buf238  # reuse
        # Source Nodes: [attn_weights_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf256, (16, 64, 128), (8192, 1, 64), 0), out=buf258)
        buf263 = buf233; del buf233  # reuse
        # Source Nodes: [attn_weights_54], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf258, buf263, 2048, 128, grid=grid(2048), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (128, 1024), (1024, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf261)
        del arg167_1
        buf262 = reinterpret_tensor(buf253, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf261, arg168_1, buf262, 131072, grid=grid(131072), stream=stream0)
        del arg168_1
        buf264 = reinterpret_tensor(buf261, (16, 128, 64), (8192, 64, 1), 0); del buf261  # reuse
        # Source Nodes: [attn_output_50, attn_weights_54], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf263, reinterpret_tensor(buf262, (16, 128, 64), (8192, 64, 1), 0), out=buf264)
        buf265 = reinterpret_tensor(buf254, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf254  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf264, buf265, 131072, grid=grid(131072), stream=stream0)
        buf266 = reinterpret_tensor(buf264, (128, 1024), (1024, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf266)
        del arg169_1
        buf267 = reinterpret_tensor(buf266, (1, 128, 1024), (131072, 1024, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf265, (1, 128, 1024), (131072, 1024, 1), 0); del buf265  # reuse
        # Source Nodes: [hidden_states_116, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf267, buf242, buf249, arg160_1, arg170_1, arg171_1, arg172_1, buf271, 128, 1024, grid=grid(128), stream=stream0)
        del arg160_1
        del arg170_1
        del arg171_1
        del arg172_1
        buf272 = reinterpret_tensor(buf248, (128, 4096), (4096, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 4096), (1, 1024), 0), out=buf272)
        del arg173_1
        buf273 = reinterpret_tensor(buf272, (1, 128, 4096), (524288, 4096, 1), 0); del buf272  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf273, arg174_1, 524288, grid=grid(524288), stream=stream0)
        del arg174_1
        buf274 = reinterpret_tensor(buf271, (128, 1024), (1024, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg175_1, (4096, 1024), (1, 4096), 0), out=buf274)
        del arg175_1
        buf278 = reinterpret_tensor(buf249, (1, 128, 1024), (131072, 1024, 1), 0); del buf249  # reuse
        # Source Nodes: [hidden_states_123, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf267, buf274, arg176_1, arg177_1, arg178_1, buf278, 128, 1024, grid=grid(128), stream=stream0)
        del arg177_1
        del arg178_1
        buf279 = reinterpret_tensor(buf242, (128, 1024), (1024, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 1024), (1, 1024), 0), out=buf279)
        del arg179_1
        buf280 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg181_1
        buf281 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf280, arg182_1, buf281, 131072, grid=grid(131072), stream=stream0)
        del arg182_1
        buf282 = reinterpret_tensor(buf280, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf280  # reuse
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf279, arg180_1, buf282, 131072, grid=grid(131072), stream=stream0)
        del arg180_1
        buf283 = buf263; del buf263  # reuse
        # Source Nodes: [attn_weights_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf281, (16, 64, 128), (8192, 1, 64), 0), out=buf283)
        buf288 = buf258; del buf258  # reuse
        # Source Nodes: [attn_weights_59], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf283, buf288, 2048, 128, grid=grid(2048), stream=stream0)
        buf286 = reinterpret_tensor(buf282, (128, 1024), (1024, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf286)
        del arg183_1
        buf287 = reinterpret_tensor(buf278, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, arg184_1, buf287, 131072, grid=grid(131072), stream=stream0)
        del arg184_1
        buf289 = reinterpret_tensor(buf286, (16, 128, 64), (8192, 64, 1), 0); del buf286  # reuse
        # Source Nodes: [attn_output_55, attn_weights_59], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf288, reinterpret_tensor(buf287, (16, 128, 64), (8192, 64, 1), 0), out=buf289)
        buf290 = reinterpret_tensor(buf279, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf279  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf289, buf290, 131072, grid=grid(131072), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (128, 1024), (1024, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf291)
        del arg185_1
        buf292 = reinterpret_tensor(buf291, (1, 128, 1024), (131072, 1024, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (1, 128, 1024), (131072, 1024, 1), 0); del buf290  # reuse
        # Source Nodes: [hidden_states_127, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf292, buf267, buf274, arg176_1, arg186_1, arg187_1, arg188_1, buf296, 128, 1024, grid=grid(128), stream=stream0)
        del arg176_1
        del arg186_1
        del arg187_1
        del arg188_1
        buf297 = reinterpret_tensor(buf273, (128, 4096), (4096, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 4096), (1, 1024), 0), out=buf297)
        del arg189_1
        buf298 = reinterpret_tensor(buf297, (1, 128, 4096), (524288, 4096, 1), 0); del buf297  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf298, arg190_1, 524288, grid=grid(524288), stream=stream0)
        del arg190_1
        buf299 = reinterpret_tensor(buf296, (128, 1024), (1024, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg191_1, (4096, 1024), (1, 4096), 0), out=buf299)
        del arg191_1
        buf303 = reinterpret_tensor(buf274, (1, 128, 1024), (131072, 1024, 1), 0); del buf274  # reuse
        # Source Nodes: [hidden_states_134, residual_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf292, buf299, arg192_1, arg193_1, arg194_1, buf303, 128, 1024, grid=grid(128), stream=stream0)
        del arg193_1
        del arg194_1
        buf304 = reinterpret_tensor(buf267, (128, 1024), (1024, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg195_1, (1024, 1024), (1, 1024), 0), out=buf304)
        del arg195_1
        buf305 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg197_1, (1024, 1024), (1, 1024), 0), out=buf305)
        del arg197_1
        buf306 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf305, arg198_1, buf306, 131072, grid=grid(131072), stream=stream0)
        del arg198_1
        buf307 = reinterpret_tensor(buf305, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf305  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf304, arg196_1, buf307, 131072, grid=grid(131072), stream=stream0)
        del arg196_1
        buf308 = buf288; del buf288  # reuse
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf306, (16, 64, 128), (8192, 1, 64), 0), out=buf308)
        buf313 = buf283; del buf283  # reuse
        # Source Nodes: [attn_weights_64], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf308, buf313, 2048, 128, grid=grid(2048), stream=stream0)
        buf311 = reinterpret_tensor(buf307, (128, 1024), (1024, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg199_1, (1024, 1024), (1, 1024), 0), out=buf311)
        del arg199_1
        buf312 = reinterpret_tensor(buf303, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf303  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf311, arg200_1, buf312, 131072, grid=grid(131072), stream=stream0)
        del arg200_1
        buf314 = reinterpret_tensor(buf311, (16, 128, 64), (8192, 64, 1), 0); del buf311  # reuse
        # Source Nodes: [attn_output_60, attn_weights_64], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf313, reinterpret_tensor(buf312, (16, 128, 64), (8192, 64, 1), 0), out=buf314)
        buf315 = reinterpret_tensor(buf304, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf304  # reuse
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf314, buf315, 131072, grid=grid(131072), stream=stream0)
        buf316 = reinterpret_tensor(buf314, (128, 1024), (1024, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), out=buf316)
        del arg201_1
        buf317 = reinterpret_tensor(buf316, (1, 128, 1024), (131072, 1024, 1), 0); del buf316  # reuse
        buf321 = reinterpret_tensor(buf315, (1, 128, 1024), (131072, 1024, 1), 0); del buf315  # reuse
        # Source Nodes: [hidden_states_138, residual_24, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf317, buf292, buf299, arg192_1, arg202_1, arg203_1, arg204_1, buf321, 128, 1024, grid=grid(128), stream=stream0)
        del arg192_1
        del arg202_1
        del arg203_1
        del arg204_1
        buf322 = reinterpret_tensor(buf298, (128, 4096), (4096, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf321, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 4096), (1, 1024), 0), out=buf322)
        del arg205_1
        buf323 = reinterpret_tensor(buf322, (1, 128, 4096), (524288, 4096, 1), 0); del buf322  # reuse
        # Source Nodes: [hidden_states_139], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf323, arg206_1, 524288, grid=grid(524288), stream=stream0)
        del arg206_1
        buf324 = reinterpret_tensor(buf321, (128, 1024), (1024, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg207_1, (4096, 1024), (1, 4096), 0), out=buf324)
        del arg207_1
        buf328 = reinterpret_tensor(buf299, (1, 128, 1024), (131072, 1024, 1), 0); del buf299  # reuse
        # Source Nodes: [hidden_states_145, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf317, buf324, arg208_1, arg209_1, arg210_1, buf328, 128, 1024, grid=grid(128), stream=stream0)
        del arg209_1
        del arg210_1
        buf329 = reinterpret_tensor(buf292, (128, 1024), (1024, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 1024), (1, 1024), 0), out=buf329)
        del arg211_1
        buf330 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), out=buf330)
        del arg213_1
        buf331 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf330, arg214_1, buf331, 131072, grid=grid(131072), stream=stream0)
        del arg214_1
        buf332 = reinterpret_tensor(buf330, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf330  # reuse
        # Source Nodes: [contiguous_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf329, arg212_1, buf332, 131072, grid=grid(131072), stream=stream0)
        del arg212_1
        buf333 = buf313; del buf313  # reuse
        # Source Nodes: [attn_weights_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf331, (16, 64, 128), (8192, 1, 64), 0), out=buf333)
        buf338 = buf308; del buf308  # reuse
        # Source Nodes: [attn_weights_69], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf333, buf338, 2048, 128, grid=grid(2048), stream=stream0)
        buf336 = reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf336)
        del arg215_1
        buf337 = reinterpret_tensor(buf328, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf328  # reuse
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf336, arg216_1, buf337, 131072, grid=grid(131072), stream=stream0)
        del arg216_1
        buf339 = reinterpret_tensor(buf336, (16, 128, 64), (8192, 64, 1), 0); del buf336  # reuse
        # Source Nodes: [attn_output_65, attn_weights_69], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf338, reinterpret_tensor(buf337, (16, 128, 64), (8192, 64, 1), 0), out=buf339)
        buf340 = reinterpret_tensor(buf329, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf329  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf339, buf340, 131072, grid=grid(131072), stream=stream0)
        buf341 = reinterpret_tensor(buf339, (128, 1024), (1024, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf341)
        del arg217_1
        buf342 = reinterpret_tensor(buf341, (1, 128, 1024), (131072, 1024, 1), 0); del buf341  # reuse
        buf346 = reinterpret_tensor(buf340, (1, 128, 1024), (131072, 1024, 1), 0); del buf340  # reuse
        # Source Nodes: [hidden_states_149, residual_26, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf342, buf317, buf324, arg208_1, arg218_1, arg219_1, arg220_1, buf346, 128, 1024, grid=grid(128), stream=stream0)
        del arg208_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf347 = reinterpret_tensor(buf323, (128, 4096), (4096, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 4096), (1, 1024), 0), out=buf347)
        del arg221_1
        buf348 = reinterpret_tensor(buf347, (1, 128, 4096), (524288, 4096, 1), 0); del buf347  # reuse
        # Source Nodes: [hidden_states_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf348, arg222_1, 524288, grid=grid(524288), stream=stream0)
        del arg222_1
        buf349 = reinterpret_tensor(buf346, (128, 1024), (1024, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg223_1, (4096, 1024), (1, 4096), 0), out=buf349)
        del arg223_1
        buf353 = reinterpret_tensor(buf324, (1, 128, 1024), (131072, 1024, 1), 0); del buf324  # reuse
        # Source Nodes: [hidden_states_156, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf342, buf349, arg224_1, arg225_1, arg226_1, buf353, 128, 1024, grid=grid(128), stream=stream0)
        del arg225_1
        del arg226_1
        buf354 = reinterpret_tensor(buf317, (128, 1024), (1024, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 1024), (1, 1024), 0), out=buf354)
        del arg227_1
        buf355 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), out=buf355)
        del arg229_1
        buf356 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf355, arg230_1, buf356, 131072, grid=grid(131072), stream=stream0)
        del arg230_1
        buf357 = reinterpret_tensor(buf355, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf355  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf354, arg228_1, buf357, 131072, grid=grid(131072), stream=stream0)
        del arg228_1
        buf358 = buf338; del buf338  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf357, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf356, (16, 64, 128), (8192, 1, 64), 0), out=buf358)
        buf363 = buf333; del buf333  # reuse
        # Source Nodes: [attn_weights_74], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf358, buf363, 2048, 128, grid=grid(2048), stream=stream0)
        buf361 = reinterpret_tensor(buf357, (128, 1024), (1024, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf361)
        del arg231_1
        buf362 = reinterpret_tensor(buf353, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf353  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf361, arg232_1, buf362, 131072, grid=grid(131072), stream=stream0)
        del arg232_1
        buf364 = reinterpret_tensor(buf361, (16, 128, 64), (8192, 64, 1), 0); del buf361  # reuse
        # Source Nodes: [attn_output_70, attn_weights_74], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf363, reinterpret_tensor(buf362, (16, 128, 64), (8192, 64, 1), 0), out=buf364)
        buf365 = reinterpret_tensor(buf354, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf354  # reuse
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf364, buf365, 131072, grid=grid(131072), stream=stream0)
        buf366 = reinterpret_tensor(buf364, (128, 1024), (1024, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg233_1
        buf367 = reinterpret_tensor(buf366, (1, 128, 1024), (131072, 1024, 1), 0); del buf366  # reuse
        buf371 = reinterpret_tensor(buf365, (1, 128, 1024), (131072, 1024, 1), 0); del buf365  # reuse
        # Source Nodes: [hidden_states_160, residual_28, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf367, buf342, buf349, arg224_1, arg234_1, arg235_1, arg236_1, buf371, 128, 1024, grid=grid(128), stream=stream0)
        del arg224_1
        del arg234_1
        del arg235_1
        del arg236_1
        buf372 = reinterpret_tensor(buf348, (128, 4096), (4096, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 4096), (1, 1024), 0), out=buf372)
        del arg237_1
        buf373 = reinterpret_tensor(buf372, (1, 128, 4096), (524288, 4096, 1), 0); del buf372  # reuse
        # Source Nodes: [hidden_states_161], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf373, arg238_1, 524288, grid=grid(524288), stream=stream0)
        del arg238_1
        buf374 = reinterpret_tensor(buf371, (128, 1024), (1024, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg239_1, (4096, 1024), (1, 4096), 0), out=buf374)
        del arg239_1
        buf378 = reinterpret_tensor(buf349, (1, 128, 1024), (131072, 1024, 1), 0); del buf349  # reuse
        # Source Nodes: [hidden_states_167, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf367, buf374, arg240_1, arg241_1, arg242_1, buf378, 128, 1024, grid=grid(128), stream=stream0)
        del arg241_1
        del arg242_1
        buf379 = reinterpret_tensor(buf342, (128, 1024), (1024, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 1024), (1, 1024), 0), out=buf379)
        del arg243_1
        buf380 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg245_1, (1024, 1024), (1, 1024), 0), out=buf380)
        del arg245_1
        buf381 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf380, arg246_1, buf381, 131072, grid=grid(131072), stream=stream0)
        del arg246_1
        buf382 = reinterpret_tensor(buf380, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf380  # reuse
        # Source Nodes: [contiguous_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf379, arg244_1, buf382, 131072, grid=grid(131072), stream=stream0)
        del arg244_1
        buf383 = buf363; del buf363  # reuse
        # Source Nodes: [attn_weights_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf382, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf381, (16, 64, 128), (8192, 1, 64), 0), out=buf383)
        buf388 = buf358; del buf358  # reuse
        # Source Nodes: [attn_weights_79], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf383, buf388, 2048, 128, grid=grid(2048), stream=stream0)
        buf386 = reinterpret_tensor(buf382, (128, 1024), (1024, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 1024), (1, 1024), 0), out=buf386)
        del arg247_1
        buf387 = reinterpret_tensor(buf378, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf378  # reuse
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf386, arg248_1, buf387, 131072, grid=grid(131072), stream=stream0)
        del arg248_1
        buf389 = reinterpret_tensor(buf386, (16, 128, 64), (8192, 64, 1), 0); del buf386  # reuse
        # Source Nodes: [attn_output_75, attn_weights_79], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf388, reinterpret_tensor(buf387, (16, 128, 64), (8192, 64, 1), 0), out=buf389)
        buf390 = reinterpret_tensor(buf379, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf379  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf389, buf390, 131072, grid=grid(131072), stream=stream0)
        buf391 = reinterpret_tensor(buf389, (128, 1024), (1024, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf390, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg249_1, (1024, 1024), (1, 1024), 0), out=buf391)
        del arg249_1
        buf392 = reinterpret_tensor(buf391, (1, 128, 1024), (131072, 1024, 1), 0); del buf391  # reuse
        buf396 = reinterpret_tensor(buf390, (1, 128, 1024), (131072, 1024, 1), 0); del buf390  # reuse
        # Source Nodes: [hidden_states_171, residual_30, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf392, buf367, buf374, arg240_1, arg250_1, arg251_1, arg252_1, buf396, 128, 1024, grid=grid(128), stream=stream0)
        del arg240_1
        del arg250_1
        del arg251_1
        del arg252_1
        buf397 = reinterpret_tensor(buf373, (128, 4096), (4096, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 4096), (1, 1024), 0), out=buf397)
        del arg253_1
        buf398 = reinterpret_tensor(buf397, (1, 128, 4096), (524288, 4096, 1), 0); del buf397  # reuse
        # Source Nodes: [hidden_states_172], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf398, arg254_1, 524288, grid=grid(524288), stream=stream0)
        del arg254_1
        buf399 = reinterpret_tensor(buf396, (128, 1024), (1024, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg255_1, (4096, 1024), (1, 4096), 0), out=buf399)
        del arg255_1
        buf403 = reinterpret_tensor(buf374, (1, 128, 1024), (131072, 1024, 1), 0); del buf374  # reuse
        # Source Nodes: [hidden_states_178, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf392, buf399, arg256_1, arg257_1, arg258_1, buf403, 128, 1024, grid=grid(128), stream=stream0)
        del arg257_1
        del arg258_1
        buf404 = reinterpret_tensor(buf367, (128, 1024), (1024, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 1024), (1, 1024), 0), out=buf404)
        del arg259_1
        buf405 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg261_1, (1024, 1024), (1, 1024), 0), out=buf405)
        del arg261_1
        buf406 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf405, arg262_1, buf406, 131072, grid=grid(131072), stream=stream0)
        del arg262_1
        buf407 = reinterpret_tensor(buf405, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf405  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf404, arg260_1, buf407, 131072, grid=grid(131072), stream=stream0)
        del arg260_1
        buf408 = buf388; del buf388  # reuse
        # Source Nodes: [attn_weights_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf406, (16, 64, 128), (8192, 1, 64), 0), out=buf408)
        buf413 = buf383; del buf383  # reuse
        # Source Nodes: [attn_weights_84], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf408, buf413, 2048, 128, grid=grid(2048), stream=stream0)
        buf411 = reinterpret_tensor(buf407, (128, 1024), (1024, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf411)
        del arg263_1
        buf412 = reinterpret_tensor(buf403, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf403  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf411, arg264_1, buf412, 131072, grid=grid(131072), stream=stream0)
        del arg264_1
        buf414 = reinterpret_tensor(buf411, (16, 128, 64), (8192, 64, 1), 0); del buf411  # reuse
        # Source Nodes: [attn_output_80, attn_weights_84], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf413, reinterpret_tensor(buf412, (16, 128, 64), (8192, 64, 1), 0), out=buf414)
        buf415 = reinterpret_tensor(buf404, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf404  # reuse
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf414, buf415, 131072, grid=grid(131072), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (128, 1024), (1024, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf415, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), out=buf416)
        del arg265_1
        buf417 = reinterpret_tensor(buf416, (1, 128, 1024), (131072, 1024, 1), 0); del buf416  # reuse
        buf421 = reinterpret_tensor(buf415, (1, 128, 1024), (131072, 1024, 1), 0); del buf415  # reuse
        # Source Nodes: [hidden_states_182, residual_32, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf417, buf392, buf399, arg256_1, arg266_1, arg267_1, arg268_1, buf421, 128, 1024, grid=grid(128), stream=stream0)
        del arg256_1
        del arg266_1
        del arg267_1
        del arg268_1
        buf422 = reinterpret_tensor(buf398, (128, 4096), (4096, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 4096), (1, 1024), 0), out=buf422)
        del arg269_1
        buf423 = reinterpret_tensor(buf422, (1, 128, 4096), (524288, 4096, 1), 0); del buf422  # reuse
        # Source Nodes: [hidden_states_183], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf423, arg270_1, 524288, grid=grid(524288), stream=stream0)
        del arg270_1
        buf424 = reinterpret_tensor(buf421, (128, 1024), (1024, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg271_1, (4096, 1024), (1, 4096), 0), out=buf424)
        del arg271_1
        buf428 = reinterpret_tensor(buf399, (1, 128, 1024), (131072, 1024, 1), 0); del buf399  # reuse
        # Source Nodes: [hidden_states_189, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf417, buf424, arg272_1, arg273_1, arg274_1, buf428, 128, 1024, grid=grid(128), stream=stream0)
        del arg273_1
        del arg274_1
        buf429 = reinterpret_tensor(buf392, (128, 1024), (1024, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg275_1, (1024, 1024), (1, 1024), 0), out=buf429)
        del arg275_1
        buf430 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg277_1, (1024, 1024), (1, 1024), 0), out=buf430)
        del arg277_1
        buf431 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf430, arg278_1, buf431, 131072, grid=grid(131072), stream=stream0)
        del arg278_1
        buf432 = reinterpret_tensor(buf430, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf430  # reuse
        # Source Nodes: [contiguous_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf429, arg276_1, buf432, 131072, grid=grid(131072), stream=stream0)
        del arg276_1
        buf433 = buf413; del buf413  # reuse
        # Source Nodes: [attn_weights_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf431, (16, 64, 128), (8192, 1, 64), 0), out=buf433)
        buf438 = buf408; del buf408  # reuse
        # Source Nodes: [attn_weights_89], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf433, buf438, 2048, 128, grid=grid(2048), stream=stream0)
        buf436 = reinterpret_tensor(buf432, (128, 1024), (1024, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), out=buf436)
        del arg279_1
        buf437 = reinterpret_tensor(buf428, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf428  # reuse
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf436, arg280_1, buf437, 131072, grid=grid(131072), stream=stream0)
        del arg280_1
        buf439 = reinterpret_tensor(buf436, (16, 128, 64), (8192, 64, 1), 0); del buf436  # reuse
        # Source Nodes: [attn_output_85, attn_weights_89], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf438, reinterpret_tensor(buf437, (16, 128, 64), (8192, 64, 1), 0), out=buf439)
        buf440 = reinterpret_tensor(buf429, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf429  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf439, buf440, 131072, grid=grid(131072), stream=stream0)
        buf441 = reinterpret_tensor(buf439, (128, 1024), (1024, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), out=buf441)
        del arg281_1
        buf442 = reinterpret_tensor(buf441, (1, 128, 1024), (131072, 1024, 1), 0); del buf441  # reuse
        buf446 = reinterpret_tensor(buf440, (1, 128, 1024), (131072, 1024, 1), 0); del buf440  # reuse
        # Source Nodes: [hidden_states_193, residual_34, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf442, buf417, buf424, arg272_1, arg282_1, arg283_1, arg284_1, buf446, 128, 1024, grid=grid(128), stream=stream0)
        del arg272_1
        del arg282_1
        del arg283_1
        del arg284_1
        buf447 = reinterpret_tensor(buf423, (128, 4096), (4096, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 4096), (1, 1024), 0), out=buf447)
        del arg285_1
        buf448 = reinterpret_tensor(buf447, (1, 128, 4096), (524288, 4096, 1), 0); del buf447  # reuse
        # Source Nodes: [hidden_states_194], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf448, arg286_1, 524288, grid=grid(524288), stream=stream0)
        del arg286_1
        buf449 = reinterpret_tensor(buf446, (128, 1024), (1024, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg287_1, (4096, 1024), (1, 4096), 0), out=buf449)
        del arg287_1
        buf453 = reinterpret_tensor(buf424, (1, 128, 1024), (131072, 1024, 1), 0); del buf424  # reuse
        # Source Nodes: [hidden_states_200, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf442, buf449, arg288_1, arg289_1, arg290_1, buf453, 128, 1024, grid=grid(128), stream=stream0)
        del arg289_1
        del arg290_1
        buf454 = reinterpret_tensor(buf417, (128, 1024), (1024, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 1024), (1, 1024), 0), out=buf454)
        del arg291_1
        buf455 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf455)
        del arg293_1
        buf456 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf455, arg294_1, buf456, 131072, grid=grid(131072), stream=stream0)
        del arg294_1
        buf457 = reinterpret_tensor(buf455, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf455  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf454, arg292_1, buf457, 131072, grid=grid(131072), stream=stream0)
        del arg292_1
        buf458 = buf438; del buf438  # reuse
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf457, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf456, (16, 64, 128), (8192, 1, 64), 0), out=buf458)
        buf463 = buf433; del buf433  # reuse
        # Source Nodes: [attn_weights_94], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf458, buf463, 2048, 128, grid=grid(2048), stream=stream0)
        buf461 = reinterpret_tensor(buf457, (128, 1024), (1024, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf461)
        del arg295_1
        buf462 = reinterpret_tensor(buf453, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf453  # reuse
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf461, arg296_1, buf462, 131072, grid=grid(131072), stream=stream0)
        del arg296_1
        buf464 = reinterpret_tensor(buf461, (16, 128, 64), (8192, 64, 1), 0); del buf461  # reuse
        # Source Nodes: [attn_output_90, attn_weights_94], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf463, reinterpret_tensor(buf462, (16, 128, 64), (8192, 64, 1), 0), out=buf464)
        buf465 = reinterpret_tensor(buf454, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf454  # reuse
        # Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf464, buf465, 131072, grid=grid(131072), stream=stream0)
        buf466 = reinterpret_tensor(buf464, (128, 1024), (1024, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg297_1, (1024, 1024), (1, 1024), 0), out=buf466)
        del arg297_1
        buf467 = reinterpret_tensor(buf466, (1, 128, 1024), (131072, 1024, 1), 0); del buf466  # reuse
        buf471 = reinterpret_tensor(buf465, (1, 128, 1024), (131072, 1024, 1), 0); del buf465  # reuse
        # Source Nodes: [hidden_states_204, residual_36, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf467, buf442, buf449, arg288_1, arg298_1, arg299_1, arg300_1, buf471, 128, 1024, grid=grid(128), stream=stream0)
        del arg288_1
        del arg298_1
        del arg299_1
        del arg300_1
        buf472 = reinterpret_tensor(buf448, (128, 4096), (4096, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg301_1, (1024, 4096), (1, 1024), 0), out=buf472)
        del arg301_1
        buf473 = reinterpret_tensor(buf472, (1, 128, 4096), (524288, 4096, 1), 0); del buf472  # reuse
        # Source Nodes: [hidden_states_205], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf473, arg302_1, 524288, grid=grid(524288), stream=stream0)
        del arg302_1
        buf474 = reinterpret_tensor(buf471, (128, 1024), (1024, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf473, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg303_1, (4096, 1024), (1, 4096), 0), out=buf474)
        del arg303_1
        buf478 = reinterpret_tensor(buf449, (1, 128, 1024), (131072, 1024, 1), 0); del buf449  # reuse
        # Source Nodes: [hidden_states_211, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf467, buf474, arg304_1, arg305_1, arg306_1, buf478, 128, 1024, grid=grid(128), stream=stream0)
        del arg305_1
        del arg306_1
        buf479 = reinterpret_tensor(buf442, (128, 1024), (1024, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 1024), (1, 1024), 0), out=buf479)
        del arg307_1
        buf480 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf480)
        del arg309_1
        buf481 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf480, arg310_1, buf481, 131072, grid=grid(131072), stream=stream0)
        del arg310_1
        buf482 = reinterpret_tensor(buf480, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf480  # reuse
        # Source Nodes: [contiguous_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf479, arg308_1, buf482, 131072, grid=grid(131072), stream=stream0)
        del arg308_1
        buf483 = buf463; del buf463  # reuse
        # Source Nodes: [attn_weights_95], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf482, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf481, (16, 64, 128), (8192, 1, 64), 0), out=buf483)
        buf488 = buf458; del buf458  # reuse
        # Source Nodes: [attn_weights_99], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf483, buf488, 2048, 128, grid=grid(2048), stream=stream0)
        buf486 = reinterpret_tensor(buf482, (128, 1024), (1024, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf486)
        del arg311_1
        buf487 = reinterpret_tensor(buf478, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf478  # reuse
        # Source Nodes: [value_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf486, arg312_1, buf487, 131072, grid=grid(131072), stream=stream0)
        del arg312_1
        buf489 = reinterpret_tensor(buf486, (16, 128, 64), (8192, 64, 1), 0); del buf486  # reuse
        # Source Nodes: [attn_output_95, attn_weights_99], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf488, reinterpret_tensor(buf487, (16, 128, 64), (8192, 64, 1), 0), out=buf489)
        buf490 = reinterpret_tensor(buf479, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf479  # reuse
        # Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf489, buf490, 131072, grid=grid(131072), stream=stream0)
        buf491 = reinterpret_tensor(buf489, (128, 1024), (1024, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 1024), (1, 1024), 0), out=buf491)
        del arg313_1
        buf492 = reinterpret_tensor(buf491, (1, 128, 1024), (131072, 1024, 1), 0); del buf491  # reuse
        buf496 = reinterpret_tensor(buf490, (1, 128, 1024), (131072, 1024, 1), 0); del buf490  # reuse
        # Source Nodes: [hidden_states_215, residual_38, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf492, buf467, buf474, arg304_1, arg314_1, arg315_1, arg316_1, buf496, 128, 1024, grid=grid(128), stream=stream0)
        del arg304_1
        del arg314_1
        del arg315_1
        del arg316_1
        buf497 = reinterpret_tensor(buf473, (128, 4096), (4096, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 4096), (1, 1024), 0), out=buf497)
        del arg317_1
        buf498 = reinterpret_tensor(buf497, (1, 128, 4096), (524288, 4096, 1), 0); del buf497  # reuse
        # Source Nodes: [hidden_states_216], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf498, arg318_1, 524288, grid=grid(524288), stream=stream0)
        del arg318_1
        buf499 = reinterpret_tensor(buf496, (128, 1024), (1024, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg319_1, (4096, 1024), (1, 4096), 0), out=buf499)
        del arg319_1
        buf503 = reinterpret_tensor(buf474, (1, 128, 1024), (131072, 1024, 1), 0); del buf474  # reuse
        # Source Nodes: [hidden_states_222, residual_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf492, buf499, arg320_1, arg321_1, arg322_1, buf503, 128, 1024, grid=grid(128), stream=stream0)
        del arg321_1
        del arg322_1
        buf504 = reinterpret_tensor(buf467, (128, 1024), (1024, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg323_1, (1024, 1024), (1, 1024), 0), out=buf504)
        del arg323_1
        buf505 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 1024), (1, 1024), 0), out=buf505)
        del arg325_1
        buf506 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf505, arg326_1, buf506, 131072, grid=grid(131072), stream=stream0)
        del arg326_1
        buf507 = reinterpret_tensor(buf505, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf505  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf504, arg324_1, buf507, 131072, grid=grid(131072), stream=stream0)
        del arg324_1
        buf508 = buf488; del buf488  # reuse
        # Source Nodes: [attn_weights_100], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf507, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf506, (16, 64, 128), (8192, 1, 64), 0), out=buf508)
        buf513 = buf483; del buf483  # reuse
        # Source Nodes: [attn_weights_104], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf508, buf513, 2048, 128, grid=grid(2048), stream=stream0)
        buf511 = reinterpret_tensor(buf507, (128, 1024), (1024, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg327_1, (1024, 1024), (1, 1024), 0), out=buf511)
        del arg327_1
        buf512 = reinterpret_tensor(buf503, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf503  # reuse
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf511, arg328_1, buf512, 131072, grid=grid(131072), stream=stream0)
        del arg328_1
        buf514 = reinterpret_tensor(buf511, (16, 128, 64), (8192, 64, 1), 0); del buf511  # reuse
        # Source Nodes: [attn_output_100, attn_weights_104], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf513, reinterpret_tensor(buf512, (16, 128, 64), (8192, 64, 1), 0), out=buf514)
        buf515 = reinterpret_tensor(buf504, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf504  # reuse
        # Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf514, buf515, 131072, grid=grid(131072), stream=stream0)
        buf516 = reinterpret_tensor(buf514, (128, 1024), (1024, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg329_1, (1024, 1024), (1, 1024), 0), out=buf516)
        del arg329_1
        buf517 = reinterpret_tensor(buf516, (1, 128, 1024), (131072, 1024, 1), 0); del buf516  # reuse
        buf521 = reinterpret_tensor(buf515, (1, 128, 1024), (131072, 1024, 1), 0); del buf515  # reuse
        # Source Nodes: [hidden_states_226, residual_40, residual_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf517, buf492, buf499, arg320_1, arg330_1, arg331_1, arg332_1, buf521, 128, 1024, grid=grid(128), stream=stream0)
        del arg320_1
        del arg330_1
        del arg331_1
        del arg332_1
        buf522 = reinterpret_tensor(buf498, (128, 4096), (4096, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf521, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg333_1, (1024, 4096), (1, 1024), 0), out=buf522)
        del arg333_1
        buf523 = reinterpret_tensor(buf522, (1, 128, 4096), (524288, 4096, 1), 0); del buf522  # reuse
        # Source Nodes: [hidden_states_227], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf523, arg334_1, 524288, grid=grid(524288), stream=stream0)
        del arg334_1
        buf524 = reinterpret_tensor(buf521, (128, 1024), (1024, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg335_1, (4096, 1024), (1, 4096), 0), out=buf524)
        del arg335_1
        buf528 = reinterpret_tensor(buf499, (1, 128, 1024), (131072, 1024, 1), 0); del buf499  # reuse
        # Source Nodes: [hidden_states_233, residual_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf517, buf524, arg336_1, arg337_1, arg338_1, buf528, 128, 1024, grid=grid(128), stream=stream0)
        del arg337_1
        del arg338_1
        buf529 = reinterpret_tensor(buf492, (128, 1024), (1024, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg339_1, (1024, 1024), (1, 1024), 0), out=buf529)
        del arg339_1
        buf530 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf530)
        del arg341_1
        buf531 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf530, arg342_1, buf531, 131072, grid=grid(131072), stream=stream0)
        del arg342_1
        buf532 = reinterpret_tensor(buf530, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf530  # reuse
        # Source Nodes: [contiguous_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf529, arg340_1, buf532, 131072, grid=grid(131072), stream=stream0)
        del arg340_1
        buf533 = buf513; del buf513  # reuse
        # Source Nodes: [attn_weights_105], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf531, (16, 64, 128), (8192, 1, 64), 0), out=buf533)
        buf538 = buf508; del buf508  # reuse
        # Source Nodes: [attn_weights_109], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf533, buf538, 2048, 128, grid=grid(2048), stream=stream0)
        buf536 = reinterpret_tensor(buf532, (128, 1024), (1024, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), out=buf536)
        del arg343_1
        buf537 = reinterpret_tensor(buf528, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf528  # reuse
        # Source Nodes: [value_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf536, arg344_1, buf537, 131072, grid=grid(131072), stream=stream0)
        del arg344_1
        buf539 = reinterpret_tensor(buf536, (16, 128, 64), (8192, 64, 1), 0); del buf536  # reuse
        # Source Nodes: [attn_output_105, attn_weights_109], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf538, reinterpret_tensor(buf537, (16, 128, 64), (8192, 64, 1), 0), out=buf539)
        buf540 = reinterpret_tensor(buf529, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf529  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf539, buf540, 131072, grid=grid(131072), stream=stream0)
        buf541 = reinterpret_tensor(buf539, (128, 1024), (1024, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf541)
        del arg345_1
        buf542 = reinterpret_tensor(buf541, (1, 128, 1024), (131072, 1024, 1), 0); del buf541  # reuse
        buf546 = reinterpret_tensor(buf540, (1, 128, 1024), (131072, 1024, 1), 0); del buf540  # reuse
        # Source Nodes: [hidden_states_237, residual_42, residual_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf542, buf517, buf524, arg336_1, arg346_1, arg347_1, arg348_1, buf546, 128, 1024, grid=grid(128), stream=stream0)
        del arg336_1
        del arg346_1
        del arg347_1
        del arg348_1
        buf547 = reinterpret_tensor(buf523, (128, 4096), (4096, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg349_1, (1024, 4096), (1, 1024), 0), out=buf547)
        del arg349_1
        buf548 = reinterpret_tensor(buf547, (1, 128, 4096), (524288, 4096, 1), 0); del buf547  # reuse
        # Source Nodes: [hidden_states_238], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf548, arg350_1, 524288, grid=grid(524288), stream=stream0)
        del arg350_1
        buf549 = reinterpret_tensor(buf546, (128, 1024), (1024, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg351_1, (4096, 1024), (1, 4096), 0), out=buf549)
        del arg351_1
        buf553 = reinterpret_tensor(buf524, (1, 128, 1024), (131072, 1024, 1), 0); del buf524  # reuse
        # Source Nodes: [hidden_states_244, residual_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf542, buf549, arg352_1, arg353_1, arg354_1, buf553, 128, 1024, grid=grid(128), stream=stream0)
        del arg353_1
        del arg354_1
        buf554 = reinterpret_tensor(buf517, (128, 1024), (1024, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg355_1, (1024, 1024), (1, 1024), 0), out=buf554)
        del arg355_1
        buf555 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), out=buf555)
        del arg357_1
        buf556 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf555, arg358_1, buf556, 131072, grid=grid(131072), stream=stream0)
        del arg358_1
        buf557 = reinterpret_tensor(buf555, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf555  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf554, arg356_1, buf557, 131072, grid=grid(131072), stream=stream0)
        del arg356_1
        buf558 = buf538; del buf538  # reuse
        # Source Nodes: [attn_weights_110], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf556, (16, 64, 128), (8192, 1, 64), 0), out=buf558)
        buf563 = buf533; del buf533  # reuse
        # Source Nodes: [attn_weights_114], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf558, buf563, 2048, 128, grid=grid(2048), stream=stream0)
        buf561 = reinterpret_tensor(buf557, (128, 1024), (1024, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), out=buf561)
        del arg359_1
        buf562 = reinterpret_tensor(buf553, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf553  # reuse
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf561, arg360_1, buf562, 131072, grid=grid(131072), stream=stream0)
        del arg360_1
        buf564 = reinterpret_tensor(buf561, (16, 128, 64), (8192, 64, 1), 0); del buf561  # reuse
        # Source Nodes: [attn_output_110, attn_weights_114], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf563, reinterpret_tensor(buf562, (16, 128, 64), (8192, 64, 1), 0), out=buf564)
        buf565 = reinterpret_tensor(buf554, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf554  # reuse
        # Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf564, buf565, 131072, grid=grid(131072), stream=stream0)
        buf566 = reinterpret_tensor(buf564, (128, 1024), (1024, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf566)
        del arg361_1
        buf567 = reinterpret_tensor(buf566, (1, 128, 1024), (131072, 1024, 1), 0); del buf566  # reuse
        buf571 = reinterpret_tensor(buf565, (1, 128, 1024), (131072, 1024, 1), 0); del buf565  # reuse
        # Source Nodes: [hidden_states_248, residual_44, residual_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf567, buf542, buf549, arg352_1, arg362_1, arg363_1, arg364_1, buf571, 128, 1024, grid=grid(128), stream=stream0)
        del arg352_1
        del arg362_1
        del arg363_1
        del arg364_1
        buf572 = reinterpret_tensor(buf548, (128, 4096), (4096, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf571, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg365_1, (1024, 4096), (1, 1024), 0), out=buf572)
        del arg365_1
        buf573 = reinterpret_tensor(buf572, (1, 128, 4096), (524288, 4096, 1), 0); del buf572  # reuse
        # Source Nodes: [hidden_states_249], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf573, arg366_1, 524288, grid=grid(524288), stream=stream0)
        del arg366_1
        buf574 = reinterpret_tensor(buf571, (128, 1024), (1024, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg367_1, (4096, 1024), (1, 4096), 0), out=buf574)
        del arg367_1
        buf578 = reinterpret_tensor(buf549, (1, 128, 1024), (131072, 1024, 1), 0); del buf549  # reuse
        # Source Nodes: [hidden_states_255, residual_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf567, buf574, arg368_1, arg369_1, arg370_1, buf578, 128, 1024, grid=grid(128), stream=stream0)
        del arg369_1
        del arg370_1
        buf579 = reinterpret_tensor(buf542, (128, 1024), (1024, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg371_1, (1024, 1024), (1, 1024), 0), out=buf579)
        del arg371_1
        buf580 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf580)
        del arg373_1
        buf581 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf580, arg374_1, buf581, 131072, grid=grid(131072), stream=stream0)
        del arg374_1
        buf582 = reinterpret_tensor(buf580, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf580  # reuse
        # Source Nodes: [contiguous_71], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf579, arg372_1, buf582, 131072, grid=grid(131072), stream=stream0)
        del arg372_1
        buf583 = buf563; del buf563  # reuse
        # Source Nodes: [attn_weights_115], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf581, (16, 64, 128), (8192, 1, 64), 0), out=buf583)
        buf588 = buf558; del buf558  # reuse
        # Source Nodes: [attn_weights_119], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf583, buf588, 2048, 128, grid=grid(2048), stream=stream0)
        del buf583
        buf586 = reinterpret_tensor(buf582, (128, 1024), (1024, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg375_1, (1024, 1024), (1, 1024), 0), out=buf586)
        del arg375_1
        buf587 = reinterpret_tensor(buf578, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf578  # reuse
        # Source Nodes: [value_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf586, arg376_1, buf587, 131072, grid=grid(131072), stream=stream0)
        del arg376_1
        buf589 = reinterpret_tensor(buf586, (16, 128, 64), (8192, 64, 1), 0); del buf586  # reuse
        # Source Nodes: [attn_output_115, attn_weights_119], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf588, reinterpret_tensor(buf587, (16, 128, 64), (8192, 64, 1), 0), out=buf589)
        del buf588
        buf590 = reinterpret_tensor(buf579, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf579  # reuse
        # Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf589, buf590, 131072, grid=grid(131072), stream=stream0)
        buf591 = reinterpret_tensor(buf589, (128, 1024), (1024, 1), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf590, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 1024), (1, 1024), 0), out=buf591)
        del arg377_1
        buf592 = reinterpret_tensor(buf591, (1, 128, 1024), (131072, 1024, 1), 0); del buf591  # reuse
        buf596 = reinterpret_tensor(buf590, (1, 128, 1024), (131072, 1024, 1), 0); del buf590  # reuse
        # Source Nodes: [hidden_states_259, residual_46, residual_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf592, buf567, buf574, arg368_1, arg378_1, arg379_1, arg380_1, buf596, 128, 1024, grid=grid(128), stream=stream0)
        del arg368_1
        del arg378_1
        del arg379_1
        del arg380_1
        del buf567
        buf597 = reinterpret_tensor(buf573, (128, 4096), (4096, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg381_1, (1024, 4096), (1, 1024), 0), out=buf597)
        del arg381_1
        buf598 = reinterpret_tensor(buf597, (1, 128, 4096), (524288, 4096, 1), 0); del buf597  # reuse
        # Source Nodes: [hidden_states_260], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf598, arg382_1, 524288, grid=grid(524288), stream=stream0)
        del arg382_1
        buf599 = reinterpret_tensor(buf596, (128, 1024), (1024, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg383_1, (4096, 1024), (1, 4096), 0), out=buf599)
        del arg383_1
        del buf598
        buf603 = reinterpret_tensor(buf574, (1, 128, 1024), (131072, 1024, 1), 0); del buf574  # reuse
        # Source Nodes: [hidden_states_265, hidden_states_266], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf592, buf599, arg384_1, arg385_1, arg386_1, buf603, 128, 1024, grid=grid(128), stream=stream0)
        del arg384_1
        del arg385_1
        del arg386_1
        del buf592
        del buf599
        buf604 = empty((128, 256008), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg387_1, (1024, 256008), (1, 1024), 0), out=buf604)
        del arg387_1
        del buf603
        buf605 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_9.run(buf604, buf605, 512, 64002, grid=grid(512), stream=stream0)
        buf606 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_10.run(buf605, buf606, 128, 4, grid=grid(128), stream=stream0)
        buf607 = buf605; del buf605  # reuse
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_11.run(buf604, buf606, buf607, 512, 64002, grid=grid(512), stream=stream0)
        buf608 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_12.run(buf607, buf608, 128, 4, grid=grid(128), stream=stream0)
        del buf607
        buf609 = empty((), device='cuda', dtype=torch.float32)
        buf611 = buf609; del buf609  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_13.run(buf611, arg390_1, buf604, buf606, buf608, 1, 128, grid=grid(1), stream=stream0)
        del arg390_1
        return (buf611, reinterpret_tensor(buf604, (1, 128, 256008), (32769024, 256008, 1), 0), buf6, buf12, buf31, buf37, buf56, buf62, buf81, buf87, buf106, buf112, buf131, buf137, buf156, buf162, buf181, buf187, buf206, buf212, buf231, buf237, buf256, buf262, buf281, buf287, buf306, buf312, buf331, buf337, buf356, buf362, buf381, buf387, buf406, buf412, buf431, buf437, buf456, buf462, buf481, buf487, buf506, buf512, buf531, buf537, buf556, buf562, buf581, buf587, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256008, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((256008, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((2050, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg390_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XGLMForCausalLM', benchmark_compiled_module)
