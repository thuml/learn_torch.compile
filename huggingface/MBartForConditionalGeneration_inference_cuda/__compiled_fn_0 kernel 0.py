
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


# kernel path: /tmp/torchinductor_youkaichao/oq/coqxhyrocqkeix2ubuggkhs5jawanm5da72ufvmkf6wlzioif3rr.py
# Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, hidden_states_3, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# embed_pos => embedding_1
# hidden_states => add_1
# hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# hidden_states_3 => add_4, add_5, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp33_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
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
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp33_mean_next, tmp33_m2_next, tmp33_weight_next = triton_helpers.welford_reduce(
            tmp32, tmp33_mean, tmp33_m2, tmp33_weight,
        )
        tmp33_mean = tl.where(rmask & xmask, tmp33_mean_next, tmp33_mean)
        tmp33_m2 = tl.where(rmask & xmask, tmp33_m2_next, tmp33_m2)
        tmp33_weight = tl.where(rmask & xmask, tmp33_weight_next, tmp33_weight)
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tmp33_tmp, tmp34_tmp, tmp35_tmp = triton_helpers.welford(
        tmp33_mean, tmp33_m2, tmp33_weight, 1
    )
    tmp33 = tmp33_tmp[:, None]
    tmp34 = tmp34_tmp[:, None]
    tmp35 = tmp35_tmp[:, None]
    tmp38_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp38_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp38_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(out_ptr2 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp38_mean_next, tmp38_m2_next, tmp38_weight_next = triton_helpers.welford_reduce(
            tmp37, tmp38_mean, tmp38_m2, tmp38_weight,
        )
        tmp38_mean = tl.where(rmask & xmask, tmp38_mean_next, tmp38_mean)
        tmp38_m2 = tl.where(rmask & xmask, tmp38_m2_next, tmp38_m2)
        tmp38_weight = tl.where(rmask & xmask, tmp38_weight_next, tmp38_weight)
    tmp38_tmp, tmp39_tmp, tmp40_tmp = triton_helpers.welford(
        tmp38_mean, tmp38_m2, tmp38_weight, 1
    )
    tmp38 = tmp38_tmp[:, None]
    tmp39 = tmp39_tmp[:, None]
    tmp40 = tmp40_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tmp41 - tmp33
        tmp43 = 1024.0
        tmp44 = tmp39 / tmp43
        tmp45 = 1e-05
        tmp46 = tmp44 + tmp45
        tmp47 = tl.math.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp48 * tmp49
        tmp52 = tmp50 + tmp51
        tl.store(out_ptr5 + (r1 + (1024*x0)), tmp52, rmask & xmask)
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eu6wdnvxiydpqqrsbtdgvoduex3f4yce4uvfbhpvjfbrsgw625.py
# Source Nodes: [hidden_states_7, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_7 => add_7, add_8, mul_6, mul_7, rsqrt_2, sub_4, var_mean_2
# residual_1 => add_6
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
# Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
# hidden_states_8 => add_9, erf, mul_10, mul_8, mul_9
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqok42btqvxrg67mknadfvju3ydd2rgoky7c53szn3offkkmqfg.py
# Source Nodes: [hidden_states_14, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_14 => add_11, add_12, mul_11, mul_12, rsqrt_3, sub_5, var_mean_3
# residual_1 => add_6
# residual_2 => add_10
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vse3yju3zpjmslw2w2emahwj4vugnb5u6yszkkw5q3xwjjonbt.py
# Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
# eq => eq
# masked_fill_ => full_default, where
# ne => ne
# sum_1 => sum_1
triton_per_fused_eq_masked_fill_ne_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_masked_fill_ne_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/an/can7pkkw5s2oy3vakuksxdmdy2wqluisktyecuue4prxxv5tq3rb.py
# Source Nodes: [add_27, clone_1, eq, hidden_states_136, hidden_states_137, hidden_states_139, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.select_scatter, aten.slice_scatter, aten.view]
# add_27 => add_91
# clone_1 => clone_1
# eq => eq
# hidden_states_136 => add_92
# hidden_states_137 => add_93, add_94, mul_102, mul_103, rsqrt_26, sub_39, var_mean_26
# hidden_states_139 => add_95, add_96, mul_104, mul_105, rsqrt_27, sub_40, var_mean_27
# inputs_embeds_1 => mul_101
# l__mod___model_decoder_embed_tokens => embedding_2, view_243
# masked_fill_ => full_default, where
# positions_2 => embedding_3
# setitem => copy, slice_scatter
# setitem_1 => copy_1, select_scatter
triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr6, xnumel, rnumel):
    xnumel = 1024
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
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (2048 + r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp56 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp25 = tmp24 + 50265
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert((0 <= tmp27) & (tmp27 < 50265), "index out of bounds: 0 <= tmp27 < 50265")
    tmp28 = tl.load(in_ptr2 + (r1 + (1024*tmp27)), rmask, other=0.0)
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = tl.full([1], 1024, tl.int32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 / tmp41
    tmp43 = tmp33 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp49 = tmp32 - tmp42
    tmp50 = 1024.0
    tmp51 = tmp48 / tmp50
    tmp52 = 1e-05
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask & xmask, tmp60, 0)
    tmp63 = tl.broadcast_to(tmp60, [RBLOCK])
    tmp65 = tl.where(rmask & xmask, tmp63, 0)
    tmp66 = triton_helpers.promote_to_tensor(tl.sum(tmp65, 0))
    tmp67 = tmp66 / tmp41
    tmp68 = tmp60 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = tl.broadcast_to(tmp69, [RBLOCK])
    tmp72 = tl.where(rmask & xmask, tmp70, 0)
    tmp73 = triton_helpers.promote_to_tensor(tl.sum(tmp72, 0))
    tmp74 = tmp59 - tmp67
    tmp75 = tmp73 / tmp50
    tmp76 = tmp75 + tmp52
    tmp77 = tl.math.rsqrt(tmp76)
    tmp78 = tmp74 * tmp77
    tmp80 = tmp78 * tmp79
    tmp82 = tmp80 + tmp81
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp59, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp82, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2cyn45xrltqjphyr754dupw3cn5erm5r6a5bn55u66pvzjldlj.py
# Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
# attn_weights_27 => amax_12, div_12, exp_12, sub_41, sum_14
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3hqkcgncyjoxw2f4w4zazeak7xin5mgnebjqfkcbvxrygu7fnp.py
# Source Nodes: [attn_output_63], Original ATen: [aten.clone]
# attn_output_63 => clone_104
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ws/cws3rhllwjoyudxvxjchxitaehapzkfdsudfs4euqfswd776oyq2.py
# Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
# lm_logits => add_229
# masked_lm_loss => amax_36, exp_36, sub_101, sum_38
triton_red_fused__log_softmax_add_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_11', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dmce6ssw6u4cxwz5ssdc7bwijpmuvnh36a2jpgxmi7owqdlus5.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_36, full_default_4, ne_2, ne_3, neg, sum_39, sum_40, where_3
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_12', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg34_1, (4096, ), (1, ))
    assert_size_stride(arg35_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg66_1, (4096, ), (1, ))
    assert_size_stride(arg67_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg82_1, (4096, ), (1, ))
    assert_size_stride(arg83_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg98_1, (4096, ), (1, ))
    assert_size_stride(arg99_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg114_1, (4096, ), (1, ))
    assert_size_stride(arg115_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg130_1, (4096, ), (1, ))
    assert_size_stride(arg131_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg146_1, (4096, ), (1, ))
    assert_size_stride(arg147_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg162_1, (4096, ), (1, ))
    assert_size_stride(arg163_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg178_1, (4096, ), (1, ))
    assert_size_stride(arg179_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg194_1, (4096, ), (1, ))
    assert_size_stride(arg195_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg225_1, (4096, ), (1, ))
    assert_size_stride(arg226_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg247_1, (1024, ), (1, ))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg251_1, (4096, ), (1, ))
    assert_size_stride(arg252_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg277_1, (4096, ), (1, ))
    assert_size_stride(arg278_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg299_1, (1024, ), (1, ))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg303_1, (4096, ), (1, ))
    assert_size_stride(arg304_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg329_1, (4096, ), (1, ))
    assert_size_stride(arg330_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg355_1, (4096, ), (1, ))
    assert_size_stride(arg356_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg381_1, (4096, ), (1, ))
    assert_size_stride(arg382_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg389_1, (1024, ), (1, ))
    assert_size_stride(arg390_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg397_1, (1024, ), (1, ))
    assert_size_stride(arg398_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg399_1, (1024, ), (1, ))
    assert_size_stride(arg400_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg403_1, (1024, ), (1, ))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg407_1, (4096, ), (1, ))
    assert_size_stride(arg408_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg409_1, (1024, ), (1, ))
    assert_size_stride(arg410_1, (1024, ), (1, ))
    assert_size_stride(arg411_1, (1024, ), (1, ))
    assert_size_stride(arg412_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg415_1, (1024, ), (1, ))
    assert_size_stride(arg416_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg427_1, (1024, ), (1, ))
    assert_size_stride(arg428_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg429_1, (1024, ), (1, ))
    assert_size_stride(arg430_1, (1024, ), (1, ))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg433_1, (4096, ), (1, ))
    assert_size_stride(arg434_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg435_1, (1024, ), (1, ))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg439_1, (1024, ), (1, ))
    assert_size_stride(arg440_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg453_1, (1024, ), (1, ))
    assert_size_stride(arg454_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg455_1, (1024, ), (1, ))
    assert_size_stride(arg456_1, (1024, ), (1, ))
    assert_size_stride(arg457_1, (1024, ), (1, ))
    assert_size_stride(arg458_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg459_1, (4096, ), (1, ))
    assert_size_stride(arg460_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (1024, ), (1, ))
    assert_size_stride(arg464_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg471_1, (1024, ), (1, ))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg475_1, (1024, ), (1, ))
    assert_size_stride(arg476_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg481_1, (1024, ), (1, ))
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg485_1, (4096, ), (1, ))
    assert_size_stride(arg486_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (1024, ), (1, ))
    assert_size_stride(arg500_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg501_1, (1024, ), (1, ))
    assert_size_stride(arg502_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg507_1, (1024, ), (1, ))
    assert_size_stride(arg508_1, (1024, ), (1, ))
    assert_size_stride(arg509_1, (1024, ), (1, ))
    assert_size_stride(arg510_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg511_1, (4096, ), (1, ))
    assert_size_stride(arg512_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (1024, ), (1, ))
    assert_size_stride(arg515_1, (1024, ), (1, ))
    assert_size_stride(arg516_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg517_1, (1, 50265), (50265, 1))
    assert_size_stride(arg518_1, (1, 1024), (1024, 1))
    assert_size_stride(arg519_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, hidden_states_3, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg519_1, arg2_1, arg0_1, arg3_1, arg4_1, arg5_1, arg6_1, buf3, buf7, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg519_1
        del arg5_1
        del arg6_1
        buf8 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg7_1
        buf9 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf9)
        del arg9_1
        buf10 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), out=buf10)
        del arg11_1
        buf11 = reinterpret_tensor(buf7, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf8, arg8_1, buf11, 1048576, grid=grid(1048576), stream=stream0)
        del arg8_1
        buf12 = reinterpret_tensor(buf8, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf9, arg10_1, buf12, 1048576, grid=grid(1048576), stream=stream0)
        del arg10_1
        buf13 = reinterpret_tensor(buf9, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf10, arg12_1, buf13, 1048576, grid=grid(1048576), stream=stream0)
        del arg12_1
        del buf10
        # Source Nodes: [], Original ATen: []
        buf14 = aten._scaled_dot_product_efficient_attention(buf11, buf12, buf13, None, True, scale=1.0)
        buf15 = buf14[0]
        del buf14
        buf19 = reinterpret_tensor(buf15, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf19, 1048576, grid=grid(1048576), stream=stream0)
        buf20 = reinterpret_tensor(buf13, (1024, 1024), (1024, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 1024), (1, 1024), 0), out=buf20)
        del arg13_1
        buf24 = reinterpret_tensor(buf19, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf19  # reuse
        # Source Nodes: [hidden_states_7, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf20, arg14_1, arg15_1, arg16_1, buf24, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg15_1
        del arg16_1
        buf25 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg17_1, (1024, 4096), (1, 1024), 0), out=buf25)
        del arg17_1
        buf26 = reinterpret_tensor(buf25, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf25  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf26, arg18_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg18_1
        buf27 = reinterpret_tensor(buf24, (1024, 1024), (1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg19_1, (4096, 1024), (1, 4096), 0), out=buf27)
        del arg19_1
        buf28 = reinterpret_tensor(buf27, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf27  # reuse
        buf32 = reinterpret_tensor(buf12, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf12  # reuse
        # Source Nodes: [hidden_states_14, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf28, buf3, buf20, arg14_1, arg20_1, arg21_1, arg22_1, buf32, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg14_1
        del arg20_1
        del arg21_1
        del arg22_1
        buf33 = reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf33)
        del arg23_1
        buf34 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg25_1
        buf35 = reinterpret_tensor(buf11, (1024, 1024), (1024, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), out=buf35)
        del arg27_1
        buf36 = reinterpret_tensor(buf32, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf33, arg24_1, buf36, 1048576, grid=grid(1048576), stream=stream0)
        del arg24_1
        buf37 = reinterpret_tensor(buf33, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf34, arg26_1, buf37, 1048576, grid=grid(1048576), stream=stream0)
        del arg26_1
        buf38 = reinterpret_tensor(buf34, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf35, arg28_1, buf38, 1048576, grid=grid(1048576), stream=stream0)
        del arg28_1
        del buf35
        # Source Nodes: [], Original ATen: []
        buf39 = aten._scaled_dot_product_efficient_attention(buf36, buf37, buf38, None, True, scale=1.0)
        buf40 = buf39[0]
        del buf39
        buf44 = reinterpret_tensor(buf40, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf44, 1048576, grid=grid(1048576), stream=stream0)
        buf45 = reinterpret_tensor(buf38, (1024, 1024), (1024, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 1024), (1, 1024), 0), out=buf45)
        del arg29_1
        buf49 = reinterpret_tensor(buf44, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf44  # reuse
        # Source Nodes: [hidden_states_18, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf28, buf45, arg30_1, arg31_1, arg32_1, buf49, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg31_1
        del arg32_1
        buf50 = reinterpret_tensor(buf26, (1024, 4096), (4096, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg33_1, (1024, 4096), (1, 1024), 0), out=buf50)
        del arg33_1
        buf51 = reinterpret_tensor(buf50, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf50  # reuse
        # Source Nodes: [hidden_states_19], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf51, arg34_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg34_1
        buf52 = reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg35_1, (4096, 1024), (1, 4096), 0), out=buf52)
        del arg35_1
        buf53 = reinterpret_tensor(buf52, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf52  # reuse
        buf57 = reinterpret_tensor(buf37, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf37  # reuse
        # Source Nodes: [hidden_states_25, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf53, buf28, buf45, arg30_1, arg36_1, arg37_1, arg38_1, buf57, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg30_1
        del arg36_1
        del arg37_1
        del arg38_1
        buf58 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf58)
        del arg39_1
        buf59 = reinterpret_tensor(buf28, (1024, 1024), (1024, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf59)
        del arg41_1
        buf60 = reinterpret_tensor(buf36, (1024, 1024), (1024, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), out=buf60)
        del arg43_1
        buf61 = reinterpret_tensor(buf57, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf58, arg40_1, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del arg40_1
        buf62 = reinterpret_tensor(buf58, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf59, arg42_1, buf62, 1048576, grid=grid(1048576), stream=stream0)
        del arg42_1
        buf63 = reinterpret_tensor(buf59, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf60, arg44_1, buf63, 1048576, grid=grid(1048576), stream=stream0)
        del arg44_1
        del buf60
        # Source Nodes: [], Original ATen: []
        buf64 = aten._scaled_dot_product_efficient_attention(buf61, buf62, buf63, None, True, scale=1.0)
        buf65 = buf64[0]
        del buf64
        buf69 = reinterpret_tensor(buf65, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf69, 1048576, grid=grid(1048576), stream=stream0)
        buf70 = reinterpret_tensor(buf63, (1024, 1024), (1024, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 1024), (1, 1024), 0), out=buf70)
        del arg45_1
        buf74 = reinterpret_tensor(buf69, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf53, buf70, arg46_1, arg47_1, arg48_1, buf74, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg47_1
        del arg48_1
        buf75 = reinterpret_tensor(buf51, (1024, 4096), (4096, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 4096), (1, 1024), 0), out=buf75)
        del arg49_1
        buf76 = reinterpret_tensor(buf75, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf75  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf76, arg50_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg50_1
        buf77 = reinterpret_tensor(buf74, (1024, 1024), (1024, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg51_1, (4096, 1024), (1, 4096), 0), out=buf77)
        del arg51_1
        buf78 = reinterpret_tensor(buf77, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf77  # reuse
        buf82 = reinterpret_tensor(buf62, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf62  # reuse
        # Source Nodes: [hidden_states_36, residual_5, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf78, buf53, buf70, arg46_1, arg52_1, arg53_1, arg54_1, buf82, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg46_1
        del arg52_1
        del arg53_1
        del arg54_1
        buf83 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf83)
        del arg55_1
        buf84 = reinterpret_tensor(buf53, (1024, 1024), (1024, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf84)
        del arg57_1
        buf85 = reinterpret_tensor(buf61, (1024, 1024), (1024, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), out=buf85)
        del arg59_1
        buf86 = reinterpret_tensor(buf82, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf83, arg56_1, buf86, 1048576, grid=grid(1048576), stream=stream0)
        del arg56_1
        buf87 = reinterpret_tensor(buf83, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf84, arg58_1, buf87, 1048576, grid=grid(1048576), stream=stream0)
        del arg58_1
        buf88 = reinterpret_tensor(buf84, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf85, arg60_1, buf88, 1048576, grid=grid(1048576), stream=stream0)
        del arg60_1
        del buf85
        # Source Nodes: [], Original ATen: []
        buf89 = aten._scaled_dot_product_efficient_attention(buf86, buf87, buf88, None, True, scale=1.0)
        buf90 = buf89[0]
        del buf89
        buf94 = reinterpret_tensor(buf90, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf94, 1048576, grid=grid(1048576), stream=stream0)
        buf95 = reinterpret_tensor(buf88, (1024, 1024), (1024, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 1024), (1, 1024), 0), out=buf95)
        del arg61_1
        buf99 = reinterpret_tensor(buf94, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
        # Source Nodes: [hidden_states_40, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf78, buf95, arg62_1, arg63_1, arg64_1, buf99, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg63_1
        del arg64_1
        buf100 = reinterpret_tensor(buf76, (1024, 4096), (4096, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 4096), (1, 1024), 0), out=buf100)
        del arg65_1
        buf101 = reinterpret_tensor(buf100, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf100  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf101, arg66_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg66_1
        buf102 = reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 1024), (1, 4096), 0), out=buf102)
        del arg67_1
        buf103 = reinterpret_tensor(buf102, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf102  # reuse
        buf107 = reinterpret_tensor(buf87, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_47, residual_7, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf103, buf78, buf95, arg62_1, arg68_1, arg69_1, arg70_1, buf107, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg62_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf108 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf108)
        del arg71_1
        buf109 = reinterpret_tensor(buf78, (1024, 1024), (1024, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf109)
        del arg73_1
        buf110 = reinterpret_tensor(buf86, (1024, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), out=buf110)
        del arg75_1
        buf111 = reinterpret_tensor(buf107, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf108, arg72_1, buf111, 1048576, grid=grid(1048576), stream=stream0)
        del arg72_1
        buf112 = reinterpret_tensor(buf108, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf109, arg74_1, buf112, 1048576, grid=grid(1048576), stream=stream0)
        del arg74_1
        buf113 = reinterpret_tensor(buf109, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf110, arg76_1, buf113, 1048576, grid=grid(1048576), stream=stream0)
        del arg76_1
        del buf110
        # Source Nodes: [], Original ATen: []
        buf114 = aten._scaled_dot_product_efficient_attention(buf111, buf112, buf113, None, True, scale=1.0)
        buf115 = buf114[0]
        del buf114
        buf119 = reinterpret_tensor(buf115, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf119, 1048576, grid=grid(1048576), stream=stream0)
        buf120 = reinterpret_tensor(buf113, (1024, 1024), (1024, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 1024), (1, 1024), 0), out=buf120)
        del arg77_1
        buf124 = reinterpret_tensor(buf119, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_51, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf103, buf120, arg78_1, arg79_1, arg80_1, buf124, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg79_1
        del arg80_1
        buf125 = reinterpret_tensor(buf101, (1024, 4096), (4096, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 4096), (1, 1024), 0), out=buf125)
        del arg81_1
        buf126 = reinterpret_tensor(buf125, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf125  # reuse
        # Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf126, arg82_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg82_1
        buf127 = reinterpret_tensor(buf124, (1024, 1024), (1024, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg83_1, (4096, 1024), (1, 4096), 0), out=buf127)
        del arg83_1
        buf128 = reinterpret_tensor(buf127, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf127  # reuse
        buf132 = reinterpret_tensor(buf112, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_58, residual_10, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf128, buf103, buf120, arg78_1, arg84_1, arg85_1, arg86_1, buf132, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg78_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf133 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf133)
        del arg87_1
        buf134 = reinterpret_tensor(buf103, (1024, 1024), (1024, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf134)
        del arg89_1
        buf135 = reinterpret_tensor(buf111, (1024, 1024), (1024, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), out=buf135)
        del arg91_1
        buf136 = reinterpret_tensor(buf132, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf133, arg88_1, buf136, 1048576, grid=grid(1048576), stream=stream0)
        del arg88_1
        buf137 = reinterpret_tensor(buf133, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf134, arg90_1, buf137, 1048576, grid=grid(1048576), stream=stream0)
        del arg90_1
        buf138 = reinterpret_tensor(buf134, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf135, arg92_1, buf138, 1048576, grid=grid(1048576), stream=stream0)
        del arg92_1
        del buf135
        # Source Nodes: [], Original ATen: []
        buf139 = aten._scaled_dot_product_efficient_attention(buf136, buf137, buf138, None, True, scale=1.0)
        buf140 = buf139[0]
        del buf139
        buf144 = reinterpret_tensor(buf140, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf140  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf144, 1048576, grid=grid(1048576), stream=stream0)
        buf145 = reinterpret_tensor(buf138, (1024, 1024), (1024, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 1024), (1, 1024), 0), out=buf145)
        del arg93_1
        buf149 = reinterpret_tensor(buf144, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf144  # reuse
        # Source Nodes: [hidden_states_62, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf128, buf145, arg94_1, arg95_1, arg96_1, buf149, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg95_1
        del arg96_1
        buf150 = reinterpret_tensor(buf126, (1024, 4096), (4096, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg97_1, (1024, 4096), (1, 1024), 0), out=buf150)
        del arg97_1
        buf151 = reinterpret_tensor(buf150, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf150  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf151, arg98_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg98_1
        buf152 = reinterpret_tensor(buf149, (1024, 1024), (1024, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 1024), (1, 4096), 0), out=buf152)
        del arg99_1
        buf153 = reinterpret_tensor(buf152, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf137, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf137  # reuse
        # Source Nodes: [hidden_states_69, residual_11, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf153, buf128, buf145, arg94_1, arg100_1, arg101_1, arg102_1, buf157, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg94_1
        buf158 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf158)
        del arg103_1
        buf159 = reinterpret_tensor(buf128, (1024, 1024), (1024, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf159)
        del arg105_1
        buf160 = reinterpret_tensor(buf136, (1024, 1024), (1024, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), out=buf160)
        del arg107_1
        buf161 = reinterpret_tensor(buf157, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf158, arg104_1, buf161, 1048576, grid=grid(1048576), stream=stream0)
        del arg104_1
        buf162 = reinterpret_tensor(buf158, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf159, arg106_1, buf162, 1048576, grid=grid(1048576), stream=stream0)
        del arg106_1
        buf163 = reinterpret_tensor(buf159, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf160, arg108_1, buf163, 1048576, grid=grid(1048576), stream=stream0)
        del arg108_1
        del buf160
        # Source Nodes: [], Original ATen: []
        buf164 = aten._scaled_dot_product_efficient_attention(buf161, buf162, buf163, None, True, scale=1.0)
        buf165 = buf164[0]
        del buf164
        buf169 = reinterpret_tensor(buf165, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf165  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf169, 1048576, grid=grid(1048576), stream=stream0)
        buf170 = reinterpret_tensor(buf163, (1024, 1024), (1024, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 1024), (1, 1024), 0), out=buf170)
        del arg109_1
        buf174 = reinterpret_tensor(buf169, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf169  # reuse
        # Source Nodes: [hidden_states_73, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf153, buf170, arg110_1, arg111_1, arg112_1, buf174, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg111_1
        del arg112_1
        buf175 = reinterpret_tensor(buf151, (1024, 4096), (4096, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg113_1, (1024, 4096), (1, 1024), 0), out=buf175)
        del arg113_1
        buf176 = reinterpret_tensor(buf175, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf175  # reuse
        # Source Nodes: [hidden_states_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf176, arg114_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg114_1
        buf177 = reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg115_1, (4096, 1024), (1, 4096), 0), out=buf177)
        del arg115_1
        buf178 = reinterpret_tensor(buf177, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
        buf182 = reinterpret_tensor(buf162, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf162  # reuse
        # Source Nodes: [hidden_states_80, residual_13, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf178, buf153, buf170, arg110_1, arg116_1, arg117_1, arg118_1, buf182, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg110_1
        del arg116_1
        del arg117_1
        del arg118_1
        buf183 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf183)
        del arg119_1
        buf184 = reinterpret_tensor(buf153, (1024, 1024), (1024, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg121_1
        buf185 = reinterpret_tensor(buf161, (1024, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), out=buf185)
        del arg123_1
        buf186 = reinterpret_tensor(buf182, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf183, arg120_1, buf186, 1048576, grid=grid(1048576), stream=stream0)
        del arg120_1
        buf187 = reinterpret_tensor(buf183, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf184, arg122_1, buf187, 1048576, grid=grid(1048576), stream=stream0)
        del arg122_1
        buf188 = reinterpret_tensor(buf184, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf185, arg124_1, buf188, 1048576, grid=grid(1048576), stream=stream0)
        del arg124_1
        del buf185
        # Source Nodes: [], Original ATen: []
        buf189 = aten._scaled_dot_product_efficient_attention(buf186, buf187, buf188, None, True, scale=1.0)
        buf190 = buf189[0]
        del buf189
        buf194 = reinterpret_tensor(buf190, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf194, 1048576, grid=grid(1048576), stream=stream0)
        buf195 = reinterpret_tensor(buf188, (1024, 1024), (1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 1024), (1, 1024), 0), out=buf195)
        del arg125_1
        buf199 = reinterpret_tensor(buf194, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf194  # reuse
        # Source Nodes: [hidden_states_84, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf178, buf195, arg126_1, arg127_1, arg128_1, buf199, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg127_1
        del arg128_1
        buf200 = reinterpret_tensor(buf176, (1024, 4096), (4096, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg129_1, (1024, 4096), (1, 1024), 0), out=buf200)
        del arg129_1
        buf201 = reinterpret_tensor(buf200, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf200  # reuse
        # Source Nodes: [hidden_states_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf201, arg130_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg130_1
        buf202 = reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 1024), (1, 4096), 0), out=buf202)
        del arg131_1
        buf203 = reinterpret_tensor(buf202, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf202  # reuse
        buf207 = reinterpret_tensor(buf187, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf187  # reuse
        # Source Nodes: [hidden_states_91, residual_15, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf203, buf178, buf195, arg126_1, arg132_1, arg133_1, arg134_1, buf207, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg126_1
        del arg132_1
        del arg133_1
        del arg134_1
        buf208 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg135_1
        buf209 = reinterpret_tensor(buf178, (1024, 1024), (1024, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf209)
        del arg137_1
        buf210 = reinterpret_tensor(buf186, (1024, 1024), (1024, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), out=buf210)
        del arg139_1
        buf211 = reinterpret_tensor(buf207, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf208, arg136_1, buf211, 1048576, grid=grid(1048576), stream=stream0)
        del arg136_1
        buf212 = reinterpret_tensor(buf208, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf209, arg138_1, buf212, 1048576, grid=grid(1048576), stream=stream0)
        del arg138_1
        buf213 = reinterpret_tensor(buf209, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf210, arg140_1, buf213, 1048576, grid=grid(1048576), stream=stream0)
        del arg140_1
        del buf210
        # Source Nodes: [], Original ATen: []
        buf214 = aten._scaled_dot_product_efficient_attention(buf211, buf212, buf213, None, True, scale=1.0)
        buf215 = buf214[0]
        del buf214
        buf219 = reinterpret_tensor(buf215, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf219, 1048576, grid=grid(1048576), stream=stream0)
        buf220 = reinterpret_tensor(buf213, (1024, 1024), (1024, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 1024), (1, 1024), 0), out=buf220)
        del arg141_1
        buf224 = reinterpret_tensor(buf219, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf219  # reuse
        # Source Nodes: [hidden_states_95, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf203, buf220, arg142_1, arg143_1, arg144_1, buf224, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg143_1
        del arg144_1
        buf225 = reinterpret_tensor(buf201, (1024, 4096), (4096, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 4096), (1, 1024), 0), out=buf225)
        del arg145_1
        buf226 = reinterpret_tensor(buf225, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf225  # reuse
        # Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf226, arg146_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg146_1
        buf227 = reinterpret_tensor(buf224, (1024, 1024), (1024, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 1024), (1, 4096), 0), out=buf227)
        del arg147_1
        buf228 = reinterpret_tensor(buf227, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf227  # reuse
        buf232 = reinterpret_tensor(buf212, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf212  # reuse
        # Source Nodes: [hidden_states_102, residual_17, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf228, buf203, buf220, arg142_1, arg148_1, arg149_1, arg150_1, buf232, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg142_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf233 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf233)
        del arg151_1
        buf234 = reinterpret_tensor(buf203, (1024, 1024), (1024, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf234)
        del arg153_1
        buf235 = reinterpret_tensor(buf211, (1024, 1024), (1024, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), out=buf235)
        del arg155_1
        buf236 = reinterpret_tensor(buf232, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf233, arg152_1, buf236, 1048576, grid=grid(1048576), stream=stream0)
        del arg152_1
        buf237 = reinterpret_tensor(buf233, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf234, arg154_1, buf237, 1048576, grid=grid(1048576), stream=stream0)
        del arg154_1
        buf238 = reinterpret_tensor(buf234, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf235, arg156_1, buf238, 1048576, grid=grid(1048576), stream=stream0)
        del arg156_1
        del buf235
        # Source Nodes: [], Original ATen: []
        buf239 = aten._scaled_dot_product_efficient_attention(buf236, buf237, buf238, None, True, scale=1.0)
        buf240 = buf239[0]
        del buf239
        buf244 = reinterpret_tensor(buf240, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf240  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf244, 1048576, grid=grid(1048576), stream=stream0)
        buf245 = reinterpret_tensor(buf238, (1024, 1024), (1024, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg157_1
        buf249 = reinterpret_tensor(buf244, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf244  # reuse
        # Source Nodes: [hidden_states_106, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf228, buf245, arg158_1, arg159_1, arg160_1, buf249, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg159_1
        del arg160_1
        buf250 = reinterpret_tensor(buf226, (1024, 4096), (4096, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 4096), (1, 1024), 0), out=buf250)
        del arg161_1
        buf251 = reinterpret_tensor(buf250, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf250  # reuse
        # Source Nodes: [hidden_states_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf251, arg162_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg162_1
        buf252 = reinterpret_tensor(buf249, (1024, 1024), (1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg163_1, (4096, 1024), (1, 4096), 0), out=buf252)
        del arg163_1
        buf253 = reinterpret_tensor(buf252, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf252  # reuse
        buf257 = reinterpret_tensor(buf237, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf237  # reuse
        # Source Nodes: [hidden_states_113, residual_19, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf253, buf228, buf245, arg158_1, arg164_1, arg165_1, arg166_1, buf257, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg158_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf258 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf258)
        del arg167_1
        buf259 = reinterpret_tensor(buf228, (1024, 1024), (1024, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg169_1
        buf260 = reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), out=buf260)
        del arg171_1
        buf261 = reinterpret_tensor(buf257, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf258, arg168_1, buf261, 1048576, grid=grid(1048576), stream=stream0)
        del arg168_1
        buf262 = reinterpret_tensor(buf258, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf259, arg170_1, buf262, 1048576, grid=grid(1048576), stream=stream0)
        del arg170_1
        buf263 = reinterpret_tensor(buf259, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf260, arg172_1, buf263, 1048576, grid=grid(1048576), stream=stream0)
        del arg172_1
        # Source Nodes: [], Original ATen: []
        buf264 = aten._scaled_dot_product_efficient_attention(buf261, buf262, buf263, None, True, scale=1.0)
        buf265 = buf264[0]
        del buf264
        buf269 = reinterpret_tensor(buf265, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf265  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf269, 1048576, grid=grid(1048576), stream=stream0)
        buf270 = reinterpret_tensor(buf263, (1024, 1024), (1024, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg173_1
        buf274 = reinterpret_tensor(buf269, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf269  # reuse
        # Source Nodes: [hidden_states_117, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf253, buf270, arg174_1, arg175_1, arg176_1, buf274, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg175_1
        del arg176_1
        buf275 = reinterpret_tensor(buf251, (1024, 4096), (4096, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 4096), (1, 1024), 0), out=buf275)
        del arg177_1
        buf276 = reinterpret_tensor(buf275, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_118], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf276, arg178_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg178_1
        buf277 = reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 1024), (1, 4096), 0), out=buf277)
        del arg179_1
        buf278 = reinterpret_tensor(buf277, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf277  # reuse
        buf282 = reinterpret_tensor(buf262, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf262  # reuse
        # Source Nodes: [hidden_states_124, residual_21, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf278, buf253, buf270, arg174_1, arg180_1, arg181_1, arg182_1, buf282, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg174_1
        del arg180_1
        del arg181_1
        del arg182_1
        buf283 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf283)
        del arg183_1
        buf284 = reinterpret_tensor(buf253, (1024, 1024), (1024, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf284)
        del arg185_1
        buf285 = reinterpret_tensor(buf261, (1024, 1024), (1024, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), out=buf285)
        del arg187_1
        buf286 = reinterpret_tensor(buf282, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf283, arg184_1, buf286, 1048576, grid=grid(1048576), stream=stream0)
        del arg184_1
        buf287 = reinterpret_tensor(buf283, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf284, arg186_1, buf287, 1048576, grid=grid(1048576), stream=stream0)
        del arg186_1
        buf288 = reinterpret_tensor(buf284, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf285, arg188_1, buf288, 1048576, grid=grid(1048576), stream=stream0)
        del arg188_1
        # Source Nodes: [], Original ATen: []
        buf289 = aten._scaled_dot_product_efficient_attention(buf286, buf287, buf288, None, True, scale=1.0)
        buf290 = buf289[0]
        del buf289
        buf294 = reinterpret_tensor(buf290, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf294, 1048576, grid=grid(1048576), stream=stream0)
        buf295 = reinterpret_tensor(buf288, (1024, 1024), (1024, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 1024), (1, 1024), 0), out=buf295)
        del arg189_1
        buf299 = reinterpret_tensor(buf294, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf294  # reuse
        # Source Nodes: [hidden_states_128, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf278, buf295, arg190_1, arg191_1, arg192_1, buf299, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg191_1
        del arg192_1
        buf300 = reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 4096), (1, 1024), 0), out=buf300)
        del arg193_1
        buf301 = reinterpret_tensor(buf300, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf300  # reuse
        # Source Nodes: [hidden_states_129], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf301, arg194_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg194_1
        buf302 = reinterpret_tensor(buf299, (1024, 1024), (1024, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf301, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 1024), (1, 4096), 0), out=buf302)
        del arg195_1
        buf303 = reinterpret_tensor(buf302, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf302  # reuse
        buf335 = reinterpret_tensor(buf287, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf287  # reuse
        # Source Nodes: [hidden_states_134, hidden_states_135, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf303, buf278, buf295, arg190_1, arg196_1, arg197_1, arg198_1, buf335, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg190_1
        del arg196_1
        del arg197_1
        del arg198_1
        buf307 = empty((1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
        triton_per_fused_eq_masked_fill_ne_sum_7.run(arg518_1, buf307, 1, 1024, grid=grid(1), stream=stream0)
        buf308 = buf303; del buf303  # reuse
        buf312 = reinterpret_tensor(buf295, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf295  # reuse
        buf316 = buf278; del buf278  # reuse
        # Source Nodes: [add_27, clone_1, eq, hidden_states_136, hidden_states_137, hidden_states_139, inputs_embeds_1, l__mod___model_decoder_embed_tokens, masked_fill_, positions_2, setitem, setitem_1], Original ATen: [aten.add, aten.clone, aten.copy, aten.embedding, aten.eq, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.select_scatter, aten.slice_scatter, aten.view]
        triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_view_8.run(buf307, arg518_1, arg199_1, arg1_1, arg200_1, arg201_1, arg202_1, arg203_1, buf308, buf312, buf316, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg199_1
        del arg1_1
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del buf307
        buf317 = reinterpret_tensor(buf308, (1024, 1024), (1024, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg204_1, (1024, 1024), (1, 1024), 0), out=buf317)
        del arg204_1
        buf318 = reinterpret_tensor(buf286, (1024, 1024), (1024, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg206_1, (1024, 1024), (1, 1024), 0), out=buf318)
        del arg206_1
        buf319 = reinterpret_tensor(buf285, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf285  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf317, arg205_1, buf319, 1048576, grid=grid(1048576), stream=stream0)
        del arg205_1
        buf320 = reinterpret_tensor(buf317, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf318, arg207_1, buf320, 1048576, grid=grid(1048576), stream=stream0)
        del arg207_1
        buf321 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf320, (16, 64, 1024), (65536, 1, 64), 0), out=buf321)
        buf325 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf321, buf325, 16384, 1024, grid=grid(16384), stream=stream0)
        buf324 = reinterpret_tensor(buf320, (1024, 1024), (1024, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg208_1, (1024, 1024), (1, 1024), 0), out=buf324)
        del arg208_1
        buf326 = reinterpret_tensor(buf316, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf324, arg209_1, buf326, 1048576, grid=grid(1048576), stream=stream0)
        del arg209_1
        buf327 = reinterpret_tensor(buf324, (16, 1024, 64), (65536, 64, 1), 0); del buf324  # reuse
        # Source Nodes: [attn_output_60, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf325, reinterpret_tensor(buf326, (16, 1024, 64), (65536, 64, 1), 0), out=buf327)
        buf328 = reinterpret_tensor(buf326, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf326  # reuse
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf327, buf328, 1048576, grid=grid(1048576), stream=stream0)
        buf329 = reinterpret_tensor(buf327, (1024, 1024), (1024, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg210_1, (1024, 1024), (1, 1024), 0), out=buf329)
        del arg210_1
        buf333 = reinterpret_tensor(buf328, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf328  # reuse
        # Source Nodes: [hidden_states_143, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf312, buf329, arg211_1, arg212_1, arg213_1, buf333, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg212_1
        del arg213_1
        buf334 = reinterpret_tensor(buf319, (1024, 1024), (1024, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg214_1, (1024, 1024), (1, 1024), 0), out=buf334)
        del arg214_1
        buf336 = reinterpret_tensor(buf333, (1024, 1024), (1024, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg216_1, (1024, 1024), (1, 1024), 0), out=buf336)
        del arg216_1
        buf337 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg218_1, (1024, 1024), (1, 1024), 0), out=buf337)
        del arg218_1
        buf338 = reinterpret_tensor(buf260, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf334, arg215_1, buf338, 1048576, grid=grid(1048576), stream=stream0)
        del arg215_1
        buf339 = reinterpret_tensor(buf334, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf336, arg217_1, buf339, 1048576, grid=grid(1048576), stream=stream0)
        del arg217_1
        buf340 = reinterpret_tensor(buf336, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf337, arg219_1, buf340, 1048576, grid=grid(1048576), stream=stream0)
        del arg219_1
        del buf337
        # Source Nodes: [], Original ATen: []
        buf341 = aten._scaled_dot_product_efficient_attention(buf338, buf339, buf340, None, True, scale=1.0)
        buf342 = buf341[0]
        del buf341
        buf346 = reinterpret_tensor(buf342, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf346, 1048576, grid=grid(1048576), stream=stream0)
        buf347 = reinterpret_tensor(buf340, (1024, 1024), (1024, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg220_1, (1024, 1024), (1, 1024), 0), out=buf347)
        del arg220_1
        buf348 = reinterpret_tensor(buf347, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf347  # reuse
        buf352 = reinterpret_tensor(buf346, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf346  # reuse
        # Source Nodes: [hidden_states_147, residual_25, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf348, buf312, buf329, arg211_1, arg221_1, arg222_1, arg223_1, buf352, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg211_1
        del arg221_1
        del arg222_1
        del arg223_1
        buf353 = reinterpret_tensor(buf301, (1024, 4096), (4096, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg224_1, (1024, 4096), (1, 1024), 0), out=buf353)
        del arg224_1
        buf354 = reinterpret_tensor(buf353, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf353  # reuse
        # Source Nodes: [hidden_states_148], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf354, arg225_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg225_1
        buf355 = reinterpret_tensor(buf352, (1024, 1024), (1024, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg226_1, (4096, 1024), (1, 4096), 0), out=buf355)
        del arg226_1
        buf359 = reinterpret_tensor(buf329, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf329  # reuse
        # Source Nodes: [hidden_states_154, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf348, buf355, arg227_1, arg228_1, arg229_1, buf359, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg228_1
        del arg229_1
        buf360 = reinterpret_tensor(buf312, (1024, 1024), (1024, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg230_1, (1024, 1024), (1, 1024), 0), out=buf360)
        del arg230_1
        buf361 = reinterpret_tensor(buf339, (1024, 1024), (1024, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg232_1, (1024, 1024), (1, 1024), 0), out=buf361)
        del arg232_1
        buf362 = buf338; del buf338  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf360, arg231_1, buf362, 1048576, grid=grid(1048576), stream=stream0)
        del arg231_1
        buf363 = reinterpret_tensor(buf360, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf360  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf361, arg233_1, buf363, 1048576, grid=grid(1048576), stream=stream0)
        del arg233_1
        del buf361
        buf364 = buf325; del buf325  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf363, (16, 64, 1024), (65536, 1, 64), 0), out=buf364)
        buf368 = buf321; del buf321  # reuse
        # Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf364, buf368, 16384, 1024, grid=grid(16384), stream=stream0)
        buf367 = reinterpret_tensor(buf363, (1024, 1024), (1024, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg234_1, (1024, 1024), (1, 1024), 0), out=buf367)
        del arg234_1
        buf369 = reinterpret_tensor(buf359, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf359  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf367, arg235_1, buf369, 1048576, grid=grid(1048576), stream=stream0)
        del arg235_1
        buf370 = reinterpret_tensor(buf367, (16, 1024, 64), (65536, 64, 1), 0); del buf367  # reuse
        # Source Nodes: [attn_output_70, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf368, reinterpret_tensor(buf369, (16, 1024, 64), (65536, 64, 1), 0), out=buf370)
        buf371 = reinterpret_tensor(buf369, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf369  # reuse
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf370, buf371, 1048576, grid=grid(1048576), stream=stream0)
        buf372 = reinterpret_tensor(buf370, (1024, 1024), (1024, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg236_1, (1024, 1024), (1, 1024), 0), out=buf372)
        del arg236_1
        buf373 = reinterpret_tensor(buf372, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf372  # reuse
        buf377 = reinterpret_tensor(buf371, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf371  # reuse
        # Source Nodes: [hidden_states_158, residual_27, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf373, buf348, buf355, arg227_1, arg237_1, arg238_1, arg239_1, buf377, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg227_1
        del arg237_1
        del arg238_1
        del arg239_1
        buf378 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg240_1, (1024, 1024), (1, 1024), 0), out=buf378)
        del arg240_1
        buf379 = reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg242_1, (1024, 1024), (1, 1024), 0), out=buf379)
        del arg242_1
        buf380 = reinterpret_tensor(buf348, (1024, 1024), (1024, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg244_1, (1024, 1024), (1, 1024), 0), out=buf380)
        del arg244_1
        buf381 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf378, arg241_1, buf381, 1048576, grid=grid(1048576), stream=stream0)
        del arg241_1
        buf382 = reinterpret_tensor(buf378, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf379, arg243_1, buf382, 1048576, grid=grid(1048576), stream=stream0)
        del arg243_1
        buf383 = reinterpret_tensor(buf379, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf380, arg245_1, buf383, 1048576, grid=grid(1048576), stream=stream0)
        del arg245_1
        # Source Nodes: [], Original ATen: []
        buf384 = aten._scaled_dot_product_efficient_attention(buf381, buf382, buf383, None, True, scale=1.0)
        buf385 = buf384[0]
        del buf384
        buf389 = reinterpret_tensor(buf385, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf385  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf389, 1048576, grid=grid(1048576), stream=stream0)
        buf390 = reinterpret_tensor(buf383, (1024, 1024), (1024, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg246_1, (1024, 1024), (1, 1024), 0), out=buf390)
        del arg246_1
        buf394 = reinterpret_tensor(buf389, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf389  # reuse
        # Source Nodes: [hidden_states_162, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf373, buf390, arg247_1, arg248_1, arg249_1, buf394, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg248_1
        del arg249_1
        buf395 = reinterpret_tensor(buf354, (1024, 4096), (4096, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg250_1, (1024, 4096), (1, 1024), 0), out=buf395)
        del arg250_1
        buf396 = reinterpret_tensor(buf395, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf395  # reuse
        # Source Nodes: [hidden_states_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf396, arg251_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg251_1
        buf397 = reinterpret_tensor(buf394, (1024, 1024), (1024, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg252_1, (4096, 1024), (1, 4096), 0), out=buf397)
        del arg252_1
        buf398 = reinterpret_tensor(buf397, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf397  # reuse
        buf402 = reinterpret_tensor(buf382, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf382  # reuse
        # Source Nodes: [hidden_states_169, residual_29, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf398, buf373, buf390, arg247_1, arg253_1, arg254_1, arg255_1, buf402, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg247_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf403 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg256_1, (1024, 1024), (1, 1024), 0), out=buf403)
        del arg256_1
        buf404 = reinterpret_tensor(buf373, (1024, 1024), (1024, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg258_1, (1024, 1024), (1, 1024), 0), out=buf404)
        del arg258_1
        buf405 = buf381; del buf381  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf403, arg257_1, buf405, 1048576, grid=grid(1048576), stream=stream0)
        del arg257_1
        buf406 = reinterpret_tensor(buf403, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf403  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf404, arg259_1, buf406, 1048576, grid=grid(1048576), stream=stream0)
        del arg259_1
        buf407 = buf368; del buf368  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf405, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf406, (16, 64, 1024), (65536, 1, 64), 0), out=buf407)
        buf411 = buf364; del buf364  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf407, buf411, 16384, 1024, grid=grid(16384), stream=stream0)
        buf410 = reinterpret_tensor(buf406, (1024, 1024), (1024, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg260_1, (1024, 1024), (1, 1024), 0), out=buf410)
        del arg260_1
        buf412 = reinterpret_tensor(buf402, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf402  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf410, arg261_1, buf412, 1048576, grid=grid(1048576), stream=stream0)
        del arg261_1
        buf413 = reinterpret_tensor(buf410, (16, 1024, 64), (65536, 64, 1), 0); del buf410  # reuse
        # Source Nodes: [attn_output_80, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf411, reinterpret_tensor(buf412, (16, 1024, 64), (65536, 64, 1), 0), out=buf413)
        buf414 = reinterpret_tensor(buf412, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf412  # reuse
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf413, buf414, 1048576, grid=grid(1048576), stream=stream0)
        buf415 = reinterpret_tensor(buf413, (1024, 1024), (1024, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf414, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg262_1, (1024, 1024), (1, 1024), 0), out=buf415)
        del arg262_1
        buf419 = reinterpret_tensor(buf414, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf414  # reuse
        # Source Nodes: [hidden_states_173, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf398, buf415, arg263_1, arg264_1, arg265_1, buf419, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg264_1
        del arg265_1
        buf420 = reinterpret_tensor(buf405, (1024, 1024), (1024, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg266_1, (1024, 1024), (1, 1024), 0), out=buf420)
        del arg266_1
        buf421 = reinterpret_tensor(buf419, (1024, 1024), (1024, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg268_1, (1024, 1024), (1, 1024), 0), out=buf421)
        del arg268_1
        buf422 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg270_1, (1024, 1024), (1, 1024), 0), out=buf422)
        del arg270_1
        buf423 = reinterpret_tensor(buf380, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf420, arg267_1, buf423, 1048576, grid=grid(1048576), stream=stream0)
        del arg267_1
        buf424 = reinterpret_tensor(buf420, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf421, arg269_1, buf424, 1048576, grid=grid(1048576), stream=stream0)
        del arg269_1
        buf425 = reinterpret_tensor(buf421, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf422, arg271_1, buf425, 1048576, grid=grid(1048576), stream=stream0)
        del arg271_1
        del buf422
        # Source Nodes: [], Original ATen: []
        buf426 = aten._scaled_dot_product_efficient_attention(buf423, buf424, buf425, None, True, scale=1.0)
        buf427 = buf426[0]
        del buf426
        buf431 = reinterpret_tensor(buf427, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf427  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf431, 1048576, grid=grid(1048576), stream=stream0)
        buf432 = reinterpret_tensor(buf425, (1024, 1024), (1024, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf431, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg272_1, (1024, 1024), (1, 1024), 0), out=buf432)
        del arg272_1
        buf433 = reinterpret_tensor(buf432, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf432  # reuse
        buf437 = reinterpret_tensor(buf431, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf431  # reuse
        # Source Nodes: [hidden_states_177, residual_31, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf433, buf398, buf415, arg263_1, arg273_1, arg274_1, arg275_1, buf437, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg263_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf438 = reinterpret_tensor(buf396, (1024, 4096), (4096, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf437, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg276_1, (1024, 4096), (1, 1024), 0), out=buf438)
        del arg276_1
        buf439 = reinterpret_tensor(buf438, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf438  # reuse
        # Source Nodes: [hidden_states_178], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf439, arg277_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg277_1
        buf440 = reinterpret_tensor(buf437, (1024, 1024), (1024, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg278_1, (4096, 1024), (1, 4096), 0), out=buf440)
        del arg278_1
        buf444 = reinterpret_tensor(buf415, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf415  # reuse
        # Source Nodes: [hidden_states_184, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf433, buf440, arg279_1, arg280_1, arg281_1, buf444, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg280_1
        del arg281_1
        buf445 = reinterpret_tensor(buf398, (1024, 1024), (1024, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg282_1, (1024, 1024), (1, 1024), 0), out=buf445)
        del arg282_1
        buf446 = reinterpret_tensor(buf424, (1024, 1024), (1024, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg284_1, (1024, 1024), (1, 1024), 0), out=buf446)
        del arg284_1
        buf447 = buf423; del buf423  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf445, arg283_1, buf447, 1048576, grid=grid(1048576), stream=stream0)
        del arg283_1
        buf448 = reinterpret_tensor(buf445, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf445  # reuse
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf446, arg285_1, buf448, 1048576, grid=grid(1048576), stream=stream0)
        del arg285_1
        del buf446
        buf449 = buf411; del buf411  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf448, (16, 64, 1024), (65536, 1, 64), 0), out=buf449)
        buf453 = buf407; del buf407  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf449, buf453, 16384, 1024, grid=grid(16384), stream=stream0)
        buf452 = reinterpret_tensor(buf448, (1024, 1024), (1024, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg286_1, (1024, 1024), (1, 1024), 0), out=buf452)
        del arg286_1
        buf454 = reinterpret_tensor(buf444, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf444  # reuse
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf452, arg287_1, buf454, 1048576, grid=grid(1048576), stream=stream0)
        del arg287_1
        buf455 = reinterpret_tensor(buf452, (16, 1024, 64), (65536, 64, 1), 0); del buf452  # reuse
        # Source Nodes: [attn_output_90, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf453, reinterpret_tensor(buf454, (16, 1024, 64), (65536, 64, 1), 0), out=buf455)
        buf456 = reinterpret_tensor(buf454, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf454  # reuse
        # Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf455, buf456, 1048576, grid=grid(1048576), stream=stream0)
        buf457 = reinterpret_tensor(buf455, (1024, 1024), (1024, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf456, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg288_1, (1024, 1024), (1, 1024), 0), out=buf457)
        del arg288_1
        buf458 = reinterpret_tensor(buf457, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf457  # reuse
        buf462 = reinterpret_tensor(buf456, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf456  # reuse
        # Source Nodes: [hidden_states_188, residual_33, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf458, buf433, buf440, arg279_1, arg289_1, arg290_1, arg291_1, buf462, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg279_1
        del arg289_1
        del arg290_1
        del arg291_1
        buf463 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg292_1, (1024, 1024), (1, 1024), 0), out=buf463)
        del arg292_1
        buf464 = reinterpret_tensor(buf462, (1024, 1024), (1024, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg294_1, (1024, 1024), (1, 1024), 0), out=buf464)
        del arg294_1
        buf465 = reinterpret_tensor(buf433, (1024, 1024), (1024, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg296_1, (1024, 1024), (1, 1024), 0), out=buf465)
        del arg296_1
        buf466 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf463, arg293_1, buf466, 1048576, grid=grid(1048576), stream=stream0)
        del arg293_1
        buf467 = reinterpret_tensor(buf463, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf464, arg295_1, buf467, 1048576, grid=grid(1048576), stream=stream0)
        del arg295_1
        buf468 = reinterpret_tensor(buf464, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf465, arg297_1, buf468, 1048576, grid=grid(1048576), stream=stream0)
        del arg297_1
        # Source Nodes: [], Original ATen: []
        buf469 = aten._scaled_dot_product_efficient_attention(buf466, buf467, buf468, None, True, scale=1.0)
        buf470 = buf469[0]
        del buf469
        buf474 = reinterpret_tensor(buf470, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf470  # reuse
        # Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf474, 1048576, grid=grid(1048576), stream=stream0)
        buf475 = reinterpret_tensor(buf468, (1024, 1024), (1024, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg298_1, (1024, 1024), (1, 1024), 0), out=buf475)
        del arg298_1
        buf479 = reinterpret_tensor(buf474, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf474  # reuse
        # Source Nodes: [hidden_states_192, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf458, buf475, arg299_1, arg300_1, arg301_1, buf479, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg300_1
        del arg301_1
        buf480 = reinterpret_tensor(buf439, (1024, 4096), (4096, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg302_1, (1024, 4096), (1, 1024), 0), out=buf480)
        del arg302_1
        buf481 = reinterpret_tensor(buf480, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf480  # reuse
        # Source Nodes: [hidden_states_193], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf481, arg303_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg303_1
        buf482 = reinterpret_tensor(buf479, (1024, 1024), (1024, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg304_1, (4096, 1024), (1, 4096), 0), out=buf482)
        del arg304_1
        buf483 = reinterpret_tensor(buf482, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf482  # reuse
        buf487 = reinterpret_tensor(buf467, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf467  # reuse
        # Source Nodes: [hidden_states_199, residual_35, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf483, buf458, buf475, arg299_1, arg305_1, arg306_1, arg307_1, buf487, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg299_1
        del arg305_1
        del arg306_1
        del arg307_1
        buf488 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf487, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg308_1, (1024, 1024), (1, 1024), 0), out=buf488)
        del arg308_1
        buf489 = reinterpret_tensor(buf458, (1024, 1024), (1024, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf487, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg310_1, (1024, 1024), (1, 1024), 0), out=buf489)
        del arg310_1
        buf490 = buf466; del buf466  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf488, arg309_1, buf490, 1048576, grid=grid(1048576), stream=stream0)
        del arg309_1
        buf491 = reinterpret_tensor(buf488, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf488  # reuse
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf489, arg311_1, buf491, 1048576, grid=grid(1048576), stream=stream0)
        del arg311_1
        buf492 = buf453; del buf453  # reuse
        # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf491, (16, 64, 1024), (65536, 1, 64), 0), out=buf492)
        buf496 = buf449; del buf449  # reuse
        # Source Nodes: [attn_weights_51], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf492, buf496, 16384, 1024, grid=grid(16384), stream=stream0)
        buf495 = reinterpret_tensor(buf491, (1024, 1024), (1024, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf487, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg312_1, (1024, 1024), (1, 1024), 0), out=buf495)
        del arg312_1
        buf497 = reinterpret_tensor(buf487, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf487  # reuse
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf495, arg313_1, buf497, 1048576, grid=grid(1048576), stream=stream0)
        del arg313_1
        buf498 = reinterpret_tensor(buf495, (16, 1024, 64), (65536, 64, 1), 0); del buf495  # reuse
        # Source Nodes: [attn_output_100, attn_weights_51], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf496, reinterpret_tensor(buf497, (16, 1024, 64), (65536, 64, 1), 0), out=buf498)
        buf499 = reinterpret_tensor(buf497, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf497  # reuse
        # Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf498, buf499, 1048576, grid=grid(1048576), stream=stream0)
        buf500 = reinterpret_tensor(buf498, (1024, 1024), (1024, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf499, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg314_1, (1024, 1024), (1, 1024), 0), out=buf500)
        del arg314_1
        buf504 = reinterpret_tensor(buf499, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf499  # reuse
        # Source Nodes: [hidden_states_203, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf483, buf500, arg315_1, arg316_1, arg317_1, buf504, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg316_1
        del arg317_1
        buf505 = reinterpret_tensor(buf490, (1024, 1024), (1024, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf504, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg318_1, (1024, 1024), (1, 1024), 0), out=buf505)
        del arg318_1
        buf506 = reinterpret_tensor(buf504, (1024, 1024), (1024, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg320_1, (1024, 1024), (1, 1024), 0), out=buf506)
        del arg320_1
        buf507 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg322_1, (1024, 1024), (1, 1024), 0), out=buf507)
        del arg322_1
        buf508 = reinterpret_tensor(buf465, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf505, arg319_1, buf508, 1048576, grid=grid(1048576), stream=stream0)
        del arg319_1
        buf509 = reinterpret_tensor(buf505, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf506, arg321_1, buf509, 1048576, grid=grid(1048576), stream=stream0)
        del arg321_1
        buf510 = reinterpret_tensor(buf506, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf507, arg323_1, buf510, 1048576, grid=grid(1048576), stream=stream0)
        del arg323_1
        del buf507
        # Source Nodes: [], Original ATen: []
        buf511 = aten._scaled_dot_product_efficient_attention(buf508, buf509, buf510, None, True, scale=1.0)
        buf512 = buf511[0]
        del buf511
        buf516 = reinterpret_tensor(buf512, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf512  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf516, 1048576, grid=grid(1048576), stream=stream0)
        buf517 = reinterpret_tensor(buf510, (1024, 1024), (1024, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf516, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg324_1, (1024, 1024), (1, 1024), 0), out=buf517)
        del arg324_1
        buf518 = reinterpret_tensor(buf517, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf517  # reuse
        buf522 = reinterpret_tensor(buf516, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf516  # reuse
        # Source Nodes: [hidden_states_207, residual_37, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf518, buf483, buf500, arg315_1, arg325_1, arg326_1, arg327_1, buf522, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg315_1
        del arg325_1
        del arg326_1
        del arg327_1
        buf523 = reinterpret_tensor(buf481, (1024, 4096), (4096, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg328_1, (1024, 4096), (1, 1024), 0), out=buf523)
        del arg328_1
        buf524 = reinterpret_tensor(buf523, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf523  # reuse
        # Source Nodes: [hidden_states_208], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf524, arg329_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg329_1
        buf525 = reinterpret_tensor(buf522, (1024, 1024), (1024, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg330_1, (4096, 1024), (1, 4096), 0), out=buf525)
        del arg330_1
        buf529 = reinterpret_tensor(buf500, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf500  # reuse
        # Source Nodes: [hidden_states_214, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf518, buf525, arg331_1, arg332_1, arg333_1, buf529, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg332_1
        del arg333_1
        buf530 = reinterpret_tensor(buf483, (1024, 1024), (1024, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg334_1, (1024, 1024), (1, 1024), 0), out=buf530)
        del arg334_1
        buf531 = reinterpret_tensor(buf509, (1024, 1024), (1024, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg336_1, (1024, 1024), (1, 1024), 0), out=buf531)
        del arg336_1
        buf532 = buf508; del buf508  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf530, arg335_1, buf532, 1048576, grid=grid(1048576), stream=stream0)
        del arg335_1
        buf533 = reinterpret_tensor(buf530, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf530  # reuse
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf531, arg337_1, buf533, 1048576, grid=grid(1048576), stream=stream0)
        del arg337_1
        del buf531
        buf534 = buf496; del buf496  # reuse
        # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf533, (16, 64, 1024), (65536, 1, 64), 0), out=buf534)
        buf538 = buf492; del buf492  # reuse
        # Source Nodes: [attn_weights_57], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf534, buf538, 16384, 1024, grid=grid(16384), stream=stream0)
        buf537 = reinterpret_tensor(buf533, (1024, 1024), (1024, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg338_1, (1024, 1024), (1, 1024), 0), out=buf537)
        del arg338_1
        buf539 = reinterpret_tensor(buf529, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf529  # reuse
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf537, arg339_1, buf539, 1048576, grid=grid(1048576), stream=stream0)
        del arg339_1
        buf540 = reinterpret_tensor(buf537, (16, 1024, 64), (65536, 64, 1), 0); del buf537  # reuse
        # Source Nodes: [attn_output_110, attn_weights_57], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf538, reinterpret_tensor(buf539, (16, 1024, 64), (65536, 64, 1), 0), out=buf540)
        buf541 = reinterpret_tensor(buf539, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf539  # reuse
        # Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf540, buf541, 1048576, grid=grid(1048576), stream=stream0)
        buf542 = reinterpret_tensor(buf540, (1024, 1024), (1024, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg340_1, (1024, 1024), (1, 1024), 0), out=buf542)
        del arg340_1
        buf543 = reinterpret_tensor(buf542, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf542  # reuse
        buf547 = reinterpret_tensor(buf541, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf541  # reuse
        # Source Nodes: [hidden_states_218, residual_39, residual_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf543, buf518, buf525, arg331_1, arg341_1, arg342_1, arg343_1, buf547, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg331_1
        del arg341_1
        del arg342_1
        del arg343_1
        buf548 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf547, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg344_1, (1024, 1024), (1, 1024), 0), out=buf548)
        del arg344_1
        buf549 = reinterpret_tensor(buf547, (1024, 1024), (1024, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg346_1, (1024, 1024), (1, 1024), 0), out=buf549)
        del arg346_1
        buf550 = reinterpret_tensor(buf518, (1024, 1024), (1024, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg348_1, (1024, 1024), (1, 1024), 0), out=buf550)
        del arg348_1
        buf551 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf548, arg345_1, buf551, 1048576, grid=grid(1048576), stream=stream0)
        del arg345_1
        buf552 = reinterpret_tensor(buf548, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf549, arg347_1, buf552, 1048576, grid=grid(1048576), stream=stream0)
        del arg347_1
        buf553 = reinterpret_tensor(buf549, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf550, arg349_1, buf553, 1048576, grid=grid(1048576), stream=stream0)
        del arg349_1
        # Source Nodes: [], Original ATen: []
        buf554 = aten._scaled_dot_product_efficient_attention(buf551, buf552, buf553, None, True, scale=1.0)
        buf555 = buf554[0]
        del buf554
        buf559 = reinterpret_tensor(buf555, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf555  # reuse
        # Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf559, 1048576, grid=grid(1048576), stream=stream0)
        buf560 = reinterpret_tensor(buf553, (1024, 1024), (1024, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf559, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg350_1, (1024, 1024), (1, 1024), 0), out=buf560)
        del arg350_1
        buf564 = reinterpret_tensor(buf559, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf559  # reuse
        # Source Nodes: [hidden_states_222, residual_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf543, buf560, arg351_1, arg352_1, arg353_1, buf564, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg352_1
        del arg353_1
        buf565 = reinterpret_tensor(buf524, (1024, 4096), (4096, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg354_1, (1024, 4096), (1, 1024), 0), out=buf565)
        del arg354_1
        buf566 = reinterpret_tensor(buf565, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf565  # reuse
        # Source Nodes: [hidden_states_223], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf566, arg355_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg355_1
        buf567 = reinterpret_tensor(buf564, (1024, 1024), (1024, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf566, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg356_1, (4096, 1024), (1, 4096), 0), out=buf567)
        del arg356_1
        buf568 = reinterpret_tensor(buf567, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf567  # reuse
        buf572 = reinterpret_tensor(buf552, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf552  # reuse
        # Source Nodes: [hidden_states_229, residual_41, residual_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf568, buf543, buf560, arg351_1, arg357_1, arg358_1, arg359_1, buf572, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg351_1
        del arg357_1
        del arg358_1
        del arg359_1
        buf573 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg360_1, (1024, 1024), (1, 1024), 0), out=buf573)
        del arg360_1
        buf574 = reinterpret_tensor(buf543, (1024, 1024), (1024, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg362_1, (1024, 1024), (1, 1024), 0), out=buf574)
        del arg362_1
        buf575 = buf551; del buf551  # reuse
        # Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf573, arg361_1, buf575, 1048576, grid=grid(1048576), stream=stream0)
        del arg361_1
        buf576 = reinterpret_tensor(buf573, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf573  # reuse
        # Source Nodes: [key_states_48], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf574, arg363_1, buf576, 1048576, grid=grid(1048576), stream=stream0)
        del arg363_1
        buf577 = buf538; del buf538  # reuse
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf575, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf576, (16, 64, 1024), (65536, 1, 64), 0), out=buf577)
        buf581 = buf534; del buf534  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf577, buf581, 16384, 1024, grid=grid(16384), stream=stream0)
        buf580 = reinterpret_tensor(buf576, (1024, 1024), (1024, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg364_1, (1024, 1024), (1, 1024), 0), out=buf580)
        del arg364_1
        buf582 = reinterpret_tensor(buf572, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf572  # reuse
        # Source Nodes: [value_states_48], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf580, arg365_1, buf582, 1048576, grid=grid(1048576), stream=stream0)
        del arg365_1
        buf583 = reinterpret_tensor(buf580, (16, 1024, 64), (65536, 64, 1), 0); del buf580  # reuse
        # Source Nodes: [attn_output_120, attn_weights_63], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf581, reinterpret_tensor(buf582, (16, 1024, 64), (65536, 64, 1), 0), out=buf583)
        buf584 = reinterpret_tensor(buf582, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf582  # reuse
        # Source Nodes: [attn_output_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf583, buf584, 1048576, grid=grid(1048576), stream=stream0)
        buf585 = reinterpret_tensor(buf583, (1024, 1024), (1024, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf584, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg366_1, (1024, 1024), (1, 1024), 0), out=buf585)
        del arg366_1
        buf589 = reinterpret_tensor(buf584, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf584  # reuse
        # Source Nodes: [hidden_states_233, residual_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf568, buf585, arg367_1, arg368_1, arg369_1, buf589, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg368_1
        del arg369_1
        buf590 = reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg370_1, (1024, 1024), (1, 1024), 0), out=buf590)
        del arg370_1
        buf591 = reinterpret_tensor(buf589, (1024, 1024), (1024, 1), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg372_1, (1024, 1024), (1, 1024), 0), out=buf591)
        del arg372_1
        buf592 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg374_1, (1024, 1024), (1, 1024), 0), out=buf592)
        del arg374_1
        buf593 = reinterpret_tensor(buf550, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf590, arg371_1, buf593, 1048576, grid=grid(1048576), stream=stream0)
        del arg371_1
        buf594 = reinterpret_tensor(buf590, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf591, arg373_1, buf594, 1048576, grid=grid(1048576), stream=stream0)
        del arg373_1
        buf595 = reinterpret_tensor(buf591, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf592, arg375_1, buf595, 1048576, grid=grid(1048576), stream=stream0)
        del arg375_1
        del buf592
        # Source Nodes: [], Original ATen: []
        buf596 = aten._scaled_dot_product_efficient_attention(buf593, buf594, buf595, None, True, scale=1.0)
        buf597 = buf596[0]
        del buf596
        buf601 = reinterpret_tensor(buf597, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf597  # reuse
        # Source Nodes: [attn_output_128], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf601, 1048576, grid=grid(1048576), stream=stream0)
        buf602 = reinterpret_tensor(buf595, (1024, 1024), (1024, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf601, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg376_1, (1024, 1024), (1, 1024), 0), out=buf602)
        del arg376_1
        buf603 = reinterpret_tensor(buf602, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf602  # reuse
        buf607 = reinterpret_tensor(buf601, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf601  # reuse
        # Source Nodes: [hidden_states_237, residual_43, residual_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf603, buf568, buf585, arg367_1, arg377_1, arg378_1, arg379_1, buf607, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg367_1
        del arg377_1
        del arg378_1
        del arg379_1
        buf608 = reinterpret_tensor(buf566, (1024, 4096), (4096, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf607, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg380_1, (1024, 4096), (1, 1024), 0), out=buf608)
        del arg380_1
        buf609 = reinterpret_tensor(buf608, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf608  # reuse
        # Source Nodes: [hidden_states_238], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf609, arg381_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg381_1
        buf610 = reinterpret_tensor(buf607, (1024, 1024), (1024, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf609, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg382_1, (4096, 1024), (1, 4096), 0), out=buf610)
        del arg382_1
        buf614 = reinterpret_tensor(buf585, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf585  # reuse
        # Source Nodes: [hidden_states_244, residual_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf603, buf610, arg383_1, arg384_1, arg385_1, buf614, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg384_1
        del arg385_1
        buf615 = reinterpret_tensor(buf568, (1024, 1024), (1024, 1), 0); del buf568  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg386_1, (1024, 1024), (1, 1024), 0), out=buf615)
        del arg386_1
        buf616 = reinterpret_tensor(buf594, (1024, 1024), (1024, 1), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg388_1, (1024, 1024), (1, 1024), 0), out=buf616)
        del arg388_1
        buf617 = buf593; del buf593  # reuse
        # Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf615, arg387_1, buf617, 1048576, grid=grid(1048576), stream=stream0)
        del arg387_1
        buf618 = reinterpret_tensor(buf615, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf615  # reuse
        # Source Nodes: [key_states_52], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf616, arg389_1, buf618, 1048576, grid=grid(1048576), stream=stream0)
        del arg389_1
        del buf616
        buf619 = buf581; del buf581  # reuse
        # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf617, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf618, (16, 64, 1024), (65536, 1, 64), 0), out=buf619)
        buf623 = buf577; del buf577  # reuse
        # Source Nodes: [attn_weights_69], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf619, buf623, 16384, 1024, grid=grid(16384), stream=stream0)
        buf622 = reinterpret_tensor(buf618, (1024, 1024), (1024, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg390_1, (1024, 1024), (1, 1024), 0), out=buf622)
        del arg390_1
        buf624 = reinterpret_tensor(buf614, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf614  # reuse
        # Source Nodes: [value_states_52], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf622, arg391_1, buf624, 1048576, grid=grid(1048576), stream=stream0)
        del arg391_1
        buf625 = reinterpret_tensor(buf622, (16, 1024, 64), (65536, 64, 1), 0); del buf622  # reuse
        # Source Nodes: [attn_output_130, attn_weights_69], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf623, reinterpret_tensor(buf624, (16, 1024, 64), (65536, 64, 1), 0), out=buf625)
        buf626 = reinterpret_tensor(buf624, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf624  # reuse
        # Source Nodes: [attn_output_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf625, buf626, 1048576, grid=grid(1048576), stream=stream0)
        buf627 = reinterpret_tensor(buf625, (1024, 1024), (1024, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf626, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg392_1, (1024, 1024), (1, 1024), 0), out=buf627)
        del arg392_1
        buf628 = reinterpret_tensor(buf627, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf627  # reuse
        buf632 = reinterpret_tensor(buf626, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf626  # reuse
        # Source Nodes: [hidden_states_248, residual_45, residual_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf628, buf603, buf610, arg383_1, arg393_1, arg394_1, arg395_1, buf632, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg383_1
        del arg393_1
        del arg394_1
        del arg395_1
        buf633 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf632, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg396_1, (1024, 1024), (1, 1024), 0), out=buf633)
        del arg396_1
        buf634 = reinterpret_tensor(buf632, (1024, 1024), (1024, 1), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg398_1, (1024, 1024), (1, 1024), 0), out=buf634)
        del arg398_1
        buf635 = reinterpret_tensor(buf603, (1024, 1024), (1024, 1), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg400_1, (1024, 1024), (1, 1024), 0), out=buf635)
        del arg400_1
        buf636 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf633, arg397_1, buf636, 1048576, grid=grid(1048576), stream=stream0)
        del arg397_1
        buf637 = reinterpret_tensor(buf633, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf634, arg399_1, buf637, 1048576, grid=grid(1048576), stream=stream0)
        del arg399_1
        buf638 = reinterpret_tensor(buf634, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf635, arg401_1, buf638, 1048576, grid=grid(1048576), stream=stream0)
        del arg401_1
        # Source Nodes: [], Original ATen: []
        buf639 = aten._scaled_dot_product_efficient_attention(buf636, buf637, buf638, None, True, scale=1.0)
        buf640 = buf639[0]
        del buf639
        buf644 = reinterpret_tensor(buf640, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf640  # reuse
        # Source Nodes: [attn_output_138], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf644, 1048576, grid=grid(1048576), stream=stream0)
        buf645 = reinterpret_tensor(buf638, (1024, 1024), (1024, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg402_1, (1024, 1024), (1, 1024), 0), out=buf645)
        del arg402_1
        buf649 = reinterpret_tensor(buf644, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf644  # reuse
        # Source Nodes: [hidden_states_252, residual_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf628, buf645, arg403_1, arg404_1, arg405_1, buf649, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg404_1
        del arg405_1
        buf650 = reinterpret_tensor(buf609, (1024, 4096), (4096, 1), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf649, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg406_1, (1024, 4096), (1, 1024), 0), out=buf650)
        del arg406_1
        buf651 = reinterpret_tensor(buf650, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf650  # reuse
        # Source Nodes: [hidden_states_253], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf651, arg407_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg407_1
        buf652 = reinterpret_tensor(buf649, (1024, 1024), (1024, 1), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf651, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg408_1, (4096, 1024), (1, 4096), 0), out=buf652)
        del arg408_1
        buf653 = reinterpret_tensor(buf652, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf652  # reuse
        buf657 = reinterpret_tensor(buf637, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf637  # reuse
        # Source Nodes: [hidden_states_259, residual_47, residual_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf653, buf628, buf645, arg403_1, arg409_1, arg410_1, arg411_1, buf657, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg403_1
        del arg409_1
        del arg410_1
        del arg411_1
        buf658 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf657, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg412_1, (1024, 1024), (1, 1024), 0), out=buf658)
        del arg412_1
        buf659 = reinterpret_tensor(buf628, (1024, 1024), (1024, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf657, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg414_1, (1024, 1024), (1, 1024), 0), out=buf659)
        del arg414_1
        buf660 = buf636; del buf636  # reuse
        # Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf658, arg413_1, buf660, 1048576, grid=grid(1048576), stream=stream0)
        del arg413_1
        buf661 = reinterpret_tensor(buf658, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf658  # reuse
        # Source Nodes: [key_states_56], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf659, arg415_1, buf661, 1048576, grid=grid(1048576), stream=stream0)
        del arg415_1
        buf662 = buf623; del buf623  # reuse
        # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf660, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf661, (16, 64, 1024), (65536, 1, 64), 0), out=buf662)
        buf666 = buf619; del buf619  # reuse
        # Source Nodes: [attn_weights_75], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf662, buf666, 16384, 1024, grid=grid(16384), stream=stream0)
        buf665 = reinterpret_tensor(buf661, (1024, 1024), (1024, 1), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf657, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg416_1, (1024, 1024), (1, 1024), 0), out=buf665)
        del arg416_1
        buf667 = reinterpret_tensor(buf657, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf657  # reuse
        # Source Nodes: [value_states_56], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf665, arg417_1, buf667, 1048576, grid=grid(1048576), stream=stream0)
        del arg417_1
        buf668 = reinterpret_tensor(buf665, (16, 1024, 64), (65536, 64, 1), 0); del buf665  # reuse
        # Source Nodes: [attn_output_140, attn_weights_75], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf666, reinterpret_tensor(buf667, (16, 1024, 64), (65536, 64, 1), 0), out=buf668)
        buf669 = reinterpret_tensor(buf667, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf667  # reuse
        # Source Nodes: [attn_output_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf668, buf669, 1048576, grid=grid(1048576), stream=stream0)
        buf670 = reinterpret_tensor(buf668, (1024, 1024), (1024, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf669, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg418_1, (1024, 1024), (1, 1024), 0), out=buf670)
        del arg418_1
        buf674 = reinterpret_tensor(buf669, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf669  # reuse
        # Source Nodes: [hidden_states_263, residual_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf653, buf670, arg419_1, arg420_1, arg421_1, buf674, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg420_1
        del arg421_1
        buf675 = reinterpret_tensor(buf660, (1024, 1024), (1024, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf674, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg422_1, (1024, 1024), (1, 1024), 0), out=buf675)
        del arg422_1
        buf676 = reinterpret_tensor(buf674, (1024, 1024), (1024, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg424_1, (1024, 1024), (1, 1024), 0), out=buf676)
        del arg424_1
        buf677 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg426_1, (1024, 1024), (1, 1024), 0), out=buf677)
        del arg426_1
        buf678 = reinterpret_tensor(buf635, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf675, arg423_1, buf678, 1048576, grid=grid(1048576), stream=stream0)
        del arg423_1
        buf679 = reinterpret_tensor(buf675, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf676, arg425_1, buf679, 1048576, grid=grid(1048576), stream=stream0)
        del arg425_1
        buf680 = reinterpret_tensor(buf676, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf677, arg427_1, buf680, 1048576, grid=grid(1048576), stream=stream0)
        del arg427_1
        del buf677
        # Source Nodes: [], Original ATen: []
        buf681 = aten._scaled_dot_product_efficient_attention(buf678, buf679, buf680, None, True, scale=1.0)
        buf682 = buf681[0]
        del buf681
        buf686 = reinterpret_tensor(buf682, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf682  # reuse
        # Source Nodes: [attn_output_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf686, 1048576, grid=grid(1048576), stream=stream0)
        buf687 = reinterpret_tensor(buf680, (1024, 1024), (1024, 1), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf686, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg428_1, (1024, 1024), (1, 1024), 0), out=buf687)
        del arg428_1
        buf688 = reinterpret_tensor(buf687, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf687  # reuse
        buf692 = reinterpret_tensor(buf686, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf686  # reuse
        # Source Nodes: [hidden_states_267, residual_49, residual_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf688, buf653, buf670, arg419_1, arg429_1, arg430_1, arg431_1, buf692, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg419_1
        del arg429_1
        del arg430_1
        del arg431_1
        buf693 = reinterpret_tensor(buf651, (1024, 4096), (4096, 1), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf692, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg432_1, (1024, 4096), (1, 1024), 0), out=buf693)
        del arg432_1
        buf694 = reinterpret_tensor(buf693, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf693  # reuse
        # Source Nodes: [hidden_states_268], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf694, arg433_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg433_1
        buf695 = reinterpret_tensor(buf692, (1024, 1024), (1024, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf694, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg434_1, (4096, 1024), (1, 4096), 0), out=buf695)
        del arg434_1
        buf699 = reinterpret_tensor(buf670, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf670  # reuse
        # Source Nodes: [hidden_states_274, residual_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf688, buf695, arg435_1, arg436_1, arg437_1, buf699, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg436_1
        del arg437_1
        buf700 = reinterpret_tensor(buf653, (1024, 1024), (1024, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf699, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg438_1, (1024, 1024), (1, 1024), 0), out=buf700)
        del arg438_1
        buf701 = reinterpret_tensor(buf679, (1024, 1024), (1024, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf699, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg440_1, (1024, 1024), (1, 1024), 0), out=buf701)
        del arg440_1
        buf702 = buf678; del buf678  # reuse
        # Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf700, arg439_1, buf702, 1048576, grid=grid(1048576), stream=stream0)
        del arg439_1
        buf703 = reinterpret_tensor(buf700, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf700  # reuse
        # Source Nodes: [key_states_60], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf701, arg441_1, buf703, 1048576, grid=grid(1048576), stream=stream0)
        del arg441_1
        del buf701
        buf704 = buf666; del buf666  # reuse
        # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf703, (16, 64, 1024), (65536, 1, 64), 0), out=buf704)
        buf708 = buf662; del buf662  # reuse
        # Source Nodes: [attn_weights_81], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf704, buf708, 16384, 1024, grid=grid(16384), stream=stream0)
        buf707 = reinterpret_tensor(buf703, (1024, 1024), (1024, 1), 0); del buf703  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf699, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg442_1, (1024, 1024), (1, 1024), 0), out=buf707)
        del arg442_1
        buf709 = reinterpret_tensor(buf699, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf699  # reuse
        # Source Nodes: [value_states_60], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf707, arg443_1, buf709, 1048576, grid=grid(1048576), stream=stream0)
        del arg443_1
        buf710 = reinterpret_tensor(buf707, (16, 1024, 64), (65536, 64, 1), 0); del buf707  # reuse
        # Source Nodes: [attn_output_150, attn_weights_81], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf708, reinterpret_tensor(buf709, (16, 1024, 64), (65536, 64, 1), 0), out=buf710)
        buf711 = reinterpret_tensor(buf709, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf709  # reuse
        # Source Nodes: [attn_output_153], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf710, buf711, 1048576, grid=grid(1048576), stream=stream0)
        buf712 = reinterpret_tensor(buf710, (1024, 1024), (1024, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf711, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg444_1, (1024, 1024), (1, 1024), 0), out=buf712)
        del arg444_1
        buf713 = reinterpret_tensor(buf712, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf712  # reuse
        buf717 = reinterpret_tensor(buf711, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf711  # reuse
        # Source Nodes: [hidden_states_278, residual_51, residual_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf713, buf688, buf695, arg435_1, arg445_1, arg446_1, arg447_1, buf717, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg435_1
        del arg445_1
        del arg446_1
        del arg447_1
        buf718 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf717, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg448_1, (1024, 1024), (1, 1024), 0), out=buf718)
        del arg448_1
        buf719 = reinterpret_tensor(buf717, (1024, 1024), (1024, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg450_1, (1024, 1024), (1, 1024), 0), out=buf719)
        del arg450_1
        buf720 = reinterpret_tensor(buf688, (1024, 1024), (1024, 1), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg452_1, (1024, 1024), (1, 1024), 0), out=buf720)
        del arg452_1
        buf721 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf718, arg449_1, buf721, 1048576, grid=grid(1048576), stream=stream0)
        del arg449_1
        buf722 = reinterpret_tensor(buf718, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf719, arg451_1, buf722, 1048576, grid=grid(1048576), stream=stream0)
        del arg451_1
        buf723 = reinterpret_tensor(buf719, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf720, arg453_1, buf723, 1048576, grid=grid(1048576), stream=stream0)
        del arg453_1
        # Source Nodes: [], Original ATen: []
        buf724 = aten._scaled_dot_product_efficient_attention(buf721, buf722, buf723, None, True, scale=1.0)
        buf725 = buf724[0]
        del buf724
        buf729 = reinterpret_tensor(buf725, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf725  # reuse
        # Source Nodes: [attn_output_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf729, 1048576, grid=grid(1048576), stream=stream0)
        buf730 = reinterpret_tensor(buf723, (1024, 1024), (1024, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf729, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg454_1, (1024, 1024), (1, 1024), 0), out=buf730)
        del arg454_1
        buf734 = reinterpret_tensor(buf729, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf729  # reuse
        # Source Nodes: [hidden_states_282, residual_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf713, buf730, arg455_1, arg456_1, arg457_1, buf734, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg456_1
        del arg457_1
        buf735 = reinterpret_tensor(buf694, (1024, 4096), (4096, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf734, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg458_1, (1024, 4096), (1, 1024), 0), out=buf735)
        del arg458_1
        buf736 = reinterpret_tensor(buf735, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf735  # reuse
        # Source Nodes: [hidden_states_283], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf736, arg459_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg459_1
        buf737 = reinterpret_tensor(buf734, (1024, 1024), (1024, 1), 0); del buf734  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf736, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg460_1, (4096, 1024), (1, 4096), 0), out=buf737)
        del arg460_1
        buf738 = reinterpret_tensor(buf737, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf737  # reuse
        buf742 = reinterpret_tensor(buf722, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf722  # reuse
        # Source Nodes: [hidden_states_289, residual_53, residual_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf738, buf713, buf730, arg455_1, arg461_1, arg462_1, arg463_1, buf742, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg455_1
        del arg461_1
        del arg462_1
        del arg463_1
        buf743 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg464_1, (1024, 1024), (1, 1024), 0), out=buf743)
        del arg464_1
        buf744 = reinterpret_tensor(buf713, (1024, 1024), (1024, 1), 0); del buf713  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg466_1, (1024, 1024), (1, 1024), 0), out=buf744)
        del arg466_1
        buf745 = buf721; del buf721  # reuse
        # Source Nodes: [contiguous_98], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf743, arg465_1, buf745, 1048576, grid=grid(1048576), stream=stream0)
        del arg465_1
        buf746 = reinterpret_tensor(buf743, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf743  # reuse
        # Source Nodes: [key_states_64], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf744, arg467_1, buf746, 1048576, grid=grid(1048576), stream=stream0)
        del arg467_1
        buf747 = buf708; del buf708  # reuse
        # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf745, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf746, (16, 64, 1024), (65536, 1, 64), 0), out=buf747)
        buf751 = buf704; del buf704  # reuse
        # Source Nodes: [attn_weights_87], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf747, buf751, 16384, 1024, grid=grid(16384), stream=stream0)
        buf750 = reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg468_1, (1024, 1024), (1, 1024), 0), out=buf750)
        del arg468_1
        buf752 = reinterpret_tensor(buf742, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf742  # reuse
        # Source Nodes: [value_states_64], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf750, arg469_1, buf752, 1048576, grid=grid(1048576), stream=stream0)
        del arg469_1
        buf753 = reinterpret_tensor(buf750, (16, 1024, 64), (65536, 64, 1), 0); del buf750  # reuse
        # Source Nodes: [attn_output_160, attn_weights_87], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf751, reinterpret_tensor(buf752, (16, 1024, 64), (65536, 64, 1), 0), out=buf753)
        buf754 = reinterpret_tensor(buf752, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf752  # reuse
        # Source Nodes: [attn_output_163], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf753, buf754, 1048576, grid=grid(1048576), stream=stream0)
        buf755 = reinterpret_tensor(buf753, (1024, 1024), (1024, 1), 0); del buf753  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf754, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg470_1, (1024, 1024), (1, 1024), 0), out=buf755)
        del arg470_1
        buf759 = reinterpret_tensor(buf754, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf754  # reuse
        # Source Nodes: [hidden_states_293, residual_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf738, buf755, arg471_1, arg472_1, arg473_1, buf759, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg472_1
        del arg473_1
        buf760 = reinterpret_tensor(buf745, (1024, 1024), (1024, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf759, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg474_1, (1024, 1024), (1, 1024), 0), out=buf760)
        del arg474_1
        buf761 = reinterpret_tensor(buf759, (1024, 1024), (1024, 1), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg476_1, (1024, 1024), (1, 1024), 0), out=buf761)
        del arg476_1
        buf762 = buf744; del buf744  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg478_1, (1024, 1024), (1, 1024), 0), out=buf762)
        del arg478_1
        buf763 = reinterpret_tensor(buf720, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf720  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf760, arg475_1, buf763, 1048576, grid=grid(1048576), stream=stream0)
        del arg475_1
        buf764 = reinterpret_tensor(buf760, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf761, arg477_1, buf764, 1048576, grid=grid(1048576), stream=stream0)
        del arg477_1
        buf765 = reinterpret_tensor(buf761, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf762, arg479_1, buf765, 1048576, grid=grid(1048576), stream=stream0)
        del arg479_1
        del buf762
        # Source Nodes: [], Original ATen: []
        buf766 = aten._scaled_dot_product_efficient_attention(buf763, buf764, buf765, None, True, scale=1.0)
        buf767 = buf766[0]
        del buf766
        buf771 = reinterpret_tensor(buf767, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf767  # reuse
        # Source Nodes: [attn_output_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf771, 1048576, grid=grid(1048576), stream=stream0)
        buf772 = reinterpret_tensor(buf765, (1024, 1024), (1024, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf771, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg480_1, (1024, 1024), (1, 1024), 0), out=buf772)
        del arg480_1
        buf773 = reinterpret_tensor(buf772, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf772  # reuse
        buf777 = reinterpret_tensor(buf771, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf771  # reuse
        # Source Nodes: [hidden_states_297, residual_55, residual_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf773, buf738, buf755, arg471_1, arg481_1, arg482_1, arg483_1, buf777, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg471_1
        del arg481_1
        del arg482_1
        del arg483_1
        buf778 = reinterpret_tensor(buf736, (1024, 4096), (4096, 1), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf777, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg484_1, (1024, 4096), (1, 1024), 0), out=buf778)
        del arg484_1
        buf779 = reinterpret_tensor(buf778, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf778  # reuse
        # Source Nodes: [hidden_states_298], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf779, arg485_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg485_1
        buf780 = reinterpret_tensor(buf777, (1024, 1024), (1024, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf779, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg486_1, (4096, 1024), (1, 4096), 0), out=buf780)
        del arg486_1
        buf784 = reinterpret_tensor(buf755, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf755  # reuse
        # Source Nodes: [hidden_states_304, residual_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf773, buf780, arg487_1, arg488_1, arg489_1, buf784, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg488_1
        del arg489_1
        buf785 = reinterpret_tensor(buf738, (1024, 1024), (1024, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg490_1, (1024, 1024), (1, 1024), 0), out=buf785)
        del arg490_1
        buf786 = reinterpret_tensor(buf764, (1024, 1024), (1024, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg492_1, (1024, 1024), (1, 1024), 0), out=buf786)
        del arg492_1
        buf787 = buf763; del buf763  # reuse
        # Source Nodes: [contiguous_104], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf785, arg491_1, buf787, 1048576, grid=grid(1048576), stream=stream0)
        del arg491_1
        buf788 = reinterpret_tensor(buf785, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf785  # reuse
        # Source Nodes: [key_states_68], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf786, arg493_1, buf788, 1048576, grid=grid(1048576), stream=stream0)
        del arg493_1
        del buf786
        buf789 = buf751; del buf751  # reuse
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf787, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf788, (16, 64, 1024), (65536, 1, 64), 0), out=buf789)
        buf793 = buf747; del buf747  # reuse
        # Source Nodes: [attn_weights_93], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf789, buf793, 16384, 1024, grid=grid(16384), stream=stream0)
        del buf789
        buf792 = reinterpret_tensor(buf788, (1024, 1024), (1024, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg494_1, (1024, 1024), (1, 1024), 0), out=buf792)
        del arg494_1
        buf794 = reinterpret_tensor(buf784, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf784  # reuse
        # Source Nodes: [value_states_68], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf792, arg495_1, buf794, 1048576, grid=grid(1048576), stream=stream0)
        del arg495_1
        buf795 = reinterpret_tensor(buf792, (16, 1024, 64), (65536, 64, 1), 0); del buf792  # reuse
        # Source Nodes: [attn_output_170, attn_weights_93], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf793, reinterpret_tensor(buf794, (16, 1024, 64), (65536, 64, 1), 0), out=buf795)
        del buf793
        buf796 = reinterpret_tensor(buf794, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf794  # reuse
        # Source Nodes: [attn_output_173], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf795, buf796, 1048576, grid=grid(1048576), stream=stream0)
        buf797 = reinterpret_tensor(buf795, (1024, 1024), (1024, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf796, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg496_1, (1024, 1024), (1, 1024), 0), out=buf797)
        del arg496_1
        buf798 = reinterpret_tensor(buf797, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf797  # reuse
        buf802 = reinterpret_tensor(buf796, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf796  # reuse
        # Source Nodes: [hidden_states_308, residual_57, residual_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf798, buf773, buf780, arg487_1, arg497_1, arg498_1, arg499_1, buf802, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg487_1
        del arg497_1
        del arg498_1
        del arg499_1
        buf803 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf802, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg500_1, (1024, 1024), (1, 1024), 0), out=buf803)
        del arg500_1
        buf804 = reinterpret_tensor(buf802, (1024, 1024), (1024, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg502_1, (1024, 1024), (1, 1024), 0), out=buf804)
        del arg502_1
        buf805 = reinterpret_tensor(buf773, (1024, 1024), (1024, 1), 0); del buf773  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg504_1, (1024, 1024), (1, 1024), 0), out=buf805)
        del arg504_1
        buf806 = buf787; del buf787  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf803, arg501_1, buf806, 1048576, grid=grid(1048576), stream=stream0)
        del arg501_1
        buf807 = reinterpret_tensor(buf803, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf803  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf804, arg503_1, buf807, 1048576, grid=grid(1048576), stream=stream0)
        del arg503_1
        buf808 = reinterpret_tensor(buf804, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf805, arg505_1, buf808, 1048576, grid=grid(1048576), stream=stream0)
        del arg505_1
        del buf805
        # Source Nodes: [], Original ATen: []
        buf809 = aten._scaled_dot_product_efficient_attention(buf806, buf807, buf808, None, True, scale=1.0)
        del buf806
        buf810 = buf809[0]
        del buf809
        buf814 = reinterpret_tensor(buf810, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf810  # reuse
        # Source Nodes: [attn_output_178], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf814, 1048576, grid=grid(1048576), stream=stream0)
        buf815 = reinterpret_tensor(buf808, (1024, 1024), (1024, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf814, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg506_1, (1024, 1024), (1, 1024), 0), out=buf815)
        del arg506_1
        buf819 = reinterpret_tensor(buf814, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf814  # reuse
        # Source Nodes: [hidden_states_312, residual_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf798, buf815, arg507_1, arg508_1, arg509_1, buf819, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg508_1
        del arg509_1
        buf820 = reinterpret_tensor(buf779, (1024, 4096), (4096, 1), 0); del buf779  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf819, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg510_1, (1024, 4096), (1, 1024), 0), out=buf820)
        del arg510_1
        buf821 = reinterpret_tensor(buf820, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf820  # reuse
        # Source Nodes: [hidden_states_313], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf821, arg511_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg511_1
        buf822 = reinterpret_tensor(buf819, (1024, 1024), (1024, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf821, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg512_1, (4096, 1024), (1, 4096), 0), out=buf822)
        del arg512_1
        del buf821
        buf823 = reinterpret_tensor(buf822, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf822  # reuse
        buf827 = reinterpret_tensor(buf807, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf807  # reuse
        # Source Nodes: [hidden_states_318, hidden_states_319, residual_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf823, buf798, buf815, arg507_1, arg513_1, arg514_1, arg515_1, buf827, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg507_1
        del arg513_1
        del arg514_1
        del arg515_1
        del buf798
        del buf815
        del buf823
        buf828 = empty((1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf827, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg516_1, (1024, 50265), (1, 1024), 0), out=buf828)
        del arg516_1
        del buf827
        buf829 = reinterpret_tensor(buf828, (1, 1024, 50265), (51471360, 50265, 1), 0); del buf828  # reuse
        buf830 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf831 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten._log_softmax, aten.add]
        triton_red_fused__log_softmax_add_11.run(buf829, arg517_1, buf830, buf831, 1024, 50265, grid=grid(1024), stream=stream0)
        del arg517_1
        buf832 = empty((), device='cuda', dtype=torch.float32)
        buf834 = buf832; del buf832  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_12.run(buf834, arg518_1, buf829, buf830, buf831, 1, 1024, grid=grid(1), stream=stream0)
        del arg518_1
        return (buf834, buf829, buf335, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg519_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
