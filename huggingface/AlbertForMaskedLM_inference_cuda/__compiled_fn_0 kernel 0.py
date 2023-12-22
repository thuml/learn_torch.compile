
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


# kernel path: /tmp/torchinductor_youkaichao/va/cvasn5b22irpm45aibugbilljom6b6howg2epxcxtqspll24ju6m.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30000
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30000)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30000")
    tmp4 = tl.load(in_ptr1 + (r1 + (128*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 2)) | ~xmask, "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr3 + (r1 + (128*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert(((0 <= tmp14) & (tmp14 < 512)) | ~xmask, "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r1 + (128*tmp14)), rmask & xmask, other=0.0)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 128.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-12
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp43, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/nm/cnm52j5mzlx4qy5ga5mtxrrs6pwgl2xpzwfmpcf246qx5tam4qrd.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (4096*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74itheyc35qr3lxpgnyi26mrvh5kc7l62zpslkofdehswfuy6fp.py
# Source Nodes: [add_2, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_5
# layernormed_context_layer => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
triton_red_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 4096.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-12
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvck4tvx67szr2fm6nnzw33xdr3llelcv2juy2k6v2eurdyqwuw2.py
# Source Nodes: [add_3, add_4, ffn_output_1, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_3 => add_8
# add_4 => add_9
# ffn_output_1 => mul_8
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = tl.math.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhda3szggpnvqkschfaapa6aqyyradqxnjyxk5to2ajwoktqrt7.py
# Source Nodes: [add_5, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
# add_5 => add_10
# hidden_states_3 => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
triton_red_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 4096.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-12
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjwxpaxe2bhjumj77awur7xdlewjtiopvd5jcwha6ootnapgo64.py
# Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
# add_61 => add_112
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => add_114, add_115, mul_103, mul_104, rsqrt_25, sub_38, var_mean_25
# mul_49 => mul_99
# mul_50 => mul_100
# mul_51 => mul_101
# pow_13 => pow_13
# tanh_12 => tanh_12
triton_per_fused_add_mul_native_layer_norm_pow_tanh_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = tl.math.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = tmp15 - tmp25
    tmp33 = 128.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp42, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgqorevr3lqoljgg45cwknzav7fadrsoyy4litniublte7nsndp.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_12, exp_12, sub_39, sum_13
triton_red_fused__log_softmax_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 30000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnx4shoa5w7qeqcpaxlfd3wyqbcoejtdmhegfwtts2wh2j2uqlbx.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_24, full_default_2, ne_1, ne_2, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_forward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
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
    tmp5 = tmp4 + 30000
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30000), "index out of bounds: 0 <= tmp7 < 30000")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (30000*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30000, 128), (128, 1))
    assert_size_stride(arg1_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (4096, 128), (128, 1))
    assert_size_stride(arg6_1, (4096, ), (1, ))
    assert_size_stride(arg7_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg8_1, (4096, ), (1, ))
    assert_size_stride(arg9_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg10_1, (4096, ), (1, ))
    assert_size_stride(arg11_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg12_1, (4096, ), (1, ))
    assert_size_stride(arg13_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg18_1, (16384, ), (1, ))
    assert_size_stride(arg19_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg20_1, (4096, ), (1, ))
    assert_size_stride(arg21_1, (4096, ), (1, ))
    assert_size_stride(arg22_1, (4096, ), (1, ))
    assert_size_stride(arg23_1, (128, 4096), (4096, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (30000, 128), (128, 1))
    assert_size_stride(arg28_1, (30000, ), (1, ))
    assert_size_stride(arg29_1, (1, 512), (512, 1))
    assert_size_stride(arg30_1, (1, 512), (512, 1))
    assert_size_stride(arg31_1, (1, 512), (512, 1))
    assert_size_stride(arg32_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg31_1, arg0_1, arg29_1, arg1_1, arg30_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 512, 128, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg29_1
        del arg2_1
        del arg30_1
        del arg31_1
        del arg3_1
        del arg4_1
        buf5 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (512, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 4096), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg5_1
        del arg6_1
        buf6 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf6)
        buf7 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf7)
        buf8 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf8)
        buf9 = empty((1, 64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg8_1, buf9, 2097152, grid=grid(2097152), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg10_1, buf10, 2097152, grid=grid(2097152), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf8, arg12_1, buf11, 2097152, grid=grid(2097152), stream=stream0)
        del buf8
        # Source Nodes: [], Original ATen: []
        buf12 = aten._scaled_dot_product_efficient_attention(buf9, buf10, buf11, None, False, scale=0.125)
        buf13 = buf12[0]
        del buf12
        buf17 = reinterpret_tensor(buf9, (512, 4096), (4096, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf17)
        buf21 = reinterpret_tensor(buf13, (1, 512, 4096), (2097152, 4096, 1), 0); del buf13  # reuse
        # Source Nodes: [add_2, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf5, buf17, arg14_1, arg15_1, arg16_1, buf21, 512, 4096, grid=grid(512), stream=stream0)
        buf22 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf22)
        buf23 = reinterpret_tensor(buf22, (1, 512, 16384), (8388608, 16384, 1), 0); del buf22  # reuse
        # Source Nodes: [add_3, add_4, ffn_output_1, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf23, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf24)
        buf28 = reinterpret_tensor(buf17, (1, 512, 4096), (2097152, 4096, 1), 0); del buf17  # reuse
        # Source Nodes: [add_5, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf24, arg20_1, buf21, arg21_1, arg22_1, buf28, 512, 4096, grid=grid(512), stream=stream0)
        buf29 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf29)
        buf30 = reinterpret_tensor(buf21, (512, 4096), (4096, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf30)
        buf31 = reinterpret_tensor(buf11, (512, 4096), (4096, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf31)
        buf32 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf29, arg8_1, buf32, 2097152, grid=grid(2097152), stream=stream0)
        buf33 = reinterpret_tensor(buf29, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf30, arg10_1, buf33, 2097152, grid=grid(2097152), stream=stream0)
        buf34 = reinterpret_tensor(buf30, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf31, arg12_1, buf34, 2097152, grid=grid(2097152), stream=stream0)
        del buf31
        # Source Nodes: [], Original ATen: []
        buf35 = aten._scaled_dot_product_efficient_attention(buf32, buf33, buf34, None, False, scale=0.125)
        buf36 = buf35[0]
        del buf35
        buf40 = reinterpret_tensor(buf34, (512, 4096), (4096, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf40)
        buf44 = reinterpret_tensor(buf36, (1, 512, 4096), (2097152, 4096, 1), 0); del buf36  # reuse
        # Source Nodes: [add_7, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf28, buf40, arg14_1, arg15_1, arg16_1, buf44, 512, 4096, grid=grid(512), stream=stream0)
        buf45 = reinterpret_tensor(buf23, (512, 16384), (16384, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf45)
        buf46 = reinterpret_tensor(buf45, (1, 512, 16384), (8388608, 16384, 1), 0); del buf45  # reuse
        # Source Nodes: [add_8, add_9, ffn_output_5, mul_5, mul_6, mul_7, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf46, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf47)
        buf51 = buf28; del buf28  # reuse
        # Source Nodes: [add_10, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf47, arg20_1, buf44, arg21_1, arg22_1, buf51, 512, 4096, grid=grid(512), stream=stream0)
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf52)
        buf53 = reinterpret_tensor(buf44, (512, 4096), (4096, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf53)
        buf54 = reinterpret_tensor(buf33, (512, 4096), (4096, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf54)
        buf55 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg8_1, buf55, 2097152, grid=grid(2097152), stream=stream0)
        buf56 = reinterpret_tensor(buf52, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf53, arg10_1, buf56, 2097152, grid=grid(2097152), stream=stream0)
        buf57 = reinterpret_tensor(buf53, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf54, arg12_1, buf57, 2097152, grid=grid(2097152), stream=stream0)
        del buf54
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, False, scale=0.125)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf57, (512, 4096), (4096, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf63)
        buf67 = reinterpret_tensor(buf59, (1, 512, 4096), (2097152, 4096, 1), 0); del buf59  # reuse
        # Source Nodes: [add_12, layernormed_context_layer_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf51, buf63, arg14_1, arg15_1, arg16_1, buf67, 512, 4096, grid=grid(512), stream=stream0)
        buf68 = reinterpret_tensor(buf46, (512, 16384), (16384, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf68)
        buf69 = reinterpret_tensor(buf68, (1, 512, 16384), (8388608, 16384, 1), 0); del buf68  # reuse
        # Source Nodes: [add_13, add_14, ffn_output_9, mul_10, mul_11, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf69, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf70 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf70)
        buf74 = buf51; del buf51  # reuse
        # Source Nodes: [add_15, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf70, arg20_1, buf67, arg21_1, arg22_1, buf74, 512, 4096, grid=grid(512), stream=stream0)
        buf75 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf75)
        buf76 = reinterpret_tensor(buf67, (512, 4096), (4096, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf76)
        buf77 = reinterpret_tensor(buf56, (512, 4096), (4096, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf77)
        buf78 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf75, arg8_1, buf78, 2097152, grid=grid(2097152), stream=stream0)
        buf79 = reinterpret_tensor(buf75, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg10_1, buf79, 2097152, grid=grid(2097152), stream=stream0)
        buf80 = reinterpret_tensor(buf76, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf77, arg12_1, buf80, 2097152, grid=grid(2097152), stream=stream0)
        del buf77
        # Source Nodes: [], Original ATen: []
        buf81 = aten._scaled_dot_product_efficient_attention(buf78, buf79, buf80, None, False, scale=0.125)
        buf82 = buf81[0]
        del buf81
        buf86 = reinterpret_tensor(buf80, (512, 4096), (4096, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf86)
        buf90 = reinterpret_tensor(buf82, (1, 512, 4096), (2097152, 4096, 1), 0); del buf82  # reuse
        # Source Nodes: [add_17, layernormed_context_layer_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf74, buf86, arg14_1, arg15_1, arg16_1, buf90, 512, 4096, grid=grid(512), stream=stream0)
        buf91 = reinterpret_tensor(buf69, (512, 16384), (16384, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf91)
        buf92 = reinterpret_tensor(buf91, (1, 512, 16384), (8388608, 16384, 1), 0); del buf91  # reuse
        # Source Nodes: [add_18, add_19, ffn_output_13, mul_13, mul_14, mul_15, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf92, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf93 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf93)
        buf97 = buf74; del buf74  # reuse
        # Source Nodes: [add_20, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf93, arg20_1, buf90, arg21_1, arg22_1, buf97, 512, 4096, grid=grid(512), stream=stream0)
        buf98 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf98)
        buf99 = reinterpret_tensor(buf90, (512, 4096), (4096, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf99)
        buf100 = reinterpret_tensor(buf79, (512, 4096), (4096, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf100)
        buf101 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf98, arg8_1, buf101, 2097152, grid=grid(2097152), stream=stream0)
        buf102 = reinterpret_tensor(buf98, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf99, arg10_1, buf102, 2097152, grid=grid(2097152), stream=stream0)
        buf103 = reinterpret_tensor(buf99, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg12_1, buf103, 2097152, grid=grid(2097152), stream=stream0)
        del buf100
        # Source Nodes: [], Original ATen: []
        buf104 = aten._scaled_dot_product_efficient_attention(buf101, buf102, buf103, None, False, scale=0.125)
        buf105 = buf104[0]
        del buf104
        buf109 = reinterpret_tensor(buf103, (512, 4096), (4096, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf109)
        buf113 = reinterpret_tensor(buf105, (1, 512, 4096), (2097152, 4096, 1), 0); del buf105  # reuse
        # Source Nodes: [add_22, layernormed_context_layer_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf97, buf109, arg14_1, arg15_1, arg16_1, buf113, 512, 4096, grid=grid(512), stream=stream0)
        buf114 = reinterpret_tensor(buf92, (512, 16384), (16384, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf114)
        buf115 = reinterpret_tensor(buf114, (1, 512, 16384), (8388608, 16384, 1), 0); del buf114  # reuse
        # Source Nodes: [add_23, add_24, ffn_output_17, mul_17, mul_18, mul_19, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf115, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf116 = reinterpret_tensor(buf97, (512, 4096), (4096, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf116)
        buf120 = reinterpret_tensor(buf109, (1, 512, 4096), (2097152, 4096, 1), 0); del buf109  # reuse
        # Source Nodes: [add_25, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf116, arg20_1, buf113, arg21_1, arg22_1, buf120, 512, 4096, grid=grid(512), stream=stream0)
        buf121 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf121)
        buf122 = reinterpret_tensor(buf113, (512, 4096), (4096, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf122)
        buf123 = reinterpret_tensor(buf102, (512, 4096), (4096, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf123)
        buf124 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf121, arg8_1, buf124, 2097152, grid=grid(2097152), stream=stream0)
        buf125 = reinterpret_tensor(buf121, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf122, arg10_1, buf125, 2097152, grid=grid(2097152), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf123, arg12_1, buf126, 2097152, grid=grid(2097152), stream=stream0)
        del buf123
        # Source Nodes: [], Original ATen: []
        buf127 = aten._scaled_dot_product_efficient_attention(buf124, buf125, buf126, None, False, scale=0.125)
        buf128 = buf127[0]
        del buf127
        buf132 = reinterpret_tensor(buf126, (512, 4096), (4096, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf132)
        buf136 = reinterpret_tensor(buf128, (1, 512, 4096), (2097152, 4096, 1), 0); del buf128  # reuse
        # Source Nodes: [add_27, layernormed_context_layer_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf120, buf132, arg14_1, arg15_1, arg16_1, buf136, 512, 4096, grid=grid(512), stream=stream0)
        buf137 = reinterpret_tensor(buf115, (512, 16384), (16384, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf137)
        buf138 = reinterpret_tensor(buf137, (1, 512, 16384), (8388608, 16384, 1), 0); del buf137  # reuse
        # Source Nodes: [add_28, add_29, ffn_output_21, mul_21, mul_22, mul_23, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf138, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf139 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf139)
        buf143 = buf120; del buf120  # reuse
        # Source Nodes: [add_30, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf139, arg20_1, buf136, arg21_1, arg22_1, buf143, 512, 4096, grid=grid(512), stream=stream0)
        buf144 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf144)
        buf145 = reinterpret_tensor(buf136, (512, 4096), (4096, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf145)
        buf146 = reinterpret_tensor(buf125, (512, 4096), (4096, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf146)
        buf147 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf144, arg8_1, buf147, 2097152, grid=grid(2097152), stream=stream0)
        buf148 = reinterpret_tensor(buf144, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf145, arg10_1, buf148, 2097152, grid=grid(2097152), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf146, arg12_1, buf149, 2097152, grid=grid(2097152), stream=stream0)
        del buf146
        # Source Nodes: [], Original ATen: []
        buf150 = aten._scaled_dot_product_efficient_attention(buf147, buf148, buf149, None, False, scale=0.125)
        buf151 = buf150[0]
        del buf150
        buf155 = reinterpret_tensor(buf149, (512, 4096), (4096, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf155)
        buf159 = reinterpret_tensor(buf151, (1, 512, 4096), (2097152, 4096, 1), 0); del buf151  # reuse
        # Source Nodes: [add_32, layernormed_context_layer_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf143, buf155, arg14_1, arg15_1, arg16_1, buf159, 512, 4096, grid=grid(512), stream=stream0)
        buf160 = reinterpret_tensor(buf138, (512, 16384), (16384, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf160)
        buf161 = reinterpret_tensor(buf160, (1, 512, 16384), (8388608, 16384, 1), 0); del buf160  # reuse
        # Source Nodes: [add_33, add_34, ffn_output_25, mul_25, mul_26, mul_27, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf161, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf162 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf162)
        buf166 = buf143; del buf143  # reuse
        # Source Nodes: [add_35, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf162, arg20_1, buf159, arg21_1, arg22_1, buf166, 512, 4096, grid=grid(512), stream=stream0)
        buf167 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf167)
        buf168 = reinterpret_tensor(buf159, (512, 4096), (4096, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf168)
        buf169 = reinterpret_tensor(buf148, (512, 4096), (4096, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf169)
        buf170 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf167, arg8_1, buf170, 2097152, grid=grid(2097152), stream=stream0)
        buf171 = reinterpret_tensor(buf167, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf168, arg10_1, buf171, 2097152, grid=grid(2097152), stream=stream0)
        buf172 = reinterpret_tensor(buf168, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf169, arg12_1, buf172, 2097152, grid=grid(2097152), stream=stream0)
        del buf169
        # Source Nodes: [], Original ATen: []
        buf173 = aten._scaled_dot_product_efficient_attention(buf170, buf171, buf172, None, False, scale=0.125)
        buf174 = buf173[0]
        del buf173
        buf178 = reinterpret_tensor(buf172, (512, 4096), (4096, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf178)
        buf182 = reinterpret_tensor(buf174, (1, 512, 4096), (2097152, 4096, 1), 0); del buf174  # reuse
        # Source Nodes: [add_37, layernormed_context_layer_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf166, buf178, arg14_1, arg15_1, arg16_1, buf182, 512, 4096, grid=grid(512), stream=stream0)
        buf183 = reinterpret_tensor(buf161, (512, 16384), (16384, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf183)
        buf184 = reinterpret_tensor(buf183, (1, 512, 16384), (8388608, 16384, 1), 0); del buf183  # reuse
        # Source Nodes: [add_38, add_39, ffn_output_29, mul_29, mul_30, mul_31, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf184, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf185 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf185)
        buf189 = buf166; del buf166  # reuse
        # Source Nodes: [add_40, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf185, arg20_1, buf182, arg21_1, arg22_1, buf189, 512, 4096, grid=grid(512), stream=stream0)
        buf190 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf190)
        buf191 = reinterpret_tensor(buf182, (512, 4096), (4096, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf191)
        buf192 = reinterpret_tensor(buf171, (512, 4096), (4096, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf192)
        buf193 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf190, arg8_1, buf193, 2097152, grid=grid(2097152), stream=stream0)
        buf194 = reinterpret_tensor(buf190, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf191, arg10_1, buf194, 2097152, grid=grid(2097152), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf192, arg12_1, buf195, 2097152, grid=grid(2097152), stream=stream0)
        del buf192
        # Source Nodes: [], Original ATen: []
        buf196 = aten._scaled_dot_product_efficient_attention(buf193, buf194, buf195, None, False, scale=0.125)
        buf197 = buf196[0]
        del buf196
        buf201 = reinterpret_tensor(buf195, (512, 4096), (4096, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf201)
        buf205 = reinterpret_tensor(buf197, (1, 512, 4096), (2097152, 4096, 1), 0); del buf197  # reuse
        # Source Nodes: [add_42, layernormed_context_layer_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf189, buf201, arg14_1, arg15_1, arg16_1, buf205, 512, 4096, grid=grid(512), stream=stream0)
        buf206 = reinterpret_tensor(buf184, (512, 16384), (16384, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf206)
        buf207 = reinterpret_tensor(buf206, (1, 512, 16384), (8388608, 16384, 1), 0); del buf206  # reuse
        # Source Nodes: [add_43, add_44, ffn_output_33, mul_33, mul_34, mul_35, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf207, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf208 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf208)
        buf212 = buf189; del buf189  # reuse
        # Source Nodes: [add_45, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf208, arg20_1, buf205, arg21_1, arg22_1, buf212, 512, 4096, grid=grid(512), stream=stream0)
        buf213 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf213)
        buf214 = reinterpret_tensor(buf205, (512, 4096), (4096, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf214)
        buf215 = reinterpret_tensor(buf194, (512, 4096), (4096, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf215)
        buf216 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf213, arg8_1, buf216, 2097152, grid=grid(2097152), stream=stream0)
        buf217 = reinterpret_tensor(buf213, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf214, arg10_1, buf217, 2097152, grid=grid(2097152), stream=stream0)
        buf218 = reinterpret_tensor(buf214, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf215, arg12_1, buf218, 2097152, grid=grid(2097152), stream=stream0)
        del buf215
        # Source Nodes: [], Original ATen: []
        buf219 = aten._scaled_dot_product_efficient_attention(buf216, buf217, buf218, None, False, scale=0.125)
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf218, (512, 4096), (4096, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf224)
        buf228 = reinterpret_tensor(buf220, (1, 512, 4096), (2097152, 4096, 1), 0); del buf220  # reuse
        # Source Nodes: [add_47, layernormed_context_layer_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf212, buf224, arg14_1, arg15_1, arg16_1, buf228, 512, 4096, grid=grid(512), stream=stream0)
        buf229 = reinterpret_tensor(buf207, (512, 16384), (16384, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf229)
        buf230 = reinterpret_tensor(buf229, (1, 512, 16384), (8388608, 16384, 1), 0); del buf229  # reuse
        # Source Nodes: [add_48, add_49, ffn_output_37, mul_37, mul_38, mul_39, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf230, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf231 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf231)
        buf235 = buf212; del buf212  # reuse
        # Source Nodes: [add_50, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf231, arg20_1, buf228, arg21_1, arg22_1, buf235, 512, 4096, grid=grid(512), stream=stream0)
        buf236 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf236)
        buf237 = reinterpret_tensor(buf228, (512, 4096), (4096, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf237)
        buf238 = reinterpret_tensor(buf217, (512, 4096), (4096, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf238)
        buf239 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf236, arg8_1, buf239, 2097152, grid=grid(2097152), stream=stream0)
        buf240 = reinterpret_tensor(buf236, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf237, arg10_1, buf240, 2097152, grid=grid(2097152), stream=stream0)
        buf241 = reinterpret_tensor(buf237, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf238, arg12_1, buf241, 2097152, grid=grid(2097152), stream=stream0)
        del buf238
        # Source Nodes: [], Original ATen: []
        buf242 = aten._scaled_dot_product_efficient_attention(buf239, buf240, buf241, None, False, scale=0.125)
        buf243 = buf242[0]
        del buf242
        buf247 = reinterpret_tensor(buf241, (512, 4096), (4096, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf247)
        buf251 = reinterpret_tensor(buf243, (1, 512, 4096), (2097152, 4096, 1), 0); del buf243  # reuse
        # Source Nodes: [add_52, layernormed_context_layer_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf235, buf247, arg14_1, arg15_1, arg16_1, buf251, 512, 4096, grid=grid(512), stream=stream0)
        buf252 = reinterpret_tensor(buf230, (512, 16384), (16384, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf252)
        buf253 = reinterpret_tensor(buf252, (1, 512, 16384), (8388608, 16384, 1), 0); del buf252  # reuse
        # Source Nodes: [add_53, add_54, ffn_output_41, mul_41, mul_42, mul_43, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf253, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        buf254 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf254)
        buf258 = buf235; del buf235  # reuse
        # Source Nodes: [add_55, hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf254, arg20_1, buf251, arg21_1, arg22_1, buf258, 512, 4096, grid=grid(512), stream=stream0)
        buf259 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf259)
        del arg7_1
        buf260 = reinterpret_tensor(buf251, (512, 4096), (4096, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), out=buf260)
        del arg9_1
        buf261 = reinterpret_tensor(buf240, (512, 4096), (4096, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), out=buf261)
        del arg11_1
        buf262 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf259, arg8_1, buf262, 2097152, grid=grid(2097152), stream=stream0)
        del arg8_1
        buf263 = reinterpret_tensor(buf259, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf260, arg10_1, buf263, 2097152, grid=grid(2097152), stream=stream0)
        del arg10_1
        buf264 = reinterpret_tensor(buf260, (1, 64, 512, 64), (2097152, 32768, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf261, arg12_1, buf264, 2097152, grid=grid(2097152), stream=stream0)
        del arg12_1
        del buf261
        # Source Nodes: [], Original ATen: []
        buf265 = aten._scaled_dot_product_efficient_attention(buf262, buf263, buf264, None, False, scale=0.125)
        del buf262
        del buf263
        buf266 = buf265[0]
        del buf265
        buf270 = reinterpret_tensor(buf264, (512, 4096), (4096, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf270)
        del arg13_1
        buf274 = reinterpret_tensor(buf266, (1, 512, 4096), (2097152, 4096, 1), 0); del buf266  # reuse
        # Source Nodes: [add_57, layernormed_context_layer_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf258, buf270, arg14_1, arg15_1, arg16_1, buf274, 512, 4096, grid=grid(512), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf275 = reinterpret_tensor(buf253, (512, 16384), (16384, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), out=buf275)
        del arg17_1
        buf276 = reinterpret_tensor(buf275, (1, 512, 16384), (8388608, 16384, 1), 0); del buf275  # reuse
        # Source Nodes: [add_58, add_59, ffn_output_45, mul_45, mul_46, mul_47, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf276, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg18_1
        buf277 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), out=buf277)
        del arg19_1
        del buf276
        buf281 = buf258; del buf258  # reuse
        # Source Nodes: [add_60, sequence_outputs], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf277, arg20_1, buf274, arg21_1, arg22_1, buf281, 512, 4096, grid=grid(512), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        del buf274
        del buf277
        buf282 = reinterpret_tensor(buf4, (512, 128), (128, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg23_1, (4096, 128), (1, 4096), 0), out=buf282)
        del arg23_1
        del buf281
        buf286 = buf0; del buf0  # reuse
        # Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_5.run(buf282, arg24_1, arg25_1, arg26_1, buf286, 512, 128, grid=grid(512), stream=stream0)
        del arg24_1
        del arg25_1
        del arg26_1
        del buf282
        buf287 = empty((512, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf286, (512, 128), (128, 1), 0), reinterpret_tensor(arg27_1, (128, 30000), (1, 128), 0), alpha=1, beta=1, out=buf287)
        del arg27_1
        del arg28_1
        del buf286
        buf288 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        buf289 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_6.run(buf287, buf288, buf289, 512, 30000, grid=grid(512), stream=stream0)
        buf290 = empty((), device='cuda', dtype=torch.float32)
        buf292 = buf290; del buf290  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_7.run(buf292, arg32_1, buf287, buf288, buf289, 1, 512, grid=grid(1), stream=stream0)
        del arg32_1
        return (buf292, reinterpret_tensor(buf287, (1, 512, 30000), (15360000, 30000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((30000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg30_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg31_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg32_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForMaskedLM', benchmark_compiled_module)
