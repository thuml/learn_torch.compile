
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


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bkde6yget6jb6rtwui6pdfozxpnwdcacak7t46hhw5lyudluc4.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, embeddings_3, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
# embeddings_3 => view
# inputs_embeds => embedding
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32000
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 32000)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32000")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 4
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 4)) | ~xmask, "index out of bounds: 0 <= tmp8 < 4")
    tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert(((0 <= tmp14) & (tmp14 < 512)) | ~xmask, "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r1 + (768*tmp14)), rmask & xmask, other=0.0)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-12
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp43, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp44, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/g3/cg33v5vnrywyhtj6krjvhk7p5kzwpkthcqhgntzbjoffnqc5t6ra.py
# Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_4
# fourier_output => var_mean_1
triton_red_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((2*r1) + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tl.store(out_ptr2 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5oobkgzdtxo5ieelrkvyxfirhesqj3vowy6zuibgrf5c7fd7lm.py
# Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_1 => add_4
# fourier_output => add_5, rsqrt_1, var_mean_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-12
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3iz6b56utz6wmfzrw76rk7xro5foa3xal5h3b7jvolvbtc6i6t3.py
# Source Nodes: [add_1, fourier_output, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# add_1 => add_4
# fourier_output => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# hidden_states_1 => view_2
triton_poi_fused_add_native_layer_norm_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (2*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-12
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvig7una3bkyfkcpg7ou7rxihyrnqezyg6x7uqfbdqelgtpuwpk.py
# Source Nodes: [add_2, add_3, hidden_states_3, intermediate_output, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
# add_2 => add_7
# add_3 => add_8
# hidden_states_3 => view_4
# intermediate_output => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
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
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffkcqbcnfatlrw2ddtmup56m5quvkktm6wniddof5kkbnjuxtjm.py
# Source Nodes: [add_4, fourier_output, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_4 => add_9
# fourier_output => add_6, mul_3
# hidden_states_6 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznh4ff2mxhwbjo2k2eo72raeeomu7jovaj6qs6o4wka2prx7ly2.py
# Source Nodes: [add_49, add_50, hidden_states_85, hidden_states_87, mul_48, mul_49, mul_50, pow_13, prediction_scores, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
# add_49 => add_100
# add_50 => add_101
# hidden_states_85 => mul_101
# hidden_states_87 => add_102, add_103, mul_102, mul_103, rsqrt_25, sub_25, var_mean_25
# mul_48 => mul_98
# mul_49 => mul_99
# mul_50 => mul_100
# pow_13 => pow_13
# prediction_scores => view_52
# tanh_12 => tanh_13
triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp37 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 768, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 768.0
    tmp31 = tmp29 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp13 - tmp23
    tmp36 = tmp35 * tmp34
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp40, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcx2vg5v3erhbhrs6bkwsiidy5scfwusv25nfjmukmlzynwgqfz.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax, exp, log, sub_26, sub_27, sum_1
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp10 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (32000*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2v37lzyriqs2ygvj5jfhiw6xiuedha5asteftyzkrzcphv6noag.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_12, div, full_default_1, ne, neg, sum_2, sum_3, where_1
triton_per_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 32000
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 32000), "index out of bounds: 0 <= tmp12 < 32000")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115 = args
    args.clear()
    assert_size_stride(primals_1, (32000, 768), (768, 1))
    assert_size_stride(primals_2, (4, 768), (768, 1))
    assert_size_stride(primals_3, (512, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (3072, 768), (768, 1))
    assert_size_stride(primals_11, (3072, ), (1, ))
    assert_size_stride(primals_12, (768, 3072), (3072, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (3072, 768), (768, 1))
    assert_size_stride(primals_19, (3072, ), (1, ))
    assert_size_stride(primals_20, (768, 3072), (3072, 1))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (3072, 768), (768, 1))
    assert_size_stride(primals_27, (3072, ), (1, ))
    assert_size_stride(primals_28, (768, 3072), (3072, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (3072, 768), (768, 1))
    assert_size_stride(primals_35, (3072, ), (1, ))
    assert_size_stride(primals_36, (768, 3072), (3072, 1))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (3072, 768), (768, 1))
    assert_size_stride(primals_43, (3072, ), (1, ))
    assert_size_stride(primals_44, (768, 3072), (3072, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (3072, 768), (768, 1))
    assert_size_stride(primals_51, (3072, ), (1, ))
    assert_size_stride(primals_52, (768, 3072), (3072, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (3072, 768), (768, 1))
    assert_size_stride(primals_59, (3072, ), (1, ))
    assert_size_stride(primals_60, (768, 3072), (3072, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (3072, 768), (768, 1))
    assert_size_stride(primals_67, (3072, ), (1, ))
    assert_size_stride(primals_68, (768, 3072), (3072, 1))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (3072, 768), (768, 1))
    assert_size_stride(primals_75, (3072, ), (1, ))
    assert_size_stride(primals_76, (768, 3072), (3072, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (3072, 768), (768, 1))
    assert_size_stride(primals_83, (3072, ), (1, ))
    assert_size_stride(primals_84, (768, 3072), (3072, 1))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (3072, 768), (768, 1))
    assert_size_stride(primals_91, (3072, ), (1, ))
    assert_size_stride(primals_92, (768, 3072), (3072, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (3072, 768), (768, 1))
    assert_size_stride(primals_99, (3072, ), (1, ))
    assert_size_stride(primals_100, (768, 3072), (3072, 1))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, 768), (768, 1))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (32000, 768), (768, 1))
    assert_size_stride(primals_111, (32000, ), (1, ))
    assert_size_stride(primals_112, (1, 512), (512, 1))
    assert_size_stride(primals_113, (1, 512), (512, 1))
    assert_size_stride(primals_114, (1, 512), (512, 1))
    assert_size_stride(primals_115, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf360 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, embeddings_3, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0.run(primals_114, primals_1, primals_112, primals_2, primals_113, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf360, 512, 768, grid=grid(512), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        buf6 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [embeddings_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf5, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del primals_7
        # Source Nodes: [embedding_output], Original ATen: [aten.native_dropout]
        buf7 = aten.native_dropout(reinterpret_tensor(buf6, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        # Source Nodes: [fft_fftn], Original ATen: [aten._to_copy]
        buf10 = torch.ops.prims.convert_element_type.default(buf8, torch.complex64)
        buf11 = buf10
        del buf10
        # Source Nodes: [fft_fftn], Original ATen: [aten._fft_c2c]
        buf12 = aten._fft_c2c(buf11, [1, 2], 0, True)
        del buf11
        buf13 = buf12
        del buf12
        # Source Nodes: [outputs], Original ATen: [aten.view_as_real]
        buf14 = aten.view_as_real(buf13)
        del buf13
        buf15 = buf14
        del buf14
        buf16 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf8, buf15, buf16, buf17, buf18, 3072, 128, grid=grid(3072), stream=stream0)
        buf19 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf359 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf16, buf17, buf18, buf19, buf20, buf359, 512, 6, grid=grid(512), stream=stream0)
        buf22 = buf8; del buf8  # reuse
        buf23 = buf6; del buf6  # reuse
        # Source Nodes: [add_1, fourier_output, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf22, buf15, buf19, buf20, primals_8, primals_9, buf23, 393216, grid=grid(393216), stream=stream0)
        del buf15
        buf24 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf23, reinterpret_tensor(primals_10, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf24)
        del primals_11
        buf25 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf26 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, hidden_states_3, intermediate_output, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf24, buf25, buf26, 1572864, grid=grid(1572864), stream=stream0)
        buf27 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf26, reinterpret_tensor(primals_12, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf27)
        del primals_13
        # Source Nodes: [hidden_states_4], Original ATen: [aten.native_dropout]
        buf28 = aten.native_dropout(reinterpret_tensor(buf27, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf34 = reinterpret_tensor(buf27, (1, 512, 768), (393216, 768, 1), 0); del buf27  # reuse
        buf35 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf358 = reinterpret_tensor(buf20, (1, 512, 1), (512, 1, 1), 0); del buf20  # reuse
        # Source Nodes: [add_4, fourier_output, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf29, buf22, primals_8, primals_9, primals_14, primals_15, buf34, buf35, buf358, 512, 768, grid=grid(512), stream=stream0)
        del primals_15
        del primals_9
        # Source Nodes: [fft_fftn_1, hidden_states_6], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf36 = torch.ops.prims.convert_element_type.default(buf35, torch.complex64)
        buf37 = buf36
        del buf36
        # Source Nodes: [fft_fftn_1], Original ATen: [aten._fft_c2c]
        buf38 = aten._fft_c2c(buf37, [1, 2], 0, True)
        del buf37
        buf39 = buf38
        del buf38
        # Source Nodes: [outputs_1], Original ATen: [aten.view_as_real]
        buf40 = aten.view_as_real(buf39)
        del buf39
        buf41 = buf40
        del buf40
        buf42 = buf18; del buf18  # reuse
        buf43 = buf17; del buf17  # reuse
        buf44 = buf16; del buf16  # reuse
        # Source Nodes: [add_5, fourier_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf35, buf41, buf42, buf43, buf44, 3072, 128, grid=grid(3072), stream=stream0)
        buf45 = buf19; del buf19  # reuse
        buf46 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf357 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, fourier_output_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf42, buf43, buf44, buf45, buf46, buf357, 512, 6, grid=grid(512), stream=stream0)
        buf48 = buf35; del buf35  # reuse
        buf49 = reinterpret_tensor(buf29, (512, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [add_5, fourier_output_2, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf48, buf41, buf45, buf46, primals_16, primals_17, buf49, 393216, grid=grid(393216), stream=stream0)
        del buf41
        buf50 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf49, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf50)
        del primals_19
        buf51 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf52 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7, hidden_states_10, intermediate_output_1, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf50, buf51, buf52, 1572864, grid=grid(1572864), stream=stream0)
        buf53 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, buf52, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf53)
        del primals_21
        # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
        buf54 = aten.native_dropout(reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf60 = reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
        buf61 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf356 = reinterpret_tensor(buf46, (1, 512, 1), (512, 1, 1), 0); del buf46  # reuse
        # Source Nodes: [add_8, fourier_output_2, hidden_states_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf55, buf48, primals_16, primals_17, primals_22, primals_23, buf60, buf61, buf356, 512, 768, grid=grid(512), stream=stream0)
        del primals_17
        del primals_23
        # Source Nodes: [fft_fftn_2, hidden_states_13], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf62 = torch.ops.prims.convert_element_type.default(buf61, torch.complex64)
        buf63 = buf62
        del buf62
        # Source Nodes: [fft_fftn_2], Original ATen: [aten._fft_c2c]
        buf64 = aten._fft_c2c(buf63, [1, 2], 0, True)
        del buf63
        buf65 = buf64
        del buf64
        # Source Nodes: [outputs_2], Original ATen: [aten.view_as_real]
        buf66 = aten.view_as_real(buf65)
        del buf65
        buf67 = buf66
        del buf66
        buf68 = buf44; del buf44  # reuse
        buf69 = buf43; del buf43  # reuse
        buf70 = buf42; del buf42  # reuse
        # Source Nodes: [add_9, fourier_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf61, buf67, buf68, buf69, buf70, 3072, 128, grid=grid(3072), stream=stream0)
        buf71 = buf45; del buf45  # reuse
        buf72 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf355 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, fourier_output_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf68, buf69, buf70, buf71, buf72, buf355, 512, 6, grid=grid(512), stream=stream0)
        buf74 = buf61; del buf61  # reuse
        buf75 = reinterpret_tensor(buf55, (512, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [add_9, fourier_output_4, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf74, buf67, buf71, buf72, primals_24, primals_25, buf75, 393216, grid=grid(393216), stream=stream0)
        del buf67
        buf76 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, buf75, reinterpret_tensor(primals_26, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf76)
        del primals_27
        buf77 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf78 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_11, hidden_states_17, intermediate_output_2, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf76, buf77, buf78, 1572864, grid=grid(1572864), stream=stream0)
        buf79 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf78, reinterpret_tensor(primals_28, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf79)
        del primals_29
        # Source Nodes: [hidden_states_18], Original ATen: [aten.native_dropout]
        buf80 = aten.native_dropout(reinterpret_tensor(buf79, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf81 = buf80[0]
        buf82 = buf80[1]
        del buf80
        buf86 = reinterpret_tensor(buf79, (1, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
        buf87 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf354 = reinterpret_tensor(buf72, (1, 512, 1), (512, 1, 1), 0); del buf72  # reuse
        # Source Nodes: [add_12, fourier_output_4, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf81, buf74, primals_24, primals_25, primals_30, primals_31, buf86, buf87, buf354, 512, 768, grid=grid(512), stream=stream0)
        del primals_25
        del primals_31
        # Source Nodes: [fft_fftn_3, hidden_states_20], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf88 = torch.ops.prims.convert_element_type.default(buf87, torch.complex64)
        buf89 = buf88
        del buf88
        # Source Nodes: [fft_fftn_3], Original ATen: [aten._fft_c2c]
        buf90 = aten._fft_c2c(buf89, [1, 2], 0, True)
        del buf89
        buf91 = buf90
        del buf90
        # Source Nodes: [outputs_3], Original ATen: [aten.view_as_real]
        buf92 = aten.view_as_real(buf91)
        del buf91
        buf93 = buf92
        del buf92
        buf94 = buf70; del buf70  # reuse
        buf95 = buf69; del buf69  # reuse
        buf96 = buf68; del buf68  # reuse
        # Source Nodes: [add_13, fourier_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf87, buf93, buf94, buf95, buf96, 3072, 128, grid=grid(3072), stream=stream0)
        buf97 = buf71; del buf71  # reuse
        buf98 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf353 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, fourier_output_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf94, buf95, buf96, buf97, buf98, buf353, 512, 6, grid=grid(512), stream=stream0)
        buf100 = buf87; del buf87  # reuse
        buf101 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [add_13, fourier_output_6, hidden_states_22], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf100, buf93, buf97, buf98, primals_32, primals_33, buf101, 393216, grid=grid(393216), stream=stream0)
        del buf93
        buf102 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf101, reinterpret_tensor(primals_34, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf102)
        del primals_35
        buf103 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf104 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, add_15, hidden_states_24, intermediate_output_3, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf102, buf103, buf104, 1572864, grid=grid(1572864), stream=stream0)
        buf105 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf104, reinterpret_tensor(primals_36, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf105)
        del primals_37
        # Source Nodes: [hidden_states_25], Original ATen: [aten.native_dropout]
        buf106 = aten.native_dropout(reinterpret_tensor(buf105, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf107 = buf106[0]
        buf108 = buf106[1]
        del buf106
        buf112 = reinterpret_tensor(buf105, (1, 512, 768), (393216, 768, 1), 0); del buf105  # reuse
        buf113 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf352 = reinterpret_tensor(buf98, (1, 512, 1), (512, 1, 1), 0); del buf98  # reuse
        # Source Nodes: [add_16, fourier_output_6, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf107, buf100, primals_32, primals_33, primals_38, primals_39, buf112, buf113, buf352, 512, 768, grid=grid(512), stream=stream0)
        del primals_33
        del primals_39
        # Source Nodes: [fft_fftn_4, hidden_states_27], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf114 = torch.ops.prims.convert_element_type.default(buf113, torch.complex64)
        buf115 = buf114
        del buf114
        # Source Nodes: [fft_fftn_4], Original ATen: [aten._fft_c2c]
        buf116 = aten._fft_c2c(buf115, [1, 2], 0, True)
        del buf115
        buf117 = buf116
        del buf116
        # Source Nodes: [outputs_4], Original ATen: [aten.view_as_real]
        buf118 = aten.view_as_real(buf117)
        del buf117
        buf119 = buf118
        del buf118
        buf120 = buf96; del buf96  # reuse
        buf121 = buf95; del buf95  # reuse
        buf122 = buf94; del buf94  # reuse
        # Source Nodes: [add_17, fourier_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf113, buf119, buf120, buf121, buf122, 3072, 128, grid=grid(3072), stream=stream0)
        buf123 = buf97; del buf97  # reuse
        buf124 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf351 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, fourier_output_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf120, buf121, buf122, buf123, buf124, buf351, 512, 6, grid=grid(512), stream=stream0)
        buf126 = buf113; del buf113  # reuse
        buf127 = reinterpret_tensor(buf107, (512, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [add_17, fourier_output_8, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf126, buf119, buf123, buf124, primals_40, primals_41, buf127, 393216, grid=grid(393216), stream=stream0)
        del buf119
        buf128 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, buf127, reinterpret_tensor(primals_42, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf128)
        del primals_43
        buf129 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf130 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, hidden_states_31, intermediate_output_4, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf128, buf129, buf130, 1572864, grid=grid(1572864), stream=stream0)
        buf131 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, buf130, reinterpret_tensor(primals_44, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf131)
        del primals_45
        # Source Nodes: [hidden_states_32], Original ATen: [aten.native_dropout]
        buf132 = aten.native_dropout(reinterpret_tensor(buf131, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf133 = buf132[0]
        buf134 = buf132[1]
        del buf132
        buf138 = reinterpret_tensor(buf131, (1, 512, 768), (393216, 768, 1), 0); del buf131  # reuse
        buf139 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf350 = reinterpret_tensor(buf124, (1, 512, 1), (512, 1, 1), 0); del buf124  # reuse
        # Source Nodes: [add_20, fourier_output_8, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf133, buf126, primals_40, primals_41, primals_46, primals_47, buf138, buf139, buf350, 512, 768, grid=grid(512), stream=stream0)
        del primals_41
        del primals_47
        # Source Nodes: [fft_fftn_5, hidden_states_34], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf140 = torch.ops.prims.convert_element_type.default(buf139, torch.complex64)
        buf141 = buf140
        del buf140
        # Source Nodes: [fft_fftn_5], Original ATen: [aten._fft_c2c]
        buf142 = aten._fft_c2c(buf141, [1, 2], 0, True)
        del buf141
        buf143 = buf142
        del buf142
        # Source Nodes: [outputs_5], Original ATen: [aten.view_as_real]
        buf144 = aten.view_as_real(buf143)
        del buf143
        buf145 = buf144
        del buf144
        buf146 = buf122; del buf122  # reuse
        buf147 = buf121; del buf121  # reuse
        buf148 = buf120; del buf120  # reuse
        # Source Nodes: [add_21, fourier_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf139, buf145, buf146, buf147, buf148, 3072, 128, grid=grid(3072), stream=stream0)
        buf149 = buf123; del buf123  # reuse
        buf150 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf349 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, fourier_output_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf146, buf147, buf148, buf149, buf150, buf349, 512, 6, grid=grid(512), stream=stream0)
        buf152 = buf139; del buf139  # reuse
        buf153 = reinterpret_tensor(buf133, (512, 768), (768, 1), 0); del buf133  # reuse
        # Source Nodes: [add_21, fourier_output_10, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf152, buf145, buf149, buf150, primals_48, primals_49, buf153, 393216, grid=grid(393216), stream=stream0)
        del buf145
        buf154 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, buf153, reinterpret_tensor(primals_50, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf154)
        del primals_51
        buf155 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf156 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, add_23, hidden_states_38, intermediate_output_5, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf154, buf155, buf156, 1572864, grid=grid(1572864), stream=stream0)
        buf157 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, buf156, reinterpret_tensor(primals_52, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf157)
        del primals_53
        # Source Nodes: [hidden_states_39], Original ATen: [aten.native_dropout]
        buf158 = aten.native_dropout(reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf159 = buf158[0]
        buf160 = buf158[1]
        del buf158
        buf164 = reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0); del buf157  # reuse
        buf165 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf348 = reinterpret_tensor(buf150, (1, 512, 1), (512, 1, 1), 0); del buf150  # reuse
        # Source Nodes: [add_24, fourier_output_10, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf159, buf152, primals_48, primals_49, primals_54, primals_55, buf164, buf165, buf348, 512, 768, grid=grid(512), stream=stream0)
        del primals_49
        del primals_55
        # Source Nodes: [fft_fftn_6, hidden_states_41], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf166 = torch.ops.prims.convert_element_type.default(buf165, torch.complex64)
        buf167 = buf166
        del buf166
        # Source Nodes: [fft_fftn_6], Original ATen: [aten._fft_c2c]
        buf168 = aten._fft_c2c(buf167, [1, 2], 0, True)
        del buf167
        buf169 = buf168
        del buf168
        # Source Nodes: [outputs_6], Original ATen: [aten.view_as_real]
        buf170 = aten.view_as_real(buf169)
        del buf169
        buf171 = buf170
        del buf170
        buf172 = buf148; del buf148  # reuse
        buf173 = buf147; del buf147  # reuse
        buf174 = buf146; del buf146  # reuse
        # Source Nodes: [add_25, fourier_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf165, buf171, buf172, buf173, buf174, 3072, 128, grid=grid(3072), stream=stream0)
        buf175 = buf149; del buf149  # reuse
        buf176 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf347 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, fourier_output_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf172, buf173, buf174, buf175, buf176, buf347, 512, 6, grid=grid(512), stream=stream0)
        buf178 = buf165; del buf165  # reuse
        buf179 = reinterpret_tensor(buf159, (512, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [add_25, fourier_output_12, hidden_states_43], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf178, buf171, buf175, buf176, primals_56, primals_57, buf179, 393216, grid=grid(393216), stream=stream0)
        del buf171
        buf180 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, buf179, reinterpret_tensor(primals_58, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf180)
        del primals_59
        buf181 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf182 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, add_27, hidden_states_45, intermediate_output_6, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf180, buf181, buf182, 1572864, grid=grid(1572864), stream=stream0)
        buf183 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, buf182, reinterpret_tensor(primals_60, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf183)
        del primals_61
        # Source Nodes: [hidden_states_46], Original ATen: [aten.native_dropout]
        buf184 = aten.native_dropout(reinterpret_tensor(buf183, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf190 = reinterpret_tensor(buf183, (1, 512, 768), (393216, 768, 1), 0); del buf183  # reuse
        buf191 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf346 = reinterpret_tensor(buf176, (1, 512, 1), (512, 1, 1), 0); del buf176  # reuse
        # Source Nodes: [add_28, fourier_output_12, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf185, buf178, primals_56, primals_57, primals_62, primals_63, buf190, buf191, buf346, 512, 768, grid=grid(512), stream=stream0)
        del primals_57
        del primals_63
        # Source Nodes: [fft_fftn_7, hidden_states_48], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf192 = torch.ops.prims.convert_element_type.default(buf191, torch.complex64)
        buf193 = buf192
        del buf192
        # Source Nodes: [fft_fftn_7], Original ATen: [aten._fft_c2c]
        buf194 = aten._fft_c2c(buf193, [1, 2], 0, True)
        del buf193
        buf195 = buf194
        del buf194
        # Source Nodes: [outputs_7], Original ATen: [aten.view_as_real]
        buf196 = aten.view_as_real(buf195)
        del buf195
        buf197 = buf196
        del buf196
        buf198 = buf174; del buf174  # reuse
        buf199 = buf173; del buf173  # reuse
        buf200 = buf172; del buf172  # reuse
        # Source Nodes: [add_29, fourier_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf191, buf197, buf198, buf199, buf200, 3072, 128, grid=grid(3072), stream=stream0)
        buf201 = buf175; del buf175  # reuse
        buf202 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf345 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, fourier_output_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf198, buf199, buf200, buf201, buf202, buf345, 512, 6, grid=grid(512), stream=stream0)
        buf204 = buf191; del buf191  # reuse
        buf205 = reinterpret_tensor(buf185, (512, 768), (768, 1), 0); del buf185  # reuse
        # Source Nodes: [add_29, fourier_output_14, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf204, buf197, buf201, buf202, primals_64, primals_65, buf205, 393216, grid=grid(393216), stream=stream0)
        del buf197
        buf206 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, buf205, reinterpret_tensor(primals_66, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf206)
        del primals_67
        buf207 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf208 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31, hidden_states_52, intermediate_output_7, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf206, buf207, buf208, 1572864, grid=grid(1572864), stream=stream0)
        buf209 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, buf208, reinterpret_tensor(primals_68, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf209)
        del primals_69
        # Source Nodes: [hidden_states_53], Original ATen: [aten.native_dropout]
        buf210 = aten.native_dropout(reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf216 = reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        buf217 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf344 = reinterpret_tensor(buf202, (1, 512, 1), (512, 1, 1), 0); del buf202  # reuse
        # Source Nodes: [add_32, fourier_output_14, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf211, buf204, primals_64, primals_65, primals_70, primals_71, buf216, buf217, buf344, 512, 768, grid=grid(512), stream=stream0)
        del primals_65
        del primals_71
        # Source Nodes: [fft_fftn_8, hidden_states_55], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf218 = torch.ops.prims.convert_element_type.default(buf217, torch.complex64)
        buf219 = buf218
        del buf218
        # Source Nodes: [fft_fftn_8], Original ATen: [aten._fft_c2c]
        buf220 = aten._fft_c2c(buf219, [1, 2], 0, True)
        del buf219
        buf221 = buf220
        del buf220
        # Source Nodes: [outputs_8], Original ATen: [aten.view_as_real]
        buf222 = aten.view_as_real(buf221)
        del buf221
        buf223 = buf222
        del buf222
        buf224 = buf200; del buf200  # reuse
        buf225 = buf199; del buf199  # reuse
        buf226 = buf198; del buf198  # reuse
        # Source Nodes: [add_33, fourier_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf217, buf223, buf224, buf225, buf226, 3072, 128, grid=grid(3072), stream=stream0)
        buf227 = buf201; del buf201  # reuse
        buf228 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf343 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, fourier_output_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf224, buf225, buf226, buf227, buf228, buf343, 512, 6, grid=grid(512), stream=stream0)
        buf230 = buf217; del buf217  # reuse
        buf231 = reinterpret_tensor(buf211, (512, 768), (768, 1), 0); del buf211  # reuse
        # Source Nodes: [add_33, fourier_output_16, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf230, buf223, buf227, buf228, primals_72, primals_73, buf231, 393216, grid=grid(393216), stream=stream0)
        del buf223
        buf232 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, buf231, reinterpret_tensor(primals_74, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf232)
        del primals_75
        buf233 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf234 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, add_35, hidden_states_59, intermediate_output_8, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf232, buf233, buf234, 1572864, grid=grid(1572864), stream=stream0)
        buf235 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf234, reinterpret_tensor(primals_76, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf235)
        del primals_77
        # Source Nodes: [hidden_states_60], Original ATen: [aten.native_dropout]
        buf236 = aten.native_dropout(reinterpret_tensor(buf235, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf242 = reinterpret_tensor(buf235, (1, 512, 768), (393216, 768, 1), 0); del buf235  # reuse
        buf243 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf342 = reinterpret_tensor(buf228, (1, 512, 1), (512, 1, 1), 0); del buf228  # reuse
        # Source Nodes: [add_36, fourier_output_16, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf237, buf230, primals_72, primals_73, primals_78, primals_79, buf242, buf243, buf342, 512, 768, grid=grid(512), stream=stream0)
        del primals_73
        del primals_79
        # Source Nodes: [fft_fftn_9, hidden_states_62], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf244 = torch.ops.prims.convert_element_type.default(buf243, torch.complex64)
        buf245 = buf244
        del buf244
        # Source Nodes: [fft_fftn_9], Original ATen: [aten._fft_c2c]
        buf246 = aten._fft_c2c(buf245, [1, 2], 0, True)
        del buf245
        buf247 = buf246
        del buf246
        # Source Nodes: [outputs_9], Original ATen: [aten.view_as_real]
        buf248 = aten.view_as_real(buf247)
        del buf247
        buf249 = buf248
        del buf248
        buf250 = buf226; del buf226  # reuse
        buf251 = buf225; del buf225  # reuse
        buf252 = buf224; del buf224  # reuse
        # Source Nodes: [add_37, fourier_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf243, buf249, buf250, buf251, buf252, 3072, 128, grid=grid(3072), stream=stream0)
        buf253 = buf227; del buf227  # reuse
        buf254 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf341 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, fourier_output_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf250, buf251, buf252, buf253, buf254, buf341, 512, 6, grid=grid(512), stream=stream0)
        buf256 = buf243; del buf243  # reuse
        buf257 = reinterpret_tensor(buf237, (512, 768), (768, 1), 0); del buf237  # reuse
        # Source Nodes: [add_37, fourier_output_18, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf256, buf249, buf253, buf254, primals_80, primals_81, buf257, 393216, grid=grid(393216), stream=stream0)
        del buf249
        buf258 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, buf257, reinterpret_tensor(primals_82, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf258)
        del primals_83
        buf259 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf260 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, hidden_states_66, intermediate_output_9, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf258, buf259, buf260, 1572864, grid=grid(1572864), stream=stream0)
        buf261 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, buf260, reinterpret_tensor(primals_84, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf261)
        del primals_85
        # Source Nodes: [hidden_states_67], Original ATen: [aten.native_dropout]
        buf262 = aten.native_dropout(reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf268 = reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0); del buf261  # reuse
        buf269 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf340 = reinterpret_tensor(buf254, (1, 512, 1), (512, 1, 1), 0); del buf254  # reuse
        # Source Nodes: [add_40, fourier_output_18, hidden_states_69], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf263, buf256, primals_80, primals_81, primals_86, primals_87, buf268, buf269, buf340, 512, 768, grid=grid(512), stream=stream0)
        del primals_81
        del primals_87
        # Source Nodes: [fft_fftn_10, hidden_states_69], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf270 = torch.ops.prims.convert_element_type.default(buf269, torch.complex64)
        buf271 = buf270
        del buf270
        # Source Nodes: [fft_fftn_10], Original ATen: [aten._fft_c2c]
        buf272 = aten._fft_c2c(buf271, [1, 2], 0, True)
        del buf271
        buf273 = buf272
        del buf272
        # Source Nodes: [outputs_10], Original ATen: [aten.view_as_real]
        buf274 = aten.view_as_real(buf273)
        del buf273
        buf275 = buf274
        del buf274
        buf276 = buf252; del buf252  # reuse
        buf277 = buf251; del buf251  # reuse
        buf278 = buf250; del buf250  # reuse
        # Source Nodes: [add_41, fourier_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf269, buf275, buf276, buf277, buf278, 3072, 128, grid=grid(3072), stream=stream0)
        buf279 = buf253; del buf253  # reuse
        buf280 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf339 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_41, fourier_output_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf276, buf277, buf278, buf279, buf280, buf339, 512, 6, grid=grid(512), stream=stream0)
        buf282 = buf269; del buf269  # reuse
        buf283 = reinterpret_tensor(buf263, (512, 768), (768, 1), 0); del buf263  # reuse
        # Source Nodes: [add_41, fourier_output_20, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf282, buf275, buf279, buf280, primals_88, primals_89, buf283, 393216, grid=grid(393216), stream=stream0)
        del buf275
        buf284 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_71], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, buf283, reinterpret_tensor(primals_90, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf284)
        del primals_91
        buf285 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf286 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43, hidden_states_73, intermediate_output_10, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf284, buf285, buf286, 1572864, grid=grid(1572864), stream=stream0)
        buf287 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf286, reinterpret_tensor(primals_92, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf287)
        del primals_93
        # Source Nodes: [hidden_states_74], Original ATen: [aten.native_dropout]
        buf288 = aten.native_dropout(reinterpret_tensor(buf287, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf294 = reinterpret_tensor(buf287, (1, 512, 768), (393216, 768, 1), 0); del buf287  # reuse
        buf295 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf338 = reinterpret_tensor(buf280, (1, 512, 1), (512, 1, 1), 0); del buf280  # reuse
        # Source Nodes: [add_44, fourier_output_20, hidden_states_76], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf289, buf282, primals_88, primals_89, primals_94, primals_95, buf294, buf295, buf338, 512, 768, grid=grid(512), stream=stream0)
        del primals_89
        del primals_95
        # Source Nodes: [fft_fftn_11, hidden_states_76], Original ATen: [aten._to_copy, aten.native_layer_norm]
        buf296 = torch.ops.prims.convert_element_type.default(buf295, torch.complex64)
        buf297 = buf296
        del buf296
        # Source Nodes: [fft_fftn_11], Original ATen: [aten._fft_c2c]
        buf298 = aten._fft_c2c(buf297, [1, 2], 0, True)
        del buf297
        buf299 = buf298
        del buf298
        # Source Nodes: [outputs_11], Original ATen: [aten.view_as_real]
        buf300 = aten.view_as_real(buf299)
        del buf299
        buf301 = buf300
        del buf300
        buf302 = buf278; del buf278  # reuse
        buf303 = buf277; del buf277  # reuse
        buf304 = buf276; del buf276  # reuse
        # Source Nodes: [add_45, fourier_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_1.run(buf295, buf301, buf302, buf303, buf304, 3072, 128, grid=grid(3072), stream=stream0)
        buf305 = buf279; del buf279  # reuse
        buf306 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf337 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_45, fourier_output_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_2.run(buf302, buf303, buf304, buf305, buf306, buf337, 512, 6, grid=grid(512), stream=stream0)
        del buf302
        del buf303
        del buf304
        buf308 = buf295; del buf295  # reuse
        buf309 = reinterpret_tensor(buf289, (512, 768), (768, 1), 0); del buf289  # reuse
        # Source Nodes: [add_45, fourier_output_22, hidden_states_78], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_3.run(buf308, buf301, buf305, buf306, primals_96, primals_97, buf309, 393216, grid=grid(393216), stream=stream0)
        del buf301
        buf310 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf309, reinterpret_tensor(primals_98, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf310)
        del primals_99
        buf311 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf312 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, add_47, hidden_states_80, intermediate_output_11, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf310, buf311, buf312, 1572864, grid=grid(1572864), stream=stream0)
        buf313 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_101, buf312, reinterpret_tensor(primals_100, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf313)
        del primals_101
        # Source Nodes: [hidden_states_81], Original ATen: [aten.native_dropout]
        buf314 = aten.native_dropout(reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf315 = buf314[0]
        buf316 = buf314[1]
        del buf314
        buf320 = reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0); del buf313  # reuse
        buf321 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf336 = reinterpret_tensor(buf306, (1, 512, 1), (512, 1, 1), 0); del buf306  # reuse
        # Source Nodes: [add_48, fourier_output_22, hidden_states_84, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf315, buf308, primals_96, primals_97, primals_102, primals_103, buf320, buf321, buf336, 512, 768, grid=grid(512), stream=stream0)
        del primals_103
        del primals_97
        buf322 = reinterpret_tensor(buf315, (512, 768), (768, 1), 0); del buf315  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_107, buf321, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf322)
        del primals_107
        buf323 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf324 = reinterpret_tensor(buf305, (1, 512, 1), (512, 1, 1), 0); del buf305  # reuse
        buf325 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf327 = reinterpret_tensor(buf325, (1, 512, 1), (512, 1, 1), 0); del buf325  # reuse
        buf328 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_49, add_50, hidden_states_85, hidden_states_87, mul_48, mul_49, mul_50, pow_13, prediction_scores, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_6.run(buf327, buf322, primals_108, primals_109, buf323, buf324, buf328, 512, 768, grid=grid(512), stream=stream0)
        del primals_109
        buf329 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_111, buf328, reinterpret_tensor(primals_110, (768, 32000), (1, 768), 0), alpha=1, beta=1, out=buf329)
        del primals_111
        buf332 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf329, buf332, 512, 32000, grid=grid(512), stream=stream0)
        buf335 = empty((), device='cuda', dtype=torch.float32)
        buf334 = empty((), device='cuda', dtype=torch.float32)
        buf361 = buf335; del buf335  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_8.run(buf361, primals_115, buf332, buf334, 1, 512, grid=grid(1), stream=stream0)
        return (buf361, reinterpret_tensor(buf329, (1, 512, 32000), (16384000, 32000, 1), 0), primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, primals_112, primals_113, buf4, buf5, buf9, buf22, buf23, buf24, buf25, buf26, buf30, buf34, buf48, buf49, buf50, buf51, buf52, buf56, buf60, buf74, buf75, buf76, buf77, buf78, buf82, buf86, buf100, buf101, buf102, buf103, buf104, buf108, buf112, buf126, buf127, buf128, buf129, buf130, buf134, buf138, buf152, buf153, buf154, buf155, buf156, buf160, buf164, buf178, buf179, buf180, buf181, buf182, buf186, buf190, buf204, buf205, buf206, buf207, buf208, buf212, buf216, buf230, buf231, buf232, buf233, buf234, buf238, buf242, buf256, buf257, buf258, buf259, buf260, buf264, buf268, buf282, buf283, buf284, buf285, buf286, buf290, buf294, buf308, buf309, buf310, buf311, buf312, buf316, buf320, buf321, buf322, buf323, buf324, buf327, buf328, buf332, buf334, reinterpret_tensor(primals_110, (32000, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), buf336, reinterpret_tensor(primals_100, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_98, (3072, 768), (768, 1), 0), buf337, buf338, reinterpret_tensor(primals_92, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_90, (3072, 768), (768, 1), 0), buf339, buf340, reinterpret_tensor(primals_84, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_82, (3072, 768), (768, 1), 0), buf341, buf342, reinterpret_tensor(primals_76, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_74, (3072, 768), (768, 1), 0), buf343, buf344, reinterpret_tensor(primals_68, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_66, (3072, 768), (768, 1), 0), buf345, buf346, reinterpret_tensor(primals_60, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_58, (3072, 768), (768, 1), 0), buf347, buf348, reinterpret_tensor(primals_52, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_50, (3072, 768), (768, 1), 0), buf349, buf350, reinterpret_tensor(primals_44, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_42, (3072, 768), (768, 1), 0), buf351, buf352, reinterpret_tensor(primals_36, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_34, (3072, 768), (768, 1), 0), buf353, buf354, reinterpret_tensor(primals_28, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 768), (768, 1), 0), buf355, buf356, reinterpret_tensor(primals_20, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_18, (3072, 768), (768, 1), 0), buf357, buf358, reinterpret_tensor(primals_12, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_10, (3072, 768), (768, 1), 0), buf359, reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf360, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_113 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_114 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_115 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GoogleFnet', benchmark_compiled_module)
