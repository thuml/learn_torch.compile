
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


# kernel path: /tmp/torchinductor_youkaichao/ok/cokz2u7folgm7dc2v3bzgntuyybxm44bx62z7utm7776my5wchum.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, hidden_states, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# hidden_states => view
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp44 = tmp38 / tmp34
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp43, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp44, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/md/cmdzruopvx4op2hkwaguz6lmgycxkgzh3m7g7l4k55ae6pfbtxpp.py
# Source Nodes: [attention_probs, attention_probs_1, attention_scores_2], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
# attention_probs => amax, div_1, exp, sub_2, sum_1
# attention_probs_1 => clone_1
# attention_scores_2 => div
triton_per_fused__softmax_add_clone_detach_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_detach_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp13, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bbhdm7nzk7povi2hr6pff7cq3htkgc3adb7tbvsfvsmcweosow.py
# Source Nodes: [projected_context_layer], Original ATen: [aten.view]
# projected_context_layer => view_18
triton_poi_fused_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (32768*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cds67mvkrudbt4nuh44hfcjkq6itmn22hxtq6kggb245nso3spbj.py
# Source Nodes: [add_2, ffn_output, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# ffn_output => view_20
# layernormed_context_layer => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp20, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp24, rmask & xmask)
    tmp25 = 4096.0
    tmp26 = tmp7 / tmp25
    tmp27 = 1e-12
    tmp28 = tmp26 + tmp27
    tmp29 = tl.math.rsqrt(tmp28)
    tmp30 = tmp29 / tmp25
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6fvzbrz4lbuukwiqqmuflnusaivxatuwyjcn2vmf4fso2hgnp3.py
# Source Nodes: [add_3, add_4, ffn_output_1, ffn_output_3, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
# add_3 => add_8
# add_4 => add_9
# ffn_output_1 => mul_8
# ffn_output_3 => view_22
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtsr2dilh553hcn3wqtoakihvaiwnz6pbfwiwzrzltrubpxgatm.py
# Source Nodes: [add_5, hidden_states_3, layernormed_context_layer, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_5 => add_10
# hidden_states_3 => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
# layernormed_context_layer => add_7, mul_4
# mixed_query_layer_1 => view_24
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r1 + (4096*x0)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp18 - tmp10
        tmp20 = 4096.0
        tmp21 = tmp16 / tmp20
        tmp22 = 1e-12
        tmp23 = tmp21 + tmp22
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp25, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp29, rmask & xmask)
    tmp30 = 4096.0
    tmp31 = tmp16 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp34 / tmp30
    tl.store(out_ptr4 + (x0), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbneo7elu7bqici6qennou6bps2fyeg2nqoeekqpm7bvu5smulog.py
# Source Nodes: [add_7, ffn_output_4, hidden_states_3, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_7 => add_14
# ffn_output_4 => view_42
# hidden_states_3 => add_12, mul_10
# layernormed_context_layer_1 => add_15, add_16, mul_11, mul_12, rsqrt_3, sub_6, var_mean_3
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r1 + (4096*x0)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp18 - tmp10
        tmp20 = 4096.0
        tmp21 = tmp16 / tmp20
        tmp22 = 1e-12
        tmp23 = tmp21 + tmp22
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp25, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp29, rmask & xmask)
    tmp30 = 4096.0
    tmp31 = tmp16 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp34 / tmp30
    tl.store(out_ptr4 + (x0), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnbkaomzfb6jwm5yfiebu3x6f66azi2spcjyiiht6dxoctojkz5.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_37
# start_loss => amax_12, exp_12, log, sub_38, sub_39, sum_13
triton_per_fused__log_softmax_clone_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstpsvc3n7anhhi34huzywghh4oubl3y43nvivpsl3dpw6r2u6zm.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_38
# end_loss => amax_13, exp_13, log_1, sub_40, sub_41, sum_16
triton_per_fused__log_softmax_clone_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxmdvemkobzmkbcmq63t7iah2cjiwrzkhywg22f6pxoo3ijqmt3.py
# Source Nodes: [add_61, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# add_61 => add_112
# end_loss => convert_element_type_1, div_25, ne_3, neg_1, sum_17, sum_18, where_3
# end_positions => clamp_max_1, clamp_min_1
# loss => div_26
# start_loss => convert_element_type, div_24, full_default_1, full_default_2, ne, neg, sum_14, sum_15, where_1
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*fp32', 7: '*i1', 8: '*i64', 9: '*i1', 10: '*i64', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tmp12 + 512
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 512), "index out of bounds: 0 <= tmp15 < 512")
    tmp16 = tl.load(in_ptr2 + (tmp15), None, eviction_policy='evict_last')
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tmp6.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tl.where(tmp11, tmp10, tmp2)
    tmp24 = tmp23 + 512
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 512), "index out of bounds: 0 <= tmp26 < 512")
    tmp27 = tl.load(in_ptr3 + (tmp26), None, eviction_policy='evict_last')
    tmp28 = -tmp27
    tmp29 = tl.where(tmp11, tmp28, tmp18)
    tmp30 = tmp11.to(tl.int64)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = 2.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
    args.clear()
    assert_size_stride(primals_1, (30000, 128), (128, 1))
    assert_size_stride(primals_2, (2, 128), (128, 1))
    assert_size_stride(primals_3, (512, 128), (128, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (4096, 128), (128, 1))
    assert_size_stride(primals_7, (4096, ), (1, ))
    assert_size_stride(primals_8, (4096, 4096), (4096, 1))
    assert_size_stride(primals_9, (4096, ), (1, ))
    assert_size_stride(primals_10, (4096, 4096), (4096, 1))
    assert_size_stride(primals_11, (4096, ), (1, ))
    assert_size_stride(primals_12, (4096, 4096), (4096, 1))
    assert_size_stride(primals_13, (4096, ), (1, ))
    assert_size_stride(primals_14, (4096, 4096), (4096, 1))
    assert_size_stride(primals_15, (4096, ), (1, ))
    assert_size_stride(primals_16, (4096, ), (1, ))
    assert_size_stride(primals_17, (4096, ), (1, ))
    assert_size_stride(primals_18, (16384, 4096), (4096, 1))
    assert_size_stride(primals_19, (16384, ), (1, ))
    assert_size_stride(primals_20, (4096, 16384), (16384, 1))
    assert_size_stride(primals_21, (4096, ), (1, ))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_23, (4096, ), (1, ))
    assert_size_stride(primals_24, (2, 4096), (4096, 1))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (1, 512), (512, 1))
    assert_size_stride(primals_27, (1, 512), (512, 1))
    assert_size_stride(primals_28, (1, 512), (512, 1))
    assert_size_stride(primals_29, (1, ), (1, ))
    assert_size_stride(primals_30, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf5 = empty((512, 128), device='cuda', dtype=torch.float32)
        buf369 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, hidden_states, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0.run(primals_28, primals_1, primals_26, primals_2, primals_27, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf369, 512, 128, grid=grid(512), stream=stream0)
        del buf0
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        buf6 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf5, reinterpret_tensor(primals_6, (128, 4096), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del primals_7
        buf7 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf6, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf7)
        buf8 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf6, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf8)
        buf9 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf6, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf9)
        buf10 = empty((64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf8, (64, 64, 512), (64, 1, 4096), 0), out=buf10)
        buf13 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf368 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_1, attention_scores_2], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf10, buf13, buf368, 32768, 512, grid=grid(32768), stream=stream0)
        buf14 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf9, (64, 512, 64), (64, 4096, 1), 0), out=buf14)
        buf15 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf14, buf15, 2097152, grid=grid(2097152), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (512, 4096), (4096, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf15, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf16)
        buf20 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf21 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf367 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, ffn_output, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf6, buf16, primals_15, primals_16, primals_17, buf20, buf21, buf367, 512, 4096, grid=grid(512), stream=stream0)
        buf22 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf21, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf22)
        buf23 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf24 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, add_4, ffn_output_1, ffn_output_3, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf22, buf23, buf24, 8388608, grid=grid(8388608), stream=stream0)
        buf25 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf24, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf25)
        buf26 = reinterpret_tensor(buf25, (1, 512, 4096), (2097152, 4096, 1), 0); del buf25  # reuse
        buf30 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf31 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf366 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, hidden_states_3, layernormed_context_layer, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf26, primals_21, buf20, primals_16, primals_17, primals_22, primals_23, buf30, buf31, buf366, 512, 4096, grid=grid(512), stream=stream0)
        buf32 = reinterpret_tensor(buf26, (512, 4096), (4096, 1), 0); del buf26  # reuse
        # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf31, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf32)
        buf33 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf31, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf33)
        buf34 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf31, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf34)
        buf35 = buf10; del buf10  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf33, (64, 64, 512), (64, 1, 4096), 0), out=buf35)
        buf38 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf365 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_2, attention_probs_3, attention_scores_5], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf35, buf38, buf365, 32768, 512, grid=grid(32768), stream=stream0)
        buf39 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf34, (64, 512, 64), (64, 4096, 1), 0), out=buf39)
        buf40 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_1], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf39, buf40, 2097152, grid=grid(2097152), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf41)
        buf42 = reinterpret_tensor(buf41, (1, 512, 4096), (2097152, 4096, 1), 0); del buf41  # reuse
        buf46 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf47 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf364 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, ffn_output_4, hidden_states_3, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf42, buf30, primals_22, primals_23, primals_15, primals_16, primals_17, buf46, buf47, buf364, 512, 4096, grid=grid(512), stream=stream0)
        buf48 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf47, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf48)
        buf49 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf50 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, add_9, ffn_output_5, ffn_output_7, mul_5, mul_6, mul_7, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf48, buf49, buf50, 8388608, grid=grid(8388608), stream=stream0)
        buf51 = reinterpret_tensor(buf42, (512, 4096), (4096, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf50, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf51)
        buf52 = reinterpret_tensor(buf51, (1, 512, 4096), (2097152, 4096, 1), 0); del buf51  # reuse
        buf56 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf57 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf363 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, hidden_states_6, layernormed_context_layer_1, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf52, primals_21, buf46, primals_16, primals_17, primals_22, primals_23, buf56, buf57, buf363, 512, 4096, grid=grid(512), stream=stream0)
        buf58 = reinterpret_tensor(buf52, (512, 4096), (4096, 1), 0); del buf52  # reuse
        # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf57, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf58)
        buf59 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf57, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf59)
        buf60 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf57, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf60)
        buf61 = buf35; del buf35  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf59, (64, 64, 512), (64, 1, 4096), 0), out=buf61)
        buf64 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf362 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_4, attention_probs_5, attention_scores_8], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf61, buf64, buf362, 32768, 512, grid=grid(32768), stream=stream0)
        buf65 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf60, (64, 512, 64), (64, 4096, 1), 0), out=buf65)
        buf66 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_2], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf65, buf66, 2097152, grid=grid(2097152), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (512, 4096), (4096, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf66, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf67)
        buf68 = reinterpret_tensor(buf67, (1, 512, 4096), (2097152, 4096, 1), 0); del buf67  # reuse
        buf72 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf73 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf361 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, ffn_output_8, hidden_states_6, layernormed_context_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf68, buf56, primals_22, primals_23, primals_15, primals_16, primals_17, buf72, buf73, buf361, 512, 4096, grid=grid(512), stream=stream0)
        buf74 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf73, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf74)
        buf75 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf76 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, add_14, ffn_output_11, ffn_output_9, mul_10, mul_11, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf74, buf75, buf76, 8388608, grid=grid(8388608), stream=stream0)
        buf77 = reinterpret_tensor(buf68, (512, 4096), (4096, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf77)
        buf78 = reinterpret_tensor(buf77, (1, 512, 4096), (2097152, 4096, 1), 0); del buf77  # reuse
        buf82 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf83 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf360 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, hidden_states_9, layernormed_context_layer_2, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf78, primals_21, buf72, primals_16, primals_17, primals_22, primals_23, buf82, buf83, buf360, 512, 4096, grid=grid(512), stream=stream0)
        buf84 = reinterpret_tensor(buf78, (512, 4096), (4096, 1), 0); del buf78  # reuse
        # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf83, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf84)
        buf85 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf83, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf85)
        buf86 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf83, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf86)
        buf87 = buf61; del buf61  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf85, (64, 64, 512), (64, 1, 4096), 0), out=buf87)
        buf90 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf359 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_6, attention_probs_7, attention_scores_11], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf87, buf90, buf359, 32768, 512, grid=grid(32768), stream=stream0)
        buf91 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf86, (64, 512, 64), (64, 4096, 1), 0), out=buf91)
        buf92 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_3], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf91, buf92, 2097152, grid=grid(2097152), stream=stream0)
        buf93 = reinterpret_tensor(buf91, (512, 4096), (4096, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf92, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf93)
        buf94 = reinterpret_tensor(buf93, (1, 512, 4096), (2097152, 4096, 1), 0); del buf93  # reuse
        buf98 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf99 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf358 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, ffn_output_12, hidden_states_9, layernormed_context_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf94, buf82, primals_22, primals_23, primals_15, primals_16, primals_17, buf98, buf99, buf358, 512, 4096, grid=grid(512), stream=stream0)
        buf100 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf99, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf100)
        buf101 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf102 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, ffn_output_13, ffn_output_15, mul_13, mul_14, mul_15, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf100, buf101, buf102, 8388608, grid=grid(8388608), stream=stream0)
        buf103 = reinterpret_tensor(buf94, (512, 4096), (4096, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf102, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf103)
        buf104 = reinterpret_tensor(buf103, (1, 512, 4096), (2097152, 4096, 1), 0); del buf103  # reuse
        buf108 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf109 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf357 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, hidden_states_12, layernormed_context_layer_3, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf104, primals_21, buf98, primals_16, primals_17, primals_22, primals_23, buf108, buf109, buf357, 512, 4096, grid=grid(512), stream=stream0)
        buf110 = reinterpret_tensor(buf104, (512, 4096), (4096, 1), 0); del buf104  # reuse
        # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf109, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf110)
        buf111 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf109, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf111)
        buf112 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf109, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf112)
        buf113 = buf87; del buf87  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf111, (64, 64, 512), (64, 1, 4096), 0), out=buf113)
        buf116 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf356 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_8, attention_probs_9, attention_scores_14], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf113, buf116, buf356, 32768, 512, grid=grid(32768), stream=stream0)
        buf117 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf112, (64, 512, 64), (64, 4096, 1), 0), out=buf117)
        buf118 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_4], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf117, buf118, 2097152, grid=grid(2097152), stream=stream0)
        buf119 = reinterpret_tensor(buf117, (512, 4096), (4096, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf118, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf119)
        buf120 = reinterpret_tensor(buf119, (1, 512, 4096), (2097152, 4096, 1), 0); del buf119  # reuse
        buf124 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf125 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf355 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, ffn_output_16, hidden_states_12, layernormed_context_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf120, buf108, primals_22, primals_23, primals_15, primals_16, primals_17, buf124, buf125, buf355, 512, 4096, grid=grid(512), stream=stream0)
        buf126 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf125, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf126)
        buf127 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf128 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, add_24, ffn_output_17, ffn_output_19, mul_17, mul_18, mul_19, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf126, buf127, buf128, 8388608, grid=grid(8388608), stream=stream0)
        buf129 = reinterpret_tensor(buf120, (512, 4096), (4096, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf128, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf129)
        buf130 = reinterpret_tensor(buf129, (1, 512, 4096), (2097152, 4096, 1), 0); del buf129  # reuse
        buf134 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf135 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf354 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, hidden_states_15, layernormed_context_layer_4, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf130, primals_21, buf124, primals_16, primals_17, primals_22, primals_23, buf134, buf135, buf354, 512, 4096, grid=grid(512), stream=stream0)
        buf136 = reinterpret_tensor(buf130, (512, 4096), (4096, 1), 0); del buf130  # reuse
        # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf135, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf136)
        buf137 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf135, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf137)
        buf138 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf135, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf138)
        buf139 = buf113; del buf113  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf137, (64, 64, 512), (64, 1, 4096), 0), out=buf139)
        buf142 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf353 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_10, attention_probs_11, attention_scores_17], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf139, buf142, buf353, 32768, 512, grid=grid(32768), stream=stream0)
        buf143 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf138, (64, 512, 64), (64, 4096, 1), 0), out=buf143)
        buf144 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_5], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf143, buf144, 2097152, grid=grid(2097152), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (512, 4096), (4096, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf145)
        buf146 = reinterpret_tensor(buf145, (1, 512, 4096), (2097152, 4096, 1), 0); del buf145  # reuse
        buf150 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf151 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf352 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, ffn_output_20, hidden_states_15, layernormed_context_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf146, buf134, primals_22, primals_23, primals_15, primals_16, primals_17, buf150, buf151, buf352, 512, 4096, grid=grid(512), stream=stream0)
        buf152 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf151, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf152)
        buf153 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf154 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, add_29, ffn_output_21, ffn_output_23, mul_21, mul_22, mul_23, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf152, buf153, buf154, 8388608, grid=grid(8388608), stream=stream0)
        buf155 = reinterpret_tensor(buf146, (512, 4096), (4096, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf154, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf155)
        buf156 = reinterpret_tensor(buf155, (1, 512, 4096), (2097152, 4096, 1), 0); del buf155  # reuse
        buf160 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf161 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf351 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, hidden_states_18, layernormed_context_layer_5, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf156, primals_21, buf150, primals_16, primals_17, primals_22, primals_23, buf160, buf161, buf351, 512, 4096, grid=grid(512), stream=stream0)
        buf162 = reinterpret_tensor(buf156, (512, 4096), (4096, 1), 0); del buf156  # reuse
        # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf161, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf162)
        buf163 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf161, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf163)
        buf164 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf161, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf164)
        buf165 = buf139; del buf139  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf162, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf163, (64, 64, 512), (64, 1, 4096), 0), out=buf165)
        buf168 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf350 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_12, attention_probs_13, attention_scores_20], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf165, buf168, buf350, 32768, 512, grid=grid(32768), stream=stream0)
        buf169 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf164, (64, 512, 64), (64, 4096, 1), 0), out=buf169)
        buf170 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_6], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf169, buf170, 2097152, grid=grid(2097152), stream=stream0)
        buf171 = reinterpret_tensor(buf169, (512, 4096), (4096, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf170, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf171)
        buf172 = reinterpret_tensor(buf171, (1, 512, 4096), (2097152, 4096, 1), 0); del buf171  # reuse
        buf176 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf177 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf349 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, ffn_output_24, hidden_states_18, layernormed_context_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf172, buf160, primals_22, primals_23, primals_15, primals_16, primals_17, buf176, buf177, buf349, 512, 4096, grid=grid(512), stream=stream0)
        buf178 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf177, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf178)
        buf179 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf180 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, add_34, ffn_output_25, ffn_output_27, mul_25, mul_26, mul_27, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf178, buf179, buf180, 8388608, grid=grid(8388608), stream=stream0)
        buf181 = reinterpret_tensor(buf172, (512, 4096), (4096, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf180, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf181)
        buf182 = reinterpret_tensor(buf181, (1, 512, 4096), (2097152, 4096, 1), 0); del buf181  # reuse
        buf186 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf187 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf348 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, hidden_states_21, layernormed_context_layer_6, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf182, primals_21, buf176, primals_16, primals_17, primals_22, primals_23, buf186, buf187, buf348, 512, 4096, grid=grid(512), stream=stream0)
        buf188 = reinterpret_tensor(buf182, (512, 4096), (4096, 1), 0); del buf182  # reuse
        # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf187, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf188)
        buf189 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf187, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf189)
        buf190 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf187, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf190)
        buf191 = buf165; del buf165  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf189, (64, 64, 512), (64, 1, 4096), 0), out=buf191)
        buf194 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf347 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_14, attention_probs_15, attention_scores_23], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf191, buf194, buf347, 32768, 512, grid=grid(32768), stream=stream0)
        buf195 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf190, (64, 512, 64), (64, 4096, 1), 0), out=buf195)
        buf196 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_7], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf195, buf196, 2097152, grid=grid(2097152), stream=stream0)
        buf197 = reinterpret_tensor(buf195, (512, 4096), (4096, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf196, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf197)
        buf198 = reinterpret_tensor(buf197, (1, 512, 4096), (2097152, 4096, 1), 0); del buf197  # reuse
        buf202 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf203 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf346 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, ffn_output_28, hidden_states_21, layernormed_context_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf198, buf186, primals_22, primals_23, primals_15, primals_16, primals_17, buf202, buf203, buf346, 512, 4096, grid=grid(512), stream=stream0)
        buf204 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf203, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf204)
        buf205 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf206 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, ffn_output_29, ffn_output_31, mul_29, mul_30, mul_31, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf204, buf205, buf206, 8388608, grid=grid(8388608), stream=stream0)
        buf207 = reinterpret_tensor(buf198, (512, 4096), (4096, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf206, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf207)
        buf208 = reinterpret_tensor(buf207, (1, 512, 4096), (2097152, 4096, 1), 0); del buf207  # reuse
        buf212 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf213 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf345 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, hidden_states_24, layernormed_context_layer_7, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf208, primals_21, buf202, primals_16, primals_17, primals_22, primals_23, buf212, buf213, buf345, 512, 4096, grid=grid(512), stream=stream0)
        buf214 = reinterpret_tensor(buf208, (512, 4096), (4096, 1), 0); del buf208  # reuse
        # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf213, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf214)
        buf215 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf213, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf215)
        buf216 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf213, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf216)
        buf217 = buf191; del buf191  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf215, (64, 64, 512), (64, 1, 4096), 0), out=buf217)
        buf220 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf344 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_16, attention_probs_17, attention_scores_26], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf217, buf220, buf344, 32768, 512, grid=grid(32768), stream=stream0)
        buf221 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf216, (64, 512, 64), (64, 4096, 1), 0), out=buf221)
        buf222 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_8], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf221, buf222, 2097152, grid=grid(2097152), stream=stream0)
        buf223 = reinterpret_tensor(buf221, (512, 4096), (4096, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf222, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf223)
        buf224 = reinterpret_tensor(buf223, (1, 512, 4096), (2097152, 4096, 1), 0); del buf223  # reuse
        buf228 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf229 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf343 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, ffn_output_32, hidden_states_24, layernormed_context_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf224, buf212, primals_22, primals_23, primals_15, primals_16, primals_17, buf228, buf229, buf343, 512, 4096, grid=grid(512), stream=stream0)
        buf230 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf229, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf230)
        buf231 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf232 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_43, add_44, ffn_output_33, ffn_output_35, mul_33, mul_34, mul_35, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf230, buf231, buf232, 8388608, grid=grid(8388608), stream=stream0)
        buf233 = reinterpret_tensor(buf224, (512, 4096), (4096, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf232, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf233)
        buf234 = reinterpret_tensor(buf233, (1, 512, 4096), (2097152, 4096, 1), 0); del buf233  # reuse
        buf238 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf239 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf342 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_45, hidden_states_27, layernormed_context_layer_8, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf234, primals_21, buf228, primals_16, primals_17, primals_22, primals_23, buf238, buf239, buf342, 512, 4096, grid=grid(512), stream=stream0)
        buf240 = reinterpret_tensor(buf234, (512, 4096), (4096, 1), 0); del buf234  # reuse
        # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf239, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf240)
        buf241 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf239, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf241)
        buf242 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf239, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf242)
        buf243 = buf217; del buf217  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf241, (64, 64, 512), (64, 1, 4096), 0), out=buf243)
        buf246 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf341 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_18, attention_probs_19, attention_scores_29], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf243, buf246, buf341, 32768, 512, grid=grid(32768), stream=stream0)
        buf247 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf242, (64, 512, 64), (64, 4096, 1), 0), out=buf247)
        buf248 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_9], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf247, buf248, 2097152, grid=grid(2097152), stream=stream0)
        buf249 = reinterpret_tensor(buf247, (512, 4096), (4096, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf248, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf249)
        buf250 = reinterpret_tensor(buf249, (1, 512, 4096), (2097152, 4096, 1), 0); del buf249  # reuse
        buf254 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf255 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf340 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47, ffn_output_36, hidden_states_27, layernormed_context_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf250, buf238, primals_22, primals_23, primals_15, primals_16, primals_17, buf254, buf255, buf340, 512, 4096, grid=grid(512), stream=stream0)
        buf256 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf255, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf256)
        buf257 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf258 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, add_49, ffn_output_37, ffn_output_39, mul_37, mul_38, mul_39, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf256, buf257, buf258, 8388608, grid=grid(8388608), stream=stream0)
        buf259 = reinterpret_tensor(buf250, (512, 4096), (4096, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf258, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf259)
        buf260 = reinterpret_tensor(buf259, (1, 512, 4096), (2097152, 4096, 1), 0); del buf259  # reuse
        buf264 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf265 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf339 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, hidden_states_30, layernormed_context_layer_9, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf260, primals_21, buf254, primals_16, primals_17, primals_22, primals_23, buf264, buf265, buf339, 512, 4096, grid=grid(512), stream=stream0)
        buf266 = reinterpret_tensor(buf260, (512, 4096), (4096, 1), 0); del buf260  # reuse
        # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf265, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf266)
        buf267 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf265, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf267)
        buf268 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf265, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf268)
        buf269 = buf243; del buf243  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf267, (64, 64, 512), (64, 1, 4096), 0), out=buf269)
        buf272 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf338 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_20, attention_probs_21, attention_scores_32], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf269, buf272, buf338, 32768, 512, grid=grid(32768), stream=stream0)
        buf273 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf268, (64, 512, 64), (64, 4096, 1), 0), out=buf273)
        buf274 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_10], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf273, buf274, 2097152, grid=grid(2097152), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (512, 4096), (4096, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf274, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf275)
        buf276 = reinterpret_tensor(buf275, (1, 512, 4096), (2097152, 4096, 1), 0); del buf275  # reuse
        buf280 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf281 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf337 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_52, ffn_output_40, hidden_states_30, layernormed_context_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf276, buf264, primals_22, primals_23, primals_15, primals_16, primals_17, buf280, buf281, buf337, 512, 4096, grid=grid(512), stream=stream0)
        buf282 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf281, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf282)
        buf283 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf284 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, add_54, ffn_output_41, ffn_output_43, mul_41, mul_42, mul_43, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf282, buf283, buf284, 8388608, grid=grid(8388608), stream=stream0)
        buf285 = reinterpret_tensor(buf276, (512, 4096), (4096, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf284, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf285)
        buf286 = reinterpret_tensor(buf285, (1, 512, 4096), (2097152, 4096, 1), 0); del buf285  # reuse
        buf290 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf291 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf336 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_55, hidden_states_33, layernormed_context_layer_10, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf286, primals_21, buf280, primals_16, primals_17, primals_22, primals_23, buf290, buf291, buf336, 512, 4096, grid=grid(512), stream=stream0)
        buf292 = reinterpret_tensor(buf286, (512, 4096), (4096, 1), 0); del buf286  # reuse
        # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf291, reinterpret_tensor(primals_8, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf292)
        del primals_9
        buf293 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf291, reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf293)
        del primals_11
        buf294 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf291, reinterpret_tensor(primals_12, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf294)
        del primals_13
        buf295 = buf269; del buf269  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(buf293, (64, 64, 512), (64, 1, 4096), 0), out=buf295)
        buf298 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        buf335 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_22, attention_probs_23, attention_scores_35], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_1.run(buf295, buf298, buf335, 32768, 512, grid=grid(32768), stream=stream0)
        del buf295
        buf299 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (64, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf294, (64, 512, 64), (64, 4096, 1), 0), out=buf299)
        buf300 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_11], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf299, buf300, 2097152, grid=grid(2097152), stream=stream0)
        buf301 = reinterpret_tensor(buf299, (512, 4096), (4096, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf300, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf301)
        buf302 = reinterpret_tensor(buf301, (1, 512, 4096), (2097152, 4096, 1), 0); del buf301  # reuse
        buf306 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf307 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf334 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_57, ffn_output_44, hidden_states_33, layernormed_context_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf302, buf290, primals_22, primals_23, primals_15, primals_16, primals_17, buf306, buf307, buf334, 512, 4096, grid=grid(512), stream=stream0)
        del primals_15
        buf308 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf307, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf308)
        del primals_19
        buf309 = empty((1, 512, 16384), device='cuda', dtype=torch.float32)
        buf310 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_58, add_59, ffn_output_45, ffn_output_47, mul_45, mul_46, mul_47, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_4.run(buf308, buf309, buf310, 8388608, grid=grid(8388608), stream=stream0)
        buf311 = reinterpret_tensor(buf302, (512, 4096), (4096, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf310, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), out=buf311)
        buf312 = reinterpret_tensor(buf311, (1, 512, 4096), (2097152, 4096, 1), 0); del buf311  # reuse
        buf316 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf317 = empty((512, 4096), device='cuda', dtype=torch.float32)
        buf333 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, layernormed_context_layer_11, logits, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf312, primals_21, buf306, primals_16, primals_17, primals_22, primals_23, buf316, buf317, buf333, 512, 4096, grid=grid(512), stream=stream0)
        del buf312
        del primals_17
        del primals_21
        del primals_23
        buf318 = empty((512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf317, reinterpret_tensor(primals_24, (4096, 2), (1, 4096), 0), out=buf318)
        buf319 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf323 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_7.run(buf318, primals_25, buf319, buf323, 1, 512, grid=grid(1), stream=stream0)
        buf320 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf327 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_8.run(buf318, primals_25, buf320, buf327, 1, 512, grid=grid(1), stream=stream0)
        del buf318
        del primals_25
        buf324 = empty((1, ), device='cuda', dtype=torch.bool)
        buf328 = empty((1, ), device='cuda', dtype=torch.bool)
        buf370 = empty((), device='cuda', dtype=torch.float32)
        buf329 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf330 = empty((1, 1), device='cuda', dtype=torch.int64)
        buf331 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf332 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [add_61, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_9.run(primals_29, primals_30, buf323, buf327, buf324, buf328, buf370, buf329, buf330, buf331, buf332, 1, grid=grid(1), stream=stream0)
        del primals_29
        del primals_30
        return (buf370, buf319, buf320, primals_4, primals_16, primals_22, primals_28, primals_26, primals_27, buf4, buf5, reinterpret_tensor(buf6, (512, 4096), (4096, 1), 0), buf15, buf20, buf21, buf22, buf23, buf24, buf30, buf31, buf40, buf46, buf47, buf48, buf49, buf50, buf56, buf57, buf66, buf72, buf73, buf74, buf75, buf76, buf82, buf83, buf92, buf98, buf99, buf100, buf101, buf102, buf108, buf109, buf118, buf124, buf125, buf126, buf127, buf128, buf134, buf135, buf144, buf150, buf151, buf152, buf153, buf154, buf160, buf161, buf170, buf176, buf177, buf178, buf179, buf180, buf186, buf187, buf196, buf202, buf203, buf204, buf205, buf206, buf212, buf213, buf222, buf228, buf229, buf230, buf231, buf232, buf238, buf239, buf248, buf254, buf255, buf256, buf257, buf258, buf264, buf265, buf274, buf280, buf281, buf282, buf283, buf284, buf290, buf291, buf300, buf306, buf307, buf308, buf309, buf310, buf316, buf317, buf323, buf324, buf327, buf328, buf329, buf330, buf331, buf332, reinterpret_tensor(primals_24, (2, 4096), (4096, 1), 0), buf333, reinterpret_tensor(primals_20, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_18, (16384, 4096), (4096, 1), 0), buf334, reinterpret_tensor(primals_14, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf298, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf294, (64, 64, 512), (64, 1, 4096), 0), buf335, reinterpret_tensor(buf292, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf293, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(primals_12, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_10, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_8, (4096, 4096), (4096, 1), 0), buf336, buf337, reinterpret_tensor(buf272, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf268, (64, 64, 512), (64, 1, 4096), 0), buf338, reinterpret_tensor(buf266, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf267, (64, 512, 64), (64, 4096, 1), 0), buf339, buf340, reinterpret_tensor(buf246, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf242, (64, 64, 512), (64, 1, 4096), 0), buf341, reinterpret_tensor(buf240, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf241, (64, 512, 64), (64, 4096, 1), 0), buf342, buf343, reinterpret_tensor(buf220, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf216, (64, 64, 512), (64, 1, 4096), 0), buf344, reinterpret_tensor(buf214, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf215, (64, 512, 64), (64, 4096, 1), 0), buf345, buf346, reinterpret_tensor(buf194, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf190, (64, 64, 512), (64, 1, 4096), 0), buf347, reinterpret_tensor(buf188, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf189, (64, 512, 64), (64, 4096, 1), 0), buf348, buf349, reinterpret_tensor(buf168, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf164, (64, 64, 512), (64, 1, 4096), 0), buf350, reinterpret_tensor(buf162, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf163, (64, 512, 64), (64, 4096, 1), 0), buf351, buf352, reinterpret_tensor(buf142, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf138, (64, 64, 512), (64, 1, 4096), 0), buf353, reinterpret_tensor(buf136, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf137, (64, 512, 64), (64, 4096, 1), 0), buf354, buf355, reinterpret_tensor(buf116, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf112, (64, 64, 512), (64, 1, 4096), 0), buf356, reinterpret_tensor(buf110, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf111, (64, 512, 64), (64, 4096, 1), 0), buf357, buf358, reinterpret_tensor(buf90, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf86, (64, 64, 512), (64, 1, 4096), 0), buf359, reinterpret_tensor(buf84, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf85, (64, 512, 64), (64, 4096, 1), 0), buf360, buf361, reinterpret_tensor(buf64, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf60, (64, 64, 512), (64, 1, 4096), 0), buf362, reinterpret_tensor(buf58, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf59, (64, 512, 64), (64, 4096, 1), 0), buf363, buf364, reinterpret_tensor(buf38, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf34, (64, 64, 512), (64, 1, 4096), 0), buf365, reinterpret_tensor(buf32, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf33, (64, 512, 64), (64, 4096, 1), 0), buf366, buf367, reinterpret_tensor(buf13, (64, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf9, (64, 64, 512), (64, 1, 4096), 0), buf368, reinterpret_tensor(buf7, (64, 64, 512), (64, 1, 4096), 0), reinterpret_tensor(buf8, (64, 512, 64), (64, 4096, 1), 0), reinterpret_tensor(primals_6, (4096, 128), (128, 1), 0), buf369, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_27 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_28 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_30 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForQuestionAnswering', benchmark_compiled_module)
