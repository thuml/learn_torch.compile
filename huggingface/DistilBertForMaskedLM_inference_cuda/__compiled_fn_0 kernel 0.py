
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgdrhv36j3ozwh65nurjnhfy7d33vbc56wvpgkl6boojov5b6pw.py
# Source Nodes: [embeddings, embeddings_1, input_embeds, position_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# embeddings => add
# embeddings_1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# input_embeds => embedding
# position_embeddings => embedding_1
triton_red_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 30522
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 + 512
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
        tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp4 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight,
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp0 + 30522
        tmp16 = tmp0 < 0
        tmp17 = tl.where(tmp16, tmp15, tmp0)
        tl.device_assert(((0 <= tmp17) & (tmp17 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp17 < 30522")
        tmp18 = tl.load(in_ptr1 + (r1 + (768*tmp17)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp5 + 512
        tmp20 = tmp5 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp5)
        tl.device_assert(((0 <= tmp21) & (tmp21 < 512)) | ~xmask, "index out of bounds: 0 <= tmp21 < 512")
        tmp22 = tl.load(in_ptr3 + (r1 + (768*tmp21)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23 - tmp12
        tmp25 = 768.0
        tmp26 = tmp13 / tmp25
        tmp27 = 1e-12
        tmp28 = tmp26 + tmp27
        tmp29 = tl.math.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp34, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/hf/chfd37rfdyz4g4ezb4c2hfdiuaafqfilpljkskppvo6l7ypsj7a7.py
# Source Nodes: [q_1], Original ATen: [aten.div]
# q_1 => div
triton_poi_fused_div_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 8.0
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6cjlnyobfzz62lp6ktyb3qhdyof4mfyautcu5f6kk2h4hldctcv.py
# Source Nodes: [scores_1, tensor, weights], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
# scores_1 => where
# tensor => full_default
# weights => amax, div_1, exp, sub_1, sum_1
triton_per_fused__softmax_lift_fresh_masked_fill_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lift_fresh_masked_fill_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp0 = 1.0
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp11 / tmp15
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlvpvaievriuuwdr5gscef24oo7k65m47azzostjhq2iphano4z.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone_2
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblbi5c62szt36c2276u5w2oa5b5djc2khgnobrsklkyr3brt62a.py
# Source Nodes: [add_1, sa_output_1], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_3
# sa_output_1 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrepwxhfryi6a6r73ybrnxvtcjnl6bp7utmei2tggupulnmxkn6.py
# Source Nodes: [x_1], Original ATen: [aten.gelu]
# x_1 => add_6, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: /tmp/torchinductor_youkaichao/ej/cejjtrasypvlc6oleoz5ybnjdzio5akgepqzg7vpfsbk3g5xiao4.py
# Source Nodes: [prediction_logits_1, prediction_logits_2], Original ATen: [aten.gelu, aten.native_layer_norm]
# prediction_logits_1 => add_45, erf_6, mul_44, mul_45, mul_46
# prediction_logits_2 => add_46, add_47, mul_47, mul_48, rsqrt_13, sub_19, var_mean_13
triton_per_fused_gelu_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvp5njnp2rao5yr7i3y6iwdy6dtxfbxvsmobxzmdake3dyqgqqe.py
# Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
# mlm_loss => amax_6
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2u/c2uiyri6gaa3qv2xmw7udunxowhmyqpxl2fuv636ri5shnfwuxtc.py
# Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
# mlm_loss => amax_6
triton_per_fused__log_softmax_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoclb3bwhmt22hfsgg4ofdbd3dswvynztxgdui46zb42nhz7qxv.py
# Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
# mlm_loss => exp_6, sub_20, sum_7
triton_red_fused__log_softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuxmp4anmxqdp4evttxgdcth4k4zsubtewp6ecqot2oioqgirdn.py
# Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
# mlm_loss => exp_6, sub_20, sum_7
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
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4drgrsvx7bmnoyybr5xyujislbszkzeke2gdexdmvzyaemcakq.py
# Source Nodes: [mlm_loss], Original ATen: [aten.nll_loss_forward]
# mlm_loss => convert_element_type, div_12, full_default_7, ne_1, ne_2, neg, sum_8, sum_9, where_7
triton_per_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 30522
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30522), "index out of bounds: 0 <= tmp7 < 30522")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (30522*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 768), (768, 1))
    assert_size_stride(arg1_1, (512, 768), (768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg15_1, (3072, ), (1, ))
    assert_size_stride(arg16_1, (768, 3072), (3072, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (3072, 768), (768, 1))
    assert_size_stride(arg31_1, (3072, ), (1, ))
    assert_size_stride(arg32_1, (768, 3072), (3072, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 768), (768, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg63_1, (3072, ), (1, ))
    assert_size_stride(arg64_1, (768, 3072), (3072, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (3072, 768), (768, 1))
    assert_size_stride(arg79_1, (3072, ), (1, ))
    assert_size_stride(arg80_1, (768, 3072), (3072, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (30522, 768), (768, 1))
    assert_size_stride(arg105_1, (30522, ), (1, ))
    assert_size_stride(arg106_1, (1, 512), (512, 1))
    assert_size_stride(arg107_1, (1, 128), (128, 1))
    assert_size_stride(arg108_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, input_embeds, position_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_0.run(arg107_1, arg0_1, arg106_1, arg1_1, arg2_1, arg3_1, buf3, 128, 768, grid=grid(128), stream=stream0)
        del arg0_1
        del arg106_1
        del arg107_1
        del arg1_1
        del arg2_1
        del arg3_1
        buf4 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), out=buf4)
        del arg4_1
        buf5 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg6_1
        del arg7_1
        buf6 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf4, arg5_1, buf6, 98304, grid=grid(98304), stream=stream0)
        del arg5_1
        buf7 = empty((12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf5, (12, 64, 128), (64, 1, 768), 0), out=buf7)
        buf11 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_1, tensor, weights], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf7, buf11, 1536, 128, grid=grid(1536), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (128, 768), (768, 1), 0); del buf6  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
        del arg8_1
        del arg9_1
        buf12 = reinterpret_tensor(buf5, (12, 128, 64), (8192, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf10, (12, 128, 64), (64, 768, 1), 0), out=buf12)
        buf13 = reinterpret_tensor(buf10, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf12, buf13, 98304, grid=grid(98304), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (128, 768), (768, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (128, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), out=buf14)
        del arg10_1
        buf18 = reinterpret_tensor(buf13, (1, 128, 768), (98304, 768, 1), 0); del buf13  # reuse
        # Source Nodes: [add_1, sa_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf14, arg11_1, buf3, arg12_1, arg13_1, buf18, 128, 768, grid=grid(128), stream=stream0)
        del arg11_1
        del arg12_1
        del arg13_1
        buf19 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 3072), (1, 768), 0), out=buf19)
        del arg14_1
        buf20 = reinterpret_tensor(buf19, (1, 128, 3072), (393216, 3072, 1), 0); del buf19  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf20, arg15_1, 393216, grid=grid(393216), stream=stream0)
        del arg15_1
        buf21 = reinterpret_tensor(buf3, (128, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg16_1, (3072, 768), (1, 3072), 0), out=buf21)
        del arg16_1
        buf25 = reinterpret_tensor(buf14, (1, 128, 768), (98304, 768, 1), 0); del buf14  # reuse
        # Source Nodes: [add_2, hidden_state_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf21, arg17_1, buf18, arg18_1, arg19_1, buf25, 128, 768, grid=grid(128), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf26 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), out=buf26)
        del arg20_1
        buf27 = reinterpret_tensor(buf18, (128, 768), (768, 1), 0); del buf18  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg23_1, reinterpret_tensor(buf25, (128, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del arg22_1
        del arg23_1
        buf28 = reinterpret_tensor(buf4, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf26, arg21_1, buf28, 98304, grid=grid(98304), stream=stream0)
        del arg21_1
        buf29 = reinterpret_tensor(buf11, (12, 128, 128), (16384, 128, 1), 0); del buf11  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf27, (12, 64, 128), (64, 1, 768), 0), out=buf29)
        buf33 = reinterpret_tensor(buf7, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf7  # reuse
        # Source Nodes: [scores_3, tensor_1, weights_2], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf29, buf33, 1536, 128, grid=grid(1536), stream=stream0)
        buf32 = reinterpret_tensor(buf28, (128, 768), (768, 1), 0); del buf28  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg25_1, reinterpret_tensor(buf25, (128, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del arg24_1
        del arg25_1
        buf34 = reinterpret_tensor(buf27, (12, 128, 64), (8192, 64, 1), 0); del buf27  # reuse
        # Source Nodes: [context_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf32, (12, 128, 64), (64, 768, 1), 0), out=buf34)
        buf35 = reinterpret_tensor(buf32, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf34, buf35, 98304, grid=grid(98304), stream=stream0)
        buf36 = reinterpret_tensor(buf34, (128, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), out=buf36)
        del arg26_1
        buf40 = reinterpret_tensor(buf35, (1, 128, 768), (98304, 768, 1), 0); del buf35  # reuse
        # Source Nodes: [add_3, sa_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf36, arg27_1, buf25, arg28_1, arg29_1, buf40, 128, 768, grid=grid(128), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        buf41 = reinterpret_tensor(buf20, (128, 3072), (3072, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (128, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 3072), (1, 768), 0), out=buf41)
        del arg30_1
        buf42 = reinterpret_tensor(buf41, (1, 128, 3072), (393216, 3072, 1), 0); del buf41  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf42, arg31_1, 393216, grid=grid(393216), stream=stream0)
        del arg31_1
        buf43 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg32_1, (3072, 768), (1, 3072), 0), out=buf43)
        del arg32_1
        buf47 = buf25; del buf25  # reuse
        # Source Nodes: [add_4, hidden_state_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf43, arg33_1, buf40, arg34_1, arg35_1, buf47, 128, 768, grid=grid(128), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        buf48 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 768), (1, 768), 0), out=buf48)
        del arg36_1
        buf49 = reinterpret_tensor(buf40, (128, 768), (768, 1), 0); del buf40  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg39_1, reinterpret_tensor(buf47, (128, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
        del arg38_1
        del arg39_1
        buf50 = reinterpret_tensor(buf26, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf26  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf48, arg37_1, buf50, 98304, grid=grid(98304), stream=stream0)
        del arg37_1
        buf51 = reinterpret_tensor(buf33, (12, 128, 128), (16384, 128, 1), 0); del buf33  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf50, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf49, (12, 64, 128), (64, 1, 768), 0), out=buf51)
        buf55 = reinterpret_tensor(buf29, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf29  # reuse
        # Source Nodes: [scores_5, tensor_2, weights_4], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf51, buf55, 1536, 128, grid=grid(1536), stream=stream0)
        buf54 = reinterpret_tensor(buf50, (128, 768), (768, 1), 0); del buf50  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg41_1, reinterpret_tensor(buf47, (128, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
        del arg40_1
        del arg41_1
        buf56 = reinterpret_tensor(buf49, (12, 128, 64), (8192, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf54, (12, 128, 64), (64, 768, 1), 0), out=buf56)
        buf57 = reinterpret_tensor(buf54, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf56, buf57, 98304, grid=grid(98304), stream=stream0)
        buf58 = reinterpret_tensor(buf56, (128, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (128, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), out=buf58)
        del arg42_1
        buf62 = reinterpret_tensor(buf57, (1, 128, 768), (98304, 768, 1), 0); del buf57  # reuse
        # Source Nodes: [add_5, sa_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf58, arg43_1, buf47, arg44_1, arg45_1, buf62, 128, 768, grid=grid(128), stream=stream0)
        del arg43_1
        del arg44_1
        del arg45_1
        buf63 = reinterpret_tensor(buf42, (128, 3072), (3072, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (128, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), out=buf63)
        del arg46_1
        buf64 = reinterpret_tensor(buf63, (1, 128, 3072), (393216, 3072, 1), 0); del buf63  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf64, arg47_1, 393216, grid=grid(393216), stream=stream0)
        del arg47_1
        buf65 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), out=buf65)
        del arg48_1
        buf69 = buf47; del buf47  # reuse
        # Source Nodes: [add_6, hidden_state_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf65, arg49_1, buf62, arg50_1, arg51_1, buf69, 128, 768, grid=grid(128), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        buf70 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), out=buf70)
        del arg52_1
        buf71 = reinterpret_tensor(buf62, (128, 768), (768, 1), 0); del buf62  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
        del arg54_1
        del arg55_1
        buf72 = reinterpret_tensor(buf48, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf48  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf70, arg53_1, buf72, 98304, grid=grid(98304), stream=stream0)
        del arg53_1
        buf73 = reinterpret_tensor(buf55, (12, 128, 128), (16384, 128, 1), 0); del buf55  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf71, (12, 64, 128), (64, 1, 768), 0), out=buf73)
        buf77 = reinterpret_tensor(buf51, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf51  # reuse
        # Source Nodes: [scores_7, tensor_3, weights_6], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf73, buf77, 1536, 128, grid=grid(1536), stream=stream0)
        buf76 = reinterpret_tensor(buf72, (128, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf76)
        del arg56_1
        del arg57_1
        buf78 = reinterpret_tensor(buf71, (12, 128, 64), (8192, 64, 1), 0); del buf71  # reuse
        # Source Nodes: [context_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf76, (12, 128, 64), (64, 768, 1), 0), out=buf78)
        buf79 = reinterpret_tensor(buf76, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf78, buf79, 98304, grid=grid(98304), stream=stream0)
        buf80 = reinterpret_tensor(buf78, (128, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (128, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), out=buf80)
        del arg58_1
        buf84 = reinterpret_tensor(buf79, (1, 128, 768), (98304, 768, 1), 0); del buf79  # reuse
        # Source Nodes: [add_7, sa_output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf80, arg59_1, buf69, arg60_1, arg61_1, buf84, 128, 768, grid=grid(128), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        buf85 = reinterpret_tensor(buf64, (128, 3072), (3072, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (128, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 3072), (1, 768), 0), out=buf85)
        del arg62_1
        buf86 = reinterpret_tensor(buf85, (1, 128, 3072), (393216, 3072, 1), 0); del buf85  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf86, arg63_1, 393216, grid=grid(393216), stream=stream0)
        del arg63_1
        buf87 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg64_1, (3072, 768), (1, 3072), 0), out=buf87)
        del arg64_1
        buf91 = buf69; del buf69  # reuse
        # Source Nodes: [add_8, hidden_state_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf87, arg65_1, buf84, arg66_1, arg67_1, buf91, 128, 768, grid=grid(128), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        buf92 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), out=buf92)
        del arg68_1
        buf93 = reinterpret_tensor(buf84, (128, 768), (768, 1), 0); del buf84  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf91, (128, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del arg70_1
        del arg71_1
        buf94 = reinterpret_tensor(buf70, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf92, arg69_1, buf94, 98304, grid=grid(98304), stream=stream0)
        del arg69_1
        buf95 = reinterpret_tensor(buf77, (12, 128, 128), (16384, 128, 1), 0); del buf77  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf93, (12, 64, 128), (64, 1, 768), 0), out=buf95)
        buf99 = reinterpret_tensor(buf73, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf73  # reuse
        # Source Nodes: [scores_9, tensor_4, weights_8], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf95, buf99, 1536, 128, grid=grid(1536), stream=stream0)
        buf98 = reinterpret_tensor(buf94, (128, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf91, (128, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf98)
        del arg72_1
        del arg73_1
        buf100 = reinterpret_tensor(buf93, (12, 128, 64), (8192, 64, 1), 0); del buf93  # reuse
        # Source Nodes: [context_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf98, (12, 128, 64), (64, 768, 1), 0), out=buf100)
        buf101 = reinterpret_tensor(buf98, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf100, buf101, 98304, grid=grid(98304), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (128, 768), (768, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (128, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), out=buf102)
        del arg74_1
        buf106 = reinterpret_tensor(buf101, (1, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
        # Source Nodes: [add_9, sa_output_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf102, arg75_1, buf91, arg76_1, arg77_1, buf106, 128, 768, grid=grid(128), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        buf107 = reinterpret_tensor(buf86, (128, 3072), (3072, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 3072), (1, 768), 0), out=buf107)
        del arg78_1
        buf108 = reinterpret_tensor(buf107, (1, 128, 3072), (393216, 3072, 1), 0); del buf107  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf108, arg79_1, 393216, grid=grid(393216), stream=stream0)
        del arg79_1
        buf109 = reinterpret_tensor(buf91, (128, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg80_1, (3072, 768), (1, 3072), 0), out=buf109)
        del arg80_1
        buf113 = reinterpret_tensor(buf102, (1, 128, 768), (98304, 768, 1), 0); del buf102  # reuse
        # Source Nodes: [add_10, hidden_state_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf109, arg81_1, buf106, arg82_1, arg83_1, buf113, 128, 768, grid=grid(128), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        buf114 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), out=buf114)
        del arg84_1
        buf115 = reinterpret_tensor(buf106, (128, 768), (768, 1), 0); del buf106  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg87_1, reinterpret_tensor(buf113, (128, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf115)
        del arg86_1
        del arg87_1
        buf116 = reinterpret_tensor(buf92, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf92  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.div]
        triton_poi_fused_div_1.run(buf114, arg85_1, buf116, 98304, grid=grid(98304), stream=stream0)
        del arg85_1
        del buf114
        buf117 = reinterpret_tensor(buf99, (12, 128, 128), (16384, 128, 1), 0); del buf99  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf115, (12, 64, 128), (64, 1, 768), 0), out=buf117)
        buf121 = reinterpret_tensor(buf95, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf95  # reuse
        # Source Nodes: [scores_11, tensor_5, weights_10], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_2.run(buf117, buf121, 1536, 128, grid=grid(1536), stream=stream0)
        del buf117
        buf120 = reinterpret_tensor(buf116, (128, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg89_1, reinterpret_tensor(buf113, (128, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
        del arg88_1
        del arg89_1
        buf122 = reinterpret_tensor(buf115, (12, 128, 64), (8192, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [context_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf120, (12, 128, 64), (64, 768, 1), 0), out=buf122)
        del buf121
        buf123 = reinterpret_tensor(buf120, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf122, buf123, 98304, grid=grid(98304), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (128, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (128, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), out=buf124)
        del arg90_1
        buf128 = reinterpret_tensor(buf123, (1, 128, 768), (98304, 768, 1), 0); del buf123  # reuse
        # Source Nodes: [add_11, sa_output_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf124, arg91_1, buf113, arg92_1, arg93_1, buf128, 128, 768, grid=grid(128), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        buf129 = reinterpret_tensor(buf108, (128, 3072), (3072, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), out=buf129)
        del arg94_1
        buf130 = reinterpret_tensor(buf129, (1, 128, 3072), (393216, 3072, 1), 0); del buf129  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf130, arg95_1, 393216, grid=grid(393216), stream=stream0)
        del arg95_1
        buf131 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), out=buf131)
        del arg96_1
        del buf130
        buf135 = buf113; del buf113  # reuse
        # Source Nodes: [add_12, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf131, arg97_1, buf128, arg98_1, arg99_1, buf135, 128, 768, grid=grid(128), stream=stream0)
        del arg97_1
        del arg98_1
        del arg99_1
        del buf128
        buf136 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), out=buf136)
        del arg100_1
        buf140 = buf135; del buf135  # reuse
        # Source Nodes: [prediction_logits_1, prediction_logits_2], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_6.run(buf136, arg101_1, arg102_1, arg103_1, buf140, 128, 768, grid=grid(128), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del buf136
        buf141 = empty((128, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_logits_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf140, (128, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf141)
        del arg104_1
        del arg105_1
        del buf140
        buf142 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf141, buf142, 512, 7631, grid=grid(512), stream=stream0)
        buf143 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_8.run(buf142, buf143, 128, 4, grid=grid(128), stream=stream0)
        buf144 = buf142; del buf142  # reuse
        # Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_9.run(buf141, buf143, buf144, 512, 7631, grid=grid(512), stream=stream0)
        buf145 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [mlm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_10.run(buf144, buf145, 128, 4, grid=grid(128), stream=stream0)
        del buf144
        buf146 = empty((), device='cuda', dtype=torch.float32)
        buf148 = buf146; del buf146  # reuse
        # Source Nodes: [mlm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_11.run(buf148, arg108_1, buf141, buf143, buf145, 1, 128, grid=grid(1), stream=stream0)
        del arg108_1
        return (buf148, reinterpret_tensor(buf141, (1, 128, 30522), (3906816, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg107_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg108_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForMaskedLM', benchmark_compiled_module)
