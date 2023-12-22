
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


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt52wcl7hjfdhfmag3zny4kh5c642ooxulyn4vqvoc2wwpvmbic.py
# Source Nodes: [embeddings, embeddings_1, input_embeds, position_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# embeddings => add
# embeddings_1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# input_embeds => embedding
# position_embeddings => embedding_1
triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (768*x0)), tmp34, rmask & xmask)
    tmp35 = 768.0
    tmp36 = tmp13 / tmp35
    tmp37 = 1e-12
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp39 / tmp35
    tl.store(out_ptr4 + (x0), tmp40, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lz/clzbaa3f46byubikdoibptgpejam6deowdr4d3iv6fdmgfkcf2zy.py
# Source Nodes: [q_1], Original ATen: [aten.div, aten.transpose]
# q_1 => div
triton_poi_fused_div_transpose_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_transpose_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (x0 + (64*x2) + (768*x1)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqyjamdrxgxrmfv4tzjif5z5latxip3qkrwdarjnfug3htrw5eq.py
# Source Nodes: [attention_mask, eq, view_3], Original ATen: [aten.eq, aten.ones, aten.view]
# attention_mask => full
# eq => eq
# view_3 => view_12
triton_poi_fused_eq_ones_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_ones_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldofq3c7d762gg6pl36chbohkp23tfqaqptx6u3wzxkapvjr6kw.py
# Source Nodes: [scores_1, tensor, weights], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
# scores_1 => where
# tensor => full_default
# weights => amax, div_1, exp, sub_1, sum_1
triton_per_fused__softmax_lift_fresh_masked_fill_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lift_fresh_masked_fill_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp14, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnlk4nvfxiehmquhhyh6axyy5w2wgckrxzyi3p3e6sn2ia3uolc.py
# Source Nodes: [sa_output], Original ATen: [aten.view]
# sa_output => view_17
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbz7qhy2t27qsougqja73r5om35y652cklnzx4neqhgudrjsmco.py
# Source Nodes: [add_1, sa_output_1, x], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_1 => add_3
# sa_output_1 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# x => view_19
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbgektjntbda6o45gz5wrojrqsmf5i6riqlxow5hpnx3a4esbiz.py
# Source Nodes: [x_1, x_2], Original ATen: [aten.gelu, aten.view]
# x_1 => add_6, erf, mul_4, mul_5, mul_6
# x_2 => view_21
triton_poi_fused_gelu_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqhelfet7xcx556yi2ir3sdi747g4t5x5erflcqvknaxvl7eaan.py
# Source Nodes: [add_2, hidden_state_1, l__mod___distilbert_transformer_layer_1_attention_q_lin, sa_output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_7
# hidden_state_1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_3, var_mean_2
# l__mod___distilbert_transformer_layer_1_attention_q_lin => view_23
# sa_output_1 => add_5, mul_3
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdtj3vzg47trre4a4llz5unjvqkvcob2v2t55sluhpmo7x2ywpr.py
# Source Nodes: [add_3, hidden_state_1, sa_output_3, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_10
# hidden_state_1 => add_9, mul_8
# sa_output_3 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_5, var_mean_3
# x_4 => view_42
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kllb3ubh2notrshait5zo5cva5g6o7axqqehsontgxztwgs3vt.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_6
# start_loss => amax_6, exp_6, log, sub_19, sub_20, sum_7
triton_per_fused__log_softmax_clone_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdfyeab5alj4bvtdopju3cts4ayy6ncxvi2jbe4pf2mamegjixw.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_7
# end_loss => amax_7, exp_7, log_1, sub_21, sub_22, sum_10
triton_per_fused__log_softmax_clone_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3khqjq66bkm6tyb5g3sw2ts5ne4husx425es2giriupw6dhwrh.py
# Source Nodes: [add_13, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# add_13 => add_45
# end_loss => convert_element_type_1, div_13, ne_3, neg_1, sum_11, sum_12, where_9
# end_positions => clamp_max_1, clamp_min_1
# loss => div_14
# start_loss => convert_element_type, div_12, full_default_6, full_default_7, ne, neg, sum_8, sum_9, where_7
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_11', 'mutated_arg_names': []},
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
    tmp4 = tl.full([1], 128, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tmp12 + 128
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 128), "index out of bounds: 0 <= tmp15 < 128")
    tmp16 = tl.load(in_ptr2 + (tmp15), None, eviction_policy='evict_last')
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tmp6.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tl.where(tmp11, tmp10, tmp2)
    tmp24 = tmp23 + 128
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 128), "index out of bounds: 0 <= tmp26 < 128")
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106 = args
    args.clear()
    assert_size_stride(primals_1, (30522, 768), (768, 1))
    assert_size_stride(primals_2, (512, 768), (768, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, 768), (768, 1))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (3072, 768), (768, 1))
    assert_size_stride(primals_16, (3072, ), (1, ))
    assert_size_stride(primals_17, (768, 3072), (3072, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, 768), (768, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768), (768, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, 768), (768, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (3072, 768), (768, 1))
    assert_size_stride(primals_32, (3072, ), (1, ))
    assert_size_stride(primals_33, (768, 3072), (3072, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 768), (768, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, 768), (768, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768), (768, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, 768), (768, 1))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (3072, 768), (768, 1))
    assert_size_stride(primals_48, (3072, ), (1, ))
    assert_size_stride(primals_49, (768, 3072), (3072, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, 768), (768, 1))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (3072, 768), (768, 1))
    assert_size_stride(primals_64, (3072, ), (1, ))
    assert_size_stride(primals_65, (768, 3072), (3072, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, 768), (768, 1))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768), (768, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, 768), (768, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (3072, 768), (768, 1))
    assert_size_stride(primals_80, (3072, ), (1, ))
    assert_size_stride(primals_81, (768, 3072), (3072, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, 768), (768, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (2, 768), (768, 1))
    assert_size_stride(primals_102, (2, ), (1, ))
    assert_size_stride(primals_103, (1, 512), (512, 1))
    assert_size_stride(primals_104, (1, 128), (128, 1))
    assert_size_stride(primals_105, (1, ), (1, ))
    assert_size_stride(primals_106, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf230 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, input_embeds, position_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0.run(primals_104, primals_1, primals_103, primals_2, primals_3, primals_4, buf3, buf4, buf230, 128, 768, grid=grid(128), stream=stream0)
        del primals_1
        del primals_2
        del primals_4
        # Source Nodes: [embeddings_1, hidden_state], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf5 = aten.native_dropout(buf4, 0.1, True)
        buf6 = buf5[0]
        buf7 = buf5[1]
        del buf5
        buf8 = reinterpret_tensor(buf4, (128, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (1, 768), 0), out=buf8)
        buf9 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del primals_8
        buf10 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
        del primals_10
        buf11 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf229 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_1], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf8, primals_6, buf11, buf229, 98304, grid=grid(98304), stream=stream0)
        del primals_6
        buf12 = empty((12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf9, (12, 64, 128), (64, 1, 768), 0), out=buf12)
        buf13 = empty((1, 1, 1, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [attention_mask, eq, view_3], Original ATen: [aten.eq, aten.ones, aten.view]
        triton_poi_fused_eq_ones_view_2.run(buf13, 128, grid=grid(128), stream=stream0)
        buf16 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_1, tensor, weights], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf12, buf16, 1536, 128, grid=grid(1536), stream=stream0)
        # Source Nodes: [scores_1, tensor, weights, weights_1], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf17 = aten.native_dropout(buf16, 0.1, True)
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = reinterpret_tensor(buf11, (12, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf10, (12, 128, 64), (64, 768, 1), 0), out=buf20)
        buf21 = buf8; del buf8  # reuse
        # Source Nodes: [sa_output], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf20, buf21, 98304, grid=grid(98304), stream=stream0)
        buf22 = reinterpret_tensor(buf20, (128, 768), (768, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf21, reinterpret_tensor(primals_11, (768, 768), (1, 768), 0), out=buf22)
        buf26 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf27 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf228 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, sa_output_1, x], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf22, primals_12, buf6, primals_13, primals_14, buf26, buf27, buf228, 128, 768, grid=grid(128), stream=stream0)
        del primals_12
        buf28 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf27, reinterpret_tensor(primals_15, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf28)
        del primals_16
        buf29 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf28, buf29, 393216, grid=grid(393216), stream=stream0)
        buf30 = buf22; del buf22  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, buf29, reinterpret_tensor(primals_17, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf30)
        del primals_18
        # Source Nodes: [ffn_output], Original ATen: [aten.native_dropout]
        buf31 = aten.native_dropout(reinterpret_tensor(buf30, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        buf37 = reinterpret_tensor(buf30, (1, 128, 768), (98304, 768, 1), 0); del buf30  # reuse
        buf38 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf227 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, hidden_state_1, l__mod___distilbert_transformer_layer_1_attention_q_lin, sa_output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf32, buf26, primals_13, primals_14, primals_19, primals_20, buf37, buf38, buf227, 128, 768, grid=grid(128), stream=stream0)
        del primals_14
        buf39 = reinterpret_tensor(buf32, (128, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf38, reinterpret_tensor(primals_21, (768, 768), (1, 768), 0), out=buf39)
        buf40 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf38, reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf40)
        del primals_24
        buf41 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, buf38, reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
        del primals_26
        buf42 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf226 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_3], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf39, primals_22, buf42, buf226, 98304, grid=grid(98304), stream=stream0)
        del primals_22
        buf43 = buf12; del buf12  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf40, (12, 64, 128), (64, 1, 768), 0), out=buf43)
        buf46 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_3, tensor, weights_2], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf43, buf46, 1536, 128, grid=grid(1536), stream=stream0)
        # Source Nodes: [scores_3, tensor, weights_2, weights_3], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf47 = aten.native_dropout(buf46, 0.1, True)
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        buf50 = reinterpret_tensor(buf42, (12, 128, 64), (8192, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [context_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf41, (12, 128, 64), (64, 768, 1), 0), out=buf50)
        buf51 = buf39; del buf39  # reuse
        # Source Nodes: [sa_output_2], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf50, buf51, 98304, grid=grid(98304), stream=stream0)
        buf52 = reinterpret_tensor(buf50, (128, 768), (768, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf51, reinterpret_tensor(primals_27, (768, 768), (1, 768), 0), out=buf52)
        buf53 = reinterpret_tensor(buf52, (1, 128, 768), (98304, 768, 1), 0); del buf52  # reuse
        buf57 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf58 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf225 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, hidden_state_1, sa_output_3, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf53, primals_28, buf37, primals_19, primals_20, primals_29, primals_30, buf57, buf58, buf225, 128, 768, grid=grid(128), stream=stream0)
        del primals_20
        del primals_28
        buf59 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_32, buf58, reinterpret_tensor(primals_31, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf59)
        del primals_32
        buf60 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf59, buf60, 393216, grid=grid(393216), stream=stream0)
        buf61 = reinterpret_tensor(buf53, (128, 768), (768, 1), 0); del buf53  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_34, buf60, reinterpret_tensor(primals_33, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf61)
        del primals_34
        # Source Nodes: [ffn_output_2], Original ATen: [aten.native_dropout]
        buf62 = aten.native_dropout(reinterpret_tensor(buf61, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        buf68 = reinterpret_tensor(buf61, (1, 128, 768), (98304, 768, 1), 0); del buf61  # reuse
        buf69 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf224 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, hidden_state_2, l__mod___distilbert_transformer_layer_2_attention_q_lin, sa_output_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf63, buf57, primals_29, primals_30, primals_35, primals_36, buf68, buf69, buf224, 128, 768, grid=grid(128), stream=stream0)
        del primals_30
        buf70 = reinterpret_tensor(buf63, (128, 768), (768, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf69, reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), out=buf70)
        buf71 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_40, buf69, reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
        del primals_40
        buf72 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, buf69, reinterpret_tensor(primals_41, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
        del primals_42
        buf73 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf223 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_5], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf70, primals_38, buf73, buf223, 98304, grid=grid(98304), stream=stream0)
        del primals_38
        buf74 = buf43; del buf43  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf71, (12, 64, 128), (64, 1, 768), 0), out=buf74)
        buf77 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_5, tensor, weights_4], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf74, buf77, 1536, 128, grid=grid(1536), stream=stream0)
        # Source Nodes: [scores_5, tensor, weights_4, weights_5], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf78 = aten.native_dropout(buf77, 0.1, True)
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = reinterpret_tensor(buf73, (12, 128, 64), (8192, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf72, (12, 128, 64), (64, 768, 1), 0), out=buf81)
        buf82 = buf70; del buf70  # reuse
        # Source Nodes: [sa_output_4], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf81, buf82, 98304, grid=grid(98304), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (128, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_43, (768, 768), (1, 768), 0), out=buf83)
        buf84 = reinterpret_tensor(buf83, (1, 128, 768), (98304, 768, 1), 0); del buf83  # reuse
        buf88 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf89 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf222 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, hidden_state_2, sa_output_5, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf84, primals_44, buf68, primals_35, primals_36, primals_45, primals_46, buf88, buf89, buf222, 128, 768, grid=grid(128), stream=stream0)
        del primals_36
        del primals_44
        buf90 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, buf89, reinterpret_tensor(primals_47, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf90)
        del primals_48
        buf91 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf90, buf91, 393216, grid=grid(393216), stream=stream0)
        buf92 = reinterpret_tensor(buf84, (128, 768), (768, 1), 0); del buf84  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_50, buf91, reinterpret_tensor(primals_49, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf92)
        del primals_50
        # Source Nodes: [ffn_output_4], Original ATen: [aten.native_dropout]
        buf93 = aten.native_dropout(reinterpret_tensor(buf92, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf94 = buf93[0]
        buf95 = buf93[1]
        del buf93
        buf99 = reinterpret_tensor(buf92, (1, 128, 768), (98304, 768, 1), 0); del buf92  # reuse
        buf100 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf221 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, hidden_state_3, l__mod___distilbert_transformer_layer_3_attention_q_lin, sa_output_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf94, buf88, primals_45, primals_46, primals_51, primals_52, buf99, buf100, buf221, 128, 768, grid=grid(128), stream=stream0)
        del primals_46
        buf101 = reinterpret_tensor(buf94, (128, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf100, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), out=buf101)
        buf102 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_56, buf100, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
        del primals_56
        buf103 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_58, buf100, reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf103)
        del primals_58
        buf104 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_7], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf101, primals_54, buf104, buf220, 98304, grid=grid(98304), stream=stream0)
        del primals_54
        buf105 = buf74; del buf74  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf102, (12, 64, 128), (64, 1, 768), 0), out=buf105)
        buf108 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_7, tensor, weights_6], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf105, buf108, 1536, 128, grid=grid(1536), stream=stream0)
        # Source Nodes: [scores_7, tensor, weights_6, weights_7], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf109 = aten.native_dropout(buf108, 0.1, True)
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf112 = reinterpret_tensor(buf104, (12, 128, 64), (8192, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [context_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf103, (12, 128, 64), (64, 768, 1), 0), out=buf112)
        buf113 = buf101; del buf101  # reuse
        # Source Nodes: [sa_output_6], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf112, buf113, 98304, grid=grid(98304), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (128, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_59, (768, 768), (1, 768), 0), out=buf114)
        buf115 = reinterpret_tensor(buf114, (1, 128, 768), (98304, 768, 1), 0); del buf114  # reuse
        buf119 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf120 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf219 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, hidden_state_3, sa_output_7, x_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf115, primals_60, buf99, primals_51, primals_52, primals_61, primals_62, buf119, buf120, buf219, 128, 768, grid=grid(128), stream=stream0)
        del primals_52
        del primals_60
        buf121 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, buf120, reinterpret_tensor(primals_63, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf121)
        del primals_64
        buf122 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_14], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf121, buf122, 393216, grid=grid(393216), stream=stream0)
        buf123 = reinterpret_tensor(buf115, (128, 768), (768, 1), 0); del buf115  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_66, buf122, reinterpret_tensor(primals_65, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf123)
        del primals_66
        # Source Nodes: [ffn_output_6], Original ATen: [aten.native_dropout]
        buf124 = aten.native_dropout(reinterpret_tensor(buf123, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf125 = buf124[0]
        buf126 = buf124[1]
        del buf124
        buf130 = reinterpret_tensor(buf123, (1, 128, 768), (98304, 768, 1), 0); del buf123  # reuse
        buf131 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf218 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, hidden_state_4, l__mod___distilbert_transformer_layer_4_attention_q_lin, sa_output_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf125, buf119, primals_61, primals_62, primals_67, primals_68, buf130, buf131, buf218, 128, 768, grid=grid(128), stream=stream0)
        del primals_62
        buf132 = reinterpret_tensor(buf125, (128, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf131, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), out=buf132)
        buf133 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf131, reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del primals_72
        buf134 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_74, buf131, reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del primals_74
        buf135 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf217 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_9], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf132, primals_70, buf135, buf217, 98304, grid=grid(98304), stream=stream0)
        del primals_70
        buf136 = buf105; del buf105  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf133, (12, 64, 128), (64, 1, 768), 0), out=buf136)
        buf139 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_9, tensor, weights_8], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf136, buf139, 1536, 128, grid=grid(1536), stream=stream0)
        # Source Nodes: [scores_9, tensor, weights_8, weights_9], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf140 = aten.native_dropout(buf139, 0.1, True)
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = reinterpret_tensor(buf135, (12, 128, 64), (8192, 64, 1), 0); del buf135  # reuse
        # Source Nodes: [context_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf134, (12, 128, 64), (64, 768, 1), 0), out=buf143)
        buf144 = buf132; del buf132  # reuse
        # Source Nodes: [sa_output_8], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf143, buf144, 98304, grid=grid(98304), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (128, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), out=buf145)
        buf146 = reinterpret_tensor(buf145, (1, 128, 768), (98304, 768, 1), 0); del buf145  # reuse
        buf150 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf151 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf216 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, hidden_state_4, sa_output_9, x_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf146, primals_76, buf130, primals_67, primals_68, primals_77, primals_78, buf150, buf151, buf216, 128, 768, grid=grid(128), stream=stream0)
        del primals_68
        del primals_76
        buf152 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_80, buf151, reinterpret_tensor(primals_79, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf152)
        del primals_80
        buf153 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_18], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf152, buf153, 393216, grid=grid(393216), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (128, 768), (768, 1), 0); del buf146  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_82, buf153, reinterpret_tensor(primals_81, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf154)
        del primals_82
        # Source Nodes: [ffn_output_8], Original ATen: [aten.native_dropout]
        buf155 = aten.native_dropout(reinterpret_tensor(buf154, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf161 = reinterpret_tensor(buf154, (1, 128, 768), (98304, 768, 1), 0); del buf154  # reuse
        buf162 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf215 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, hidden_state_5, l__mod___distilbert_transformer_layer_5_attention_q_lin, sa_output_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf156, buf150, primals_77, primals_78, primals_83, primals_84, buf161, buf162, buf215, 128, 768, grid=grid(128), stream=stream0)
        del primals_78
        buf163 = reinterpret_tensor(buf156, (128, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf162, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), out=buf163)
        buf164 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_k_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf162, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf164)
        del primals_88
        buf165 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_v_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_90, buf162, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del primals_90
        buf166 = empty((1, 12, 128, 64), device='cuda', dtype=torch.float32)
        buf214 = empty_strided((12, 64, 128), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_11], Original ATen: [aten.div, aten.transpose]
        triton_poi_fused_div_transpose_1.run(buf163, primals_86, buf166, buf214, 98304, grid=grid(98304), stream=stream0)
        del primals_86
        buf167 = buf136; del buf136  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf164, (12, 64, 128), (64, 1, 768), 0), out=buf167)
        buf170 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_11, tensor, weights_10], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_lift_fresh_masked_fill_3.run(buf13, buf167, buf170, 1536, 128, grid=grid(1536), stream=stream0)
        del buf167
        # Source Nodes: [scores_11, tensor, weights_10, weights_11], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
        buf171 = aten.native_dropout(buf170, 0.1, True)
        buf172 = buf171[0]
        buf173 = buf171[1]
        del buf171
        buf174 = reinterpret_tensor(buf166, (12, 128, 64), (8192, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [context_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf165, (12, 128, 64), (64, 768, 1), 0), out=buf174)
        buf175 = buf163; del buf163  # reuse
        # Source Nodes: [sa_output_10], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf174, buf175, 98304, grid=grid(98304), stream=stream0)
        buf176 = reinterpret_tensor(buf174, (128, 768), (768, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf175, reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), out=buf176)
        buf177 = reinterpret_tensor(buf176, (1, 128, 768), (98304, 768, 1), 0); del buf176  # reuse
        buf181 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf182 = empty((128, 768), device='cuda', dtype=torch.float32)
        buf213 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, hidden_state_5, sa_output_11, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf177, primals_92, buf161, primals_83, primals_84, primals_93, primals_94, buf181, buf182, buf213, 128, 768, grid=grid(128), stream=stream0)
        del primals_84
        del primals_92
        buf183 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf182, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf183)
        del primals_96
        buf184 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_22], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf183, buf184, 393216, grid=grid(393216), stream=stream0)
        buf185 = reinterpret_tensor(buf177, (128, 768), (768, 1), 0); del buf177  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_98, buf184, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf185)
        del primals_98
        # Source Nodes: [ffn_output_10], Original ATen: [aten.native_dropout]
        buf186 = aten.native_dropout(reinterpret_tensor(buf185, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf192 = reinterpret_tensor(buf185, (1, 128, 768), (98304, 768, 1), 0); del buf185  # reuse
        buf193 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf212 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, hidden_states, sa_output_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf187, buf181, primals_93, primals_94, primals_99, primals_100, buf192, buf193, buf212, 128, 768, grid=grid(128), stream=stream0)
        del buf187
        del primals_100
        del primals_94
        # Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf194 = aten.native_dropout(buf193, 0.1, True)
        del buf193
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf197 = empty((128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (128, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 2), (1, 768), 0), out=buf197)
        buf198 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf202 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_9.run(buf197, primals_102, buf198, buf202, 1, 128, grid=grid(1), stream=stream0)
        buf199 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf206 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_10.run(buf197, primals_102, buf199, buf206, 1, 128, grid=grid(1), stream=stream0)
        del buf197
        del primals_102
        buf203 = empty((1, ), device='cuda', dtype=torch.bool)
        buf207 = empty((1, ), device='cuda', dtype=torch.bool)
        buf231 = empty((), device='cuda', dtype=torch.float32)
        buf208 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf209 = empty((1, 1), device='cuda', dtype=torch.int64)
        buf210 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf211 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [add_13, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_11.run(primals_105, primals_106, buf202, buf206, buf203, buf207, buf231, buf208, buf209, buf210, buf211, 1, grid=grid(1), stream=stream0)
        del primals_105
        del primals_106
        return (buf231, buf198, buf199, primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, reinterpret_tensor(primals_103, (1, 128), (512, 1), 0), buf3, buf7, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), buf13, buf19, buf21, buf26, buf27, buf28, buf29, buf33, buf37, buf38, buf49, buf51, buf57, buf58, buf59, buf60, buf64, buf68, buf69, buf80, buf82, buf88, buf89, buf90, buf91, buf95, buf99, buf100, buf111, buf113, buf119, buf120, buf121, buf122, buf126, buf130, buf131, buf142, buf144, buf150, buf151, buf152, buf153, buf157, buf161, buf162, buf173, buf175, buf181, buf182, buf183, buf184, buf188, buf192, buf196, reinterpret_tensor(buf195, (128, 768), (768, 1), 0), buf202, buf203, buf206, buf207, buf208, buf209, buf210, buf211, reinterpret_tensor(primals_101, (2, 768), (768, 1), 0), buf212, reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), buf213, reinterpret_tensor(primals_91, (768, 768), (768, 1), 0), reinterpret_tensor(buf172, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf165, (12, 64, 128), (64, 1, 768), 0), buf170, buf214, reinterpret_tensor(buf164, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), buf215, reinterpret_tensor(primals_81, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 768), (768, 1), 0), buf216, reinterpret_tensor(primals_75, (768, 768), (768, 1), 0), reinterpret_tensor(buf141, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf134, (12, 64, 128), (64, 1, 768), 0), buf139, buf217, reinterpret_tensor(buf133, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (768, 1), 0), reinterpret_tensor(primals_71, (768, 768), (768, 1), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), buf218, reinterpret_tensor(primals_65, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_63, (3072, 768), (768, 1), 0), buf219, reinterpret_tensor(primals_59, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf103, (12, 64, 128), (64, 1, 768), 0), buf108, buf220, reinterpret_tensor(buf102, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (768, 1), 0), reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), buf221, reinterpret_tensor(primals_49, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 768), (768, 1), 0), buf222, reinterpret_tensor(primals_43, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf72, (12, 64, 128), (64, 1, 768), 0), buf77, buf223, reinterpret_tensor(buf71, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_41, (768, 768), (768, 1), 0), reinterpret_tensor(primals_39, (768, 768), (768, 1), 0), reinterpret_tensor(primals_37, (768, 768), (768, 1), 0), buf224, reinterpret_tensor(primals_33, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_31, (3072, 768), (768, 1), 0), buf225, reinterpret_tensor(primals_27, (768, 768), (768, 1), 0), reinterpret_tensor(buf48, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf41, (12, 64, 128), (64, 1, 768), 0), buf46, buf226, reinterpret_tensor(buf40, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (768, 1), 0), reinterpret_tensor(primals_23, (768, 768), (768, 1), 0), reinterpret_tensor(primals_21, (768, 768), (768, 1), 0), buf227, reinterpret_tensor(primals_17, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_15, (3072, 768), (768, 1), 0), buf228, reinterpret_tensor(primals_11, (768, 768), (768, 1), 0), reinterpret_tensor(buf18, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf10, (12, 64, 128), (64, 1, 768), 0), buf16, buf229, reinterpret_tensor(buf9, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (768, 1), 0), buf230, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_104 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_105 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_106 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForQuestionAnswering', benchmark_compiled_module)
