
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmsrhm7ylgoiaq5rii3zr3sfp3knhxsmxerybx3pr2mnzu5f744.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp43, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ud/cud47idjir4h5mbtkwghcn36iwgxg5sefnuixkbln6lsufzx6qi5.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chi5sk4rluhen7ipk25gcuwuiye6r5hbvfuqp273vo47wiwboipv.py
# Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_4
# fourier_output => var_mean_1
triton_red_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3w3ew6icnsqhggk7ub6glahttb4qspbf7psojbrpul5pwt53k6.py
# Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_4
# fourier_output => var_mean_1
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckowshbtun6wxaopdyo36glkihu55bclainklkawdoaix4vu2bsm.py
# Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_4
# fourier_output => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_poi_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caq5hlz4vuqozdveowfvjfx34cdyel7lv3fm63zcybgm4yho5qgk.py
# Source Nodes: [add_2, add_3, intermediate_output, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_7
# add_3 => add_8
# intermediate_output => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/fa/cfaxvsq525hgmz7f77goq26qjyr4srt76cnh35vkmx6baybqtruq.py
# Source Nodes: [add_4, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
# add_4 => add_9
# hidden_states_6 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/pe/cpe7ktl5g6cjphkigllohw6p66fjyc5hwo3bbsksdd5njvnivffc.py
# Source Nodes: [add_49, add_50, hidden_states_85, hidden_states_87, mul_48, mul_49, mul_50, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
# add_49 => add_100
# add_50 => add_101
# hidden_states_85 => mul_101
# hidden_states_87 => add_102, add_103, mul_102, mul_103, rsqrt_25, sub_25, var_mean_25
# mul_48 => mul_98
# mul_49 => mul_99
# mul_50 => mul_100
# pow_13 => pow_13
# tanh_12 => tanh_13
triton_per_fused_add_mul_native_layer_norm_pow_tanh_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 768.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp42, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbf2ph6v7ahqomm2hf3ahnnkgr7obq5oz72emoieqrcnhoowfuin.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax, exp, sub_26, sum_1
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidv73foymc5a6zfcawwehrrxlrqkraonidfxy66nf3obxxqkr7g.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type_12, div, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_1
triton_per_fused_nll_loss_forward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = tmp4 + 32000
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 32000), "index out of bounds: 0 <= tmp7 < 32000")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32000, 768), (768, 1))
    assert_size_stride(arg1_1, (4, 768), (768, 1))
    assert_size_stride(arg2_1, (512, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (3072, 768), (768, 1))
    assert_size_stride(arg10_1, (3072, ), (1, ))
    assert_size_stride(arg11_1, (768, 3072), (3072, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (3072, 768), (768, 1))
    assert_size_stride(arg18_1, (3072, ), (1, ))
    assert_size_stride(arg19_1, (768, 3072), (3072, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (3072, 768), (768, 1))
    assert_size_stride(arg26_1, (3072, ), (1, ))
    assert_size_stride(arg27_1, (768, 3072), (3072, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (3072, 768), (768, 1))
    assert_size_stride(arg34_1, (3072, ), (1, ))
    assert_size_stride(arg35_1, (768, 3072), (3072, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (3072, 768), (768, 1))
    assert_size_stride(arg42_1, (3072, ), (1, ))
    assert_size_stride(arg43_1, (768, 3072), (3072, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (3072, 768), (768, 1))
    assert_size_stride(arg50_1, (3072, ), (1, ))
    assert_size_stride(arg51_1, (768, 3072), (3072, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (3072, 768), (768, 1))
    assert_size_stride(arg58_1, (3072, ), (1, ))
    assert_size_stride(arg59_1, (768, 3072), (3072, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (3072, 768), (768, 1))
    assert_size_stride(arg66_1, (3072, ), (1, ))
    assert_size_stride(arg67_1, (768, 3072), (3072, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (3072, 768), (768, 1))
    assert_size_stride(arg74_1, (3072, ), (1, ))
    assert_size_stride(arg75_1, (768, 3072), (3072, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (3072, 768), (768, 1))
    assert_size_stride(arg82_1, (3072, ), (1, ))
    assert_size_stride(arg83_1, (768, 3072), (3072, 1))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (3072, 768), (768, 1))
    assert_size_stride(arg90_1, (3072, ), (1, ))
    assert_size_stride(arg91_1, (768, 3072), (3072, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (3072, 768), (768, 1))
    assert_size_stride(arg98_1, (3072, ), (1, ))
    assert_size_stride(arg99_1, (768, 3072), (3072, 1))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (32000, 768), (768, 1))
    assert_size_stride(arg110_1, (32000, ), (1, ))
    assert_size_stride(arg111_1, (1, 512), (512, 1))
    assert_size_stride(arg112_1, (1, 512), (512, 1))
    assert_size_stride(arg113_1, (1, 512), (512, 1))
    assert_size_stride(arg114_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg113_1, arg0_1, arg111_1, arg1_1, arg112_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 512, 768, grid=grid(512), stream=stream0)
        del arg0_1
        del arg111_1
        del arg112_1
        del arg113_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf5 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf5)
        del arg5_1
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg6_1, 393216, grid=grid(393216), stream=stream0)
        del arg6_1
        # Source Nodes: [fft_fftn], Original ATen: [aten._to_copy]
        buf7 = torch.ops.prims.convert_element_type.default(reinterpret_tensor(buf6, (1, 512, 768), (0, 768, 1), 0), torch.complex64)
        buf8 = buf7
        del buf7
        # Source Nodes: [fft_fftn], Original ATen: [aten._fft_c2c]
        buf9 = aten._fft_c2c(buf8, [1, 2], 0, True)
        del buf8
        buf10 = buf9
        del buf9
        # Source Nodes: [outputs], Original ATen: [aten.view_as_real]
        buf11 = aten.view_as_real(buf10)
        del buf10
        buf12 = buf11
        del buf11
        buf13 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 512, 1, 6), (3072, 6, 3072, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf6, buf12, buf13, buf14, buf15, 3072, 128, grid=grid(3072), stream=stream0)
        buf16 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf13, buf14, buf15, buf16, buf17, 512, 6, grid=grid(512), stream=stream0)
        buf19 = reinterpret_tensor(buf6, (1, 512, 768), (393216, 768, 1), 0); del buf6  # reuse
        # Source Nodes: [add_1, fourier_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf19, buf12, buf16, buf17, arg7_1, arg8_1, 393216, grid=grid(393216), stream=stream0)
        del arg7_1
        del arg8_1
        del buf12
        buf20 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (512, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 3072), (1, 768), 0), out=buf20)
        del arg9_1
        buf21 = reinterpret_tensor(buf20, (1, 512, 3072), (1572864, 3072, 1), 0); del buf20  # reuse
        # Source Nodes: [add_2, add_3, intermediate_output, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf21, arg10_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg10_1
        buf22 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg11_1, (3072, 768), (1, 3072), 0), out=buf22)
        del arg11_1
        buf26 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf22, arg12_1, buf19, arg13_1, arg14_1, buf26, 512, 768, grid=grid(512), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        # Source Nodes: [fft_fftn_1], Original ATen: [aten._to_copy]
        buf27 = torch.ops.prims.convert_element_type.default(buf26, torch.complex64)
        buf28 = buf27
        del buf27
        # Source Nodes: [fft_fftn_1], Original ATen: [aten._fft_c2c]
        buf29 = aten._fft_c2c(buf28, [1, 2], 0, True)
        del buf28
        buf30 = buf29
        del buf29
        # Source Nodes: [outputs_1], Original ATen: [aten.view_as_real]
        buf31 = aten.view_as_real(buf30)
        del buf30
        buf32 = buf31
        del buf31
        buf33 = buf15; del buf15  # reuse
        buf34 = buf14; del buf14  # reuse
        buf35 = buf13; del buf13  # reuse
        # Source Nodes: [add_5, fourier_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf26, buf32, buf33, buf34, buf35, 3072, 128, grid=grid(3072), stream=stream0)
        buf36 = buf17; del buf17  # reuse
        buf37 = buf16; del buf16  # reuse
        # Source Nodes: [add_5, fourier_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf33, buf34, buf35, buf36, buf37, 512, 6, grid=grid(512), stream=stream0)
        buf39 = buf26; del buf26  # reuse
        # Source Nodes: [add_5, fourier_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf39, buf32, buf36, buf37, arg15_1, arg16_1, 393216, grid=grid(393216), stream=stream0)
        del arg15_1
        del arg16_1
        del buf32
        buf40 = reinterpret_tensor(buf21, (512, 3072), (3072, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf40)
        del arg17_1
        buf41 = reinterpret_tensor(buf40, (1, 512, 3072), (1572864, 3072, 1), 0); del buf40  # reuse
        # Source Nodes: [add_6, add_7, intermediate_output_1, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf41, arg18_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg18_1
        buf42 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf42)
        del arg19_1
        buf46 = buf19; del buf19  # reuse
        # Source Nodes: [add_8, hidden_states_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf42, arg20_1, buf39, arg21_1, arg22_1, buf46, 512, 768, grid=grid(512), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        # Source Nodes: [fft_fftn_2], Original ATen: [aten._to_copy]
        buf47 = torch.ops.prims.convert_element_type.default(buf46, torch.complex64)
        buf48 = buf47
        del buf47
        # Source Nodes: [fft_fftn_2], Original ATen: [aten._fft_c2c]
        buf49 = aten._fft_c2c(buf48, [1, 2], 0, True)
        del buf48
        buf50 = buf49
        del buf49
        # Source Nodes: [outputs_2], Original ATen: [aten.view_as_real]
        buf51 = aten.view_as_real(buf50)
        del buf50
        buf52 = buf51
        del buf51
        buf53 = buf35; del buf35  # reuse
        buf54 = buf34; del buf34  # reuse
        buf55 = buf33; del buf33  # reuse
        # Source Nodes: [add_9, fourier_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf46, buf52, buf53, buf54, buf55, 3072, 128, grid=grid(3072), stream=stream0)
        buf56 = buf37; del buf37  # reuse
        buf57 = buf36; del buf36  # reuse
        # Source Nodes: [add_9, fourier_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf53, buf54, buf55, buf56, buf57, 512, 6, grid=grid(512), stream=stream0)
        buf59 = buf46; del buf46  # reuse
        # Source Nodes: [add_9, fourier_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf59, buf52, buf56, buf57, arg23_1, arg24_1, 393216, grid=grid(393216), stream=stream0)
        del arg23_1
        del arg24_1
        del buf52
        buf60 = reinterpret_tensor(buf41, (512, 3072), (3072, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 3072), (1, 768), 0), out=buf60)
        del arg25_1
        buf61 = reinterpret_tensor(buf60, (1, 512, 3072), (1572864, 3072, 1), 0); del buf60  # reuse
        # Source Nodes: [add_10, add_11, intermediate_output_2, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf61, arg26_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg26_1
        buf62 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg27_1, (3072, 768), (1, 3072), 0), out=buf62)
        del arg27_1
        buf66 = buf39; del buf39  # reuse
        # Source Nodes: [add_12, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf62, arg28_1, buf59, arg29_1, arg30_1, buf66, 512, 768, grid=grid(512), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        # Source Nodes: [fft_fftn_3], Original ATen: [aten._to_copy]
        buf67 = torch.ops.prims.convert_element_type.default(buf66, torch.complex64)
        buf68 = buf67
        del buf67
        # Source Nodes: [fft_fftn_3], Original ATen: [aten._fft_c2c]
        buf69 = aten._fft_c2c(buf68, [1, 2], 0, True)
        del buf68
        buf70 = buf69
        del buf69
        # Source Nodes: [outputs_3], Original ATen: [aten.view_as_real]
        buf71 = aten.view_as_real(buf70)
        del buf70
        buf72 = buf71
        del buf71
        buf73 = buf55; del buf55  # reuse
        buf74 = buf54; del buf54  # reuse
        buf75 = buf53; del buf53  # reuse
        # Source Nodes: [add_13, fourier_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf66, buf72, buf73, buf74, buf75, 3072, 128, grid=grid(3072), stream=stream0)
        buf76 = buf57; del buf57  # reuse
        buf77 = buf56; del buf56  # reuse
        # Source Nodes: [add_13, fourier_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf73, buf74, buf75, buf76, buf77, 512, 6, grid=grid(512), stream=stream0)
        buf79 = buf66; del buf66  # reuse
        # Source Nodes: [add_13, fourier_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf79, buf72, buf76, buf77, arg31_1, arg32_1, 393216, grid=grid(393216), stream=stream0)
        del arg31_1
        del arg32_1
        del buf72
        buf80 = reinterpret_tensor(buf61, (512, 3072), (3072, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (512, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 3072), (1, 768), 0), out=buf80)
        del arg33_1
        buf81 = reinterpret_tensor(buf80, (1, 512, 3072), (1572864, 3072, 1), 0); del buf80  # reuse
        # Source Nodes: [add_14, add_15, intermediate_output_3, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf81, arg34_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg34_1
        buf82 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg35_1, (3072, 768), (1, 3072), 0), out=buf82)
        del arg35_1
        buf86 = buf59; del buf59  # reuse
        # Source Nodes: [add_16, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf82, arg36_1, buf79, arg37_1, arg38_1, buf86, 512, 768, grid=grid(512), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        # Source Nodes: [fft_fftn_4], Original ATen: [aten._to_copy]
        buf87 = torch.ops.prims.convert_element_type.default(buf86, torch.complex64)
        buf88 = buf87
        del buf87
        # Source Nodes: [fft_fftn_4], Original ATen: [aten._fft_c2c]
        buf89 = aten._fft_c2c(buf88, [1, 2], 0, True)
        del buf88
        buf90 = buf89
        del buf89
        # Source Nodes: [outputs_4], Original ATen: [aten.view_as_real]
        buf91 = aten.view_as_real(buf90)
        del buf90
        buf92 = buf91
        del buf91
        buf93 = buf75; del buf75  # reuse
        buf94 = buf74; del buf74  # reuse
        buf95 = buf73; del buf73  # reuse
        # Source Nodes: [add_17, fourier_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf86, buf92, buf93, buf94, buf95, 3072, 128, grid=grid(3072), stream=stream0)
        buf96 = buf77; del buf77  # reuse
        buf97 = buf76; del buf76  # reuse
        # Source Nodes: [add_17, fourier_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf93, buf94, buf95, buf96, buf97, 512, 6, grid=grid(512), stream=stream0)
        buf99 = buf86; del buf86  # reuse
        # Source Nodes: [add_17, fourier_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf99, buf92, buf96, buf97, arg39_1, arg40_1, 393216, grid=grid(393216), stream=stream0)
        del arg39_1
        del arg40_1
        del buf92
        buf100 = reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 3072), (1, 768), 0), out=buf100)
        del arg41_1
        buf101 = reinterpret_tensor(buf100, (1, 512, 3072), (1572864, 3072, 1), 0); del buf100  # reuse
        # Source Nodes: [add_18, add_19, intermediate_output_4, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf101, arg42_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg42_1
        buf102 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg43_1, (3072, 768), (1, 3072), 0), out=buf102)
        del arg43_1
        buf106 = buf79; del buf79  # reuse
        # Source Nodes: [add_20, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf102, arg44_1, buf99, arg45_1, arg46_1, buf106, 512, 768, grid=grid(512), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        # Source Nodes: [fft_fftn_5], Original ATen: [aten._to_copy]
        buf107 = torch.ops.prims.convert_element_type.default(buf106, torch.complex64)
        buf108 = buf107
        del buf107
        # Source Nodes: [fft_fftn_5], Original ATen: [aten._fft_c2c]
        buf109 = aten._fft_c2c(buf108, [1, 2], 0, True)
        del buf108
        buf110 = buf109
        del buf109
        # Source Nodes: [outputs_5], Original ATen: [aten.view_as_real]
        buf111 = aten.view_as_real(buf110)
        del buf110
        buf112 = buf111
        del buf111
        buf113 = buf95; del buf95  # reuse
        buf114 = buf94; del buf94  # reuse
        buf115 = buf93; del buf93  # reuse
        # Source Nodes: [add_21, fourier_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf106, buf112, buf113, buf114, buf115, 3072, 128, grid=grid(3072), stream=stream0)
        buf116 = buf97; del buf97  # reuse
        buf117 = buf96; del buf96  # reuse
        # Source Nodes: [add_21, fourier_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf113, buf114, buf115, buf116, buf117, 512, 6, grid=grid(512), stream=stream0)
        buf119 = buf106; del buf106  # reuse
        # Source Nodes: [add_21, fourier_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf119, buf112, buf116, buf117, arg47_1, arg48_1, 393216, grid=grid(393216), stream=stream0)
        del arg47_1
        del arg48_1
        del buf112
        buf120 = reinterpret_tensor(buf101, (512, 3072), (3072, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 3072), (1, 768), 0), out=buf120)
        del arg49_1
        buf121 = reinterpret_tensor(buf120, (1, 512, 3072), (1572864, 3072, 1), 0); del buf120  # reuse
        # Source Nodes: [add_22, add_23, intermediate_output_5, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf121, arg50_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg50_1
        buf122 = reinterpret_tensor(buf99, (512, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg51_1, (3072, 768), (1, 3072), 0), out=buf122)
        del arg51_1
        buf126 = reinterpret_tensor(buf102, (1, 512, 768), (393216, 768, 1), 0); del buf102  # reuse
        # Source Nodes: [add_24, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf122, arg52_1, buf119, arg53_1, arg54_1, buf126, 512, 768, grid=grid(512), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        # Source Nodes: [fft_fftn_6], Original ATen: [aten._to_copy]
        buf127 = torch.ops.prims.convert_element_type.default(buf126, torch.complex64)
        buf128 = buf127
        del buf127
        # Source Nodes: [fft_fftn_6], Original ATen: [aten._fft_c2c]
        buf129 = aten._fft_c2c(buf128, [1, 2], 0, True)
        del buf128
        buf130 = buf129
        del buf129
        # Source Nodes: [outputs_6], Original ATen: [aten.view_as_real]
        buf131 = aten.view_as_real(buf130)
        del buf130
        buf132 = buf131
        del buf131
        buf133 = buf115; del buf115  # reuse
        buf134 = buf114; del buf114  # reuse
        buf135 = buf113; del buf113  # reuse
        # Source Nodes: [add_25, fourier_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf126, buf132, buf133, buf134, buf135, 3072, 128, grid=grid(3072), stream=stream0)
        buf136 = buf117; del buf117  # reuse
        buf137 = buf116; del buf116  # reuse
        # Source Nodes: [add_25, fourier_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf133, buf134, buf135, buf136, buf137, 512, 6, grid=grid(512), stream=stream0)
        buf139 = buf126; del buf126  # reuse
        # Source Nodes: [add_25, fourier_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf139, buf132, buf136, buf137, arg55_1, arg56_1, 393216, grid=grid(393216), stream=stream0)
        del arg55_1
        del arg56_1
        del buf132
        buf140 = reinterpret_tensor(buf121, (512, 3072), (3072, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (512, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 3072), (1, 768), 0), out=buf140)
        del arg57_1
        buf141 = reinterpret_tensor(buf140, (1, 512, 3072), (1572864, 3072, 1), 0); del buf140  # reuse
        # Source Nodes: [add_26, add_27, intermediate_output_6, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf141, arg58_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg58_1
        buf142 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg59_1, (3072, 768), (1, 3072), 0), out=buf142)
        del arg59_1
        buf146 = buf119; del buf119  # reuse
        # Source Nodes: [add_28, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf142, arg60_1, buf139, arg61_1, arg62_1, buf146, 512, 768, grid=grid(512), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        # Source Nodes: [fft_fftn_7], Original ATen: [aten._to_copy]
        buf147 = torch.ops.prims.convert_element_type.default(buf146, torch.complex64)
        buf148 = buf147
        del buf147
        # Source Nodes: [fft_fftn_7], Original ATen: [aten._fft_c2c]
        buf149 = aten._fft_c2c(buf148, [1, 2], 0, True)
        del buf148
        buf150 = buf149
        del buf149
        # Source Nodes: [outputs_7], Original ATen: [aten.view_as_real]
        buf151 = aten.view_as_real(buf150)
        del buf150
        buf152 = buf151
        del buf151
        buf153 = buf135; del buf135  # reuse
        buf154 = buf134; del buf134  # reuse
        buf155 = buf133; del buf133  # reuse
        # Source Nodes: [add_29, fourier_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf146, buf152, buf153, buf154, buf155, 3072, 128, grid=grid(3072), stream=stream0)
        buf156 = buf137; del buf137  # reuse
        buf157 = buf136; del buf136  # reuse
        # Source Nodes: [add_29, fourier_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf153, buf154, buf155, buf156, buf157, 512, 6, grid=grid(512), stream=stream0)
        buf159 = buf146; del buf146  # reuse
        # Source Nodes: [add_29, fourier_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf159, buf152, buf156, buf157, arg63_1, arg64_1, 393216, grid=grid(393216), stream=stream0)
        del arg63_1
        del arg64_1
        del buf152
        buf160 = reinterpret_tensor(buf141, (512, 3072), (3072, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 768), (768, 1), 0), reinterpret_tensor(arg65_1, (768, 3072), (1, 768), 0), out=buf160)
        del arg65_1
        buf161 = reinterpret_tensor(buf160, (1, 512, 3072), (1572864, 3072, 1), 0); del buf160  # reuse
        # Source Nodes: [add_30, add_31, intermediate_output_7, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf161, arg66_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg66_1
        buf162 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg67_1, (3072, 768), (1, 3072), 0), out=buf162)
        del arg67_1
        buf166 = buf139; del buf139  # reuse
        # Source Nodes: [add_32, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf162, arg68_1, buf159, arg69_1, arg70_1, buf166, 512, 768, grid=grid(512), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        # Source Nodes: [fft_fftn_8], Original ATen: [aten._to_copy]
        buf167 = torch.ops.prims.convert_element_type.default(buf166, torch.complex64)
        buf168 = buf167
        del buf167
        # Source Nodes: [fft_fftn_8], Original ATen: [aten._fft_c2c]
        buf169 = aten._fft_c2c(buf168, [1, 2], 0, True)
        del buf168
        buf170 = buf169
        del buf169
        # Source Nodes: [outputs_8], Original ATen: [aten.view_as_real]
        buf171 = aten.view_as_real(buf170)
        del buf170
        buf172 = buf171
        del buf171
        buf173 = buf155; del buf155  # reuse
        buf174 = buf154; del buf154  # reuse
        buf175 = buf153; del buf153  # reuse
        # Source Nodes: [add_33, fourier_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf166, buf172, buf173, buf174, buf175, 3072, 128, grid=grid(3072), stream=stream0)
        buf176 = buf157; del buf157  # reuse
        buf177 = buf156; del buf156  # reuse
        # Source Nodes: [add_33, fourier_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf173, buf174, buf175, buf176, buf177, 512, 6, grid=grid(512), stream=stream0)
        buf179 = buf166; del buf166  # reuse
        # Source Nodes: [add_33, fourier_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf179, buf172, buf176, buf177, arg71_1, arg72_1, 393216, grid=grid(393216), stream=stream0)
        del arg71_1
        del arg72_1
        del buf172
        buf180 = reinterpret_tensor(buf161, (512, 3072), (3072, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 3072), (1, 768), 0), out=buf180)
        del arg73_1
        buf181 = reinterpret_tensor(buf180, (1, 512, 3072), (1572864, 3072, 1), 0); del buf180  # reuse
        # Source Nodes: [add_34, add_35, intermediate_output_8, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf181, arg74_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg74_1
        buf182 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg75_1, (3072, 768), (1, 3072), 0), out=buf182)
        del arg75_1
        buf186 = buf159; del buf159  # reuse
        # Source Nodes: [add_36, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf182, arg76_1, buf179, arg77_1, arg78_1, buf186, 512, 768, grid=grid(512), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        # Source Nodes: [fft_fftn_9], Original ATen: [aten._to_copy]
        buf187 = torch.ops.prims.convert_element_type.default(buf186, torch.complex64)
        buf188 = buf187
        del buf187
        # Source Nodes: [fft_fftn_9], Original ATen: [aten._fft_c2c]
        buf189 = aten._fft_c2c(buf188, [1, 2], 0, True)
        del buf188
        buf190 = buf189
        del buf189
        # Source Nodes: [outputs_9], Original ATen: [aten.view_as_real]
        buf191 = aten.view_as_real(buf190)
        del buf190
        buf192 = buf191
        del buf191
        buf193 = buf175; del buf175  # reuse
        buf194 = buf174; del buf174  # reuse
        buf195 = buf173; del buf173  # reuse
        # Source Nodes: [add_37, fourier_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf186, buf192, buf193, buf194, buf195, 3072, 128, grid=grid(3072), stream=stream0)
        buf196 = buf177; del buf177  # reuse
        buf197 = buf176; del buf176  # reuse
        # Source Nodes: [add_37, fourier_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf193, buf194, buf195, buf196, buf197, 512, 6, grid=grid(512), stream=stream0)
        buf199 = buf186; del buf186  # reuse
        # Source Nodes: [add_37, fourier_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf199, buf192, buf196, buf197, arg79_1, arg80_1, 393216, grid=grid(393216), stream=stream0)
        del arg79_1
        del arg80_1
        del buf192
        buf200 = reinterpret_tensor(buf181, (512, 3072), (3072, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg81_1, (768, 3072), (1, 768), 0), out=buf200)
        del arg81_1
        buf201 = reinterpret_tensor(buf200, (1, 512, 3072), (1572864, 3072, 1), 0); del buf200  # reuse
        # Source Nodes: [add_38, add_39, intermediate_output_9, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf201, arg82_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg82_1
        buf202 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg83_1, (3072, 768), (1, 3072), 0), out=buf202)
        del arg83_1
        buf206 = buf179; del buf179  # reuse
        # Source Nodes: [add_40, hidden_states_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf202, arg84_1, buf199, arg85_1, arg86_1, buf206, 512, 768, grid=grid(512), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        # Source Nodes: [fft_fftn_10], Original ATen: [aten._to_copy]
        buf207 = torch.ops.prims.convert_element_type.default(buf206, torch.complex64)
        buf208 = buf207
        del buf207
        # Source Nodes: [fft_fftn_10], Original ATen: [aten._fft_c2c]
        buf209 = aten._fft_c2c(buf208, [1, 2], 0, True)
        del buf208
        buf210 = buf209
        del buf209
        # Source Nodes: [outputs_10], Original ATen: [aten.view_as_real]
        buf211 = aten.view_as_real(buf210)
        del buf210
        buf212 = buf211
        del buf211
        buf213 = buf195; del buf195  # reuse
        buf214 = buf194; del buf194  # reuse
        buf215 = buf193; del buf193  # reuse
        # Source Nodes: [add_41, fourier_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf206, buf212, buf213, buf214, buf215, 3072, 128, grid=grid(3072), stream=stream0)
        buf216 = buf197; del buf197  # reuse
        buf217 = buf196; del buf196  # reuse
        # Source Nodes: [add_41, fourier_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf213, buf214, buf215, buf216, buf217, 512, 6, grid=grid(512), stream=stream0)
        buf219 = buf206; del buf206  # reuse
        # Source Nodes: [add_41, fourier_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf219, buf212, buf216, buf217, arg87_1, arg88_1, 393216, grid=grid(393216), stream=stream0)
        del arg87_1
        del arg88_1
        del buf212
        buf220 = reinterpret_tensor(buf201, (512, 3072), (3072, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 3072), (1, 768), 0), out=buf220)
        del arg89_1
        buf221 = reinterpret_tensor(buf220, (1, 512, 3072), (1572864, 3072, 1), 0); del buf220  # reuse
        # Source Nodes: [add_42, add_43, intermediate_output_10, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf221, arg90_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg90_1
        buf222 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg91_1, (3072, 768), (1, 3072), 0), out=buf222)
        del arg91_1
        buf226 = buf199; del buf199  # reuse
        # Source Nodes: [add_44, hidden_states_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf222, arg92_1, buf219, arg93_1, arg94_1, buf226, 512, 768, grid=grid(512), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        # Source Nodes: [fft_fftn_11], Original ATen: [aten._to_copy]
        buf227 = torch.ops.prims.convert_element_type.default(buf226, torch.complex64)
        buf228 = buf227
        del buf227
        # Source Nodes: [fft_fftn_11], Original ATen: [aten._fft_c2c]
        buf229 = aten._fft_c2c(buf228, [1, 2], 0, True)
        del buf228
        buf230 = buf229
        del buf229
        # Source Nodes: [outputs_11], Original ATen: [aten.view_as_real]
        buf231 = aten.view_as_real(buf230)
        del buf230
        buf232 = buf231
        del buf231
        buf233 = buf215; del buf215  # reuse
        buf234 = buf214; del buf214  # reuse
        buf235 = buf213; del buf213  # reuse
        # Source Nodes: [add_45, fourier_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf226, buf232, buf233, buf234, buf235, 3072, 128, grid=grid(3072), stream=stream0)
        buf236 = buf217; del buf217  # reuse
        buf237 = buf216; del buf216  # reuse
        # Source Nodes: [add_45, fourier_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf233, buf234, buf235, buf236, buf237, 512, 6, grid=grid(512), stream=stream0)
        del buf233
        del buf234
        del buf235
        buf239 = buf226; del buf226  # reuse
        # Source Nodes: [add_45, fourier_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf239, buf232, buf236, buf237, arg95_1, arg96_1, 393216, grid=grid(393216), stream=stream0)
        del arg95_1
        del arg96_1
        del buf232
        buf240 = reinterpret_tensor(buf221, (512, 3072), (3072, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (512, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 3072), (1, 768), 0), out=buf240)
        del arg97_1
        buf241 = reinterpret_tensor(buf240, (1, 512, 3072), (1572864, 3072, 1), 0); del buf240  # reuse
        # Source Nodes: [add_46, add_47, intermediate_output_11, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf241, arg98_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg98_1
        buf242 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg99_1, (3072, 768), (1, 3072), 0), out=buf242)
        del arg99_1
        del buf241
        buf246 = buf219; del buf219  # reuse
        # Source Nodes: [add_48, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf242, arg100_1, buf239, arg101_1, arg102_1, buf246, 512, 768, grid=grid(512), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del buf239
        buf247 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf247)
        del arg105_1
        buf251 = buf246; del buf246  # reuse
        # Source Nodes: [add_49, add_50, hidden_states_85, hidden_states_87, mul_48, mul_49, mul_50, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_7.run(buf247, arg106_1, arg107_1, arg108_1, buf251, 512, 768, grid=grid(512), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del buf247
        buf252 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg110_1, reinterpret_tensor(buf251, (512, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 32000), (1, 768), 0), alpha=1, beta=1, out=buf252)
        del arg109_1
        del arg110_1
        del buf251
        buf253 = reinterpret_tensor(buf237, (512, 1), (1, 512), 0); del buf237  # reuse
        buf254 = reinterpret_tensor(buf236, (512, 1), (1, 512), 0); del buf236  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf252, buf253, buf254, 512, 32000, grid=grid(512), stream=stream0)
        buf255 = empty((), device='cuda', dtype=torch.float32)
        buf257 = buf255; del buf255  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_9.run(buf257, arg114_1, buf252, buf253, buf254, 1, 512, grid=grid(1), stream=stream0)
        del arg114_1
        return (buf257, reinterpret_tensor(buf252, (1, 512, 32000), (16384000, 32000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg112_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg113_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg114_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GoogleFnet', benchmark_compiled_module)
