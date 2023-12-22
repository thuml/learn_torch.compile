
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
# Source Nodes: [add_1, hidden_states, hidden_states_1, hidden_states_3, inputs_embeds, l__mod___model_decoder_embed_tokens, positions_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add_1 => add_1
# hidden_states => add_2
# hidden_states_1 => add_3, add_4, mul_1, mul_2, rsqrt, sub, var_mean
# hidden_states_3 => add_5, add_6, mul_3, mul_4, rsqrt_1, sub_1, var_mean_1
# inputs_embeds => mul
# l__mod___model_decoder_embed_tokens => embedding
# positions_1 => embedding_1
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzxelroobrwpdtjldpce7vzu2l7huvhtzoja7f2lkb5lb5vi5kh.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/te/ctevri73phuhv5coyfmze6murtzhnfknw7wmnsrqk4nehfyp6vrv.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmnu4dvddoht6mvtnfp7avqgjavjp3wraeq3st767lf3kgiaujm.py
# Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
# attn_weights_3 => amax, div, exp, sub_2, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wz/cwz7so7tozydh5qvetn4weikxm3rx5wj7o5lrhjrtpqydintrgx3.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pi/cpitxmcj7l26fa2v65s37vnxw5ff23i4amurnaq5ypfqtidltdi5.py
# Source Nodes: [hidden_states_7, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_7 => add_10, add_9, mul_6, mul_7, rsqrt_2, sub_3, var_mean_2
# residual_1 => add_8
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pmm4733hsnbxmx7btyiqyihcvjlz36vjcluo6i5uzdyloywybh.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
# hidden_states_8 => add_11, erf, mul_10, mul_8, mul_9
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/rv/crvuqdc3oijxpy4i6ymvns7w77akzqcbeydy3argwqi2e6dv4bxp.py
# Source Nodes: [hidden_states_14, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_14 => add_13, add_14, mul_11, mul_12, rsqrt_3, sub_4, var_mean_3
# residual_1 => add_8
# residual_2 => add_12
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/q5/cq53rjj3sb7vsn7rhujzupatijg2no5ev2tqsby74pwsan3x646c.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_12, exp_12, sub_38, sum_13
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwelgxztw3hdt6gbucotzsjz3lao43rqxeo6jv7ybu5jxiosnzth.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_12, full_default_3, ne_1, ne_2, neg, sum_14, sum_15, where_2
triton_per_fused_nll_loss_forward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg1_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg7_1, (1024, ), (1, ))
    assert_size_stride(arg8_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg17_1, (4096, ), (1, ))
    assert_size_stride(arg18_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg23_1, (1024, ), (1, ))
    assert_size_stride(arg24_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg33_1, (4096, ), (1, ))
    assert_size_stride(arg34_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg49_1, (4096, ), (1, ))
    assert_size_stride(arg50_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg65_1, (4096, ), (1, ))
    assert_size_stride(arg66_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg73_1, (1024, ), (1, ))
    assert_size_stride(arg74_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg81_1, (4096, ), (1, ))
    assert_size_stride(arg82_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg97_1, (4096, ), (1, ))
    assert_size_stride(arg98_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg113_1, (4096, ), (1, ))
    assert_size_stride(arg114_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg129_1, (4096, ), (1, ))
    assert_size_stride(arg130_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg145_1, (4096, ), (1, ))
    assert_size_stride(arg146_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg161_1, (4096, ), (1, ))
    assert_size_stride(arg162_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg177_1, (4096, ), (1, ))
    assert_size_stride(arg178_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg193_1, (4096, ), (1, ))
    assert_size_stride(arg194_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg199_1, (1, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, hidden_states, hidden_states_1, hidden_states_3, inputs_embeds, l__mod___model_decoder_embed_tokens, positions_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg199_1, arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, buf7, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg199_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf8 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg6_1
        buf9 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf9)
        del arg8_1
        buf10 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf9, arg9_1, buf10, 1048576, grid=grid(1048576), stream=stream0)
        del arg9_1
        buf11 = reinterpret_tensor(buf9, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf8, arg7_1, buf11, 1048576, grid=grid(1048576), stream=stream0)
        del arg7_1
        buf12 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf10, (16, 64, 1024), (65536, 1, 64), 0), out=buf12)
        buf17 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf12, buf17, 16384, 1024, grid=grid(16384), stream=stream0)
        buf15 = reinterpret_tensor(buf11, (1024, 1024), (1024, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf15)
        del arg10_1
        buf16 = reinterpret_tensor(buf7, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf15, arg11_1, buf16, 1048576, grid=grid(1048576), stream=stream0)
        del arg11_1
        buf18 = reinterpret_tensor(buf15, (16, 1024, 64), (65536, 64, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_output, attn_weights_3], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf17, reinterpret_tensor(buf16, (16, 1024, 64), (65536, 64, 1), 0), out=buf18)
        buf19 = reinterpret_tensor(buf8, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf18, buf19, 1048576, grid=grid(1048576), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (1024, 1024), (1024, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 1024), (1, 1024), 0), out=buf20)
        del arg12_1
        buf24 = reinterpret_tensor(buf19, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf19  # reuse
        # Source Nodes: [hidden_states_7, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf3, buf20, arg13_1, arg14_1, arg15_1, buf24, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg14_1
        del arg15_1
        buf25 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg16_1, (1024, 4096), (1, 1024), 0), out=buf25)
        del arg16_1
        buf26 = reinterpret_tensor(buf25, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf25  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf26, arg17_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg17_1
        buf27 = reinterpret_tensor(buf24, (1024, 1024), (1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg18_1, (4096, 1024), (1, 4096), 0), out=buf27)
        del arg18_1
        buf28 = reinterpret_tensor(buf27, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf27  # reuse
        buf32 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf28, buf3, buf20, arg13_1, arg19_1, arg20_1, arg21_1, buf32, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg13_1
        del arg19_1
        del arg20_1
        del arg21_1
        buf33 = reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), out=buf33)
        del arg22_1
        buf34 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg24_1
        buf35 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf34, arg25_1, buf35, 1048576, grid=grid(1048576), stream=stream0)
        del arg25_1
        buf36 = reinterpret_tensor(buf34, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf33, arg23_1, buf36, 1048576, grid=grid(1048576), stream=stream0)
        del arg23_1
        buf37 = buf17; del buf17  # reuse
        # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf35, (16, 64, 1024), (65536, 1, 64), 0), out=buf37)
        buf42 = buf12; del buf12  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf37, buf42, 16384, 1024, grid=grid(16384), stream=stream0)
        buf40 = reinterpret_tensor(buf36, (1024, 1024), (1024, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf40)
        del arg26_1
        buf41 = reinterpret_tensor(buf32, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf40, arg27_1, buf41, 1048576, grid=grid(1048576), stream=stream0)
        del arg27_1
        buf43 = reinterpret_tensor(buf40, (16, 1024, 64), (65536, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [attn_output_5, attn_weights_7], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf42, reinterpret_tensor(buf41, (16, 1024, 64), (65536, 64, 1), 0), out=buf43)
        buf44 = reinterpret_tensor(buf33, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf43, buf44, 1048576, grid=grid(1048576), stream=stream0)
        buf45 = reinterpret_tensor(buf43, (1024, 1024), (1024, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 1024), (1, 1024), 0), out=buf45)
        del arg28_1
        buf49 = reinterpret_tensor(buf44, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf44  # reuse
        # Source Nodes: [hidden_states_18, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf28, buf45, arg29_1, arg30_1, arg31_1, buf49, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg30_1
        del arg31_1
        buf50 = reinterpret_tensor(buf26, (1024, 4096), (4096, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg32_1, (1024, 4096), (1, 1024), 0), out=buf50)
        del arg32_1
        buf51 = reinterpret_tensor(buf50, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf50  # reuse
        # Source Nodes: [hidden_states_19], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf51, arg33_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg33_1
        buf52 = reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg34_1, (4096, 1024), (1, 4096), 0), out=buf52)
        del arg34_1
        buf53 = reinterpret_tensor(buf52, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf52  # reuse
        buf57 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf53, buf28, buf45, arg29_1, arg35_1, arg36_1, arg37_1, buf57, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg29_1
        del arg35_1
        del arg36_1
        del arg37_1
        buf58 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), out=buf58)
        del arg38_1
        buf59 = reinterpret_tensor(buf28, (1024, 1024), (1024, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf59)
        del arg40_1
        buf60 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf59, arg41_1, buf60, 1048576, grid=grid(1048576), stream=stream0)
        del arg41_1
        buf61 = reinterpret_tensor(buf59, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf58, arg39_1, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del arg39_1
        buf62 = buf42; del buf42  # reuse
        # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf60, (16, 64, 1024), (65536, 1, 64), 0), out=buf62)
        buf67 = buf37; del buf37  # reuse
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf62, buf67, 16384, 1024, grid=grid(16384), stream=stream0)
        buf65 = reinterpret_tensor(buf61, (1024, 1024), (1024, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf65)
        del arg42_1
        buf66 = reinterpret_tensor(buf57, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf65, arg43_1, buf66, 1048576, grid=grid(1048576), stream=stream0)
        del arg43_1
        buf68 = reinterpret_tensor(buf65, (16, 1024, 64), (65536, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [attn_output_10, attn_weights_11], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf67, reinterpret_tensor(buf66, (16, 1024, 64), (65536, 64, 1), 0), out=buf68)
        buf69 = reinterpret_tensor(buf58, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf68, buf69, 1048576, grid=grid(1048576), stream=stream0)
        buf70 = reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg44_1, (1024, 1024), (1, 1024), 0), out=buf70)
        del arg44_1
        buf74 = reinterpret_tensor(buf69, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf53, buf70, arg45_1, arg46_1, arg47_1, buf74, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg46_1
        del arg47_1
        buf75 = reinterpret_tensor(buf51, (1024, 4096), (4096, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg48_1, (1024, 4096), (1, 1024), 0), out=buf75)
        del arg48_1
        buf76 = reinterpret_tensor(buf75, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf75  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf76, arg49_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg49_1
        buf77 = reinterpret_tensor(buf74, (1024, 1024), (1024, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg50_1, (4096, 1024), (1, 4096), 0), out=buf77)
        del arg50_1
        buf78 = reinterpret_tensor(buf77, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf77  # reuse
        buf82 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36, residual_5, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf78, buf53, buf70, arg45_1, arg51_1, arg52_1, arg53_1, buf82, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg45_1
        del arg51_1
        del arg52_1
        del arg53_1
        buf83 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), out=buf83)
        del arg54_1
        buf84 = reinterpret_tensor(buf53, (1024, 1024), (1024, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf84)
        del arg56_1
        buf85 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf84, arg57_1, buf85, 1048576, grid=grid(1048576), stream=stream0)
        del arg57_1
        buf86 = reinterpret_tensor(buf84, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf84  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf83, arg55_1, buf86, 1048576, grid=grid(1048576), stream=stream0)
        del arg55_1
        buf87 = buf67; del buf67  # reuse
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf85, (16, 64, 1024), (65536, 1, 64), 0), out=buf87)
        buf92 = buf62; del buf62  # reuse
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf87, buf92, 16384, 1024, grid=grid(16384), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (1024, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf90)
        del arg58_1
        buf91 = reinterpret_tensor(buf82, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf82  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf90, arg59_1, buf91, 1048576, grid=grid(1048576), stream=stream0)
        del arg59_1
        buf93 = reinterpret_tensor(buf90, (16, 1024, 64), (65536, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [attn_output_15, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf92, reinterpret_tensor(buf91, (16, 1024, 64), (65536, 64, 1), 0), out=buf93)
        buf94 = reinterpret_tensor(buf83, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf93, buf94, 1048576, grid=grid(1048576), stream=stream0)
        buf95 = reinterpret_tensor(buf93, (1024, 1024), (1024, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 1024), (1, 1024), 0), out=buf95)
        del arg60_1
        buf99 = reinterpret_tensor(buf94, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
        # Source Nodes: [hidden_states_40, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf78, buf95, arg61_1, arg62_1, arg63_1, buf99, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg62_1
        del arg63_1
        buf100 = reinterpret_tensor(buf76, (1024, 4096), (4096, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 4096), (1, 1024), 0), out=buf100)
        del arg64_1
        buf101 = reinterpret_tensor(buf100, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf100  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf101, arg65_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg65_1
        buf102 = reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg66_1, (4096, 1024), (1, 4096), 0), out=buf102)
        del arg66_1
        buf103 = reinterpret_tensor(buf102, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf102  # reuse
        buf107 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_47, residual_7, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf103, buf78, buf95, arg61_1, arg67_1, arg68_1, arg69_1, buf107, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg61_1
        del arg67_1
        del arg68_1
        del arg69_1
        buf108 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), out=buf108)
        del arg70_1
        buf109 = reinterpret_tensor(buf78, (1024, 1024), (1024, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf109)
        del arg72_1
        buf110 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf109, arg73_1, buf110, 1048576, grid=grid(1048576), stream=stream0)
        del arg73_1
        buf111 = reinterpret_tensor(buf109, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf108, arg71_1, buf111, 1048576, grid=grid(1048576), stream=stream0)
        del arg71_1
        buf112 = buf92; del buf92  # reuse
        # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf110, (16, 64, 1024), (65536, 1, 64), 0), out=buf112)
        buf117 = buf87; del buf87  # reuse
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf112, buf117, 16384, 1024, grid=grid(16384), stream=stream0)
        buf115 = reinterpret_tensor(buf111, (1024, 1024), (1024, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf115)
        del arg74_1
        buf116 = reinterpret_tensor(buf107, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf115, arg75_1, buf116, 1048576, grid=grid(1048576), stream=stream0)
        del arg75_1
        buf118 = reinterpret_tensor(buf115, (16, 1024, 64), (65536, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [attn_output_20, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf117, reinterpret_tensor(buf116, (16, 1024, 64), (65536, 64, 1), 0), out=buf118)
        buf119 = reinterpret_tensor(buf108, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf118, buf119, 1048576, grid=grid(1048576), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (1024, 1024), (1024, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg76_1, (1024, 1024), (1, 1024), 0), out=buf120)
        del arg76_1
        buf124 = reinterpret_tensor(buf119, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_51, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf103, buf120, arg77_1, arg78_1, arg79_1, buf124, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg78_1
        del arg79_1
        buf125 = reinterpret_tensor(buf101, (1024, 4096), (4096, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 4096), (1, 1024), 0), out=buf125)
        del arg80_1
        buf126 = reinterpret_tensor(buf125, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf125  # reuse
        # Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf126, arg81_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg81_1
        buf127 = reinterpret_tensor(buf124, (1024, 1024), (1024, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg82_1, (4096, 1024), (1, 4096), 0), out=buf127)
        del arg82_1
        buf128 = reinterpret_tensor(buf127, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf127  # reuse
        buf132 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58, residual_10, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf128, buf103, buf120, arg77_1, arg83_1, arg84_1, arg85_1, buf132, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg77_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf133 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), out=buf133)
        del arg86_1
        buf134 = reinterpret_tensor(buf103, (1024, 1024), (1024, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf134)
        del arg88_1
        buf135 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf134, arg89_1, buf135, 1048576, grid=grid(1048576), stream=stream0)
        del arg89_1
        buf136 = reinterpret_tensor(buf134, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf133, arg87_1, buf136, 1048576, grid=grid(1048576), stream=stream0)
        del arg87_1
        buf137 = buf117; del buf117  # reuse
        # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf135, (16, 64, 1024), (65536, 1, 64), 0), out=buf137)
        buf142 = buf112; del buf112  # reuse
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf137, buf142, 16384, 1024, grid=grid(16384), stream=stream0)
        buf140 = reinterpret_tensor(buf136, (1024, 1024), (1024, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf140)
        del arg90_1
        buf141 = reinterpret_tensor(buf132, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf140, arg91_1, buf141, 1048576, grid=grid(1048576), stream=stream0)
        del arg91_1
        buf143 = reinterpret_tensor(buf140, (16, 1024, 64), (65536, 64, 1), 0); del buf140  # reuse
        # Source Nodes: [attn_output_25, attn_weights_23], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf142, reinterpret_tensor(buf141, (16, 1024, 64), (65536, 64, 1), 0), out=buf143)
        buf144 = reinterpret_tensor(buf133, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf143, buf144, 1048576, grid=grid(1048576), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (1024, 1024), (1024, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 1024), (1, 1024), 0), out=buf145)
        del arg92_1
        buf149 = reinterpret_tensor(buf144, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf144  # reuse
        # Source Nodes: [hidden_states_62, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf128, buf145, arg93_1, arg94_1, arg95_1, buf149, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg94_1
        del arg95_1
        buf150 = reinterpret_tensor(buf126, (1024, 4096), (4096, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg96_1, (1024, 4096), (1, 1024), 0), out=buf150)
        del arg96_1
        buf151 = reinterpret_tensor(buf150, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf150  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf151, arg97_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg97_1
        buf152 = reinterpret_tensor(buf149, (1024, 1024), (1024, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg98_1, (4096, 1024), (1, 4096), 0), out=buf152)
        del arg98_1
        buf153 = reinterpret_tensor(buf152, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf152  # reuse
        buf157 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_69, residual_11, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf153, buf128, buf145, arg93_1, arg99_1, arg100_1, arg101_1, buf157, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg100_1
        del arg101_1
        del arg93_1
        del arg99_1
        buf158 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), out=buf158)
        del arg102_1
        buf159 = reinterpret_tensor(buf128, (1024, 1024), (1024, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf159)
        del arg104_1
        buf160 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf159, arg105_1, buf160, 1048576, grid=grid(1048576), stream=stream0)
        del arg105_1
        buf161 = reinterpret_tensor(buf159, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf158, arg103_1, buf161, 1048576, grid=grid(1048576), stream=stream0)
        del arg103_1
        buf162 = buf142; del buf142  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf160, (16, 64, 1024), (65536, 1, 64), 0), out=buf162)
        buf167 = buf137; del buf137  # reuse
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf162, buf167, 16384, 1024, grid=grid(16384), stream=stream0)
        buf165 = reinterpret_tensor(buf161, (1024, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf165)
        del arg106_1
        buf166 = reinterpret_tensor(buf157, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf157  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf165, arg107_1, buf166, 1048576, grid=grid(1048576), stream=stream0)
        del arg107_1
        buf168 = reinterpret_tensor(buf165, (16, 1024, 64), (65536, 64, 1), 0); del buf165  # reuse
        # Source Nodes: [attn_output_30, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf167, reinterpret_tensor(buf166, (16, 1024, 64), (65536, 64, 1), 0), out=buf168)
        buf169 = reinterpret_tensor(buf158, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf158  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf168, buf169, 1048576, grid=grid(1048576), stream=stream0)
        buf170 = reinterpret_tensor(buf168, (1024, 1024), (1024, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 1024), (1, 1024), 0), out=buf170)
        del arg108_1
        buf174 = reinterpret_tensor(buf169, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf169  # reuse
        # Source Nodes: [hidden_states_73, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf153, buf170, arg109_1, arg110_1, arg111_1, buf174, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg110_1
        del arg111_1
        buf175 = reinterpret_tensor(buf151, (1024, 4096), (4096, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 4096), (1, 1024), 0), out=buf175)
        del arg112_1
        buf176 = reinterpret_tensor(buf175, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf175  # reuse
        # Source Nodes: [hidden_states_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf176, arg113_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg113_1
        buf177 = reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 1024), (1, 4096), 0), out=buf177)
        del arg114_1
        buf178 = reinterpret_tensor(buf177, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
        buf182 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80, residual_13, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf178, buf153, buf170, arg109_1, arg115_1, arg116_1, arg117_1, buf182, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg109_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf183 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), out=buf183)
        del arg118_1
        buf184 = reinterpret_tensor(buf153, (1024, 1024), (1024, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg120_1
        buf185 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf184, arg121_1, buf185, 1048576, grid=grid(1048576), stream=stream0)
        del arg121_1
        buf186 = reinterpret_tensor(buf184, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf184  # reuse
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf183, arg119_1, buf186, 1048576, grid=grid(1048576), stream=stream0)
        del arg119_1
        buf187 = buf167; del buf167  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf185, (16, 64, 1024), (65536, 1, 64), 0), out=buf187)
        buf192 = buf162; del buf162  # reuse
        # Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf187, buf192, 16384, 1024, grid=grid(16384), stream=stream0)
        buf190 = reinterpret_tensor(buf186, (1024, 1024), (1024, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf190)
        del arg122_1
        buf191 = reinterpret_tensor(buf182, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf182  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf190, arg123_1, buf191, 1048576, grid=grid(1048576), stream=stream0)
        del arg123_1
        buf193 = reinterpret_tensor(buf190, (16, 1024, 64), (65536, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_output_35, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf192, reinterpret_tensor(buf191, (16, 1024, 64), (65536, 64, 1), 0), out=buf193)
        buf194 = reinterpret_tensor(buf183, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf183  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf193, buf194, 1048576, grid=grid(1048576), stream=stream0)
        buf195 = reinterpret_tensor(buf193, (1024, 1024), (1024, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf195)
        del arg124_1
        buf199 = reinterpret_tensor(buf194, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf194  # reuse
        # Source Nodes: [hidden_states_84, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf178, buf195, arg125_1, arg126_1, arg127_1, buf199, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg126_1
        del arg127_1
        buf200 = reinterpret_tensor(buf176, (1024, 4096), (4096, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), out=buf200)
        del arg128_1
        buf201 = reinterpret_tensor(buf200, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf200  # reuse
        # Source Nodes: [hidden_states_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf201, arg129_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg129_1
        buf202 = reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), out=buf202)
        del arg130_1
        buf203 = reinterpret_tensor(buf202, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf202  # reuse
        buf207 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_91, residual_15, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf203, buf178, buf195, arg125_1, arg131_1, arg132_1, arg133_1, buf207, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg125_1
        del arg131_1
        del arg132_1
        del arg133_1
        buf208 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg134_1
        buf209 = reinterpret_tensor(buf178, (1024, 1024), (1024, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf209)
        del arg136_1
        buf210 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf209, arg137_1, buf210, 1048576, grid=grid(1048576), stream=stream0)
        del arg137_1
        buf211 = reinterpret_tensor(buf209, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf208, arg135_1, buf211, 1048576, grid=grid(1048576), stream=stream0)
        del arg135_1
        buf212 = buf192; del buf192  # reuse
        # Source Nodes: [attn_weights_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf210, (16, 64, 1024), (65536, 1, 64), 0), out=buf212)
        buf217 = buf187; del buf187  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf212, buf217, 16384, 1024, grid=grid(16384), stream=stream0)
        buf215 = reinterpret_tensor(buf211, (1024, 1024), (1024, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf215)
        del arg138_1
        buf216 = reinterpret_tensor(buf207, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf207  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf215, arg139_1, buf216, 1048576, grid=grid(1048576), stream=stream0)
        del arg139_1
        buf218 = reinterpret_tensor(buf215, (16, 1024, 64), (65536, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [attn_output_40, attn_weights_35], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf217, reinterpret_tensor(buf216, (16, 1024, 64), (65536, 64, 1), 0), out=buf218)
        buf219 = reinterpret_tensor(buf208, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf218, buf219, 1048576, grid=grid(1048576), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (1024, 1024), (1024, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 1024), (1, 1024), 0), out=buf220)
        del arg140_1
        buf224 = reinterpret_tensor(buf219, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf219  # reuse
        # Source Nodes: [hidden_states_95, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf203, buf220, arg141_1, arg142_1, arg143_1, buf224, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg142_1
        del arg143_1
        buf225 = reinterpret_tensor(buf201, (1024, 4096), (4096, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 4096), (1, 1024), 0), out=buf225)
        del arg144_1
        buf226 = reinterpret_tensor(buf225, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf225  # reuse
        # Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf226, arg145_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg145_1
        buf227 = reinterpret_tensor(buf224, (1024, 1024), (1024, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 1024), (1, 4096), 0), out=buf227)
        del arg146_1
        buf228 = reinterpret_tensor(buf227, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf227  # reuse
        buf232 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102, residual_17, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf228, buf203, buf220, arg141_1, arg147_1, arg148_1, arg149_1, buf232, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg141_1
        del arg147_1
        del arg148_1
        del arg149_1
        buf233 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), out=buf233)
        del arg150_1
        buf234 = reinterpret_tensor(buf203, (1024, 1024), (1024, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf234)
        del arg152_1
        buf235 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf234, arg153_1, buf235, 1048576, grid=grid(1048576), stream=stream0)
        del arg153_1
        buf236 = reinterpret_tensor(buf234, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf233, arg151_1, buf236, 1048576, grid=grid(1048576), stream=stream0)
        del arg151_1
        buf237 = buf217; del buf217  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf235, (16, 64, 1024), (65536, 1, 64), 0), out=buf237)
        buf242 = buf212; del buf212  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf237, buf242, 16384, 1024, grid=grid(16384), stream=stream0)
        buf240 = reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf240)
        del arg154_1
        buf241 = reinterpret_tensor(buf232, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf240, arg155_1, buf241, 1048576, grid=grid(1048576), stream=stream0)
        del arg155_1
        buf243 = reinterpret_tensor(buf240, (16, 1024, 64), (65536, 64, 1), 0); del buf240  # reuse
        # Source Nodes: [attn_output_45, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf242, reinterpret_tensor(buf241, (16, 1024, 64), (65536, 64, 1), 0), out=buf243)
        buf244 = reinterpret_tensor(buf233, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf243, buf244, 1048576, grid=grid(1048576), stream=stream0)
        buf245 = reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg156_1
        buf249 = reinterpret_tensor(buf244, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf244  # reuse
        # Source Nodes: [hidden_states_106, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf228, buf245, arg157_1, arg158_1, arg159_1, buf249, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg158_1
        del arg159_1
        buf250 = reinterpret_tensor(buf226, (1024, 4096), (4096, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 4096), (1, 1024), 0), out=buf250)
        del arg160_1
        buf251 = reinterpret_tensor(buf250, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf250  # reuse
        # Source Nodes: [hidden_states_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf251, arg161_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg161_1
        buf252 = reinterpret_tensor(buf249, (1024, 1024), (1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg162_1, (4096, 1024), (1, 4096), 0), out=buf252)
        del arg162_1
        buf253 = reinterpret_tensor(buf252, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf252  # reuse
        buf257 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_113, residual_19, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf253, buf228, buf245, arg157_1, arg163_1, arg164_1, arg165_1, buf257, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg157_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf258 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), out=buf258)
        del arg166_1
        buf259 = reinterpret_tensor(buf228, (1024, 1024), (1024, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg168_1
        buf260 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf259, arg169_1, buf260, 1048576, grid=grid(1048576), stream=stream0)
        del arg169_1
        buf261 = reinterpret_tensor(buf259, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf258, arg167_1, buf261, 1048576, grid=grid(1048576), stream=stream0)
        del arg167_1
        buf262 = buf242; del buf242  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf260, (16, 64, 1024), (65536, 1, 64), 0), out=buf262)
        buf267 = buf237; del buf237  # reuse
        # Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf262, buf267, 16384, 1024, grid=grid(16384), stream=stream0)
        buf265 = reinterpret_tensor(buf261, (1024, 1024), (1024, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf265)
        del arg170_1
        buf266 = reinterpret_tensor(buf257, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf257  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf265, arg171_1, buf266, 1048576, grid=grid(1048576), stream=stream0)
        del arg171_1
        buf268 = reinterpret_tensor(buf265, (16, 1024, 64), (65536, 64, 1), 0); del buf265  # reuse
        # Source Nodes: [attn_output_50, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf267, reinterpret_tensor(buf266, (16, 1024, 64), (65536, 64, 1), 0), out=buf268)
        buf269 = reinterpret_tensor(buf258, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf268, buf269, 1048576, grid=grid(1048576), stream=stream0)
        buf270 = reinterpret_tensor(buf268, (1024, 1024), (1024, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg172_1
        buf274 = reinterpret_tensor(buf269, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf269  # reuse
        # Source Nodes: [hidden_states_117, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf253, buf270, arg173_1, arg174_1, arg175_1, buf274, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg174_1
        del arg175_1
        buf275 = reinterpret_tensor(buf251, (1024, 4096), (4096, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg176_1, (1024, 4096), (1, 1024), 0), out=buf275)
        del arg176_1
        buf276 = reinterpret_tensor(buf275, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_118], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf276, arg177_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg177_1
        buf277 = reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg178_1, (4096, 1024), (1, 4096), 0), out=buf277)
        del arg178_1
        buf278 = reinterpret_tensor(buf277, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf277  # reuse
        buf282 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_124, residual_21, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf278, buf253, buf270, arg173_1, arg179_1, arg180_1, arg181_1, buf282, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg173_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf283 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), out=buf283)
        del arg182_1
        buf284 = reinterpret_tensor(buf253, (1024, 1024), (1024, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf284)
        del arg184_1
        buf285 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf284, arg185_1, buf285, 1048576, grid=grid(1048576), stream=stream0)
        del arg185_1
        buf286 = reinterpret_tensor(buf284, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf283, arg183_1, buf286, 1048576, grid=grid(1048576), stream=stream0)
        del arg183_1
        buf287 = buf267; del buf267  # reuse
        # Source Nodes: [attn_weights_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf286, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf285, (16, 64, 1024), (65536, 1, 64), 0), out=buf287)
        buf292 = buf262; del buf262  # reuse
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf287, buf292, 16384, 1024, grid=grid(16384), stream=stream0)
        del buf287
        buf290 = reinterpret_tensor(buf286, (1024, 1024), (1024, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf290)
        del arg186_1
        buf291 = reinterpret_tensor(buf282, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf282  # reuse
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf290, arg187_1, buf291, 1048576, grid=grid(1048576), stream=stream0)
        del arg187_1
        buf293 = reinterpret_tensor(buf290, (16, 1024, 64), (65536, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [attn_output_55, attn_weights_47], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf292, reinterpret_tensor(buf291, (16, 1024, 64), (65536, 64, 1), 0), out=buf293)
        del buf292
        buf294 = reinterpret_tensor(buf283, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf283  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf293, buf294, 1048576, grid=grid(1048576), stream=stream0)
        buf295 = reinterpret_tensor(buf293, (1024, 1024), (1024, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 1024), (1, 1024), 0), out=buf295)
        del arg188_1
        buf299 = reinterpret_tensor(buf294, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf294  # reuse
        # Source Nodes: [hidden_states_128, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf278, buf295, arg189_1, arg190_1, arg191_1, buf299, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg190_1
        del arg191_1
        buf300 = reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 4096), (1, 1024), 0), out=buf300)
        del arg192_1
        buf301 = reinterpret_tensor(buf300, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf300  # reuse
        # Source Nodes: [hidden_states_129], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf301, arg193_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg193_1
        buf302 = reinterpret_tensor(buf299, (1024, 1024), (1024, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf301, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 1024), (1, 4096), 0), out=buf302)
        del arg194_1
        del buf301
        buf303 = reinterpret_tensor(buf302, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf302  # reuse
        buf307 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_135, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf303, buf278, buf295, arg189_1, arg195_1, arg196_1, arg197_1, buf307, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg189_1
        del arg195_1
        del arg196_1
        del arg197_1
        del buf278
        del buf295
        del buf303
        buf308 = empty((1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg198_1, (1024, 50265), (1, 1024), 0), out=buf308)
        del arg198_1
        del buf307
        buf309 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf308, buf309, buf310, 1024, 50265, grid=grid(1024), stream=stream0)
        buf311 = empty((), device='cuda', dtype=torch.float32)
        buf313 = buf311; del buf311  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_9.run(buf313, arg200_1, buf308, buf309, buf310, 1, 1024, grid=grid(1), stream=stream0)
        del arg200_1
        return (buf313, reinterpret_tensor(buf308, (1, 1024, 50265), (51471360, 50265, 1), 0), buf10, buf16, buf35, buf41, buf60, buf66, buf85, buf91, buf110, buf116, buf135, buf141, buf160, buf166, buf185, buf191, buf210, buf216, buf235, buf241, buf260, buf266, buf285, buf291, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg200_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForCausalLM', benchmark_compiled_module)
