
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


# kernel path: /tmp/torchinductor_youkaichao/xj/cxjqqdqpcjf75nthx3rworsb2qwqkpv2hweo2m4u67idinrgaesf.py
# Source Nodes: [embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens, positions], Original ATen: [aten.add, aten.arange, aten.embedding, aten.mul, aten.native_layer_norm]
# embed_pos => embedding_1
# hidden_states => add
# hidden_states_1 => add_1, add_2, mul_1, mul_2, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
# positions => iota
triton_red_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
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
        tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp19 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50265
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp15 < 50265")
        tmp16 = tl.load(in_ptr1 + (r1 + (512*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 512.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwo6ml6wktgfnoquwrjlfahpf4rcltqoqcuisok4wl54boidzr5.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (512*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvzluzxdowseg46yvb6gg4jjmniv3djoyjb5cwtk5fcc7syoqky.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (512*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cuswalj5kiarc2cly2uloxnj3rgpihvntqpvxmecrdiqjtbowuhd.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_5
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxwyxrf3krveg5drdfbztvryhg5ejlewhd5b3tu3enqse6qzzve.py
# Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_5 => add_3
# residual_1 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftsjssaimje6eijjvywwjxctnw2oiuakazljak3hk57jv2oxpjh.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# hidden_states_7 => add_6, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
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


# kernel path: /tmp/torchinductor_youkaichao/nj/cnj5qduo7wa7oscgrknoo3tnmkqw77izvzg65q4xz4bljvwx3b23.py
# Source Nodes: [hidden_states_91, inputs_embeds_1, inputs_embeds_2, l__mod___model_encoder_embed_tokens_1, positions_1, positions_2], Original ATen: [aten.add, aten.arange, aten.embedding, aten.mul, aten.native_layer_norm]
# hidden_states_91 => add_62
# inputs_embeds_1 => mul_67
# inputs_embeds_2 => add_60, add_61, mul_68, mul_69, rsqrt_17, sub_25, var_mean_17
# l__mod___model_encoder_embed_tokens_1 => embedding_2
# positions_1 => iota_2
# positions_2 => embedding_3
triton_red_fused_add_arange_embedding_mul_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp0 + 50265
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert(((0 <= tmp13) & (tmp13 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp13 < 50265")
        tmp14 = tl.load(in_ptr1 + (r1 + (512*tmp13)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 512.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvcyttm6plwxyf23je6y6sxxidob3zqbfadkz237jzeno4ylfp5.py
# Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
# attn_weights_19 => amax_8, div_8, exp_8, sub_26, sum_9
triton_per_fused__softmax_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_7', 'mutated_arg_names': []}
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
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp13 / tmp17
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxzypicqqwjrgyvg4wwojvd4mufsmvfgya3sc26vopuai34mdxe.py
# Source Nodes: [attn_output_43], Original ATen: [aten.clone]
# attn_output_43 => clone_70
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (4096*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxeuibduqk4wopzuqjvxcndmytou6zfgcg6zbqw2bzxdjsoknxd.py
# Source Nodes: [lm_logits], Original ATen: [aten.add]
# lm_logits => add_151
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6433920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 50265
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigpu3uk24ir64ix6wloufccokjos2zmuyvogxoaujham6pqnhjd.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_24
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 7181
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7181*x0)
        tmp1 = tl.full([1, 1], 50265, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7181*x0) + (50265*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqiqifsennrnurqwmxrdvr6gii2mjmnrxrdnqmohotakpggmhch.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_24
triton_per_fused__log_softmax_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhdawqmmeotmgvijhhbh5cebr2dwdgqv6zx2vqpujes5r6cy2w6.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_24, sub_66, sum_25
triton_red_fused__log_softmax_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 7181
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7181*x0)
        tmp1 = tl.full([1, 1], 50265, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7181*x0) + (50265*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sl/csl447j2rr3eimzl3ens7rua5o5yul4gipizrcaockcmihzdio5p.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_24, sub_66, sum_25
triton_per_fused__log_softmax_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7ykpc5kyblaxyepexgzkup3a6eann6lp7eczjxcvgfdm7odfgk.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_24, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_2
triton_per_fused_nll_loss_forward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_14', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 512), (512, 1))
    assert_size_stride(arg1_1, (512, 512), (512, 1))
    assert_size_stride(arg2_1, (50265, 512), (512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, 512), (512, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, 512), (512, 1))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (2048, 512), (512, 1))
    assert_size_stride(arg16_1, (2048, ), (1, ))
    assert_size_stride(arg17_1, (512, 2048), (2048, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, 512), (512, 1))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, 512), (512, 1))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, 512), (512, 1))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 512), (512, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (2048, 512), (512, 1))
    assert_size_stride(arg32_1, (2048, ), (1, ))
    assert_size_stride(arg33_1, (512, 2048), (2048, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, 512), (512, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 512), (512, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (2048, 512), (512, 1))
    assert_size_stride(arg48_1, (2048, ), (1, ))
    assert_size_stride(arg49_1, (512, 2048), (2048, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, 512), (512, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (2048, 512), (512, 1))
    assert_size_stride(arg64_1, (2048, ), (1, ))
    assert_size_stride(arg65_1, (512, 2048), (2048, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, 512), (512, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, 512), (512, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (512, 2048), (2048, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, 512), (512, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (2048, 512), (512, 1))
    assert_size_stride(arg96_1, (2048, ), (1, ))
    assert_size_stride(arg97_1, (512, 2048), (2048, 1))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, 512), (512, 1))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, 512), (512, 1))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (2048, 512), (512, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (512, 2048), (2048, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, 512), (512, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, 512), (512, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (2048, 512), (512, 1))
    assert_size_stride(arg128_1, (2048, ), (1, ))
    assert_size_stride(arg129_1, (512, 2048), (2048, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, 512), (512, 1))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, 512), (512, 1))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 512), (512, 1))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, 512), (512, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (512, 512), (512, 1))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (512, 512), (512, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, 512), (512, 1))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, 512), (512, 1))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, ), (1, ))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (2048, 512), (512, 1))
    assert_size_stride(arg156_1, (2048, ), (1, ))
    assert_size_stride(arg157_1, (512, 2048), (2048, 1))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, 512), (512, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, 512), (512, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, 512), (512, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, 512), (512, 1))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (512, ), (1, ))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (512, 512), (512, 1))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, 512), (512, 1))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (512, 512), (512, 1))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (2048, 512), (512, 1))
    assert_size_stride(arg182_1, (2048, ), (1, ))
    assert_size_stride(arg183_1, (512, 2048), (2048, 1))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (512, 512), (512, 1))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, 512), (512, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, 512), (512, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, 512), (512, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, 512), (512, 1))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (512, 512), (512, 1))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, 512), (512, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, 512), (512, 1))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (2048, 512), (512, 1))
    assert_size_stride(arg208_1, (2048, ), (1, ))
    assert_size_stride(arg209_1, (512, 2048), (2048, 1))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, 512), (512, 1))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, 512), (512, 1))
    assert_size_stride(arg216_1, (512, ), (1, ))
    assert_size_stride(arg217_1, (512, 512), (512, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, 512), (512, 1))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, 512), (512, 1))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, 512), (512, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, 512), (512, 1))
    assert_size_stride(arg228_1, (512, ), (1, ))
    assert_size_stride(arg229_1, (512, 512), (512, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (2048, 512), (512, 1))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (512, 2048), (2048, 1))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, 512), (512, 1))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, 512), (512, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, 512), (512, 1))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, 512), (512, 1))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, 512), (512, 1))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, 512), (512, 1))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (512, 512), (512, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, 512), (512, 1))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (2048, 512), (512, 1))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (512, 2048), (2048, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, 512), (512, 1))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, 512), (512, 1))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, 512), (512, 1))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (512, 512), (512, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, 512), (512, 1))
    assert_size_stride(arg276_1, (512, ), (1, ))
    assert_size_stride(arg277_1, (512, 512), (512, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (512, 512), (512, 1))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, 512), (512, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (512, ), (1, ))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (2048, 512), (512, 1))
    assert_size_stride(arg286_1, (2048, ), (1, ))
    assert_size_stride(arg287_1, (512, 2048), (2048, 1))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, 512), (512, 1))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, 512), (512, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, 512), (512, 1))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (512, 512), (512, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, 512), (512, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, 512), (512, 1))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, 512), (512, 1))
    assert_size_stride(arg306_1, (512, ), (1, ))
    assert_size_stride(arg307_1, (512, 512), (512, 1))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (2048, 512), (512, 1))
    assert_size_stride(arg312_1, (2048, ), (1, ))
    assert_size_stride(arg313_1, (512, 2048), (2048, 1))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (512, 512), (512, 1))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (512, 512), (512, 1))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, 512), (512, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, 512), (512, 1))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (512, ), (1, ))
    assert_size_stride(arg327_1, (512, 512), (512, 1))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, 512), (512, 1))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (512, 512), (512, 1))
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, 512), (512, 1))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (512, ), (1, ))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (2048, 512), (512, 1))
    assert_size_stride(arg338_1, (2048, ), (1, ))
    assert_size_stride(arg339_1, (512, 2048), (2048, 1))
    assert_size_stride(arg340_1, (512, ), (1, ))
    assert_size_stride(arg341_1, (512, ), (1, ))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (50265, 512), (512, 1))
    assert_size_stride(arg344_1, (1, 50265), (50265, 1))
    assert_size_stride(arg345_1, (1, 128), (128, 1))
    assert_size_stride(arg346_1, (1, 128), (128, 1))
    assert_size_stride(arg347_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_encoder_embed_tokens, positions], Original ATen: [aten.add, aten.arange, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg347_1, arg2_1, arg0_1, arg3_1, arg4_1, buf3, 128, 512, grid=grid(128), stream=stream0)
        del arg0_1
        del arg347_1
        del arg3_1
        del arg4_1
        buf4 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf4)
        del arg5_1
        buf5 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf5)
        del arg7_1
        buf6 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 512), (1, 512), 0), out=buf6)
        del arg9_1
        buf7 = empty((1, 16, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg6_1, buf7, 65536, grid=grid(65536), stream=stream0)
        del arg6_1
        buf8 = reinterpret_tensor(buf4, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg8_1, buf8, 65536, grid=grid(65536), stream=stream0)
        del arg8_1
        buf9 = reinterpret_tensor(buf5, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg10_1, buf9, 65536, grid=grid(65536), stream=stream0)
        del arg10_1
        del buf6
        # Source Nodes: [], Original ATen: []
        buf10 = aten._scaled_dot_product_efficient_attention(buf7, buf8, buf9, None, True, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf11, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf15, 65536, grid=grid(65536), stream=stream0)
        buf16 = reinterpret_tensor(buf9, (128, 512), (512, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (128, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 512), (1, 512), 0), out=buf16)
        del arg11_1
        buf20 = reinterpret_tensor(buf15, (1, 128, 512), (65536, 512, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg12_1, arg13_1, arg14_1, buf20, 128, 512, grid=grid(128), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf21 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (128, 512), (512, 1), 0), reinterpret_tensor(arg15_1, (512, 2048), (1, 512), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf21, (1, 128, 2048), (262144, 2048, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg16_1, 262144, grid=grid(262144), stream=stream0)
        del arg16_1
        buf23 = reinterpret_tensor(buf3, (128, 512), (512, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 512), (1, 2048), 0), out=buf23)
        del arg17_1
        buf27 = reinterpret_tensor(buf16, (1, 128, 512), (65536, 512, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg18_1, arg19_1, arg20_1, buf27, 128, 512, grid=grid(128), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (128, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 512), (1, 512), 0), out=buf28)
        del arg21_1
        buf29 = reinterpret_tensor(buf20, (128, 512), (512, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (128, 512), (512, 1), 0), reinterpret_tensor(arg23_1, (512, 512), (1, 512), 0), out=buf29)
        del arg23_1
        buf30 = reinterpret_tensor(buf8, (128, 512), (512, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (128, 512), (512, 1), 0), reinterpret_tensor(arg25_1, (512, 512), (1, 512), 0), out=buf30)
        del arg25_1
        buf31 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg22_1, buf31, 65536, grid=grid(65536), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf28, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg24_1, buf32, 65536, grid=grid(65536), stream=stream0)
        del arg24_1
        buf33 = reinterpret_tensor(buf29, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg26_1, buf33, 65536, grid=grid(65536), stream=stream0)
        del arg26_1
        del buf30
        # Source Nodes: [], Original ATen: []
        buf34 = aten._scaled_dot_product_efficient_attention(buf31, buf32, buf33, None, True, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf35, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf39, 65536, grid=grid(65536), stream=stream0)
        buf40 = reinterpret_tensor(buf33, (128, 512), (512, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (128, 512), (512, 1), 0), reinterpret_tensor(arg27_1, (512, 512), (1, 512), 0), out=buf40)
        del arg27_1
        buf44 = reinterpret_tensor(buf39, (1, 128, 512), (65536, 512, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_16, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg28_1, arg29_1, arg30_1, buf44, 128, 512, grid=grid(128), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf45 = reinterpret_tensor(buf22, (128, 2048), (2048, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (128, 512), (512, 1), 0), reinterpret_tensor(arg31_1, (512, 2048), (1, 512), 0), out=buf45)
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (1, 128, 2048), (262144, 2048, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg32_1, 262144, grid=grid(262144), stream=stream0)
        del arg32_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 512), (1, 2048), 0), out=buf47)
        del arg33_1
        buf51 = buf27; del buf27  # reuse
        # Source Nodes: [hidden_states_22, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg34_1, arg35_1, arg36_1, buf51, 128, 512, grid=grid(128), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf52)
        del arg37_1
        buf53 = reinterpret_tensor(buf44, (128, 512), (512, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), reinterpret_tensor(arg39_1, (512, 512), (1, 512), 0), out=buf53)
        del arg39_1
        buf54 = reinterpret_tensor(buf32, (128, 512), (512, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), out=buf54)
        del arg41_1
        buf55 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg38_1, buf55, 65536, grid=grid(65536), stream=stream0)
        del arg38_1
        buf56 = reinterpret_tensor(buf52, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg40_1, buf56, 65536, grid=grid(65536), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf53, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg42_1, buf57, 65536, grid=grid(65536), stream=stream0)
        del arg42_1
        del buf54
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, True, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf59, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf59  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf63, 65536, grid=grid(65536), stream=stream0)
        buf64 = reinterpret_tensor(buf57, (128, 512), (512, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (128, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf64)
        del arg43_1
        buf68 = reinterpret_tensor(buf63, (1, 128, 512), (65536, 512, 1), 0); del buf63  # reuse
        # Source Nodes: [hidden_states_27, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg44_1, arg45_1, arg46_1, buf68, 128, 512, grid=grid(128), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf69 = reinterpret_tensor(buf46, (128, 2048), (2048, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (128, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 2048), (1, 512), 0), out=buf69)
        del arg47_1
        buf70 = reinterpret_tensor(buf69, (1, 128, 2048), (262144, 2048, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg48_1, 262144, grid=grid(262144), stream=stream0)
        del arg48_1
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg49_1, (2048, 512), (1, 2048), 0), out=buf71)
        del arg49_1
        buf75 = buf51; del buf51  # reuse
        # Source Nodes: [hidden_states_33, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg50_1, arg51_1, arg52_1, buf75, 128, 512, grid=grid(128), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf76)
        del arg53_1
        buf77 = reinterpret_tensor(buf68, (128, 512), (512, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf77)
        del arg55_1
        buf78 = reinterpret_tensor(buf56, (128, 512), (512, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(arg57_1, (512, 512), (1, 512), 0), out=buf78)
        del arg57_1
        buf79 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg54_1, buf79, 65536, grid=grid(65536), stream=stream0)
        del arg54_1
        buf80 = reinterpret_tensor(buf76, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg56_1, buf80, 65536, grid=grid(65536), stream=stream0)
        del arg56_1
        buf81 = reinterpret_tensor(buf77, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg58_1, buf81, 65536, grid=grid(65536), stream=stream0)
        del arg58_1
        del buf78
        # Source Nodes: [], Original ATen: []
        buf82 = aten._scaled_dot_product_efficient_attention(buf79, buf80, buf81, None, True, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf83, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf87, 65536, grid=grid(65536), stream=stream0)
        buf88 = reinterpret_tensor(buf81, (128, 512), (512, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (128, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf87, (1, 128, 512), (65536, 512, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_38, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg60_1, arg61_1, arg62_1, buf92, 128, 512, grid=grid(128), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf70, (128, 2048), (2048, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (128, 512), (512, 1), 0), reinterpret_tensor(arg63_1, (512, 2048), (1, 512), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (1, 128, 2048), (262144, 2048, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg64_1, 262144, grid=grid(262144), stream=stream0)
        del arg64_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg65_1, (2048, 512), (1, 2048), 0), out=buf95)
        del arg65_1
        buf99 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_44, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg66_1, arg67_1, arg68_1, buf99, 128, 512, grid=grid(128), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 512), (1, 512), 0), out=buf100)
        del arg69_1
        buf101 = reinterpret_tensor(buf92, (128, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), out=buf101)
        del arg71_1
        buf102 = reinterpret_tensor(buf80, (128, 512), (512, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 512), (1, 512), 0), out=buf102)
        del arg73_1
        buf103 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg70_1, buf103, 65536, grid=grid(65536), stream=stream0)
        del arg70_1
        buf104 = reinterpret_tensor(buf100, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg72_1, buf104, 65536, grid=grid(65536), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf101, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg74_1, buf105, 65536, grid=grid(65536), stream=stream0)
        del arg74_1
        del buf102
        # Source Nodes: [], Original ATen: []
        buf106 = aten._scaled_dot_product_efficient_attention(buf103, buf104, buf105, None, True, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf107, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf107  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf111, 65536, grid=grid(65536), stream=stream0)
        buf112 = reinterpret_tensor(buf105, (128, 512), (512, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (128, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf112)
        del arg75_1
        buf116 = reinterpret_tensor(buf111, (1, 128, 512), (65536, 512, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_49, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg76_1, arg77_1, arg78_1, buf116, 128, 512, grid=grid(128), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf117 = reinterpret_tensor(buf94, (128, 2048), (2048, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (128, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf117)
        del arg79_1
        buf118 = reinterpret_tensor(buf117, (1, 128, 2048), (262144, 2048, 1), 0); del buf117  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg80_1, 262144, grid=grid(262144), stream=stream0)
        del arg80_1
        buf119 = reinterpret_tensor(buf99, (128, 512), (512, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 512), (1, 2048), 0), out=buf119)
        del arg81_1
        buf123 = reinterpret_tensor(buf112, (1, 128, 512), (65536, 512, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_55, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg82_1, arg83_1, arg84_1, buf123, 128, 512, grid=grid(128), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (128, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf124)
        del arg85_1
        buf125 = reinterpret_tensor(buf116, (128, 512), (512, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (128, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf125)
        del arg87_1
        buf126 = reinterpret_tensor(buf104, (128, 512), (512, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (128, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 512), (1, 512), 0), out=buf126)
        del arg89_1
        buf127 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg86_1, buf127, 65536, grid=grid(65536), stream=stream0)
        del arg86_1
        buf128 = reinterpret_tensor(buf124, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg88_1, buf128, 65536, grid=grid(65536), stream=stream0)
        del arg88_1
        buf129 = reinterpret_tensor(buf125, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg90_1, buf129, 65536, grid=grid(65536), stream=stream0)
        del arg90_1
        del buf126
        # Source Nodes: [], Original ATen: []
        buf130 = aten._scaled_dot_product_efficient_attention(buf127, buf128, buf129, None, True, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf131, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf131  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf135, 65536, grid=grid(65536), stream=stream0)
        buf136 = reinterpret_tensor(buf129, (128, 512), (512, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf136)
        del arg91_1
        buf140 = reinterpret_tensor(buf135, (1, 128, 512), (65536, 512, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_60, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg92_1, arg93_1, arg94_1, buf140, 128, 512, grid=grid(128), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf141 = reinterpret_tensor(buf118, (128, 2048), (2048, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 2048), (1, 512), 0), out=buf141)
        del arg95_1
        buf142 = reinterpret_tensor(buf141, (1, 128, 2048), (262144, 2048, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg96_1, 262144, grid=grid(262144), stream=stream0)
        del arg96_1
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 512), (1, 2048), 0), out=buf143)
        del arg97_1
        buf147 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg98_1, arg99_1, arg100_1, buf147, 128, 512, grid=grid(128), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf148 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (128, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf148)
        del arg101_1
        buf149 = reinterpret_tensor(buf140, (128, 512), (512, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (128, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), out=buf149)
        del arg103_1
        buf150 = reinterpret_tensor(buf128, (128, 512), (512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (128, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), out=buf150)
        del arg105_1
        buf151 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf148, arg102_1, buf151, 65536, grid=grid(65536), stream=stream0)
        del arg102_1
        buf152 = reinterpret_tensor(buf148, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf149, arg104_1, buf152, 65536, grid=grid(65536), stream=stream0)
        del arg104_1
        buf153 = reinterpret_tensor(buf149, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf150, arg106_1, buf153, 65536, grid=grid(65536), stream=stream0)
        del arg106_1
        del buf150
        # Source Nodes: [], Original ATen: []
        buf154 = aten._scaled_dot_product_efficient_attention(buf151, buf152, buf153, None, True, scale=1.0)
        buf155 = buf154[0]
        del buf154
        buf159 = reinterpret_tensor(buf155, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf159, 65536, grid=grid(65536), stream=stream0)
        buf160 = reinterpret_tensor(buf153, (128, 512), (512, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (128, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf160)
        del arg107_1
        buf164 = reinterpret_tensor(buf159, (1, 128, 512), (65536, 512, 1), 0); del buf159  # reuse
        # Source Nodes: [hidden_states_71, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf147, buf160, arg108_1, arg109_1, arg110_1, buf164, 128, 512, grid=grid(128), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        buf165 = reinterpret_tensor(buf142, (128, 2048), (2048, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (128, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 2048), (1, 512), 0), out=buf165)
        del arg111_1
        buf166 = reinterpret_tensor(buf165, (1, 128, 2048), (262144, 2048, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf166, arg112_1, 262144, grid=grid(262144), stream=stream0)
        del arg112_1
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg113_1, (2048, 512), (1, 2048), 0), out=buf167)
        del arg113_1
        buf171 = buf147; del buf147  # reuse
        # Source Nodes: [hidden_states_77, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf164, buf167, arg114_1, arg115_1, arg116_1, buf171, 128, 512, grid=grid(128), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf172 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (128, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), out=buf172)
        del arg117_1
        buf173 = reinterpret_tensor(buf164, (128, 512), (512, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (128, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 512), (1, 512), 0), out=buf173)
        del arg119_1
        buf174 = reinterpret_tensor(buf152, (128, 512), (512, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (128, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf174)
        del arg121_1
        buf175 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf172, arg118_1, buf175, 65536, grid=grid(65536), stream=stream0)
        del arg118_1
        buf176 = reinterpret_tensor(buf172, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf173, arg120_1, buf176, 65536, grid=grid(65536), stream=stream0)
        del arg120_1
        buf177 = reinterpret_tensor(buf173, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf174, arg122_1, buf177, 65536, grid=grid(65536), stream=stream0)
        del arg122_1
        # Source Nodes: [], Original ATen: []
        buf178 = aten._scaled_dot_product_efficient_attention(buf175, buf176, buf177, None, True, scale=1.0)
        buf179 = buf178[0]
        del buf178
        buf183 = reinterpret_tensor(buf179, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf183, 65536, grid=grid(65536), stream=stream0)
        buf184 = reinterpret_tensor(buf177, (128, 512), (512, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), out=buf184)
        del arg123_1
        buf188 = reinterpret_tensor(buf183, (1, 128, 512), (65536, 512, 1), 0); del buf183  # reuse
        # Source Nodes: [hidden_states_82, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf171, buf184, arg124_1, arg125_1, arg126_1, buf188, 128, 512, grid=grid(128), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf189 = reinterpret_tensor(buf166, (128, 2048), (2048, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (128, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 2048), (1, 512), 0), out=buf189)
        del arg127_1
        buf190 = reinterpret_tensor(buf189, (1, 128, 2048), (262144, 2048, 1), 0); del buf189  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf190, arg128_1, 262144, grid=grid(262144), stream=stream0)
        del arg128_1
        buf191 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg129_1, (2048, 512), (1, 2048), 0), out=buf191)
        del arg129_1
        buf217 = buf171; del buf171  # reuse
        # Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf188, buf191, arg130_1, arg131_1, arg132_1, buf217, 128, 512, grid=grid(128), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        buf198 = reinterpret_tensor(buf191, (1, 128, 512), (65536, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [hidden_states_91, inputs_embeds_1, inputs_embeds_2, l__mod___model_encoder_embed_tokens_1, positions_1, positions_2], Original ATen: [aten.add, aten.arange, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_6.run(arg346_1, arg2_1, arg133_1, arg134_1, arg1_1, buf198, 128, 512, grid=grid(128), stream=stream0)
        del arg133_1
        del arg134_1
        del arg1_1
        del arg2_1
        del arg346_1
        buf199 = reinterpret_tensor(buf188, (128, 512), (512, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 512), (512, 1), 0), reinterpret_tensor(arg135_1, (512, 512), (1, 512), 0), out=buf199)
        del arg135_1
        buf200 = reinterpret_tensor(buf176, (128, 512), (512, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 512), (512, 1), 0), reinterpret_tensor(arg137_1, (512, 512), (1, 512), 0), out=buf200)
        del arg137_1
        buf201 = buf175; del buf175  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf199, arg136_1, buf201, 65536, grid=grid(65536), stream=stream0)
        del arg136_1
        buf202 = reinterpret_tensor(buf199, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf199  # reuse
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf200, arg138_1, buf202, 65536, grid=grid(65536), stream=stream0)
        del arg138_1
        buf203 = reinterpret_tensor(buf190, (16, 128, 128), (16384, 128, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf202, (16, 32, 128), (4096, 1, 32), 0), out=buf203)
        buf207 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf203, buf207, 2048, 128, grid=grid(2048), stream=stream0)
        buf206 = reinterpret_tensor(buf202, (128, 512), (512, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 512), (1, 512), 0), out=buf206)
        del arg139_1
        buf208 = buf201; del buf201  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf206, arg140_1, buf208, 65536, grid=grid(65536), stream=stream0)
        del arg140_1
        buf209 = reinterpret_tensor(buf206, (16, 128, 32), (4096, 32, 1), 0); del buf206  # reuse
        # Source Nodes: [attn_output_40, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf207, reinterpret_tensor(buf208, (16, 128, 32), (4096, 32, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf208, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf208  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf209, buf210, 65536, grid=grid(65536), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (128, 512), (512, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (128, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 512), (1, 512), 0), out=buf211)
        del arg141_1
        buf215 = reinterpret_tensor(buf210, (1, 128, 512), (65536, 512, 1), 0); del buf210  # reuse
        # Source Nodes: [hidden_states_95, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf198, buf211, arg142_1, arg143_1, arg144_1, buf215, 128, 512, grid=grid(128), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        buf216 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 512), (1, 512), 0), out=buf216)
        del arg145_1
        buf218 = reinterpret_tensor(buf198, (128, 512), (512, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 512), (1, 512), 0), out=buf218)
        del arg147_1
        buf219 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 512), (1, 512), 0), out=buf219)
        del arg149_1
        buf220 = reinterpret_tensor(buf174, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf216, arg146_1, buf220, 65536, grid=grid(65536), stream=stream0)
        del arg146_1
        buf221 = reinterpret_tensor(buf216, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf218, arg148_1, buf221, 65536, grid=grid(65536), stream=stream0)
        del arg148_1
        buf222 = reinterpret_tensor(buf218, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf219, arg150_1, buf222, 65536, grid=grid(65536), stream=stream0)
        del arg150_1
        del buf219
        # Source Nodes: [], Original ATen: []
        buf223 = aten._scaled_dot_product_efficient_attention(buf220, buf221, buf222, None, True, scale=1.0)
        buf224 = buf223[0]
        del buf223
        buf228 = reinterpret_tensor(buf224, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf224  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf228, 65536, grid=grid(65536), stream=stream0)
        buf229 = reinterpret_tensor(buf222, (128, 512), (512, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 512), (1, 512), 0), out=buf229)
        del arg151_1
        buf233 = reinterpret_tensor(buf228, (1, 128, 512), (65536, 512, 1), 0); del buf228  # reuse
        # Source Nodes: [hidden_states_99, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf215, buf229, arg152_1, arg153_1, arg154_1, buf233, 128, 512, grid=grid(128), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        buf234 = reinterpret_tensor(buf207, (128, 2048), (2048, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (128, 512), (512, 1), 0), reinterpret_tensor(arg155_1, (512, 2048), (1, 512), 0), out=buf234)
        del arg155_1
        buf235 = reinterpret_tensor(buf234, (1, 128, 2048), (262144, 2048, 1), 0); del buf234  # reuse
        # Source Nodes: [hidden_states_101], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf235, arg156_1, 262144, grid=grid(262144), stream=stream0)
        del arg156_1
        buf236 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg157_1, (2048, 512), (1, 2048), 0), out=buf236)
        del arg157_1
        buf240 = buf215; del buf215  # reuse
        # Source Nodes: [hidden_states_105, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf233, buf236, arg158_1, arg159_1, arg160_1, buf240, 128, 512, grid=grid(128), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        buf241 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 512), (1, 512), 0), out=buf241)
        del arg161_1
        buf242 = reinterpret_tensor(buf233, (128, 512), (512, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 512), (1, 512), 0), out=buf242)
        del arg163_1
        buf243 = buf221; del buf221  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf241, arg162_1, buf243, 65536, grid=grid(65536), stream=stream0)
        del arg162_1
        buf244 = reinterpret_tensor(buf241, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf241  # reuse
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf242, arg164_1, buf244, 65536, grid=grid(65536), stream=stream0)
        del arg164_1
        buf245 = reinterpret_tensor(buf235, (16, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
        # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf244, (16, 32, 128), (4096, 1, 32), 0), out=buf245)
        buf249 = buf203; del buf203  # reuse
        # Source Nodes: [attn_weights_25], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf245, buf249, 2048, 128, grid=grid(2048), stream=stream0)
        buf248 = reinterpret_tensor(buf244, (128, 512), (512, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg165_1, (512, 512), (1, 512), 0), out=buf248)
        del arg165_1
        buf250 = buf243; del buf243  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf248, arg166_1, buf250, 65536, grid=grid(65536), stream=stream0)
        del arg166_1
        buf251 = reinterpret_tensor(buf248, (16, 128, 32), (4096, 32, 1), 0); del buf248  # reuse
        # Source Nodes: [attn_output_50, attn_weights_25], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf249, reinterpret_tensor(buf250, (16, 128, 32), (4096, 32, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf250, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf250  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf251, buf252, 65536, grid=grid(65536), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (128, 512), (512, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (128, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 512), (1, 512), 0), out=buf253)
        del arg167_1
        buf257 = reinterpret_tensor(buf252, (1, 128, 512), (65536, 512, 1), 0); del buf252  # reuse
        # Source Nodes: [hidden_states_110, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf240, buf253, arg168_1, arg169_1, arg170_1, buf257, 128, 512, grid=grid(128), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        buf258 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (128, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 512), (1, 512), 0), out=buf258)
        del arg171_1
        buf259 = reinterpret_tensor(buf240, (128, 512), (512, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 512), (1, 512), 0), out=buf259)
        del arg173_1
        buf260 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 512), (1, 512), 0), out=buf260)
        del arg175_1
        buf261 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf258, arg172_1, buf261, 65536, grid=grid(65536), stream=stream0)
        del arg172_1
        buf262 = reinterpret_tensor(buf258, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf259, arg174_1, buf262, 65536, grid=grid(65536), stream=stream0)
        del arg174_1
        buf263 = reinterpret_tensor(buf259, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf260, arg176_1, buf263, 65536, grid=grid(65536), stream=stream0)
        del arg176_1
        del buf260
        # Source Nodes: [], Original ATen: []
        buf264 = aten._scaled_dot_product_efficient_attention(buf261, buf262, buf263, None, True, scale=1.0)
        buf265 = buf264[0]
        del buf264
        buf269 = reinterpret_tensor(buf265, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf265  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf269, 65536, grid=grid(65536), stream=stream0)
        buf270 = reinterpret_tensor(buf263, (128, 512), (512, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (128, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), out=buf270)
        del arg177_1
        buf274 = reinterpret_tensor(buf269, (1, 128, 512), (65536, 512, 1), 0); del buf269  # reuse
        # Source Nodes: [hidden_states_114, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf257, buf270, arg178_1, arg179_1, arg180_1, buf274, 128, 512, grid=grid(128), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        buf275 = reinterpret_tensor(buf249, (128, 2048), (2048, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), reinterpret_tensor(arg181_1, (512, 2048), (1, 512), 0), out=buf275)
        del arg181_1
        buf276 = reinterpret_tensor(buf275, (1, 128, 2048), (262144, 2048, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_116], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf276, arg182_1, 262144, grid=grid(262144), stream=stream0)
        del arg182_1
        buf277 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg183_1, (2048, 512), (1, 2048), 0), out=buf277)
        del arg183_1
        buf281 = buf257; del buf257  # reuse
        # Source Nodes: [hidden_states_120, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf274, buf277, arg184_1, arg185_1, arg186_1, buf281, 128, 512, grid=grid(128), stream=stream0)
        del arg184_1
        del arg185_1
        del arg186_1
        buf282 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 512), (1, 512), 0), out=buf282)
        del arg187_1
        buf283 = reinterpret_tensor(buf274, (128, 512), (512, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 512), (512, 1), 0), reinterpret_tensor(arg189_1, (512, 512), (1, 512), 0), out=buf283)
        del arg189_1
        buf284 = buf262; del buf262  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf282, arg188_1, buf284, 65536, grid=grid(65536), stream=stream0)
        del arg188_1
        buf285 = reinterpret_tensor(buf282, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf282  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf283, arg190_1, buf285, 65536, grid=grid(65536), stream=stream0)
        del arg190_1
        buf286 = reinterpret_tensor(buf276, (16, 128, 128), (16384, 128, 1), 0); del buf276  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf285, (16, 32, 128), (4096, 1, 32), 0), out=buf286)
        buf290 = buf245; del buf245  # reuse
        # Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf286, buf290, 2048, 128, grid=grid(2048), stream=stream0)
        buf289 = reinterpret_tensor(buf285, (128, 512), (512, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 512), (1, 512), 0), out=buf289)
        del arg191_1
        buf291 = buf284; del buf284  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf289, arg192_1, buf291, 65536, grid=grid(65536), stream=stream0)
        del arg192_1
        buf292 = reinterpret_tensor(buf289, (16, 128, 32), (4096, 32, 1), 0); del buf289  # reuse
        # Source Nodes: [attn_output_60, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf290, reinterpret_tensor(buf291, (16, 128, 32), (4096, 32, 1), 0), out=buf292)
        buf293 = reinterpret_tensor(buf291, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf291  # reuse
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf292, buf293, 65536, grid=grid(65536), stream=stream0)
        buf294 = reinterpret_tensor(buf292, (128, 512), (512, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (128, 512), (512, 1), 0), reinterpret_tensor(arg193_1, (512, 512), (1, 512), 0), out=buf294)
        del arg193_1
        buf298 = reinterpret_tensor(buf293, (1, 128, 512), (65536, 512, 1), 0); del buf293  # reuse
        # Source Nodes: [hidden_states_125, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf281, buf294, arg194_1, arg195_1, arg196_1, buf298, 128, 512, grid=grid(128), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        buf299 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 512), (512, 1), 0), reinterpret_tensor(arg197_1, (512, 512), (1, 512), 0), out=buf299)
        del arg197_1
        buf300 = reinterpret_tensor(buf281, (128, 512), (512, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg199_1, (512, 512), (1, 512), 0), out=buf300)
        del arg199_1
        buf301 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg201_1, (512, 512), (1, 512), 0), out=buf301)
        del arg201_1
        buf302 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf299, arg198_1, buf302, 65536, grid=grid(65536), stream=stream0)
        del arg198_1
        buf303 = reinterpret_tensor(buf299, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf300, arg200_1, buf303, 65536, grid=grid(65536), stream=stream0)
        del arg200_1
        buf304 = reinterpret_tensor(buf300, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf301, arg202_1, buf304, 65536, grid=grid(65536), stream=stream0)
        del arg202_1
        del buf301
        # Source Nodes: [], Original ATen: []
        buf305 = aten._scaled_dot_product_efficient_attention(buf302, buf303, buf304, None, True, scale=1.0)
        buf306 = buf305[0]
        del buf305
        buf310 = reinterpret_tensor(buf306, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf306  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf310, 65536, grid=grid(65536), stream=stream0)
        buf311 = reinterpret_tensor(buf304, (128, 512), (512, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 512), (512, 1), 0), reinterpret_tensor(arg203_1, (512, 512), (1, 512), 0), out=buf311)
        del arg203_1
        buf315 = reinterpret_tensor(buf310, (1, 128, 512), (65536, 512, 1), 0); del buf310  # reuse
        # Source Nodes: [hidden_states_129, residual_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf298, buf311, arg204_1, arg205_1, arg206_1, buf315, 128, 512, grid=grid(128), stream=stream0)
        del arg204_1
        del arg205_1
        del arg206_1
        buf316 = reinterpret_tensor(buf290, (128, 2048), (2048, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (128, 512), (512, 1), 0), reinterpret_tensor(arg207_1, (512, 2048), (1, 512), 0), out=buf316)
        del arg207_1
        buf317 = reinterpret_tensor(buf316, (1, 128, 2048), (262144, 2048, 1), 0); del buf316  # reuse
        # Source Nodes: [hidden_states_131], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf317, arg208_1, 262144, grid=grid(262144), stream=stream0)
        del arg208_1
        buf318 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg209_1, (2048, 512), (1, 2048), 0), out=buf318)
        del arg209_1
        buf322 = buf298; del buf298  # reuse
        # Source Nodes: [hidden_states_135, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf315, buf318, arg210_1, arg211_1, arg212_1, buf322, 128, 512, grid=grid(128), stream=stream0)
        del arg210_1
        del arg211_1
        del arg212_1
        buf323 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 512), (512, 1), 0), reinterpret_tensor(arg213_1, (512, 512), (1, 512), 0), out=buf323)
        del arg213_1
        buf324 = reinterpret_tensor(buf315, (128, 512), (512, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 512), (1, 512), 0), out=buf324)
        del arg215_1
        buf325 = buf303; del buf303  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf323, arg214_1, buf325, 65536, grid=grid(65536), stream=stream0)
        del arg214_1
        buf326 = reinterpret_tensor(buf323, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf323  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf324, arg216_1, buf326, 65536, grid=grid(65536), stream=stream0)
        del arg216_1
        buf327 = reinterpret_tensor(buf317, (16, 128, 128), (16384, 128, 1), 0); del buf317  # reuse
        # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf326, (16, 32, 128), (4096, 1, 32), 0), out=buf327)
        buf331 = buf286; del buf286  # reuse
        # Source Nodes: [attn_weights_37], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf327, buf331, 2048, 128, grid=grid(2048), stream=stream0)
        buf330 = reinterpret_tensor(buf326, (128, 512), (512, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 512), (512, 1), 0), reinterpret_tensor(arg217_1, (512, 512), (1, 512), 0), out=buf330)
        del arg217_1
        buf332 = buf325; del buf325  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf330, arg218_1, buf332, 65536, grid=grid(65536), stream=stream0)
        del arg218_1
        buf333 = reinterpret_tensor(buf330, (16, 128, 32), (4096, 32, 1), 0); del buf330  # reuse
        # Source Nodes: [attn_output_70, attn_weights_37], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf331, reinterpret_tensor(buf332, (16, 128, 32), (4096, 32, 1), 0), out=buf333)
        buf334 = reinterpret_tensor(buf332, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf332  # reuse
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf333, buf334, 65536, grid=grid(65536), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (128, 512), (512, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (128, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 512), (1, 512), 0), out=buf335)
        del arg219_1
        buf339 = reinterpret_tensor(buf334, (1, 128, 512), (65536, 512, 1), 0); del buf334  # reuse
        # Source Nodes: [hidden_states_140, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf322, buf335, arg220_1, arg221_1, arg222_1, buf339, 128, 512, grid=grid(128), stream=stream0)
        del arg220_1
        del arg221_1
        del arg222_1
        buf340 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (128, 512), (512, 1), 0), reinterpret_tensor(arg223_1, (512, 512), (1, 512), 0), out=buf340)
        del arg223_1
        buf341 = reinterpret_tensor(buf322, (128, 512), (512, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg225_1, (512, 512), (1, 512), 0), out=buf341)
        del arg225_1
        buf342 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg227_1, (512, 512), (1, 512), 0), out=buf342)
        del arg227_1
        buf343 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf340, arg224_1, buf343, 65536, grid=grid(65536), stream=stream0)
        del arg224_1
        buf344 = reinterpret_tensor(buf340, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf341, arg226_1, buf344, 65536, grid=grid(65536), stream=stream0)
        del arg226_1
        buf345 = reinterpret_tensor(buf341, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf342, arg228_1, buf345, 65536, grid=grid(65536), stream=stream0)
        del arg228_1
        del buf342
        # Source Nodes: [], Original ATen: []
        buf346 = aten._scaled_dot_product_efficient_attention(buf343, buf344, buf345, None, True, scale=1.0)
        buf347 = buf346[0]
        del buf346
        buf351 = reinterpret_tensor(buf347, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf347  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf351, 65536, grid=grid(65536), stream=stream0)
        buf352 = reinterpret_tensor(buf345, (128, 512), (512, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (128, 512), (512, 1), 0), reinterpret_tensor(arg229_1, (512, 512), (1, 512), 0), out=buf352)
        del arg229_1
        buf356 = reinterpret_tensor(buf351, (1, 128, 512), (65536, 512, 1), 0); del buf351  # reuse
        # Source Nodes: [hidden_states_144, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf339, buf352, arg230_1, arg231_1, arg232_1, buf356, 128, 512, grid=grid(128), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        buf357 = reinterpret_tensor(buf331, (128, 2048), (2048, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 512), (512, 1), 0), reinterpret_tensor(arg233_1, (512, 2048), (1, 512), 0), out=buf357)
        del arg233_1
        buf358 = reinterpret_tensor(buf357, (1, 128, 2048), (262144, 2048, 1), 0); del buf357  # reuse
        # Source Nodes: [hidden_states_146], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf358, arg234_1, 262144, grid=grid(262144), stream=stream0)
        del arg234_1
        buf359 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg235_1, (2048, 512), (1, 2048), 0), out=buf359)
        del arg235_1
        buf363 = buf339; del buf339  # reuse
        # Source Nodes: [hidden_states_150, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf356, buf359, arg236_1, arg237_1, arg238_1, buf363, 128, 512, grid=grid(128), stream=stream0)
        del arg236_1
        del arg237_1
        del arg238_1
        buf364 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (128, 512), (512, 1), 0), reinterpret_tensor(arg239_1, (512, 512), (1, 512), 0), out=buf364)
        del arg239_1
        buf365 = reinterpret_tensor(buf356, (128, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (128, 512), (512, 1), 0), reinterpret_tensor(arg241_1, (512, 512), (1, 512), 0), out=buf365)
        del arg241_1
        buf366 = buf344; del buf344  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf364, arg240_1, buf366, 65536, grid=grid(65536), stream=stream0)
        del arg240_1
        buf367 = reinterpret_tensor(buf364, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf364  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf365, arg242_1, buf367, 65536, grid=grid(65536), stream=stream0)
        del arg242_1
        buf368 = reinterpret_tensor(buf358, (16, 128, 128), (16384, 128, 1), 0); del buf358  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf367, (16, 32, 128), (4096, 1, 32), 0), out=buf368)
        buf372 = buf327; del buf327  # reuse
        # Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf368, buf372, 2048, 128, grid=grid(2048), stream=stream0)
        buf371 = reinterpret_tensor(buf367, (128, 512), (512, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (128, 512), (512, 1), 0), reinterpret_tensor(arg243_1, (512, 512), (1, 512), 0), out=buf371)
        del arg243_1
        buf373 = buf366; del buf366  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf371, arg244_1, buf373, 65536, grid=grid(65536), stream=stream0)
        del arg244_1
        buf374 = reinterpret_tensor(buf371, (16, 128, 32), (4096, 32, 1), 0); del buf371  # reuse
        # Source Nodes: [attn_output_80, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf372, reinterpret_tensor(buf373, (16, 128, 32), (4096, 32, 1), 0), out=buf374)
        buf375 = reinterpret_tensor(buf373, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf373  # reuse
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf374, buf375, 65536, grid=grid(65536), stream=stream0)
        buf376 = reinterpret_tensor(buf374, (128, 512), (512, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (128, 512), (512, 1), 0), reinterpret_tensor(arg245_1, (512, 512), (1, 512), 0), out=buf376)
        del arg245_1
        buf380 = reinterpret_tensor(buf375, (1, 128, 512), (65536, 512, 1), 0); del buf375  # reuse
        # Source Nodes: [hidden_states_155, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf363, buf376, arg246_1, arg247_1, arg248_1, buf380, 128, 512, grid=grid(128), stream=stream0)
        del arg246_1
        del arg247_1
        del arg248_1
        buf381 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (128, 512), (512, 1), 0), reinterpret_tensor(arg249_1, (512, 512), (1, 512), 0), out=buf381)
        del arg249_1
        buf382 = reinterpret_tensor(buf363, (128, 512), (512, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 512), (1, 512), 0), out=buf382)
        del arg251_1
        buf383 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg253_1, (512, 512), (1, 512), 0), out=buf383)
        del arg253_1
        buf384 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf381, arg250_1, buf384, 65536, grid=grid(65536), stream=stream0)
        del arg250_1
        buf385 = reinterpret_tensor(buf381, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf382, arg252_1, buf385, 65536, grid=grid(65536), stream=stream0)
        del arg252_1
        buf386 = reinterpret_tensor(buf382, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf383, arg254_1, buf386, 65536, grid=grid(65536), stream=stream0)
        del arg254_1
        del buf383
        # Source Nodes: [], Original ATen: []
        buf387 = aten._scaled_dot_product_efficient_attention(buf384, buf385, buf386, None, True, scale=1.0)
        buf388 = buf387[0]
        del buf387
        buf392 = reinterpret_tensor(buf388, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf388  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf392, 65536, grid=grid(65536), stream=stream0)
        buf393 = reinterpret_tensor(buf386, (128, 512), (512, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (128, 512), (512, 1), 0), reinterpret_tensor(arg255_1, (512, 512), (1, 512), 0), out=buf393)
        del arg255_1
        buf397 = reinterpret_tensor(buf392, (1, 128, 512), (65536, 512, 1), 0); del buf392  # reuse
        # Source Nodes: [hidden_states_159, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf380, buf393, arg256_1, arg257_1, arg258_1, buf397, 128, 512, grid=grid(128), stream=stream0)
        del arg256_1
        del arg257_1
        del arg258_1
        buf398 = reinterpret_tensor(buf372, (128, 2048), (2048, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (128, 512), (512, 1), 0), reinterpret_tensor(arg259_1, (512, 2048), (1, 512), 0), out=buf398)
        del arg259_1
        buf399 = reinterpret_tensor(buf398, (1, 128, 2048), (262144, 2048, 1), 0); del buf398  # reuse
        # Source Nodes: [hidden_states_161], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf399, arg260_1, 262144, grid=grid(262144), stream=stream0)
        del arg260_1
        buf400 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 512), (1, 2048), 0), out=buf400)
        del arg261_1
        buf404 = buf380; del buf380  # reuse
        # Source Nodes: [hidden_states_165, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf397, buf400, arg262_1, arg263_1, arg264_1, buf404, 128, 512, grid=grid(128), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        buf405 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (128, 512), (512, 1), 0), reinterpret_tensor(arg265_1, (512, 512), (1, 512), 0), out=buf405)
        del arg265_1
        buf406 = reinterpret_tensor(buf397, (128, 512), (512, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (128, 512), (512, 1), 0), reinterpret_tensor(arg267_1, (512, 512), (1, 512), 0), out=buf406)
        del arg267_1
        buf407 = buf385; del buf385  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf405, arg266_1, buf407, 65536, grid=grid(65536), stream=stream0)
        del arg266_1
        buf408 = reinterpret_tensor(buf405, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf405  # reuse
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf406, arg268_1, buf408, 65536, grid=grid(65536), stream=stream0)
        del arg268_1
        buf409 = reinterpret_tensor(buf399, (16, 128, 128), (16384, 128, 1), 0); del buf399  # reuse
        # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf408, (16, 32, 128), (4096, 1, 32), 0), out=buf409)
        buf413 = buf368; del buf368  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf409, buf413, 2048, 128, grid=grid(2048), stream=stream0)
        buf412 = reinterpret_tensor(buf408, (128, 512), (512, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (128, 512), (512, 1), 0), reinterpret_tensor(arg269_1, (512, 512), (1, 512), 0), out=buf412)
        del arg269_1
        buf414 = buf407; del buf407  # reuse
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf412, arg270_1, buf414, 65536, grid=grid(65536), stream=stream0)
        del arg270_1
        buf415 = reinterpret_tensor(buf412, (16, 128, 32), (4096, 32, 1), 0); del buf412  # reuse
        # Source Nodes: [attn_output_90, attn_weights_49], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf413, reinterpret_tensor(buf414, (16, 128, 32), (4096, 32, 1), 0), out=buf415)
        buf416 = reinterpret_tensor(buf414, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf414  # reuse
        # Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf415, buf416, 65536, grid=grid(65536), stream=stream0)
        buf417 = reinterpret_tensor(buf415, (128, 512), (512, 1), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (128, 512), (512, 1), 0), reinterpret_tensor(arg271_1, (512, 512), (1, 512), 0), out=buf417)
        del arg271_1
        buf421 = reinterpret_tensor(buf416, (1, 128, 512), (65536, 512, 1), 0); del buf416  # reuse
        # Source Nodes: [hidden_states_170, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf404, buf417, arg272_1, arg273_1, arg274_1, buf421, 128, 512, grid=grid(128), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        buf422 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 512), (512, 1), 0), reinterpret_tensor(arg275_1, (512, 512), (1, 512), 0), out=buf422)
        del arg275_1
        buf423 = reinterpret_tensor(buf404, (128, 512), (512, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg277_1, (512, 512), (1, 512), 0), out=buf423)
        del arg277_1
        buf424 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg279_1, (512, 512), (1, 512), 0), out=buf424)
        del arg279_1
        buf425 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf422, arg276_1, buf425, 65536, grid=grid(65536), stream=stream0)
        del arg276_1
        buf426 = reinterpret_tensor(buf422, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf423, arg278_1, buf426, 65536, grid=grid(65536), stream=stream0)
        del arg278_1
        buf427 = reinterpret_tensor(buf423, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf424, arg280_1, buf427, 65536, grid=grid(65536), stream=stream0)
        del arg280_1
        del buf424
        # Source Nodes: [], Original ATen: []
        buf428 = aten._scaled_dot_product_efficient_attention(buf425, buf426, buf427, None, True, scale=1.0)
        buf429 = buf428[0]
        del buf428
        buf433 = reinterpret_tensor(buf429, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf429  # reuse
        # Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf433, 65536, grid=grid(65536), stream=stream0)
        buf434 = reinterpret_tensor(buf427, (128, 512), (512, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (128, 512), (512, 1), 0), reinterpret_tensor(arg281_1, (512, 512), (1, 512), 0), out=buf434)
        del arg281_1
        buf438 = reinterpret_tensor(buf433, (1, 128, 512), (65536, 512, 1), 0); del buf433  # reuse
        # Source Nodes: [hidden_states_174, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf421, buf434, arg282_1, arg283_1, arg284_1, buf438, 128, 512, grid=grid(128), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        buf439 = reinterpret_tensor(buf413, (128, 2048), (2048, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (128, 512), (512, 1), 0), reinterpret_tensor(arg285_1, (512, 2048), (1, 512), 0), out=buf439)
        del arg285_1
        buf440 = reinterpret_tensor(buf439, (1, 128, 2048), (262144, 2048, 1), 0); del buf439  # reuse
        # Source Nodes: [hidden_states_176], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf440, arg286_1, 262144, grid=grid(262144), stream=stream0)
        del arg286_1
        buf441 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg287_1, (2048, 512), (1, 2048), 0), out=buf441)
        del arg287_1
        buf445 = buf421; del buf421  # reuse
        # Source Nodes: [hidden_states_180, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf438, buf441, arg288_1, arg289_1, arg290_1, buf445, 128, 512, grid=grid(128), stream=stream0)
        del arg288_1
        del arg289_1
        del arg290_1
        buf446 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (128, 512), (512, 1), 0), reinterpret_tensor(arg291_1, (512, 512), (1, 512), 0), out=buf446)
        del arg291_1
        buf447 = reinterpret_tensor(buf438, (128, 512), (512, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (128, 512), (512, 1), 0), reinterpret_tensor(arg293_1, (512, 512), (1, 512), 0), out=buf447)
        del arg293_1
        buf448 = buf426; del buf426  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf446, arg292_1, buf448, 65536, grid=grid(65536), stream=stream0)
        del arg292_1
        buf449 = reinterpret_tensor(buf446, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf446  # reuse
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf447, arg294_1, buf449, 65536, grid=grid(65536), stream=stream0)
        del arg294_1
        buf450 = reinterpret_tensor(buf440, (16, 128, 128), (16384, 128, 1), 0); del buf440  # reuse
        # Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf448, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf449, (16, 32, 128), (4096, 1, 32), 0), out=buf450)
        buf454 = buf409; del buf409  # reuse
        # Source Nodes: [attn_weights_55], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf450, buf454, 2048, 128, grid=grid(2048), stream=stream0)
        buf453 = reinterpret_tensor(buf449, (128, 512), (512, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (128, 512), (512, 1), 0), reinterpret_tensor(arg295_1, (512, 512), (1, 512), 0), out=buf453)
        del arg295_1
        buf455 = buf448; del buf448  # reuse
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf453, arg296_1, buf455, 65536, grid=grid(65536), stream=stream0)
        del arg296_1
        buf456 = reinterpret_tensor(buf453, (16, 128, 32), (4096, 32, 1), 0); del buf453  # reuse
        # Source Nodes: [attn_output_100, attn_weights_55], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf454, reinterpret_tensor(buf455, (16, 128, 32), (4096, 32, 1), 0), out=buf456)
        buf457 = reinterpret_tensor(buf455, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf455  # reuse
        # Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf456, buf457, 65536, grid=grid(65536), stream=stream0)
        buf458 = reinterpret_tensor(buf456, (128, 512), (512, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (128, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 512), (1, 512), 0), out=buf458)
        del arg297_1
        buf462 = reinterpret_tensor(buf457, (1, 128, 512), (65536, 512, 1), 0); del buf457  # reuse
        # Source Nodes: [hidden_states_185, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf445, buf458, arg298_1, arg299_1, arg300_1, buf462, 128, 512, grid=grid(128), stream=stream0)
        del arg298_1
        del arg299_1
        del arg300_1
        buf463 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (128, 512), (512, 1), 0), reinterpret_tensor(arg301_1, (512, 512), (1, 512), 0), out=buf463)
        del arg301_1
        buf464 = reinterpret_tensor(buf445, (128, 512), (512, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 512), (1, 512), 0), out=buf464)
        del arg303_1
        buf465 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg305_1, (512, 512), (1, 512), 0), out=buf465)
        del arg305_1
        buf466 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf463, arg302_1, buf466, 65536, grid=grid(65536), stream=stream0)
        del arg302_1
        buf467 = reinterpret_tensor(buf463, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf464, arg304_1, buf467, 65536, grid=grid(65536), stream=stream0)
        del arg304_1
        buf468 = reinterpret_tensor(buf464, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf465, arg306_1, buf468, 65536, grid=grid(65536), stream=stream0)
        del arg306_1
        del buf465
        # Source Nodes: [], Original ATen: []
        buf469 = aten._scaled_dot_product_efficient_attention(buf466, buf467, buf468, None, True, scale=1.0)
        buf470 = buf469[0]
        del buf469
        buf474 = reinterpret_tensor(buf470, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf470  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf474, 65536, grid=grid(65536), stream=stream0)
        buf475 = reinterpret_tensor(buf468, (128, 512), (512, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 512), (512, 1), 0), reinterpret_tensor(arg307_1, (512, 512), (1, 512), 0), out=buf475)
        del arg307_1
        buf479 = reinterpret_tensor(buf474, (1, 128, 512), (65536, 512, 1), 0); del buf474  # reuse
        # Source Nodes: [hidden_states_189, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf462, buf475, arg308_1, arg309_1, arg310_1, buf479, 128, 512, grid=grid(128), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        buf480 = reinterpret_tensor(buf454, (128, 2048), (2048, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (128, 512), (512, 1), 0), reinterpret_tensor(arg311_1, (512, 2048), (1, 512), 0), out=buf480)
        del arg311_1
        buf481 = reinterpret_tensor(buf480, (1, 128, 2048), (262144, 2048, 1), 0); del buf480  # reuse
        # Source Nodes: [hidden_states_191], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf481, arg312_1, 262144, grid=grid(262144), stream=stream0)
        del arg312_1
        buf482 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg313_1, (2048, 512), (1, 2048), 0), out=buf482)
        del arg313_1
        buf486 = buf462; del buf462  # reuse
        # Source Nodes: [hidden_states_195, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf479, buf482, arg314_1, arg315_1, arg316_1, buf486, 128, 512, grid=grid(128), stream=stream0)
        del arg314_1
        del arg315_1
        del arg316_1
        buf487 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (128, 512), (512, 1), 0), reinterpret_tensor(arg317_1, (512, 512), (1, 512), 0), out=buf487)
        del arg317_1
        buf488 = reinterpret_tensor(buf479, (128, 512), (512, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (128, 512), (512, 1), 0), reinterpret_tensor(arg319_1, (512, 512), (1, 512), 0), out=buf488)
        del arg319_1
        buf489 = buf467; del buf467  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_1.run(buf487, arg318_1, buf489, 65536, grid=grid(65536), stream=stream0)
        del arg318_1
        buf490 = reinterpret_tensor(buf487, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf487  # reuse
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf488, arg320_1, buf490, 65536, grid=grid(65536), stream=stream0)
        del arg320_1
        buf491 = reinterpret_tensor(buf481, (16, 128, 128), (16384, 128, 1), 0); del buf481  # reuse
        # Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf490, (16, 32, 128), (4096, 1, 32), 0), out=buf491)
        buf495 = buf450; del buf450  # reuse
        # Source Nodes: [attn_weights_61], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf491, buf495, 2048, 128, grid=grid(2048), stream=stream0)
        del buf491
        buf494 = reinterpret_tensor(buf490, (128, 512), (512, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (128, 512), (512, 1), 0), reinterpret_tensor(arg321_1, (512, 512), (1, 512), 0), out=buf494)
        del arg321_1
        buf496 = buf489; del buf489  # reuse
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf494, arg322_1, buf496, 65536, grid=grid(65536), stream=stream0)
        del arg322_1
        buf497 = reinterpret_tensor(buf494, (16, 128, 32), (4096, 32, 1), 0); del buf494  # reuse
        # Source Nodes: [attn_output_110, attn_weights_61], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf495, reinterpret_tensor(buf496, (16, 128, 32), (4096, 32, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf496, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf496  # reuse
        # Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf497, buf498, 65536, grid=grid(65536), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (128, 512), (512, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (128, 512), (512, 1), 0), reinterpret_tensor(arg323_1, (512, 512), (1, 512), 0), out=buf499)
        del arg323_1
        buf503 = reinterpret_tensor(buf498, (1, 128, 512), (65536, 512, 1), 0); del buf498  # reuse
        # Source Nodes: [hidden_states_200, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf486, buf499, arg324_1, arg325_1, arg326_1, buf503, 128, 512, grid=grid(128), stream=stream0)
        del arg324_1
        del arg325_1
        del arg326_1
        buf504 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (128, 512), (512, 1), 0), reinterpret_tensor(arg327_1, (512, 512), (1, 512), 0), out=buf504)
        del arg327_1
        buf505 = reinterpret_tensor(buf486, (128, 512), (512, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg329_1, (512, 512), (1, 512), 0), out=buf505)
        del arg329_1
        buf506 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(arg331_1, (512, 512), (1, 512), 0), out=buf506)
        del arg331_1
        buf507 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf504, arg328_1, buf507, 65536, grid=grid(65536), stream=stream0)
        del arg328_1
        buf508 = reinterpret_tensor(buf504, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf505, arg330_1, buf508, 65536, grid=grid(65536), stream=stream0)
        del arg330_1
        buf509 = reinterpret_tensor(buf505, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf506, arg332_1, buf509, 65536, grid=grid(65536), stream=stream0)
        del arg332_1
        del buf506
        # Source Nodes: [], Original ATen: []
        buf510 = aten._scaled_dot_product_efficient_attention(buf507, buf508, buf509, None, True, scale=1.0)
        del buf507
        del buf508
        buf511 = buf510[0]
        del buf510
        buf515 = reinterpret_tensor(buf511, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf511  # reuse
        # Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf515, 65536, grid=grid(65536), stream=stream0)
        buf516 = reinterpret_tensor(buf509, (128, 512), (512, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (128, 512), (512, 1), 0), reinterpret_tensor(arg333_1, (512, 512), (1, 512), 0), out=buf516)
        del arg333_1
        buf520 = reinterpret_tensor(buf515, (1, 128, 512), (65536, 512, 1), 0); del buf515  # reuse
        # Source Nodes: [hidden_states_204, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf503, buf516, arg334_1, arg335_1, arg336_1, buf520, 128, 512, grid=grid(128), stream=stream0)
        del arg334_1
        del arg335_1
        del arg336_1
        buf521 = reinterpret_tensor(buf495, (128, 2048), (2048, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (128, 512), (512, 1), 0), reinterpret_tensor(arg337_1, (512, 2048), (1, 512), 0), out=buf521)
        del arg337_1
        buf522 = reinterpret_tensor(buf521, (1, 128, 2048), (262144, 2048, 1), 0); del buf521  # reuse
        # Source Nodes: [hidden_states_206], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf522, arg338_1, 262144, grid=grid(262144), stream=stream0)
        del arg338_1
        buf523 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg339_1, (2048, 512), (1, 2048), 0), out=buf523)
        del arg339_1
        del buf522
        buf527 = buf503; del buf503  # reuse
        # Source Nodes: [hidden_states_210, hidden_states_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf520, buf523, arg340_1, arg341_1, arg342_1, buf527, 128, 512, grid=grid(128), stream=stream0)
        del arg340_1
        del arg341_1
        del arg342_1
        del buf520
        del buf523
        buf528 = empty((128, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (128, 512), (512, 1), 0), reinterpret_tensor(arg343_1, (512, 50265), (1, 512), 0), out=buf528)
        del arg343_1
        del buf527
        buf529 = reinterpret_tensor(buf528, (1, 128, 50265), (6433920, 50265, 1), 0); del buf528  # reuse
        # Source Nodes: [lm_logits], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf529, arg344_1, 6433920, grid=grid(6433920), stream=stream0)
        del arg344_1
        buf530 = empty_strided((128, 1, 7), (7, 896, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf529, buf530, 896, 7181, grid=grid(896), stream=stream0)
        buf531 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_11.run(buf530, buf531, 128, 7, grid=grid(128), stream=stream0)
        buf532 = buf530; del buf530  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_12.run(buf529, buf531, buf532, 896, 7181, grid=grid(896), stream=stream0)
        buf533 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_13.run(buf532, buf533, 128, 7, grid=grid(128), stream=stream0)
        del buf532
        buf534 = empty((), device='cuda', dtype=torch.float32)
        buf536 = buf534; del buf534  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_14.run(buf536, arg345_1, buf529, buf531, buf533, 1, 128, grid=grid(1), stream=stream0)
        del arg345_1
        return (buf536, buf529, buf217, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((50265, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg346_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg347_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
