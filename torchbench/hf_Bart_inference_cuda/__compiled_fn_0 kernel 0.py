
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtb2qz5vh7lw5c2jfypgypx7awqdqw4a3iosjn4r7p6pdqkqnlj.py
# Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_model_model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# embed_pos => embedding_1
# hidden_states => add_1
# hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_model_model_encoder_embed_tokens => embedding
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 512
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 50265), "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50265
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert((0 <= tmp15) & (tmp15 < 50265), "index out of bounds: 0 <= tmp15 < 50265")
        tmp16 = tl.load(in_ptr1 + (r2 + (768*tmp15)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 768.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5mveqnbfi5oz5folstdtaygis35tvtns72mnqtuzcr5hbeyqrs.py
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
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 12)) + (768*x1) + (393216*(x2 // 12))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgd4o5ehfq34csnnqm4rxooxzblyibjjogfftg6iebow22rd6fo.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 12)) + (768*x1) + (393216*(x2 // 12))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsoaleneg4f2wlkxiidd6docbrktrtyodbg3hqhq2iyny62nrt3.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 512
    x2 = (xindex // 393216)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*x2) + (3072*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4cnfshfmzupyeod5v26ib4tfr7rmjm76uxwiasklmx34i3blnk.py
# Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_5 => add_4
# residual_1 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eb33qvtsuzmxv5zbgmwatdjcjh32qidnh2v72unhdps6u3ycgx.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# hidden_states_7 => add_7, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/u6/cu63winbpb3nhwyskjungxvxykmxdsrww2wglrzuysudktqnn3ct.py
# Source Nodes: [key_states_12], Original ATen: [aten.clone]
# key_states_12 => clone_50
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczaocsyfh6zjm6fqutvdwosak5bkqzfmi6fzew36jhewdo36kwx.py
# Source Nodes: [contiguous_20], Original ATen: [aten.clone]
# contiguous_20 => clone_52
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covukj373wjzmvyzibhasirtdcrp72v6xy53qhfyzh7a6gnbt47k.py
# Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
# attn_weights_15 => amax_6, div_6, exp_6, sub_20, sum_7
triton_per_fused__softmax_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 24576
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0)
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
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6uxzqh2ttbrvsmej7tlii52ctdy6orrt4yxtu3wrnfrtuuevidu.py
# Source Nodes: [attn_output_33], Original ATen: [aten.clone]
# attn_output_33 => clone_54
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 512
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgnkogjzs4lbyupy4v5u4t6a4n4c2hlh2skpsfbpnyh5puns7w6.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38605824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50265, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50268, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = 0.0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co34v6ikbxaewoivpkhs22ihn6fcuzrjvtnifxyhsvtge66rr5gw.py
# Source Nodes: [lm_logits_1], Original ATen: [aten.add]
# lm_logits_1 => add_117
triton_poi_fused_add_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102942720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50265
    x1 = (xindex // 50265)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50268*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 768), (768, 1))
    assert_size_stride(arg1_1, (1026, 768), (768, 1))
    assert_size_stride(arg2_1, (50265, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (3072, ), (1, ))
    assert_size_stride(arg65_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (50265, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (3072, 768), (768, 1))
    assert_size_stride(arg125_1, (3072, ), (1, ))
    assert_size_stride(arg126_1, (768, 3072), (3072, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, 768), (768, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, 768), (768, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (3072, 768), (768, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (768, 3072), (3072, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (768, 3072), (3072, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 768), (768, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, 768), (768, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, 768), (768, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, 768), (768, 1))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (3072, 768), (768, 1))
    assert_size_stride(arg203_1, (3072, ), (1, ))
    assert_size_stride(arg204_1, (768, 3072), (3072, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, 768), (768, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, 768), (768, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, 768), (768, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, 768), (768, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, 768), (768, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, 768), (768, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, 768), (768, 1))
    assert_size_stride(arg223_1, (768, ), (1, ))
    assert_size_stride(arg224_1, (768, 768), (768, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (3072, 768), (768, 1))
    assert_size_stride(arg229_1, (3072, ), (1, ))
    assert_size_stride(arg230_1, (768, 3072), (3072, 1))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, 768), (768, 1))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, 768), (768, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, 768), (768, 1))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, 768), (768, 1))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, 768), (768, 1))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, 768), (768, 1))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (768, 768), (768, 1))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, 768), (768, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (3072, 768), (768, 1))
    assert_size_stride(arg255_1, (3072, ), (1, ))
    assert_size_stride(arg256_1, (768, 3072), (3072, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (50265, 768), (768, 1))
    assert_size_stride(arg261_1, (1, 50265), (50265, 1))
    assert_size_stride(arg262_1, (4, 512), (512, 1))
    assert_size_stride(arg263_1, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_model_model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg262_1, arg2_1, arg0_1, arg3_1, arg4_1, buf3, 2048, 768, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg262_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf4 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf4)
        del arg5_1
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), out=buf5)
        del arg7_1
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf6)
        del arg9_1
        buf7 = empty((1, 48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg6_1, buf7, 1572864, grid=grid(1572864), stream=stream0)
        del arg6_1
        buf8 = reinterpret_tensor(buf4, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg8_1, buf8, 1572864, grid=grid(1572864), stream=stream0)
        del arg8_1
        buf9 = reinterpret_tensor(buf5, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg10_1, buf9, 1572864, grid=grid(1572864), stream=stream0)
        del arg10_1
        # Source Nodes: [], Original ATen: []
        buf10 = aten._scaled_dot_product_efficient_attention(buf7, buf8, buf9, None, True, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf9, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf11, buf15, 1572864, grid=grid(1572864), stream=stream0)
        buf16 = reinterpret_tensor(buf11, (2048, 768), (768, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf16)
        del arg11_1
        buf20 = reinterpret_tensor(buf15, (4, 512, 768), (393216, 768, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg12_1, arg13_1, arg14_1, buf20, 2048, 768, grid=grid(2048), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf21 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf21, (4, 512, 3072), (1572864, 3072, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg16_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg16_1
        buf23 = reinterpret_tensor(buf3, (2048, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), out=buf23)
        del arg17_1
        buf27 = reinterpret_tensor(buf16, (4, 512, 768), (393216, 768, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg18_1, arg19_1, arg20_1, buf27, 2048, 768, grid=grid(2048), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf28)
        del arg21_1
        buf29 = reinterpret_tensor(buf20, (2048, 768), (768, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), out=buf29)
        del arg23_1
        buf30 = reinterpret_tensor(buf8, (2048, 768), (768, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), out=buf30)
        del arg25_1
        buf31 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg22_1, buf31, 1572864, grid=grid(1572864), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf28, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg24_1, buf32, 1572864, grid=grid(1572864), stream=stream0)
        del arg24_1
        buf33 = reinterpret_tensor(buf29, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg26_1, buf33, 1572864, grid=grid(1572864), stream=stream0)
        del arg26_1
        # Source Nodes: [], Original ATen: []
        buf34 = aten._scaled_dot_product_efficient_attention(buf31, buf32, buf33, None, True, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf33, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf35, buf39, 1572864, grid=grid(1572864), stream=stream0)
        buf40 = reinterpret_tensor(buf35, (2048, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (2048, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf40)
        del arg27_1
        buf44 = reinterpret_tensor(buf39, (4, 512, 768), (393216, 768, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_16, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg28_1, arg29_1, arg30_1, buf44, 2048, 768, grid=grid(2048), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf45 = reinterpret_tensor(buf22, (2048, 3072), (3072, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (2048, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), out=buf45)
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (4, 512, 3072), (1572864, 3072, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg32_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg32_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), out=buf47)
        del arg33_1
        buf51 = buf27; del buf27  # reuse
        # Source Nodes: [hidden_states_22, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg34_1, arg35_1, arg36_1, buf51, 2048, 768, grid=grid(2048), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (2048, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), out=buf52)
        del arg37_1
        buf53 = reinterpret_tensor(buf44, (2048, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (2048, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), out=buf53)
        del arg39_1
        buf54 = reinterpret_tensor(buf32, (2048, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (2048, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf54)
        del arg41_1
        buf55 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg38_1, buf55, 1572864, grid=grid(1572864), stream=stream0)
        del arg38_1
        buf56 = reinterpret_tensor(buf52, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg40_1, buf56, 1572864, grid=grid(1572864), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf53, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg42_1, buf57, 1572864, grid=grid(1572864), stream=stream0)
        del arg42_1
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, True, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf57, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf59, buf63, 1572864, grid=grid(1572864), stream=stream0)
        buf64 = reinterpret_tensor(buf59, (2048, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (2048, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf64)
        del arg43_1
        buf68 = reinterpret_tensor(buf63, (4, 512, 768), (393216, 768, 1), 0); del buf63  # reuse
        # Source Nodes: [hidden_states_27, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg44_1, arg45_1, arg46_1, buf68, 2048, 768, grid=grid(2048), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf69 = reinterpret_tensor(buf46, (2048, 3072), (3072, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (2048, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf69)
        del arg47_1
        buf70 = reinterpret_tensor(buf69, (4, 512, 3072), (1572864, 3072, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg48_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg48_1
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf71)
        del arg49_1
        buf75 = buf51; del buf51  # reuse
        # Source Nodes: [hidden_states_33, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg50_1, arg51_1, arg52_1, buf75, 2048, 768, grid=grid(2048), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (2048, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), out=buf76)
        del arg53_1
        buf77 = reinterpret_tensor(buf68, (2048, 768), (768, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (2048, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), out=buf77)
        del arg55_1
        buf78 = reinterpret_tensor(buf56, (2048, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (2048, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf78)
        del arg57_1
        buf79 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg54_1, buf79, 1572864, grid=grid(1572864), stream=stream0)
        del arg54_1
        buf80 = reinterpret_tensor(buf76, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg56_1, buf80, 1572864, grid=grid(1572864), stream=stream0)
        del arg56_1
        buf81 = reinterpret_tensor(buf77, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg58_1, buf81, 1572864, grid=grid(1572864), stream=stream0)
        del arg58_1
        # Source Nodes: [], Original ATen: []
        buf82 = aten._scaled_dot_product_efficient_attention(buf79, buf80, buf81, None, True, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf81, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf81  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf83, buf87, 1572864, grid=grid(1572864), stream=stream0)
        buf88 = reinterpret_tensor(buf83, (2048, 768), (768, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf87, (4, 512, 768), (393216, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_38, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg60_1, arg61_1, arg62_1, buf92, 2048, 768, grid=grid(2048), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf70, (2048, 3072), (3072, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (2048, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (4, 512, 3072), (1572864, 3072, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg64_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg64_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg65_1
        buf99 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_44, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg66_1, arg67_1, arg68_1, buf99, 2048, 768, grid=grid(2048), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (2048, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf100)
        del arg69_1
        buf101 = reinterpret_tensor(buf92, (2048, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (2048, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), out=buf101)
        del arg71_1
        buf102 = reinterpret_tensor(buf80, (2048, 768), (768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (2048, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf102)
        del arg73_1
        buf103 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg70_1, buf103, 1572864, grid=grid(1572864), stream=stream0)
        del arg70_1
        buf104 = reinterpret_tensor(buf100, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg72_1, buf104, 1572864, grid=grid(1572864), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf101, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg74_1, buf105, 1572864, grid=grid(1572864), stream=stream0)
        del arg74_1
        # Source Nodes: [], Original ATen: []
        buf106 = aten._scaled_dot_product_efficient_attention(buf103, buf104, buf105, None, True, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf105, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf105  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf107, buf111, 1572864, grid=grid(1572864), stream=stream0)
        buf112 = reinterpret_tensor(buf107, (2048, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (2048, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), out=buf112)
        del arg75_1
        buf116 = reinterpret_tensor(buf111, (4, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_49, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg76_1, arg77_1, arg78_1, buf116, 2048, 768, grid=grid(2048), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf117 = reinterpret_tensor(buf94, (2048, 3072), (3072, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (2048, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), out=buf117)
        del arg79_1
        buf118 = reinterpret_tensor(buf117, (4, 512, 3072), (1572864, 3072, 1), 0); del buf117  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg80_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg80_1
        buf119 = reinterpret_tensor(buf99, (2048, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), out=buf119)
        del arg81_1
        buf123 = reinterpret_tensor(buf112, (4, 512, 768), (393216, 768, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_55, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg82_1, arg83_1, arg84_1, buf123, 2048, 768, grid=grid(2048), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (2048, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), out=buf124)
        del arg85_1
        buf125 = reinterpret_tensor(buf116, (2048, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (2048, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf125)
        del arg87_1
        buf126 = reinterpret_tensor(buf104, (2048, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (2048, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf126)
        del arg89_1
        buf127 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg86_1, buf127, 1572864, grid=grid(1572864), stream=stream0)
        del arg86_1
        buf128 = reinterpret_tensor(buf124, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg88_1, buf128, 1572864, grid=grid(1572864), stream=stream0)
        del arg88_1
        buf129 = reinterpret_tensor(buf125, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg90_1, buf129, 1572864, grid=grid(1572864), stream=stream0)
        del arg90_1
        # Source Nodes: [], Original ATen: []
        buf130 = aten._scaled_dot_product_efficient_attention(buf127, buf128, buf129, None, True, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf129, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf131, buf135, 1572864, grid=grid(1572864), stream=stream0)
        buf136 = reinterpret_tensor(buf131, (2048, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (2048, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf136)
        del arg91_1
        buf140 = reinterpret_tensor(buf135, (4, 512, 768), (393216, 768, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_60, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg92_1, arg93_1, arg94_1, buf140, 2048, 768, grid=grid(2048), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf141 = reinterpret_tensor(buf118, (2048, 3072), (3072, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2048, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), out=buf141)
        del arg95_1
        buf142 = reinterpret_tensor(buf141, (4, 512, 3072), (1572864, 3072, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg96_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg96_1
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), out=buf143)
        del arg97_1
        buf169 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg98_1, arg99_1, arg100_1, buf169, 2048, 768, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf150 = reinterpret_tensor(buf143, (4, 512, 768), (393216, 768, 1), 0); del buf143  # reuse
        # Source Nodes: [add_15, hidden_states_69, hidden_states_70, inputs_embeds_1, l__mod___model_model_model_decoder_embed_tokens, positions_2], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg263_1, arg101_1, arg1_1, arg102_1, arg103_1, buf150, 2048, 768, grid=grid(2048), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg1_1
        del arg263_1
        buf151 = reinterpret_tensor(buf140, (2048, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (2048, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), out=buf151)
        del arg104_1
        buf152 = reinterpret_tensor(buf128, (2048, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (2048, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), out=buf152)
        del arg106_1
        buf153 = reinterpret_tensor(buf127, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf127  # reuse
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf152, arg107_1, buf153, 1572864, grid=grid(1572864), stream=stream0)
        del arg107_1
        buf154 = reinterpret_tensor(buf152, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf151, arg105_1, buf154, 1572864, grid=grid(1572864), stream=stream0)
        del arg105_1
        buf155 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf153, (48, 64, 512), (32768, 1, 64), 0), out=buf155)
        buf160 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf155, buf160, 24576, 512, grid=grid(24576), stream=stream0)
        buf158 = reinterpret_tensor(buf154, (2048, 768), (768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (2048, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf158)
        del arg108_1
        buf159 = reinterpret_tensor(buf151, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf151  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf158, arg109_1, buf159, 1572864, grid=grid(1572864), stream=stream0)
        del arg109_1
        buf161 = reinterpret_tensor(buf158, (48, 512, 64), (32768, 64, 1), 0); del buf158  # reuse
        # Source Nodes: [attn_output_30, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf160, reinterpret_tensor(buf159, (48, 512, 64), (32768, 64, 1), 0), out=buf161)
        buf162 = reinterpret_tensor(buf126, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf126  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf161, buf162, 1572864, grid=grid(1572864), stream=stream0)
        buf163 = reinterpret_tensor(buf161, (2048, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (2048, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), out=buf163)
        del arg110_1
        buf167 = reinterpret_tensor(buf162, (4, 512, 768), (393216, 768, 1), 0); del buf162  # reuse
        # Source Nodes: [hidden_states_74, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf150, buf163, arg111_1, arg112_1, arg113_1, buf167, 2048, 768, grid=grid(2048), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        buf168 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), out=buf168)
        del arg114_1
        buf170 = reinterpret_tensor(buf150, (2048, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf170)
        del arg116_1
        buf171 = reinterpret_tensor(buf102, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf170, arg117_1, buf171, 1572864, grid=grid(1572864), stream=stream0)
        del arg117_1
        buf172 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf172)
        del arg118_1
        buf173 = reinterpret_tensor(buf78, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf172, arg119_1, buf173, 1572864, grid=grid(1572864), stream=stream0)
        del arg119_1
        buf174 = reinterpret_tensor(buf172, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf168, arg115_1, buf174, 1572864, grid=grid(1572864), stream=stream0)
        del arg115_1
        # Source Nodes: [], Original ATen: []
        buf175 = aten._scaled_dot_product_efficient_attention(buf174, reinterpret_tensor(buf171, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf173, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf176 = buf175[0]
        del buf175
        buf180 = reinterpret_tensor(buf174, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf176, buf180, 1572864, grid=grid(1572864), stream=stream0)
        buf181 = reinterpret_tensor(buf176, (2048, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (2048, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), out=buf181)
        del arg120_1
        buf185 = reinterpret_tensor(buf180, (4, 512, 768), (393216, 768, 1), 0); del buf180  # reuse
        # Source Nodes: [hidden_states_78, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf167, buf181, arg121_1, arg122_1, arg123_1, buf185, 2048, 768, grid=grid(2048), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        buf186 = reinterpret_tensor(buf142, (2048, 3072), (3072, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (2048, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 3072), (1, 768), 0), out=buf186)
        del arg124_1
        buf187 = reinterpret_tensor(buf186, (4, 512, 3072), (1572864, 3072, 1), 0); del buf186  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf187, arg125_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg125_1
        buf188 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg126_1, (3072, 768), (1, 3072), 0), out=buf188)
        del arg126_1
        buf192 = buf167; del buf167  # reuse
        # Source Nodes: [hidden_states_84, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf185, buf188, arg127_1, arg128_1, arg129_1, buf192, 2048, 768, grid=grid(2048), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        buf193 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (2048, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 768), (1, 768), 0), out=buf193)
        del arg130_1
        buf194 = reinterpret_tensor(buf185, (2048, 768), (768, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (2048, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), out=buf194)
        del arg132_1
        buf195 = reinterpret_tensor(buf168, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf168  # reuse
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf194, arg133_1, buf195, 1572864, grid=grid(1572864), stream=stream0)
        del arg133_1
        buf196 = reinterpret_tensor(buf194, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf194  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf193, arg131_1, buf196, 1572864, grid=grid(1572864), stream=stream0)
        del arg131_1
        buf197 = buf160; del buf160  # reuse
        # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf195, (48, 64, 512), (32768, 1, 64), 0), out=buf197)
        buf202 = buf155; del buf155  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf197, buf202, 24576, 512, grid=grid(24576), stream=stream0)
        buf200 = reinterpret_tensor(buf196, (2048, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (2048, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), out=buf200)
        del arg134_1
        buf201 = reinterpret_tensor(buf193, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf193  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf200, arg135_1, buf201, 1572864, grid=grid(1572864), stream=stream0)
        del arg135_1
        buf203 = reinterpret_tensor(buf200, (48, 512, 64), (32768, 64, 1), 0); del buf200  # reuse
        # Source Nodes: [attn_output_40, attn_weights_21], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf202, reinterpret_tensor(buf201, (48, 512, 64), (32768, 64, 1), 0), out=buf203)
        buf204 = reinterpret_tensor(buf54, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf203, buf204, 1572864, grid=grid(1572864), stream=stream0)
        buf205 = reinterpret_tensor(buf203, (2048, 768), (768, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (2048, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), out=buf205)
        del arg136_1
        buf209 = reinterpret_tensor(buf204, (4, 512, 768), (393216, 768, 1), 0); del buf204  # reuse
        # Source Nodes: [hidden_states_89, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf192, buf205, arg137_1, arg138_1, arg139_1, buf209, 2048, 768, grid=grid(2048), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        buf210 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (2048, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf210)
        del arg140_1
        buf211 = reinterpret_tensor(buf192, (2048, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), out=buf211)
        del arg142_1
        buf212 = reinterpret_tensor(buf30, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf211, arg143_1, buf212, 1572864, grid=grid(1572864), stream=stream0)
        del arg143_1
        buf213 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 768), (1, 768), 0), out=buf213)
        del arg144_1
        buf214 = reinterpret_tensor(buf6, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf213, arg145_1, buf214, 1572864, grid=grid(1572864), stream=stream0)
        del arg145_1
        buf215 = reinterpret_tensor(buf213, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf210, arg141_1, buf215, 1572864, grid=grid(1572864), stream=stream0)
        del arg141_1
        # Source Nodes: [], Original ATen: []
        buf216 = aten._scaled_dot_product_efficient_attention(buf215, reinterpret_tensor(buf212, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf214, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf217 = buf216[0]
        del buf216
        buf221 = reinterpret_tensor(buf215, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf217, buf221, 1572864, grid=grid(1572864), stream=stream0)
        buf222 = reinterpret_tensor(buf217, (2048, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), out=buf222)
        del arg146_1
        buf226 = reinterpret_tensor(buf221, (4, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        # Source Nodes: [hidden_states_93, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf209, buf222, arg147_1, arg148_1, arg149_1, buf226, 2048, 768, grid=grid(2048), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        buf227 = reinterpret_tensor(buf187, (2048, 3072), (3072, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (2048, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 3072), (1, 768), 0), out=buf227)
        del arg150_1
        buf228 = reinterpret_tensor(buf227, (4, 512, 3072), (1572864, 3072, 1), 0); del buf227  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf228, arg151_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg151_1
        buf229 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg152_1, (3072, 768), (1, 3072), 0), out=buf229)
        del arg152_1
        buf233 = buf209; del buf209  # reuse
        # Source Nodes: [hidden_states_99, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf226, buf229, arg153_1, arg154_1, arg155_1, buf233, 2048, 768, grid=grid(2048), stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        buf234 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), out=buf234)
        del arg156_1
        buf235 = reinterpret_tensor(buf226, (2048, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf235)
        del arg158_1
        buf236 = reinterpret_tensor(buf210, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf210  # reuse
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf235, arg159_1, buf236, 1572864, grid=grid(1572864), stream=stream0)
        del arg159_1
        buf237 = reinterpret_tensor(buf235, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf234, arg157_1, buf237, 1572864, grid=grid(1572864), stream=stream0)
        del arg157_1
        buf238 = buf202; del buf202  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf236, (48, 64, 512), (32768, 1, 64), 0), out=buf238)
        buf243 = buf197; del buf197  # reuse
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf238, buf243, 24576, 512, grid=grid(24576), stream=stream0)
        buf241 = reinterpret_tensor(buf237, (2048, 768), (768, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), out=buf241)
        del arg160_1
        buf242 = reinterpret_tensor(buf234, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf241, arg161_1, buf242, 1572864, grid=grid(1572864), stream=stream0)
        del arg161_1
        buf244 = reinterpret_tensor(buf241, (48, 512, 64), (32768, 64, 1), 0); del buf241  # reuse
        # Source Nodes: [attn_output_50, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf243, reinterpret_tensor(buf242, (48, 512, 64), (32768, 64, 1), 0), out=buf244)
        buf245 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf244, buf245, 1572864, grid=grid(1572864), stream=stream0)
        buf246 = reinterpret_tensor(buf244, (2048, 768), (768, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (2048, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), out=buf246)
        del arg162_1
        buf250 = reinterpret_tensor(buf245, (4, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        # Source Nodes: [hidden_states_104, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf233, buf246, arg163_1, arg164_1, arg165_1, buf250, 2048, 768, grid=grid(2048), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        buf251 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (2048, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), out=buf251)
        del arg166_1
        buf252 = reinterpret_tensor(buf233, (2048, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), out=buf252)
        del arg168_1
        buf253 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf252, arg169_1, buf253, 1572864, grid=grid(1572864), stream=stream0)
        del arg169_1
        buf254 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), out=buf254)
        del arg170_1
        buf255 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf254, arg171_1, buf255, 1572864, grid=grid(1572864), stream=stream0)
        del arg171_1
        buf256 = reinterpret_tensor(buf254, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf251, arg167_1, buf256, 1572864, grid=grid(1572864), stream=stream0)
        del arg167_1
        # Source Nodes: [], Original ATen: []
        buf257 = aten._scaled_dot_product_efficient_attention(buf256, reinterpret_tensor(buf253, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf255, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf258 = buf257[0]
        del buf257
        buf262 = reinterpret_tensor(buf256, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf256  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf258, buf262, 1572864, grid=grid(1572864), stream=stream0)
        buf263 = reinterpret_tensor(buf258, (2048, 768), (768, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (2048, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf263)
        del arg172_1
        buf267 = reinterpret_tensor(buf262, (4, 512, 768), (393216, 768, 1), 0); del buf262  # reuse
        # Source Nodes: [hidden_states_108, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf250, buf263, arg173_1, arg174_1, arg175_1, buf267, 2048, 768, grid=grid(2048), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        buf268 = reinterpret_tensor(buf228, (2048, 3072), (3072, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (2048, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 3072), (1, 768), 0), out=buf268)
        del arg176_1
        buf269 = reinterpret_tensor(buf268, (4, 512, 3072), (1572864, 3072, 1), 0); del buf268  # reuse
        # Source Nodes: [hidden_states_110], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf269, arg177_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg177_1
        buf270 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), out=buf270)
        del arg178_1
        buf274 = buf250; del buf250  # reuse
        # Source Nodes: [hidden_states_114, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf267, buf270, arg179_1, arg180_1, arg181_1, buf274, 2048, 768, grid=grid(2048), stream=stream0)
        del arg179_1
        del arg180_1
        del arg181_1
        buf275 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2048, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), out=buf275)
        del arg182_1
        buf276 = reinterpret_tensor(buf267, (2048, 768), (768, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2048, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), out=buf276)
        del arg184_1
        buf277 = reinterpret_tensor(buf251, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf276, arg185_1, buf277, 1572864, grid=grid(1572864), stream=stream0)
        del arg185_1
        buf278 = reinterpret_tensor(buf276, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf275, arg183_1, buf278, 1572864, grid=grid(1572864), stream=stream0)
        del arg183_1
        buf279 = buf243; del buf243  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf277, (48, 64, 512), (32768, 1, 64), 0), out=buf279)
        buf284 = buf238; del buf238  # reuse
        # Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf279, buf284, 24576, 512, grid=grid(24576), stream=stream0)
        buf282 = reinterpret_tensor(buf278, (2048, 768), (768, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2048, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), out=buf282)
        del arg186_1
        buf283 = reinterpret_tensor(buf275, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf275  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf282, arg187_1, buf283, 1572864, grid=grid(1572864), stream=stream0)
        del arg187_1
        buf285 = reinterpret_tensor(buf282, (48, 512, 64), (32768, 64, 1), 0); del buf282  # reuse
        # Source Nodes: [attn_output_60, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf284, reinterpret_tensor(buf283, (48, 512, 64), (32768, 64, 1), 0), out=buf285)
        buf286 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf285, buf286, 1572864, grid=grid(1572864), stream=stream0)
        buf287 = reinterpret_tensor(buf285, (2048, 768), (768, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (2048, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), out=buf287)
        del arg188_1
        buf291 = reinterpret_tensor(buf286, (4, 512, 768), (393216, 768, 1), 0); del buf286  # reuse
        # Source Nodes: [hidden_states_119, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf274, buf287, arg189_1, arg190_1, arg191_1, buf291, 2048, 768, grid=grid(2048), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        buf292 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (2048, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 768), (1, 768), 0), out=buf292)
        del arg192_1
        buf293 = reinterpret_tensor(buf274, (2048, 768), (768, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg194_1, (768, 768), (1, 768), 0), out=buf293)
        del arg194_1
        buf294 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf293, arg195_1, buf294, 1572864, grid=grid(1572864), stream=stream0)
        del arg195_1
        buf295 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 768), (1, 768), 0), out=buf295)
        del arg196_1
        buf296 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf295, arg197_1, buf296, 1572864, grid=grid(1572864), stream=stream0)
        del arg197_1
        buf297 = reinterpret_tensor(buf295, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf292, arg193_1, buf297, 1572864, grid=grid(1572864), stream=stream0)
        del arg193_1
        # Source Nodes: [], Original ATen: []
        buf298 = aten._scaled_dot_product_efficient_attention(buf297, reinterpret_tensor(buf294, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf296, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf299 = buf298[0]
        del buf298
        buf303 = reinterpret_tensor(buf297, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf297  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf299, buf303, 1572864, grid=grid(1572864), stream=stream0)
        buf304 = reinterpret_tensor(buf299, (2048, 768), (768, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (2048, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 768), (1, 768), 0), out=buf304)
        del arg198_1
        buf308 = reinterpret_tensor(buf303, (4, 512, 768), (393216, 768, 1), 0); del buf303  # reuse
        # Source Nodes: [hidden_states_123, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf291, buf304, arg199_1, arg200_1, arg201_1, buf308, 2048, 768, grid=grid(2048), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        buf309 = reinterpret_tensor(buf269, (2048, 3072), (3072, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (2048, 768), (768, 1), 0), reinterpret_tensor(arg202_1, (768, 3072), (1, 768), 0), out=buf309)
        del arg202_1
        buf310 = reinterpret_tensor(buf309, (4, 512, 3072), (1572864, 3072, 1), 0); del buf309  # reuse
        # Source Nodes: [hidden_states_125], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf310, arg203_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg203_1
        buf311 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg204_1, (3072, 768), (1, 3072), 0), out=buf311)
        del arg204_1
        buf315 = buf291; del buf291  # reuse
        # Source Nodes: [hidden_states_129, residual_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf308, buf311, arg205_1, arg206_1, arg207_1, buf315, 2048, 768, grid=grid(2048), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        buf316 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (2048, 768), (768, 1), 0), reinterpret_tensor(arg208_1, (768, 768), (1, 768), 0), out=buf316)
        del arg208_1
        buf317 = reinterpret_tensor(buf308, (2048, 768), (768, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (2048, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 768), (1, 768), 0), out=buf317)
        del arg210_1
        buf318 = reinterpret_tensor(buf292, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf292  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf317, arg211_1, buf318, 1572864, grid=grid(1572864), stream=stream0)
        del arg211_1
        buf319 = reinterpret_tensor(buf317, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf316, arg209_1, buf319, 1572864, grid=grid(1572864), stream=stream0)
        del arg209_1
        buf320 = buf284; del buf284  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf318, (48, 64, 512), (32768, 1, 64), 0), out=buf320)
        buf325 = buf279; del buf279  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf320, buf325, 24576, 512, grid=grid(24576), stream=stream0)
        buf323 = reinterpret_tensor(buf319, (2048, 768), (768, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (2048, 768), (768, 1), 0), reinterpret_tensor(arg212_1, (768, 768), (1, 768), 0), out=buf323)
        del arg212_1
        buf324 = reinterpret_tensor(buf316, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf323, arg213_1, buf324, 1572864, grid=grid(1572864), stream=stream0)
        del arg213_1
        buf326 = reinterpret_tensor(buf323, (48, 512, 64), (32768, 64, 1), 0); del buf323  # reuse
        # Source Nodes: [attn_output_70, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf325, reinterpret_tensor(buf324, (48, 512, 64), (32768, 64, 1), 0), out=buf326)
        buf327 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf326, buf327, 1572864, grid=grid(1572864), stream=stream0)
        buf328 = reinterpret_tensor(buf326, (2048, 768), (768, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (2048, 768), (768, 1), 0), reinterpret_tensor(arg214_1, (768, 768), (1, 768), 0), out=buf328)
        del arg214_1
        buf332 = reinterpret_tensor(buf327, (4, 512, 768), (393216, 768, 1), 0); del buf327  # reuse
        # Source Nodes: [hidden_states_134, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf315, buf328, arg215_1, arg216_1, arg217_1, buf332, 2048, 768, grid=grid(2048), stream=stream0)
        del arg215_1
        del arg216_1
        del arg217_1
        buf333 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (2048, 768), (768, 1), 0), reinterpret_tensor(arg218_1, (768, 768), (1, 768), 0), out=buf333)
        del arg218_1
        buf334 = reinterpret_tensor(buf315, (2048, 768), (768, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg220_1, (768, 768), (1, 768), 0), out=buf334)
        del arg220_1
        buf335 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf334, arg221_1, buf335, 1572864, grid=grid(1572864), stream=stream0)
        del arg221_1
        buf336 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 768), (1, 768), 0), out=buf336)
        del arg222_1
        buf337 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf336, arg223_1, buf337, 1572864, grid=grid(1572864), stream=stream0)
        del arg223_1
        buf338 = reinterpret_tensor(buf336, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf333, arg219_1, buf338, 1572864, grid=grid(1572864), stream=stream0)
        del arg219_1
        # Source Nodes: [], Original ATen: []
        buf339 = aten._scaled_dot_product_efficient_attention(buf338, reinterpret_tensor(buf335, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf337, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf340 = buf339[0]
        del buf339
        buf344 = reinterpret_tensor(buf338, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf338  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf340, buf344, 1572864, grid=grid(1572864), stream=stream0)
        buf345 = reinterpret_tensor(buf340, (2048, 768), (768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (2048, 768), (768, 1), 0), reinterpret_tensor(arg224_1, (768, 768), (1, 768), 0), out=buf345)
        del arg224_1
        buf349 = reinterpret_tensor(buf344, (4, 512, 768), (393216, 768, 1), 0); del buf344  # reuse
        # Source Nodes: [hidden_states_138, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf332, buf345, arg225_1, arg226_1, arg227_1, buf349, 2048, 768, grid=grid(2048), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        buf350 = reinterpret_tensor(buf310, (2048, 3072), (3072, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (2048, 768), (768, 1), 0), reinterpret_tensor(arg228_1, (768, 3072), (1, 768), 0), out=buf350)
        del arg228_1
        buf351 = reinterpret_tensor(buf350, (4, 512, 3072), (1572864, 3072, 1), 0); del buf350  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf351, arg229_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg229_1
        buf352 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg230_1, (3072, 768), (1, 3072), 0), out=buf352)
        del arg230_1
        buf356 = buf332; del buf332  # reuse
        # Source Nodes: [hidden_states_144, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf349, buf352, arg231_1, arg232_1, arg233_1, buf356, 2048, 768, grid=grid(2048), stream=stream0)
        del arg231_1
        del arg232_1
        del arg233_1
        buf357 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (2048, 768), (768, 1), 0), reinterpret_tensor(arg234_1, (768, 768), (1, 768), 0), out=buf357)
        del arg234_1
        buf358 = reinterpret_tensor(buf349, (2048, 768), (768, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (2048, 768), (768, 1), 0), reinterpret_tensor(arg236_1, (768, 768), (1, 768), 0), out=buf358)
        del arg236_1
        buf359 = reinterpret_tensor(buf333, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf358, arg237_1, buf359, 1572864, grid=grid(1572864), stream=stream0)
        del arg237_1
        buf360 = reinterpret_tensor(buf358, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf358  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf357, arg235_1, buf360, 1572864, grid=grid(1572864), stream=stream0)
        del arg235_1
        buf361 = buf325; del buf325  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf359, (48, 64, 512), (32768, 1, 64), 0), out=buf361)
        buf366 = buf320; del buf320  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf361, buf366, 24576, 512, grid=grid(24576), stream=stream0)
        del buf361
        buf364 = reinterpret_tensor(buf360, (2048, 768), (768, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (2048, 768), (768, 1), 0), reinterpret_tensor(arg238_1, (768, 768), (1, 768), 0), out=buf364)
        del arg238_1
        buf365 = reinterpret_tensor(buf357, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf357  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf364, arg239_1, buf365, 1572864, grid=grid(1572864), stream=stream0)
        del arg239_1
        buf367 = reinterpret_tensor(buf364, (48, 512, 64), (32768, 64, 1), 0); del buf364  # reuse
        # Source Nodes: [attn_output_80, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf366, reinterpret_tensor(buf365, (48, 512, 64), (32768, 64, 1), 0), out=buf367)
        del buf366
        buf368 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf367, buf368, 1572864, grid=grid(1572864), stream=stream0)
        buf369 = reinterpret_tensor(buf367, (2048, 768), (768, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf368, (2048, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 768), (1, 768), 0), out=buf369)
        del arg240_1
        buf373 = reinterpret_tensor(buf368, (4, 512, 768), (393216, 768, 1), 0); del buf368  # reuse
        # Source Nodes: [hidden_states_149, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf356, buf369, arg241_1, arg242_1, arg243_1, buf373, 2048, 768, grid=grid(2048), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        buf374 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (2048, 768), (768, 1), 0), reinterpret_tensor(arg244_1, (768, 768), (1, 768), 0), out=buf374)
        del arg244_1
        buf375 = reinterpret_tensor(buf356, (2048, 768), (768, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg246_1, (768, 768), (1, 768), 0), out=buf375)
        del arg246_1
        buf376 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf375, arg247_1, buf376, 1572864, grid=grid(1572864), stream=stream0)
        del arg247_1
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 768), (1, 768), 0), out=buf377)
        del arg248_1
        buf378 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf377, arg249_1, buf378, 1572864, grid=grid(1572864), stream=stream0)
        del arg249_1
        buf379 = reinterpret_tensor(buf377, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf374, arg245_1, buf379, 1572864, grid=grid(1572864), stream=stream0)
        del arg245_1
        del buf374
        # Source Nodes: [], Original ATen: []
        buf380 = aten._scaled_dot_product_efficient_attention(buf379, reinterpret_tensor(buf376, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), reinterpret_tensor(buf378, (1, 48, 512, 64), (1572864, 32768, 64, 1), 0), None, True, scale=1.0)
        buf381 = buf380[0]
        del buf380
        buf385 = reinterpret_tensor(buf379, (4, 512, 12, 64), (393216, 768, 64, 1), 0); del buf379  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf381, buf385, 1572864, grid=grid(1572864), stream=stream0)
        buf386 = reinterpret_tensor(buf381, (2048, 768), (768, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (2048, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 768), (1, 768), 0), out=buf386)
        del arg250_1
        buf390 = reinterpret_tensor(buf385, (4, 512, 768), (393216, 768, 1), 0); del buf385  # reuse
        # Source Nodes: [hidden_states_153, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf373, buf386, arg251_1, arg252_1, arg253_1, buf390, 2048, 768, grid=grid(2048), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        buf391 = reinterpret_tensor(buf351, (2048, 3072), (3072, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf390, (2048, 768), (768, 1), 0), reinterpret_tensor(arg254_1, (768, 3072), (1, 768), 0), out=buf391)
        del arg254_1
        buf392 = reinterpret_tensor(buf391, (4, 512, 3072), (1572864, 3072, 1), 0); del buf391  # reuse
        # Source Nodes: [hidden_states_155], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf392, arg255_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg255_1
        buf393 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg256_1, (3072, 768), (1, 3072), 0), out=buf393)
        del arg256_1
        del buf392
        buf397 = buf373; del buf373  # reuse
        # Source Nodes: [hidden_states_159, hidden_states_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf390, buf393, arg257_1, arg258_1, arg259_1, buf397, 2048, 768, grid=grid(2048), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del buf390
        del buf393
        buf398 = empty_strided((768, 50268), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(arg260_1, buf398, 38605824, grid=grid(38605824), stream=stream0)
        del arg260_1
        buf399 = empty((2048, 50268), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (2048, 768), (768, 1), 0), buf398, out=buf399)
        del buf397
        del buf398
        buf400 = empty((4, 512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits_1], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf399, arg261_1, buf400, 102942720, grid=grid(102942720), stream=stream0)
        del arg261_1
        return (buf400, buf153, buf159, buf171, buf173, buf195, buf201, buf212, buf214, buf236, buf242, buf253, buf255, buf277, buf283, buf294, buf296, buf318, buf324, buf335, buf337, buf359, buf365, buf376, buf378, buf169, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bart', benchmark_compiled_module)
