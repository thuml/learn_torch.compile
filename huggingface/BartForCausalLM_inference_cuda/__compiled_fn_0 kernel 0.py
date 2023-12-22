
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cghpx7275xpyln7w3xlimjx6gj7z6kqfzztaeh6vax5yeezxxotx.py
# Source Nodes: [add_1, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_decoder_embed_tokens, positions_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add_1 => add_1
# hidden_states => add_2
# hidden_states_1 => add_3, add_4, mul_1, mul_2, rsqrt, sub, var_mean
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
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
# attn_weights_3 => amax, div, exp, sub_1, sum_1
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
# Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_5 => add_6
# residual_1 => add_7, add_8, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
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
# Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# hidden_states_7 => add_9, erf, mul_6, mul_7, mul_8
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


# kernel path: /tmp/torchinductor_youkaichao/57/c57jw26o6hol352gx7vmgqhvytx3z4y2tocyblnzfqbnjrypgujp.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_12, exp_12, sub_37, sum_13
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkfpl7vo5o5ecsaspngsxwucg4kcbfikdh3gxb7gliwvljfsbzt.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_12, full_default_3, ne_1, ne_2, neg, sum_14, sum_15, where_2
triton_per_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg1_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg7_1, (1024, ), (1, ))
    assert_size_stride(arg8_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg17_1, (1024, ), (1, ))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg23_1, (1024, ), (1, ))
    assert_size_stride(arg24_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg31_1, (4096, ), (1, ))
    assert_size_stride(arg32_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg47_1, (4096, ), (1, ))
    assert_size_stride(arg48_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg63_1, (4096, ), (1, ))
    assert_size_stride(arg64_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg73_1, (1024, ), (1, ))
    assert_size_stride(arg74_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg79_1, (4096, ), (1, ))
    assert_size_stride(arg80_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg95_1, (4096, ), (1, ))
    assert_size_stride(arg96_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg111_1, (4096, ), (1, ))
    assert_size_stride(arg112_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg127_1, (4096, ), (1, ))
    assert_size_stride(arg128_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg143_1, (4096, ), (1, ))
    assert_size_stride(arg144_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg159_1, (4096, ), (1, ))
    assert_size_stride(arg160_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg175_1, (4096, ), (1, ))
    assert_size_stride(arg176_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg191_1, (4096, ), (1, ))
    assert_size_stride(arg192_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg197_1, (1, 1024), (1024, 1))
    assert_size_stride(arg198_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_1, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_decoder_embed_tokens, positions_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg197_1, arg1_1, arg0_1, arg2_1, arg3_1, buf3, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg197_1
        del arg1_1
        del arg2_1
        del arg3_1
        buf4 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg4_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg4_1
        buf5 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg6_1
        buf6 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, arg7_1, buf6, 1048576, grid=grid(1048576), stream=stream0)
        del arg7_1
        buf7 = reinterpret_tensor(buf5, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, arg5_1, buf7, 1048576, grid=grid(1048576), stream=stream0)
        del arg5_1
        buf8 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf6, (16, 64, 1024), (65536, 1, 64), 0), out=buf8)
        buf13 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf8, buf13, 16384, 1024, grid=grid(16384), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf11)
        del arg8_1
        buf12 = reinterpret_tensor(buf4, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg9_1, buf12, 1048576, grid=grid(1048576), stream=stream0)
        del arg9_1
        buf14 = reinterpret_tensor(buf11, (16, 1024, 64), (65536, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output, attn_weights_3], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (16, 1024, 64), (65536, 64, 1), 0), out=buf14)
        buf15 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 1048576, grid=grid(1048576), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (1024, 1024), (1024, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg10_1
        buf20 = reinterpret_tensor(buf15, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf3, buf16, arg11_1, arg12_1, arg13_1, buf20, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg11_1
        del arg12_1
        del arg13_1
        buf21 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg14_1, (1024, 4096), (1, 1024), 0), out=buf21)
        del arg14_1
        buf22 = reinterpret_tensor(buf21, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf22, arg15_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg15_1
        buf23 = reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 1024), (1, 4096), 0), out=buf23)
        del arg16_1
        buf27 = reinterpret_tensor(buf16, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf20, buf23, arg17_1, arg18_1, arg19_1, buf27, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg20_1, (1024, 1024), (1, 1024), 0), out=buf28)
        del arg20_1
        buf29 = reinterpret_tensor(buf20, (1024, 1024), (1024, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg22_1
        buf30 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf29, arg23_1, buf30, 1048576, grid=grid(1048576), stream=stream0)
        del arg23_1
        buf31 = reinterpret_tensor(buf29, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf28, arg21_1, buf31, 1048576, grid=grid(1048576), stream=stream0)
        del arg21_1
        buf32 = buf13; del buf13  # reuse
        # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf30, (16, 64, 1024), (65536, 1, 64), 0), out=buf32)
        buf37 = buf8; del buf8  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf32, buf37, 16384, 1024, grid=grid(16384), stream=stream0)
        buf35 = reinterpret_tensor(buf31, (1024, 1024), (1024, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf35)
        del arg24_1
        buf36 = reinterpret_tensor(buf28, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf35, arg25_1, buf36, 1048576, grid=grid(1048576), stream=stream0)
        del arg25_1
        buf38 = reinterpret_tensor(buf35, (16, 1024, 64), (65536, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_5, attn_weights_7], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf37, reinterpret_tensor(buf36, (16, 1024, 64), (65536, 64, 1), 0), out=buf38)
        buf39 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf38, buf39, 1048576, grid=grid(1048576), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (1024, 1024), (1024, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf40)
        del arg26_1
        buf44 = reinterpret_tensor(buf39, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_16, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf27, buf40, arg27_1, arg28_1, arg29_1, buf44, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        buf45 = reinterpret_tensor(buf22, (1024, 4096), (4096, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg30_1, (1024, 4096), (1, 1024), 0), out=buf45)
        del arg30_1
        buf46 = reinterpret_tensor(buf45, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf46, arg31_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg31_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg32_1, (4096, 1024), (1, 4096), 0), out=buf47)
        del arg32_1
        buf51 = buf27; del buf27  # reuse
        # Source Nodes: [hidden_states_22, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf44, buf47, arg33_1, arg34_1, arg35_1, buf51, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg36_1, (1024, 1024), (1, 1024), 0), out=buf52)
        del arg36_1
        buf53 = reinterpret_tensor(buf44, (1024, 1024), (1024, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), out=buf53)
        del arg38_1
        buf54 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf53, arg39_1, buf54, 1048576, grid=grid(1048576), stream=stream0)
        del arg39_1
        buf55 = reinterpret_tensor(buf53, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf52, arg37_1, buf55, 1048576, grid=grid(1048576), stream=stream0)
        del arg37_1
        buf56 = buf37; del buf37  # reuse
        # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf54, (16, 64, 1024), (65536, 1, 64), 0), out=buf56)
        buf61 = buf32; del buf32  # reuse
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf56, buf61, 16384, 1024, grid=grid(16384), stream=stream0)
        buf59 = reinterpret_tensor(buf55, (1024, 1024), (1024, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf59)
        del arg40_1
        buf60 = reinterpret_tensor(buf52, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf59, arg41_1, buf60, 1048576, grid=grid(1048576), stream=stream0)
        del arg41_1
        buf62 = reinterpret_tensor(buf59, (16, 1024, 64), (65536, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [attn_output_10, attn_weights_11], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf61, reinterpret_tensor(buf60, (16, 1024, 64), (65536, 64, 1), 0), out=buf62)
        buf63 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf62, buf63, 1048576, grid=grid(1048576), stream=stream0)
        buf64 = reinterpret_tensor(buf62, (1024, 1024), (1024, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf64)
        del arg42_1
        buf68 = reinterpret_tensor(buf63, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf63  # reuse
        # Source Nodes: [hidden_states_27, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf51, buf64, arg43_1, arg44_1, arg45_1, buf68, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg43_1
        del arg44_1
        del arg45_1
        buf69 = reinterpret_tensor(buf46, (1024, 4096), (4096, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg46_1, (1024, 4096), (1, 1024), 0), out=buf69)
        del arg46_1
        buf70 = reinterpret_tensor(buf69, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf69  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf70, arg47_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg47_1
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg48_1, (4096, 1024), (1, 4096), 0), out=buf71)
        del arg48_1
        buf75 = buf51; del buf51  # reuse
        # Source Nodes: [hidden_states_33, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf68, buf71, arg49_1, arg50_1, arg51_1, buf75, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg52_1, (1024, 1024), (1, 1024), 0), out=buf76)
        del arg52_1
        buf77 = reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), out=buf77)
        del arg54_1
        buf78 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf77, arg55_1, buf78, 1048576, grid=grid(1048576), stream=stream0)
        del arg55_1
        buf79 = reinterpret_tensor(buf77, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf76, arg53_1, buf79, 1048576, grid=grid(1048576), stream=stream0)
        del arg53_1
        buf80 = buf61; del buf61  # reuse
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf78, (16, 64, 1024), (65536, 1, 64), 0), out=buf80)
        buf85 = buf56; del buf56  # reuse
        # Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf80, buf85, 16384, 1024, grid=grid(16384), stream=stream0)
        buf83 = reinterpret_tensor(buf79, (1024, 1024), (1024, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf83)
        del arg56_1
        buf84 = reinterpret_tensor(buf76, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf83, arg57_1, buf84, 1048576, grid=grid(1048576), stream=stream0)
        del arg57_1
        buf86 = reinterpret_tensor(buf83, (16, 1024, 64), (65536, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_15, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf85, reinterpret_tensor(buf84, (16, 1024, 64), (65536, 64, 1), 0), out=buf86)
        buf87 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf86, buf87, 1048576, grid=grid(1048576), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (1024, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf88)
        del arg58_1
        buf92 = reinterpret_tensor(buf87, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_38, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf75, buf88, arg59_1, arg60_1, arg61_1, buf92, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        buf93 = reinterpret_tensor(buf70, (1024, 4096), (4096, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg62_1, (1024, 4096), (1, 1024), 0), out=buf93)
        del arg62_1
        buf94 = reinterpret_tensor(buf93, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf94, arg63_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg63_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg64_1, (4096, 1024), (1, 4096), 0), out=buf95)
        del arg64_1
        buf99 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_44, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf92, buf95, arg65_1, arg66_1, arg67_1, buf99, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg68_1, (1024, 1024), (1, 1024), 0), out=buf100)
        del arg68_1
        buf101 = reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), out=buf101)
        del arg70_1
        buf102 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf101, arg71_1, buf102, 1048576, grid=grid(1048576), stream=stream0)
        del arg71_1
        buf103 = reinterpret_tensor(buf101, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf100, arg69_1, buf103, 1048576, grid=grid(1048576), stream=stream0)
        del arg69_1
        buf104 = buf85; del buf85  # reuse
        # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf102, (16, 64, 1024), (65536, 1, 64), 0), out=buf104)
        buf109 = buf80; del buf80  # reuse
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf104, buf109, 16384, 1024, grid=grid(16384), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (1024, 1024), (1024, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf107)
        del arg72_1
        buf108 = reinterpret_tensor(buf100, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf107, arg73_1, buf108, 1048576, grid=grid(1048576), stream=stream0)
        del arg73_1
        buf110 = reinterpret_tensor(buf107, (16, 1024, 64), (65536, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [attn_output_20, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf109, reinterpret_tensor(buf108, (16, 1024, 64), (65536, 64, 1), 0), out=buf110)
        buf111 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf110, buf111, 1048576, grid=grid(1048576), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (1024, 1024), (1024, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf112)
        del arg74_1
        buf116 = reinterpret_tensor(buf111, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_49, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf99, buf112, arg75_1, arg76_1, arg77_1, buf116, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        buf117 = reinterpret_tensor(buf94, (1024, 4096), (4096, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 4096), (1, 1024), 0), out=buf117)
        del arg78_1
        buf118 = reinterpret_tensor(buf117, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf117  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf118, arg79_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg79_1
        buf119 = reinterpret_tensor(buf99, (1024, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg80_1, (4096, 1024), (1, 4096), 0), out=buf119)
        del arg80_1
        buf123 = reinterpret_tensor(buf112, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_55, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf116, buf119, arg81_1, arg82_1, arg83_1, buf123, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg84_1, (1024, 1024), (1, 1024), 0), out=buf124)
        del arg84_1
        buf125 = reinterpret_tensor(buf116, (1024, 1024), (1024, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), out=buf125)
        del arg86_1
        buf126 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf125, arg87_1, buf126, 1048576, grid=grid(1048576), stream=stream0)
        del arg87_1
        buf127 = reinterpret_tensor(buf125, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf124, arg85_1, buf127, 1048576, grid=grid(1048576), stream=stream0)
        del arg85_1
        buf128 = buf109; del buf109  # reuse
        # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf127, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf126, (16, 64, 1024), (65536, 1, 64), 0), out=buf128)
        buf133 = buf104; del buf104  # reuse
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf128, buf133, 16384, 1024, grid=grid(16384), stream=stream0)
        buf131 = reinterpret_tensor(buf127, (1024, 1024), (1024, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf131)
        del arg88_1
        buf132 = reinterpret_tensor(buf124, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf131, arg89_1, buf132, 1048576, grid=grid(1048576), stream=stream0)
        del arg89_1
        buf134 = reinterpret_tensor(buf131, (16, 1024, 64), (65536, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [attn_output_25, attn_weights_23], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf133, reinterpret_tensor(buf132, (16, 1024, 64), (65536, 64, 1), 0), out=buf134)
        buf135 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf134, buf135, 1048576, grid=grid(1048576), stream=stream0)
        buf136 = reinterpret_tensor(buf134, (1024, 1024), (1024, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg90_1
        buf140 = reinterpret_tensor(buf135, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_60, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf123, buf136, arg91_1, arg92_1, arg93_1, buf140, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        buf141 = reinterpret_tensor(buf118, (1024, 4096), (4096, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg94_1, (1024, 4096), (1, 1024), 0), out=buf141)
        del arg94_1
        buf142 = reinterpret_tensor(buf141, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf142, arg95_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg95_1
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg96_1, (4096, 1024), (1, 4096), 0), out=buf143)
        del arg96_1
        buf147 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf140, buf143, arg97_1, arg98_1, arg99_1, buf147, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg97_1
        del arg98_1
        del arg99_1
        buf148 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg100_1, (1024, 1024), (1, 1024), 0), out=buf148)
        del arg100_1
        buf149 = reinterpret_tensor(buf140, (1024, 1024), (1024, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), out=buf149)
        del arg102_1
        buf150 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf149, arg103_1, buf150, 1048576, grid=grid(1048576), stream=stream0)
        del arg103_1
        buf151 = reinterpret_tensor(buf149, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf148, arg101_1, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del arg101_1
        buf152 = buf133; del buf133  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf150, (16, 64, 1024), (65536, 1, 64), 0), out=buf152)
        buf157 = buf128; del buf128  # reuse
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf152, buf157, 16384, 1024, grid=grid(16384), stream=stream0)
        buf155 = reinterpret_tensor(buf151, (1024, 1024), (1024, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf155)
        del arg104_1
        buf156 = reinterpret_tensor(buf148, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg105_1, buf156, 1048576, grid=grid(1048576), stream=stream0)
        del arg105_1
        buf158 = reinterpret_tensor(buf155, (16, 1024, 64), (65536, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_output_30, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf157, reinterpret_tensor(buf156, (16, 1024, 64), (65536, 64, 1), 0), out=buf158)
        buf159 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf158, buf159, 1048576, grid=grid(1048576), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (1024, 1024), (1024, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf160)
        del arg106_1
        buf164 = reinterpret_tensor(buf159, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf159  # reuse
        # Source Nodes: [hidden_states_71, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf147, buf160, arg107_1, arg108_1, arg109_1, buf164, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        buf165 = reinterpret_tensor(buf142, (1024, 4096), (4096, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg110_1, (1024, 4096), (1, 1024), 0), out=buf165)
        del arg110_1
        buf166 = reinterpret_tensor(buf165, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf166, arg111_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg111_1
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg112_1, (4096, 1024), (1, 4096), 0), out=buf167)
        del arg112_1
        buf171 = buf147; del buf147  # reuse
        # Source Nodes: [hidden_states_77, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf164, buf167, arg113_1, arg114_1, arg115_1, buf171, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        buf172 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg116_1, (1024, 1024), (1, 1024), 0), out=buf172)
        del arg116_1
        buf173 = reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), out=buf173)
        del arg118_1
        buf174 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf173, arg119_1, buf174, 1048576, grid=grid(1048576), stream=stream0)
        del arg119_1
        buf175 = reinterpret_tensor(buf173, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf172, arg117_1, buf175, 1048576, grid=grid(1048576), stream=stream0)
        del arg117_1
        buf176 = buf157; del buf157  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf175, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf174, (16, 64, 1024), (65536, 1, 64), 0), out=buf176)
        buf181 = buf152; del buf152  # reuse
        # Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf176, buf181, 16384, 1024, grid=grid(16384), stream=stream0)
        buf179 = reinterpret_tensor(buf175, (1024, 1024), (1024, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf179)
        del arg120_1
        buf180 = reinterpret_tensor(buf172, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf179, arg121_1, buf180, 1048576, grid=grid(1048576), stream=stream0)
        del arg121_1
        buf182 = reinterpret_tensor(buf179, (16, 1024, 64), (65536, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_35, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf181, reinterpret_tensor(buf180, (16, 1024, 64), (65536, 64, 1), 0), out=buf182)
        buf183 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf182, buf183, 1048576, grid=grid(1048576), stream=stream0)
        buf184 = reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg122_1
        buf188 = reinterpret_tensor(buf183, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf183  # reuse
        # Source Nodes: [hidden_states_82, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf171, buf184, arg123_1, arg124_1, arg125_1, buf188, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg123_1
        del arg124_1
        del arg125_1
        buf189 = reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg126_1, (1024, 4096), (1, 1024), 0), out=buf189)
        del arg126_1
        buf190 = reinterpret_tensor(buf189, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf189  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf190, arg127_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg127_1
        buf191 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg128_1, (4096, 1024), (1, 4096), 0), out=buf191)
        del arg128_1
        buf195 = buf171; del buf171  # reuse
        # Source Nodes: [hidden_states_88, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf188, buf191, arg129_1, arg130_1, arg131_1, buf195, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        buf196 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg132_1, (1024, 1024), (1, 1024), 0), out=buf196)
        del arg132_1
        buf197 = reinterpret_tensor(buf188, (1024, 1024), (1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), out=buf197)
        del arg134_1
        buf198 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf197, arg135_1, buf198, 1048576, grid=grid(1048576), stream=stream0)
        del arg135_1
        buf199 = reinterpret_tensor(buf197, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf196, arg133_1, buf199, 1048576, grid=grid(1048576), stream=stream0)
        del arg133_1
        buf200 = buf181; del buf181  # reuse
        # Source Nodes: [attn_weights_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf198, (16, 64, 1024), (65536, 1, 64), 0), out=buf200)
        buf205 = buf176; del buf176  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf200, buf205, 16384, 1024, grid=grid(16384), stream=stream0)
        buf203 = reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf203)
        del arg136_1
        buf204 = reinterpret_tensor(buf196, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf203, arg137_1, buf204, 1048576, grid=grid(1048576), stream=stream0)
        del arg137_1
        buf206 = reinterpret_tensor(buf203, (16, 1024, 64), (65536, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [attn_output_40, attn_weights_35], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf205, reinterpret_tensor(buf204, (16, 1024, 64), (65536, 64, 1), 0), out=buf206)
        buf207 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf206, buf207, 1048576, grid=grid(1048576), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (1024, 1024), (1024, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg138_1
        buf212 = reinterpret_tensor(buf207, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf207  # reuse
        # Source Nodes: [hidden_states_93, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf195, buf208, arg139_1, arg140_1, arg141_1, buf212, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        buf213 = reinterpret_tensor(buf190, (1024, 4096), (4096, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg142_1, (1024, 4096), (1, 1024), 0), out=buf213)
        del arg142_1
        buf214 = reinterpret_tensor(buf213, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf213  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf214, arg143_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg143_1
        buf215 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg144_1, (4096, 1024), (1, 4096), 0), out=buf215)
        del arg144_1
        buf219 = buf195; del buf195  # reuse
        # Source Nodes: [hidden_states_99, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf212, buf215, arg145_1, arg146_1, arg147_1, buf219, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg145_1
        del arg146_1
        del arg147_1
        buf220 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 1024), (1, 1024), 0), out=buf220)
        del arg148_1
        buf221 = reinterpret_tensor(buf212, (1024, 1024), (1024, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), out=buf221)
        del arg150_1
        buf222 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf221, arg151_1, buf222, 1048576, grid=grid(1048576), stream=stream0)
        del arg151_1
        buf223 = reinterpret_tensor(buf221, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf221  # reuse
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf220, arg149_1, buf223, 1048576, grid=grid(1048576), stream=stream0)
        del arg149_1
        buf224 = buf205; del buf205  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf222, (16, 64, 1024), (65536, 1, 64), 0), out=buf224)
        buf229 = buf200; del buf200  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf224, buf229, 16384, 1024, grid=grid(16384), stream=stream0)
        buf227 = reinterpret_tensor(buf223, (1024, 1024), (1024, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf227)
        del arg152_1
        buf228 = reinterpret_tensor(buf220, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf220  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf227, arg153_1, buf228, 1048576, grid=grid(1048576), stream=stream0)
        del arg153_1
        buf230 = reinterpret_tensor(buf227, (16, 1024, 64), (65536, 64, 1), 0); del buf227  # reuse
        # Source Nodes: [attn_output_45, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf229, reinterpret_tensor(buf228, (16, 1024, 64), (65536, 64, 1), 0), out=buf230)
        buf231 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf230, buf231, 1048576, grid=grid(1048576), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (1024, 1024), (1024, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf232)
        del arg154_1
        buf236 = reinterpret_tensor(buf231, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf231  # reuse
        # Source Nodes: [hidden_states_104, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf219, buf232, arg155_1, arg156_1, arg157_1, buf236, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg155_1
        del arg156_1
        del arg157_1
        buf237 = reinterpret_tensor(buf214, (1024, 4096), (4096, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg158_1, (1024, 4096), (1, 1024), 0), out=buf237)
        del arg158_1
        buf238 = reinterpret_tensor(buf237, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf237  # reuse
        # Source Nodes: [hidden_states_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf238, arg159_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg159_1
        buf239 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg160_1, (4096, 1024), (1, 4096), 0), out=buf239)
        del arg160_1
        buf243 = buf219; del buf219  # reuse
        # Source Nodes: [hidden_states_110, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf236, buf239, arg161_1, arg162_1, arg163_1, buf243, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        buf244 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg164_1, (1024, 1024), (1, 1024), 0), out=buf244)
        del arg164_1
        buf245 = reinterpret_tensor(buf236, (1024, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg166_1
        buf246 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf245, arg167_1, buf246, 1048576, grid=grid(1048576), stream=stream0)
        del arg167_1
        buf247 = reinterpret_tensor(buf245, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf245  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf244, arg165_1, buf247, 1048576, grid=grid(1048576), stream=stream0)
        del arg165_1
        buf248 = buf229; del buf229  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf246, (16, 64, 1024), (65536, 1, 64), 0), out=buf248)
        buf253 = buf224; del buf224  # reuse
        # Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf248, buf253, 16384, 1024, grid=grid(16384), stream=stream0)
        buf251 = reinterpret_tensor(buf247, (1024, 1024), (1024, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf251)
        del arg168_1
        buf252 = reinterpret_tensor(buf244, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf244  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf251, arg169_1, buf252, 1048576, grid=grid(1048576), stream=stream0)
        del arg169_1
        buf254 = reinterpret_tensor(buf251, (16, 1024, 64), (65536, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [attn_output_50, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf253, reinterpret_tensor(buf252, (16, 1024, 64), (65536, 64, 1), 0), out=buf254)
        buf255 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf254, buf255, 1048576, grid=grid(1048576), stream=stream0)
        buf256 = reinterpret_tensor(buf254, (1024, 1024), (1024, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf256)
        del arg170_1
        buf260 = reinterpret_tensor(buf255, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf255  # reuse
        # Source Nodes: [hidden_states_115, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf243, buf256, arg171_1, arg172_1, arg173_1, buf260, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg171_1
        del arg172_1
        del arg173_1
        buf261 = reinterpret_tensor(buf238, (1024, 4096), (4096, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg174_1, (1024, 4096), (1, 1024), 0), out=buf261)
        del arg174_1
        buf262 = reinterpret_tensor(buf261, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf261  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf262, arg175_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg175_1
        buf263 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg176_1, (4096, 1024), (1, 4096), 0), out=buf263)
        del arg176_1
        buf267 = buf243; del buf243  # reuse
        # Source Nodes: [hidden_states_121, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf260, buf263, arg177_1, arg178_1, arg179_1, buf267, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        buf268 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg180_1, (1024, 1024), (1, 1024), 0), out=buf268)
        del arg180_1
        buf269 = reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), out=buf269)
        del arg182_1
        buf270 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf269, arg183_1, buf270, 1048576, grid=grid(1048576), stream=stream0)
        del arg183_1
        buf271 = reinterpret_tensor(buf269, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf268, arg181_1, buf271, 1048576, grid=grid(1048576), stream=stream0)
        del arg181_1
        buf272 = buf253; del buf253  # reuse
        # Source Nodes: [attn_weights_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf270, (16, 64, 1024), (65536, 1, 64), 0), out=buf272)
        buf277 = buf248; del buf248  # reuse
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf272, buf277, 16384, 1024, grid=grid(16384), stream=stream0)
        del buf272
        buf275 = reinterpret_tensor(buf271, (1024, 1024), (1024, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf275)
        del arg184_1
        buf276 = reinterpret_tensor(buf268, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf275, arg185_1, buf276, 1048576, grid=grid(1048576), stream=stream0)
        del arg185_1
        buf278 = reinterpret_tensor(buf275, (16, 1024, 64), (65536, 64, 1), 0); del buf275  # reuse
        # Source Nodes: [attn_output_55, attn_weights_47], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf277, reinterpret_tensor(buf276, (16, 1024, 64), (65536, 64, 1), 0), out=buf278)
        del buf277
        buf279 = empty((1, 1024, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf278, buf279, 1048576, grid=grid(1048576), stream=stream0)
        buf280 = reinterpret_tensor(buf278, (1024, 1024), (1024, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg186_1
        buf284 = reinterpret_tensor(buf279, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf279  # reuse
        # Source Nodes: [hidden_states_126, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf267, buf280, arg187_1, arg188_1, arg189_1, buf284, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        buf285 = reinterpret_tensor(buf262, (1024, 4096), (4096, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg190_1, (1024, 4096), (1, 1024), 0), out=buf285)
        del arg190_1
        buf286 = reinterpret_tensor(buf285, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf285  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf286, arg191_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg191_1
        buf287 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg192_1, (4096, 1024), (1, 4096), 0), out=buf287)
        del arg192_1
        del buf286
        buf291 = buf267; del buf267  # reuse
        # Source Nodes: [hidden_states_132, hidden_states_134], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf284, buf287, arg193_1, arg194_1, arg195_1, buf291, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del buf284
        del buf287
        buf292 = empty((1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg196_1, (1024, 50265), (1, 1024), 0), out=buf292)
        del arg196_1
        del buf291
        buf293 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf294 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf292, buf293, buf294, 1024, 50265, grid=grid(1024), stream=stream0)
        buf295 = empty((), device='cuda', dtype=torch.float32)
        buf297 = buf295; del buf295  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_8.run(buf297, arg198_1, buf292, buf293, buf294, 1, 1024, grid=grid(1), stream=stream0)
        del arg198_1
        return (buf297, reinterpret_tensor(buf292, (1, 1024, 50265), (51471360, 50265, 1), 0), buf6, buf12, buf30, buf36, buf54, buf60, buf78, buf84, buf102, buf108, buf126, buf132, buf150, buf156, buf174, buf180, buf198, buf204, buf222, buf228, buf246, buf252, buf270, buf276, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg198_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForCausalLM', benchmark_compiled_module)
