
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


# kernel path: /tmp/torchinductor_youkaichao/rf/crftrsjvzpjq6wghzeat3wy2s2uvcuvaoifdrbx4u2jxajygsssq.py
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
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
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


# kernel path: /tmp/torchinductor_youkaichao/52/c52n57knqh2pmvjyr2gfnsp4hhawwv2nje6nrbnbyoktepyn62ql.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cse6cvgqiv6yvqudhypycznj5i3j7ojupp6vytm2csrqkqvihpvh.py
# Source Nodes: [add_2, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_5
# attention_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 256.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyrrghvvd3vxiuvdpmthlj5u7ezm5rp42lf54w2axich2jgklfz.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5donquaot7gdtdf6rcopjusnhvulzfdmrjzmxvqinqtbv4prxn5.py
# Source Nodes: [hidden_states_111, hidden_states_112], Original ATen: [aten.gelu, aten.native_layer_norm]
# hidden_states_111 => add_100, erf_12, mul_87, mul_88, mul_89
# hidden_states_112 => add_101, add_102, mul_90, mul_91, rsqrt_25, sub_38, var_mean_25
triton_per_fused_gelu_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_4', 'mutated_arg_names': []}
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
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = 128.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3og5y7s77wchce6jv3fix2j5ucea5ms744a5rseaqud46ue4zzl.py
# Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
# lm_loss => amax_12, exp_12, sub_39, sum_13
triton_red_fused__log_softmax_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xx6t4wu7ipu6hupksjvgx2smm4kutzgyltgqimgiwpc4aktsdc.py
# Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
# lm_loss => convert_element_type, div_24, full_default_2, ne_1, ne_2, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_forward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 511
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (1 + r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 128), (128, 1))
    assert_size_stride(arg1_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (256, 128), (128, 1))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (256, 256), (256, 1))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, 256), (256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, 256), (256, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (1024, 256), (256, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (256, 1024), (1024, 1))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, 256), (256, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, 256), (256, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, 256), (256, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, 256), (256, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (1024, 256), (256, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (256, 1024), (1024, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 256), (256, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256), (256, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, 256), (256, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, 256), (256, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (1024, 256), (256, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (256, 1024), (1024, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, 256), (256, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, 256), (256, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, 256), (256, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, 256), (256, 1))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (1024, 256), (256, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (256, 1024), (1024, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, 256), (256, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, 256), (256, 1))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 256), (256, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, 256), (256, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (1024, 256), (256, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (256, 1024), (1024, 1))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, 256), (256, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256), (256, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 256), (256, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, 256), (256, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (1024, 256), (256, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (256, 1024), (1024, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, 256), (256, 1))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256), (256, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, 256), (256, 1))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (1024, 256), (256, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (256, 1024), (1024, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, 256), (256, 1))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, 256), (256, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256), (256, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, 256), (256, 1))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (1024, 256), (256, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (256, 1024), (1024, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, 256), (256, 1))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (256, 256), (256, 1))
    assert_size_stride(arg138_1, (256, ), (1, ))
    assert_size_stride(arg139_1, (256, 256), (256, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, 256), (256, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (1024, 256), (256, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (256, 1024), (1024, 1))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, 256), (256, 1))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, 256), (256, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, 256), (256, 1))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (256, 256), (256, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (1024, 256), (256, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (256, 1024), (1024, 1))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, 256), (256, 1))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, 256), (256, 1))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, 256), (256, 1))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, 256), (256, 1))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (1024, 256), (256, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (256, 1024), (1024, 1))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, 256), (256, 1))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, 256), (256, 1))
    assert_size_stride(arg186_1, (256, ), (1, ))
    assert_size_stride(arg187_1, (256, 256), (256, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, 256), (256, 1))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (1024, 256), (256, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (256, 1024), (1024, 1))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (128, 256), (256, 1))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (30522, 128), (128, 1))
    assert_size_stride(arg204_1, (30522, ), (1, ))
    assert_size_stride(arg205_1, (1, 512), (512, 1))
    assert_size_stride(arg206_1, (1, 512), (512, 1))
    assert_size_stride(arg207_1, (1, 512), (512, 1))
    assert_size_stride(arg208_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg208_1, arg0_1, arg205_1, arg1_1, arg206_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 512, 128, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg205_1
        del arg206_1
        del arg208_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf5 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (512, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg5_1
        del arg6_1
        buf6 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg7_1, (256, 256), (1, 256), 0), out=buf6)
        del arg7_1
        buf7 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg9_1, (256, 256), (1, 256), 0), out=buf7)
        del arg9_1
        buf8 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf8)
        del arg11_1
        buf9 = empty((1, 4, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg8_1, buf9, 131072, grid=grid(131072), stream=stream0)
        del arg8_1
        buf10 = reinterpret_tensor(buf6, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg10_1, buf10, 131072, grid=grid(131072), stream=stream0)
        del arg10_1
        buf11 = reinterpret_tensor(buf7, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf8, arg12_1, buf11, 131072, grid=grid(131072), stream=stream0)
        del arg12_1
        del buf8
        # Source Nodes: [], Original ATen: []
        buf12 = aten._scaled_dot_product_efficient_attention(buf9, buf10, buf11, None, False, scale=0.125)
        buf13 = buf12[0]
        del buf12
        buf17 = reinterpret_tensor(buf9, (512, 256), (256, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 256), (256, 1), 0), reinterpret_tensor(arg13_1, (256, 256), (1, 256), 0), out=buf17)
        del arg13_1
        buf21 = reinterpret_tensor(buf13, (1, 512, 256), (131072, 256, 1), 0); del buf13  # reuse
        # Source Nodes: [add_2, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf17, arg14_1, buf5, arg15_1, arg16_1, buf21, 512, 256, grid=grid(512), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf22 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 256), (256, 1), 0), reinterpret_tensor(arg17_1, (256, 1024), (1, 256), 0), out=buf22)
        del arg17_1
        buf23 = reinterpret_tensor(buf22, (1, 512, 1024), (524288, 1024, 1), 0); del buf22  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf23, arg18_1, 524288, grid=grid(524288), stream=stream0)
        del arg18_1
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg19_1, (1024, 256), (1, 1024), 0), out=buf24)
        del arg19_1
        buf28 = reinterpret_tensor(buf17, (1, 512, 256), (131072, 256, 1), 0); del buf17  # reuse
        # Source Nodes: [add_3, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf24, arg20_1, buf21, arg21_1, arg22_1, buf28, 512, 256, grid=grid(512), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        buf29 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 256), (1, 256), 0), out=buf29)
        del arg23_1
        buf30 = reinterpret_tensor(buf21, (512, 256), (256, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 256), (1, 256), 0), out=buf30)
        del arg25_1
        buf31 = reinterpret_tensor(buf11, (512, 256), (256, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 256), (256, 1), 0), reinterpret_tensor(arg27_1, (256, 256), (1, 256), 0), out=buf31)
        del arg27_1
        buf32 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf29, arg24_1, buf32, 131072, grid=grid(131072), stream=stream0)
        del arg24_1
        buf33 = reinterpret_tensor(buf29, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf30, arg26_1, buf33, 131072, grid=grid(131072), stream=stream0)
        del arg26_1
        buf34 = reinterpret_tensor(buf30, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf31, arg28_1, buf34, 131072, grid=grid(131072), stream=stream0)
        del arg28_1
        del buf31
        # Source Nodes: [], Original ATen: []
        buf35 = aten._scaled_dot_product_efficient_attention(buf32, buf33, buf34, None, False, scale=0.125)
        buf36 = buf35[0]
        del buf35
        buf40 = reinterpret_tensor(buf34, (512, 256), (256, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 256), (256, 1), 0), reinterpret_tensor(arg29_1, (256, 256), (1, 256), 0), out=buf40)
        del arg29_1
        buf44 = reinterpret_tensor(buf36, (1, 512, 256), (131072, 256, 1), 0); del buf36  # reuse
        # Source Nodes: [add_5, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf40, arg30_1, buf28, arg31_1, arg32_1, buf44, 512, 256, grid=grid(512), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        buf45 = reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 256), (256, 1), 0), reinterpret_tensor(arg33_1, (256, 1024), (1, 256), 0), out=buf45)
        del arg33_1
        buf46 = reinterpret_tensor(buf45, (1, 512, 1024), (524288, 1024, 1), 0); del buf45  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf46, arg34_1, 524288, grid=grid(524288), stream=stream0)
        del arg34_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg35_1, (1024, 256), (1, 1024), 0), out=buf47)
        del arg35_1
        buf51 = buf28; del buf28  # reuse
        # Source Nodes: [add_6, hidden_states_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf47, arg36_1, buf44, arg37_1, arg38_1, buf51, 512, 256, grid=grid(512), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        buf52 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 256), (256, 1), 0), reinterpret_tensor(arg39_1, (256, 256), (1, 256), 0), out=buf52)
        del arg39_1
        buf53 = reinterpret_tensor(buf44, (512, 256), (256, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 256), (1, 256), 0), out=buf53)
        del arg41_1
        buf54 = reinterpret_tensor(buf33, (512, 256), (256, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 256), (256, 1), 0), reinterpret_tensor(arg43_1, (256, 256), (1, 256), 0), out=buf54)
        del arg43_1
        buf55 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg40_1, buf55, 131072, grid=grid(131072), stream=stream0)
        del arg40_1
        buf56 = reinterpret_tensor(buf52, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf53, arg42_1, buf56, 131072, grid=grid(131072), stream=stream0)
        del arg42_1
        buf57 = reinterpret_tensor(buf53, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf54, arg44_1, buf57, 131072, grid=grid(131072), stream=stream0)
        del arg44_1
        del buf54
        # Source Nodes: [], Original ATen: []
        buf58 = aten._scaled_dot_product_efficient_attention(buf55, buf56, buf57, None, False, scale=0.125)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf57, (512, 256), (256, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 256), (256, 1), 0), reinterpret_tensor(arg45_1, (256, 256), (1, 256), 0), out=buf63)
        del arg45_1
        buf67 = reinterpret_tensor(buf59, (1, 512, 256), (131072, 256, 1), 0); del buf59  # reuse
        # Source Nodes: [add_8, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf63, arg46_1, buf51, arg47_1, arg48_1, buf67, 512, 256, grid=grid(512), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        buf68 = reinterpret_tensor(buf46, (512, 1024), (1024, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 1024), (1, 256), 0), out=buf68)
        del arg49_1
        buf69 = reinterpret_tensor(buf68, (1, 512, 1024), (524288, 1024, 1), 0); del buf68  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf69, arg50_1, 524288, grid=grid(524288), stream=stream0)
        del arg50_1
        buf70 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 256), (1, 1024), 0), out=buf70)
        del arg51_1
        buf74 = buf51; del buf51  # reuse
        # Source Nodes: [add_9, hidden_states_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf70, arg52_1, buf67, arg53_1, arg54_1, buf74, 512, 256, grid=grid(512), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        buf75 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 256), (256, 1), 0), reinterpret_tensor(arg55_1, (256, 256), (1, 256), 0), out=buf75)
        del arg55_1
        buf76 = reinterpret_tensor(buf67, (512, 256), (256, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 256), (1, 256), 0), out=buf76)
        del arg57_1
        buf77 = reinterpret_tensor(buf56, (512, 256), (256, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 256), (1, 256), 0), out=buf77)
        del arg59_1
        buf78 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf75, arg56_1, buf78, 131072, grid=grid(131072), stream=stream0)
        del arg56_1
        buf79 = reinterpret_tensor(buf75, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg58_1, buf79, 131072, grid=grid(131072), stream=stream0)
        del arg58_1
        buf80 = reinterpret_tensor(buf76, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf77, arg60_1, buf80, 131072, grid=grid(131072), stream=stream0)
        del arg60_1
        del buf77
        # Source Nodes: [], Original ATen: []
        buf81 = aten._scaled_dot_product_efficient_attention(buf78, buf79, buf80, None, False, scale=0.125)
        buf82 = buf81[0]
        del buf81
        buf86 = reinterpret_tensor(buf80, (512, 256), (256, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (512, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 256), (1, 256), 0), out=buf86)
        del arg61_1
        buf90 = reinterpret_tensor(buf82, (1, 512, 256), (131072, 256, 1), 0); del buf82  # reuse
        # Source Nodes: [add_11, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf86, arg62_1, buf74, arg63_1, arg64_1, buf90, 512, 256, grid=grid(512), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        buf91 = reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 256), (256, 1), 0), reinterpret_tensor(arg65_1, (256, 1024), (1, 256), 0), out=buf91)
        del arg65_1
        buf92 = reinterpret_tensor(buf91, (1, 512, 1024), (524288, 1024, 1), 0); del buf91  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf92, arg66_1, 524288, grid=grid(524288), stream=stream0)
        del arg66_1
        buf93 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg67_1, (1024, 256), (1, 1024), 0), out=buf93)
        del arg67_1
        buf97 = buf74; del buf74  # reuse
        # Source Nodes: [add_12, hidden_states_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf93, arg68_1, buf90, arg69_1, arg70_1, buf97, 512, 256, grid=grid(512), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        buf98 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), out=buf98)
        del arg71_1
        buf99 = reinterpret_tensor(buf90, (512, 256), (256, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 256), (1, 256), 0), out=buf99)
        del arg73_1
        buf100 = reinterpret_tensor(buf79, (512, 256), (256, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 256), (1, 256), 0), out=buf100)
        del arg75_1
        buf101 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf98, arg72_1, buf101, 131072, grid=grid(131072), stream=stream0)
        del arg72_1
        buf102 = reinterpret_tensor(buf98, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf99, arg74_1, buf102, 131072, grid=grid(131072), stream=stream0)
        del arg74_1
        buf103 = reinterpret_tensor(buf99, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg76_1, buf103, 131072, grid=grid(131072), stream=stream0)
        del arg76_1
        del buf100
        # Source Nodes: [], Original ATen: []
        buf104 = aten._scaled_dot_product_efficient_attention(buf101, buf102, buf103, None, False, scale=0.125)
        buf105 = buf104[0]
        del buf104
        buf109 = reinterpret_tensor(buf103, (512, 256), (256, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 256), (1, 256), 0), out=buf109)
        del arg77_1
        buf113 = reinterpret_tensor(buf105, (1, 512, 256), (131072, 256, 1), 0); del buf105  # reuse
        # Source Nodes: [add_14, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf109, arg78_1, buf97, arg79_1, arg80_1, buf113, 512, 256, grid=grid(512), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        buf114 = reinterpret_tensor(buf92, (512, 1024), (1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (512, 256), (256, 1), 0), reinterpret_tensor(arg81_1, (256, 1024), (1, 256), 0), out=buf114)
        del arg81_1
        buf115 = reinterpret_tensor(buf114, (1, 512, 1024), (524288, 1024, 1), 0); del buf114  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf115, arg82_1, 524288, grid=grid(524288), stream=stream0)
        del arg82_1
        buf116 = reinterpret_tensor(buf97, (512, 256), (256, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg83_1, (1024, 256), (1, 1024), 0), out=buf116)
        del arg83_1
        buf120 = reinterpret_tensor(buf109, (1, 512, 256), (131072, 256, 1), 0); del buf109  # reuse
        # Source Nodes: [add_15, hidden_states_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf116, arg84_1, buf113, arg85_1, arg86_1, buf120, 512, 256, grid=grid(512), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        buf121 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 256), (256, 1), 0), reinterpret_tensor(arg87_1, (256, 256), (1, 256), 0), out=buf121)
        del arg87_1
        buf122 = reinterpret_tensor(buf113, (512, 256), (256, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 256), (1, 256), 0), out=buf122)
        del arg89_1
        buf123 = reinterpret_tensor(buf102, (512, 256), (256, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (512, 256), (256, 1), 0), reinterpret_tensor(arg91_1, (256, 256), (1, 256), 0), out=buf123)
        del arg91_1
        buf124 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf121, arg88_1, buf124, 131072, grid=grid(131072), stream=stream0)
        del arg88_1
        buf125 = reinterpret_tensor(buf121, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf122, arg90_1, buf125, 131072, grid=grid(131072), stream=stream0)
        del arg90_1
        buf126 = reinterpret_tensor(buf122, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf123, arg92_1, buf126, 131072, grid=grid(131072), stream=stream0)
        del arg92_1
        del buf123
        # Source Nodes: [], Original ATen: []
        buf127 = aten._scaled_dot_product_efficient_attention(buf124, buf125, buf126, None, False, scale=0.125)
        buf128 = buf127[0]
        del buf127
        buf132 = reinterpret_tensor(buf126, (512, 256), (256, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 256), (256, 1), 0), reinterpret_tensor(arg93_1, (256, 256), (1, 256), 0), out=buf132)
        del arg93_1
        buf136 = reinterpret_tensor(buf128, (1, 512, 256), (131072, 256, 1), 0); del buf128  # reuse
        # Source Nodes: [add_17, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf132, arg94_1, buf120, arg95_1, arg96_1, buf136, 512, 256, grid=grid(512), stream=stream0)
        del arg94_1
        del arg95_1
        del arg96_1
        buf137 = reinterpret_tensor(buf115, (512, 1024), (1024, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (512, 256), (256, 1), 0), reinterpret_tensor(arg97_1, (256, 1024), (1, 256), 0), out=buf137)
        del arg97_1
        buf138 = reinterpret_tensor(buf137, (1, 512, 1024), (524288, 1024, 1), 0); del buf137  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf138, arg98_1, 524288, grid=grid(524288), stream=stream0)
        del arg98_1
        buf139 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 256), (1, 1024), 0), out=buf139)
        del arg99_1
        buf143 = buf120; del buf120  # reuse
        # Source Nodes: [add_18, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf139, arg100_1, buf136, arg101_1, arg102_1, buf143, 512, 256, grid=grid(512), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        buf144 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 256), (256, 1), 0), reinterpret_tensor(arg103_1, (256, 256), (1, 256), 0), out=buf144)
        del arg103_1
        buf145 = reinterpret_tensor(buf136, (512, 256), (256, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 256), (256, 1), 0), reinterpret_tensor(arg105_1, (256, 256), (1, 256), 0), out=buf145)
        del arg105_1
        buf146 = reinterpret_tensor(buf125, (512, 256), (256, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 256), (1, 256), 0), out=buf146)
        del arg107_1
        buf147 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf144, arg104_1, buf147, 131072, grid=grid(131072), stream=stream0)
        del arg104_1
        buf148 = reinterpret_tensor(buf144, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf145, arg106_1, buf148, 131072, grid=grid(131072), stream=stream0)
        del arg106_1
        buf149 = reinterpret_tensor(buf145, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf146, arg108_1, buf149, 131072, grid=grid(131072), stream=stream0)
        del arg108_1
        del buf146
        # Source Nodes: [], Original ATen: []
        buf150 = aten._scaled_dot_product_efficient_attention(buf147, buf148, buf149, None, False, scale=0.125)
        buf151 = buf150[0]
        del buf150
        buf155 = reinterpret_tensor(buf149, (512, 256), (256, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (512, 256), (256, 1), 0), reinterpret_tensor(arg109_1, (256, 256), (1, 256), 0), out=buf155)
        del arg109_1
        buf159 = reinterpret_tensor(buf151, (1, 512, 256), (131072, 256, 1), 0); del buf151  # reuse
        # Source Nodes: [add_20, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf155, arg110_1, buf143, arg111_1, arg112_1, buf159, 512, 256, grid=grid(512), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        buf160 = reinterpret_tensor(buf138, (512, 1024), (1024, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 256), (256, 1), 0), reinterpret_tensor(arg113_1, (256, 1024), (1, 256), 0), out=buf160)
        del arg113_1
        buf161 = reinterpret_tensor(buf160, (1, 512, 1024), (524288, 1024, 1), 0); del buf160  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf161, arg114_1, 524288, grid=grid(524288), stream=stream0)
        del arg114_1
        buf162 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg115_1, (1024, 256), (1, 1024), 0), out=buf162)
        del arg115_1
        buf166 = buf143; del buf143  # reuse
        # Source Nodes: [add_21, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf162, arg116_1, buf159, arg117_1, arg118_1, buf166, 512, 256, grid=grid(512), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        buf167 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 256), (1, 256), 0), out=buf167)
        del arg119_1
        buf168 = reinterpret_tensor(buf159, (512, 256), (256, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 256), (256, 1), 0), reinterpret_tensor(arg121_1, (256, 256), (1, 256), 0), out=buf168)
        del arg121_1
        buf169 = reinterpret_tensor(buf148, (512, 256), (256, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 256), (256, 1), 0), reinterpret_tensor(arg123_1, (256, 256), (1, 256), 0), out=buf169)
        del arg123_1
        buf170 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf167, arg120_1, buf170, 131072, grid=grid(131072), stream=stream0)
        del arg120_1
        buf171 = reinterpret_tensor(buf167, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf168, arg122_1, buf171, 131072, grid=grid(131072), stream=stream0)
        del arg122_1
        buf172 = reinterpret_tensor(buf168, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf169, arg124_1, buf172, 131072, grid=grid(131072), stream=stream0)
        del arg124_1
        del buf169
        # Source Nodes: [], Original ATen: []
        buf173 = aten._scaled_dot_product_efficient_attention(buf170, buf171, buf172, None, False, scale=0.125)
        buf174 = buf173[0]
        del buf173
        buf178 = reinterpret_tensor(buf172, (512, 256), (256, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (512, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 256), (1, 256), 0), out=buf178)
        del arg125_1
        buf182 = reinterpret_tensor(buf174, (1, 512, 256), (131072, 256, 1), 0); del buf174  # reuse
        # Source Nodes: [add_23, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf178, arg126_1, buf166, arg127_1, arg128_1, buf182, 512, 256, grid=grid(512), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        buf183 = reinterpret_tensor(buf161, (512, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 256), (256, 1), 0), reinterpret_tensor(arg129_1, (256, 1024), (1, 256), 0), out=buf183)
        del arg129_1
        buf184 = reinterpret_tensor(buf183, (1, 512, 1024), (524288, 1024, 1), 0); del buf183  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf184, arg130_1, 524288, grid=grid(524288), stream=stream0)
        del arg130_1
        buf185 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 256), (1, 1024), 0), out=buf185)
        del arg131_1
        buf189 = buf166; del buf166  # reuse
        # Source Nodes: [add_24, hidden_states_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf185, arg132_1, buf182, arg133_1, arg134_1, buf189, 512, 256, grid=grid(512), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        buf190 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 256), (256, 1), 0), reinterpret_tensor(arg135_1, (256, 256), (1, 256), 0), out=buf190)
        del arg135_1
        buf191 = reinterpret_tensor(buf182, (512, 256), (256, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 256), (1, 256), 0), out=buf191)
        del arg137_1
        buf192 = reinterpret_tensor(buf171, (512, 256), (256, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 256), (256, 1), 0), reinterpret_tensor(arg139_1, (256, 256), (1, 256), 0), out=buf192)
        del arg139_1
        buf193 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf190, arg136_1, buf193, 131072, grid=grid(131072), stream=stream0)
        del arg136_1
        buf194 = reinterpret_tensor(buf190, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf191, arg138_1, buf194, 131072, grid=grid(131072), stream=stream0)
        del arg138_1
        buf195 = reinterpret_tensor(buf191, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf192, arg140_1, buf195, 131072, grid=grid(131072), stream=stream0)
        del arg140_1
        del buf192
        # Source Nodes: [], Original ATen: []
        buf196 = aten._scaled_dot_product_efficient_attention(buf193, buf194, buf195, None, False, scale=0.125)
        buf197 = buf196[0]
        del buf196
        buf201 = reinterpret_tensor(buf195, (512, 256), (256, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (512, 256), (256, 1), 0), reinterpret_tensor(arg141_1, (256, 256), (1, 256), 0), out=buf201)
        del arg141_1
        buf205 = reinterpret_tensor(buf197, (1, 512, 256), (131072, 256, 1), 0); del buf197  # reuse
        # Source Nodes: [add_26, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf201, arg142_1, buf189, arg143_1, arg144_1, buf205, 512, 256, grid=grid(512), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        buf206 = reinterpret_tensor(buf184, (512, 1024), (1024, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 256), (256, 1), 0), reinterpret_tensor(arg145_1, (256, 1024), (1, 256), 0), out=buf206)
        del arg145_1
        buf207 = reinterpret_tensor(buf206, (1, 512, 1024), (524288, 1024, 1), 0); del buf206  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf207, arg146_1, 524288, grid=grid(524288), stream=stream0)
        del arg146_1
        buf208 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg147_1, (1024, 256), (1, 1024), 0), out=buf208)
        del arg147_1
        buf212 = buf189; del buf189  # reuse
        # Source Nodes: [add_27, hidden_states_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf208, arg148_1, buf205, arg149_1, arg150_1, buf212, 512, 256, grid=grid(512), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        buf213 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 256), (256, 1), 0), reinterpret_tensor(arg151_1, (256, 256), (1, 256), 0), out=buf213)
        del arg151_1
        buf214 = reinterpret_tensor(buf205, (512, 256), (256, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 256), (256, 1), 0), reinterpret_tensor(arg153_1, (256, 256), (1, 256), 0), out=buf214)
        del arg153_1
        buf215 = reinterpret_tensor(buf194, (512, 256), (256, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 256), (256, 1), 0), reinterpret_tensor(arg155_1, (256, 256), (1, 256), 0), out=buf215)
        del arg155_1
        buf216 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf213, arg152_1, buf216, 131072, grid=grid(131072), stream=stream0)
        del arg152_1
        buf217 = reinterpret_tensor(buf213, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf214, arg154_1, buf217, 131072, grid=grid(131072), stream=stream0)
        del arg154_1
        buf218 = reinterpret_tensor(buf214, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf215, arg156_1, buf218, 131072, grid=grid(131072), stream=stream0)
        del arg156_1
        del buf215
        # Source Nodes: [], Original ATen: []
        buf219 = aten._scaled_dot_product_efficient_attention(buf216, buf217, buf218, None, False, scale=0.125)
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf218, (512, 256), (256, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 256), (256, 1), 0), reinterpret_tensor(arg157_1, (256, 256), (1, 256), 0), out=buf224)
        del arg157_1
        buf228 = reinterpret_tensor(buf220, (1, 512, 256), (131072, 256, 1), 0); del buf220  # reuse
        # Source Nodes: [add_29, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf224, arg158_1, buf212, arg159_1, arg160_1, buf228, 512, 256, grid=grid(512), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        buf229 = reinterpret_tensor(buf207, (512, 1024), (1024, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 256), (256, 1), 0), reinterpret_tensor(arg161_1, (256, 1024), (1, 256), 0), out=buf229)
        del arg161_1
        buf230 = reinterpret_tensor(buf229, (1, 512, 1024), (524288, 1024, 1), 0); del buf229  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf230, arg162_1, 524288, grid=grid(524288), stream=stream0)
        del arg162_1
        buf231 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg163_1, (1024, 256), (1, 1024), 0), out=buf231)
        del arg163_1
        buf235 = buf212; del buf212  # reuse
        # Source Nodes: [add_30, hidden_states_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf231, arg164_1, buf228, arg165_1, arg166_1, buf235, 512, 256, grid=grid(512), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        buf236 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 256), (256, 1), 0), reinterpret_tensor(arg167_1, (256, 256), (1, 256), 0), out=buf236)
        del arg167_1
        buf237 = reinterpret_tensor(buf228, (512, 256), (256, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 256), (256, 1), 0), reinterpret_tensor(arg169_1, (256, 256), (1, 256), 0), out=buf237)
        del arg169_1
        buf238 = reinterpret_tensor(buf217, (512, 256), (256, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 256), (256, 1), 0), reinterpret_tensor(arg171_1, (256, 256), (1, 256), 0), out=buf238)
        del arg171_1
        buf239 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf236, arg168_1, buf239, 131072, grid=grid(131072), stream=stream0)
        del arg168_1
        buf240 = reinterpret_tensor(buf236, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf237, arg170_1, buf240, 131072, grid=grid(131072), stream=stream0)
        del arg170_1
        buf241 = reinterpret_tensor(buf237, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf238, arg172_1, buf241, 131072, grid=grid(131072), stream=stream0)
        del arg172_1
        del buf238
        # Source Nodes: [], Original ATen: []
        buf242 = aten._scaled_dot_product_efficient_attention(buf239, buf240, buf241, None, False, scale=0.125)
        buf243 = buf242[0]
        del buf242
        buf247 = reinterpret_tensor(buf241, (512, 256), (256, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 256), (256, 1), 0), reinterpret_tensor(arg173_1, (256, 256), (1, 256), 0), out=buf247)
        del arg173_1
        buf251 = reinterpret_tensor(buf243, (1, 512, 256), (131072, 256, 1), 0); del buf243  # reuse
        # Source Nodes: [add_32, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf247, arg174_1, buf235, arg175_1, arg176_1, buf251, 512, 256, grid=grid(512), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        buf252 = reinterpret_tensor(buf230, (512, 1024), (1024, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (512, 256), (256, 1), 0), reinterpret_tensor(arg177_1, (256, 1024), (1, 256), 0), out=buf252)
        del arg177_1
        buf253 = reinterpret_tensor(buf252, (1, 512, 1024), (524288, 1024, 1), 0); del buf252  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf253, arg178_1, 524288, grid=grid(524288), stream=stream0)
        del arg178_1
        buf254 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 256), (1, 1024), 0), out=buf254)
        del arg179_1
        buf258 = buf235; del buf235  # reuse
        # Source Nodes: [add_33, hidden_states_100], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf254, arg180_1, buf251, arg181_1, arg182_1, buf258, 512, 256, grid=grid(512), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        buf259 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 256), (256, 1), 0), reinterpret_tensor(arg183_1, (256, 256), (1, 256), 0), out=buf259)
        del arg183_1
        buf260 = reinterpret_tensor(buf251, (512, 256), (256, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 256), (256, 1), 0), reinterpret_tensor(arg185_1, (256, 256), (1, 256), 0), out=buf260)
        del arg185_1
        buf261 = reinterpret_tensor(buf240, (512, 256), (256, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 256), (256, 1), 0), reinterpret_tensor(arg187_1, (256, 256), (1, 256), 0), out=buf261)
        del arg187_1
        buf262 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf259, arg184_1, buf262, 131072, grid=grid(131072), stream=stream0)
        del arg184_1
        buf263 = reinterpret_tensor(buf259, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf260, arg186_1, buf263, 131072, grid=grid(131072), stream=stream0)
        del arg186_1
        buf264 = reinterpret_tensor(buf260, (1, 4, 512, 64), (131072, 32768, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf261, arg188_1, buf264, 131072, grid=grid(131072), stream=stream0)
        del arg188_1
        del buf261
        # Source Nodes: [], Original ATen: []
        buf265 = aten._scaled_dot_product_efficient_attention(buf262, buf263, buf264, None, False, scale=0.125)
        del buf262
        del buf263
        buf266 = buf265[0]
        del buf265
        buf270 = reinterpret_tensor(buf264, (512, 256), (256, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (512, 256), (256, 1), 0), reinterpret_tensor(arg189_1, (256, 256), (1, 256), 0), out=buf270)
        del arg189_1
        buf274 = reinterpret_tensor(buf266, (1, 512, 256), (131072, 256, 1), 0); del buf266  # reuse
        # Source Nodes: [add_35, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf270, arg190_1, buf258, arg191_1, arg192_1, buf274, 512, 256, grid=grid(512), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        buf275 = reinterpret_tensor(buf253, (512, 1024), (1024, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 256), (256, 1), 0), reinterpret_tensor(arg193_1, (256, 1024), (1, 256), 0), out=buf275)
        del arg193_1
        buf276 = reinterpret_tensor(buf275, (1, 512, 1024), (524288, 1024, 1), 0); del buf275  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf276, arg194_1, 524288, grid=grid(524288), stream=stream0)
        del arg194_1
        buf277 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg195_1, (1024, 256), (1, 1024), 0), out=buf277)
        del arg195_1
        del buf276
        buf281 = buf258; del buf258  # reuse
        # Source Nodes: [add_36, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf277, arg196_1, buf274, arg197_1, arg198_1, buf281, 512, 256, grid=grid(512), stream=stream0)
        del arg196_1
        del arg197_1
        del arg198_1
        del buf274
        del buf277
        buf282 = reinterpret_tensor(buf4, (512, 128), (128, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (512, 256), (256, 1), 0), reinterpret_tensor(arg199_1, (256, 128), (1, 256), 0), out=buf282)
        del arg199_1
        del buf281
        buf286 = buf0; del buf0  # reuse
        # Source Nodes: [hidden_states_111, hidden_states_112], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_4.run(buf282, arg200_1, arg201_1, arg202_1, buf286, 512, 128, grid=grid(512), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del buf282
        buf287 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg204_1, reinterpret_tensor(buf286, (512, 128), (128, 1), 0), reinterpret_tensor(arg203_1, (128, 30522), (1, 128), 0), alpha=1, beta=1, out=buf287)
        del arg203_1
        del arg204_1
        del buf286
        buf288 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        buf289 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_5.run(buf287, buf288, buf289, 511, 30522, grid=grid(511), stream=stream0)
        buf290 = empty((), device='cuda', dtype=torch.float32)
        buf292 = buf290; del buf290  # reuse
        # Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_6.run(buf292, arg207_1, buf287, buf288, buf289, 1, 511, grid=grid(1), stream=stream0)
        del arg207_1
        return (buf292, reinterpret_tensor(buf287, (1, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg207_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg208_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForCausalLM', benchmark_compiled_module)
