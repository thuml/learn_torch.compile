
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


# kernel path: /tmp/torchinductor_youkaichao/td/ctdiorbjeqo2rsbyhrxmzx3u2eiepf4563wdybckqgotxoivsbvd.py
# Source Nodes: [add, embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# embeddings => add_1
# embeddings_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# position_embeddings => embedding_1
# token_type_embeddings => embedding_2
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': []}
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
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 512
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 2
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert(((0 <= tmp14) & (tmp14 < 2)) | ~xmask, "index out of bounds: 0 <= tmp14 < 2")
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


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvr3dad45rdfyrqwtlqrykks67jsgc6codnqwl4c5of3tl24pqo.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44iomysrxwcryq3epcyut4ubb42x7wzrdl2v2ua7l5hnppxgzvk.py
# Source Nodes: [conv_attn_layer, conv_kernel_layer], Original ATen: [aten.mul, aten.view]
# conv_attn_layer => mul_3
# conv_kernel_layer => view_9
triton_poi_fused_mul_view_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (384*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cneulrjeigfc7l6mvhxtncznzoig2qe7td2tryeid6wgcs6izrgn.py
# Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
# conv_kernel_layer_2 => amax, div, exp, sub_2, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((r1 + (9*x0)) % 54), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (9*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpyomxeywxf3pu5scnyda5h6mgjjctztfxooccqhrmux5jjke6nk.py
# Source Nodes: [conv_out_layer_3], Original ATen: [aten.im2col]
# conv_out_layer_3 => full_default_1
triton_poi_fused_im2col_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_im2col_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktej2gwbpzp4d4ww2fsyjidqtefuvh56pajad4dfyjsw4zeoat5.py
# Source Nodes: [conv_out_layer_3], Original ATen: [aten.im2col]
# conv_out_layer_3 => unsqueeze_8
triton_poi_fused_im2col_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_im2col_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = x0 + x1
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstiqubsyecjytxk5kqbkswmoxi3lsp7nscxhvbfu3mepy2ovdao.py
# Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
# conv_out_layer_5 => clone_1
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196608
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 384)
    y3 = yindex
    y0 = yindex % 384
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp2 = tmp1 + 1
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert((0 <= tmp4) & (tmp4 < 1), "index out of bounds: 0 <= tmp4 < 1")
    tmp5 = (-4) + x2 + y1
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1, 1], 512, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.load(in_ptr1 + ((-1536) + y3 + (384*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x2 + (9*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufykpbobkum2mwx4xm4byxsf2yf47c5m45sizhcgswxvlzikqlp.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 6
    y1 = (yindex // 6)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (6*x2) + (384*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbvtg3pcdap26ohbcqdol5xqvy7ohekzemmfhasq35guxg4zmpk.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 6
    y1 = (yindex // 6)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y0)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (6*x2) + (384*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbsgj7m7kovmftyyvsuakidov24vvfbvy2ksnrevasrb35a3ago.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (6*x1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (32768*y0)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyz23ttfz3z2qwqlinit6ff5wxd2gbxofsrynkjrs6dcj4zveqk.py
# Source Nodes: [hidden_states_1], Original ATen: [aten.view]
# hidden_states_1 => view_30
triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = (x0 // 64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (384*x1)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 12, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-384) + x0 + (384*x1)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4i2pt7kw3yeln67xnwxjcl36kwonkjwg67yhs75bhibpv3ll7nv.py
# Source Nodes: [add_3, attention_output, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_9
# attention_output => add_10, add_11, mul_4, mul_5, rsqrt_1, sub_4, var_mean_1
# hidden_states_4 => view_32
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbd2lwdw2rqqb6boskzylx52cv25ffjrjkaegmlmleth3pypoca3.py
# Source Nodes: [hidden_states_6, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_6 => view_34
# intermediate_output => add_12, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6uk66jnyqkza25ts7hme2pq5zctt352nsvgvatihfjqkgpykyg.py
# Source Nodes: [add_4, attention_output, hidden_states_9, mixed_query_layer_1, transpose_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
# add_4 => add_13
# attention_output => add_11, mul_5
# hidden_states_9 => add_14, add_15, mul_10, mul_9, rsqrt_2, sub_5, var_mean_2
# mixed_query_layer_1 => view_36
# transpose_5 => permute_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqd42falxn6vf5sym3y4c45xpsascrvpjytv5lsmi7b5t3pxuj5w.py
# Source Nodes: [add_6, attention_output_2, hidden_states_13, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_6 => add_21
# attention_output_2 => add_22, add_23, mul_12, mul_13, rsqrt_3, sub_8, var_mean_3
# hidden_states_13 => view_68
# hidden_states_9 => add_15, mul_10
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/si/csirxebzplho7vnlqwbusa3gx43zihnhmjpcfij7d36hs6cn2cbd.py
# Source Nodes: [hidden_states_110, prediction_scores, prediction_scores_1], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_110 => add_148, erf_12, mul_100, mul_101, mul_99
# prediction_scores => add_149, add_150, mul_102, mul_103, rsqrt_25, sub_50, var_mean_25
# prediction_scores_1 => view_434
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vvmhuqe7q4odgyc6hovxbp2w4hpusr3zlbi72num2iakfcjdlp.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24, exp_24, log, sub_51, sub_52, sum_25
triton_red_fused__log_softmax_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp10 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (30522*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgrl2lqd6bgfjgogf4imv4532pp2qbiztsjfcueeqn4vtqknupo.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_36, full_default_14, ne, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_17', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp10 = tmp9 + 30522
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 30522), "index out of bounds: 0 <= tmp12 < 30522")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (30522*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291 = args
    args.clear()
    assert_size_stride(primals_1, (384, 1), (1, 1))
    assert_size_stride(primals_2, (384, 1), (1, 1))
    assert_size_stride(primals_3, (384, 1), (1, 1))
    assert_size_stride(primals_4, (384, 1), (1, 1))
    assert_size_stride(primals_5, (384, 1), (1, 1))
    assert_size_stride(primals_6, (384, 1), (1, 1))
    assert_size_stride(primals_7, (384, 1), (1, 1))
    assert_size_stride(primals_8, (384, 1), (1, 1))
    assert_size_stride(primals_9, (384, 1), (1, 1))
    assert_size_stride(primals_10, (384, 1), (1, 1))
    assert_size_stride(primals_11, (384, 1), (1, 1))
    assert_size_stride(primals_12, (384, 1), (1, 1))
    assert_size_stride(primals_13, (30522, 768), (768, 1))
    assert_size_stride(primals_14, (512, 768), (768, 1))
    assert_size_stride(primals_15, (2, 768), (768, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (384, 768), (768, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (384, 768), (768, 1))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_22, (384, 768), (768, 1))
    assert_size_stride(primals_23, (384, ), (1, ))
    assert_size_stride(primals_24, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_25, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_26, (54, 384), (384, 1))
    assert_size_stride(primals_27, (54, ), (1, ))
    assert_size_stride(primals_28, (384, 768), (768, 1))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (768, 768), (768, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (3072, 768), (768, 1))
    assert_size_stride(primals_35, (3072, ), (1, ))
    assert_size_stride(primals_36, (768, 3072), (3072, 1))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (384, 768), (768, 1))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, 768), (768, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (384, 768), (768, 1))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_47, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_48, (54, 384), (384, 1))
    assert_size_stride(primals_49, (54, ), (1, ))
    assert_size_stride(primals_50, (384, 768), (768, 1))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (768, 768), (768, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (3072, 768), (768, 1))
    assert_size_stride(primals_57, (3072, ), (1, ))
    assert_size_stride(primals_58, (768, 3072), (3072, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (384, 768), (768, 1))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, 768), (768, 1))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, 768), (768, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_69, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_70, (54, 384), (384, 1))
    assert_size_stride(primals_71, (54, ), (1, ))
    assert_size_stride(primals_72, (384, 768), (768, 1))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (3072, 768), (768, 1))
    assert_size_stride(primals_79, (3072, ), (1, ))
    assert_size_stride(primals_80, (768, 3072), (3072, 1))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (384, 768), (768, 1))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (384, 768), (768, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, 768), (768, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_91, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_92, (54, 384), (384, 1))
    assert_size_stride(primals_93, (54, ), (1, ))
    assert_size_stride(primals_94, (384, 768), (768, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (768, 768), (768, 1))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768), (768, 1))
    assert_size_stride(primals_101, (3072, ), (1, ))
    assert_size_stride(primals_102, (768, 3072), (3072, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (384, 768), (768, 1))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (384, 768), (768, 1))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (384, 768), (768, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_113, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_114, (54, 384), (384, 1))
    assert_size_stride(primals_115, (54, ), (1, ))
    assert_size_stride(primals_116, (384, 768), (768, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (384, 768), (768, 1))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, 768), (768, 1))
    assert_size_stride(primals_131, (384, ), (1, ))
    assert_size_stride(primals_132, (384, 768), (768, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_135, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_136, (54, 384), (384, 1))
    assert_size_stride(primals_137, (54, ), (1, ))
    assert_size_stride(primals_138, (384, 768), (768, 1))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (768, 768), (768, 1))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (3072, 768), (768, 1))
    assert_size_stride(primals_145, (3072, ), (1, ))
    assert_size_stride(primals_146, (768, 3072), (3072, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (384, 768), (768, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (384, 768), (768, 1))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, 768), (768, 1))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_157, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_158, (54, 384), (384, 1))
    assert_size_stride(primals_159, (54, ), (1, ))
    assert_size_stride(primals_160, (384, 768), (768, 1))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (768, 768), (768, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (3072, 768), (768, 1))
    assert_size_stride(primals_167, (3072, ), (1, ))
    assert_size_stride(primals_168, (768, 3072), (3072, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (384, 768), (768, 1))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, 768), (768, 1))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, 768), (768, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_179, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_180, (54, 384), (384, 1))
    assert_size_stride(primals_181, (54, ), (1, ))
    assert_size_stride(primals_182, (384, 768), (768, 1))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (768, 768), (768, 1))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (3072, 768), (768, 1))
    assert_size_stride(primals_189, (3072, ), (1, ))
    assert_size_stride(primals_190, (768, 3072), (3072, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (384, 768), (768, 1))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, 768), (768, 1))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (384, 768), (768, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_201, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_202, (54, 384), (384, 1))
    assert_size_stride(primals_203, (54, ), (1, ))
    assert_size_stride(primals_204, (384, 768), (768, 1))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (768, 768), (768, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, ), (1, ))
    assert_size_stride(primals_210, (3072, 768), (768, 1))
    assert_size_stride(primals_211, (3072, ), (1, ))
    assert_size_stride(primals_212, (768, 3072), (3072, 1))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (384, 768), (768, 1))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (384, 768), (768, 1))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, 768), (768, 1))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_223, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_224, (54, 384), (384, 1))
    assert_size_stride(primals_225, (54, ), (1, ))
    assert_size_stride(primals_226, (384, 768), (768, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (768, 768), (768, 1))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_232, (3072, 768), (768, 1))
    assert_size_stride(primals_233, (3072, ), (1, ))
    assert_size_stride(primals_234, (768, 3072), (3072, 1))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_238, (384, 768), (768, 1))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, 768), (768, 1))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (384, 768), (768, 1))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_245, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_246, (54, 384), (384, 1))
    assert_size_stride(primals_247, (54, ), (1, ))
    assert_size_stride(primals_248, (384, 768), (768, 1))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (768, 768), (768, 1))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (3072, 768), (768, 1))
    assert_size_stride(primals_255, (3072, ), (1, ))
    assert_size_stride(primals_256, (768, 3072), (3072, 1))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (384, 768), (768, 1))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (384, 768), (768, 1))
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (384, 768), (768, 1))
    assert_size_stride(primals_265, (384, ), (1, ))
    assert_size_stride(primals_266, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_267, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_268, (54, 384), (384, 1))
    assert_size_stride(primals_269, (54, ), (1, ))
    assert_size_stride(primals_270, (384, 768), (768, 1))
    assert_size_stride(primals_271, (384, ), (1, ))
    assert_size_stride(primals_272, (768, 768), (768, 1))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (768, ), (1, ))
    assert_size_stride(primals_275, (768, ), (1, ))
    assert_size_stride(primals_276, (3072, 768), (768, 1))
    assert_size_stride(primals_277, (3072, ), (1, ))
    assert_size_stride(primals_278, (768, 3072), (3072, 1))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_280, (768, ), (1, ))
    assert_size_stride(primals_281, (768, ), (1, ))
    assert_size_stride(primals_282, (768, 768), (768, 1))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (30522, 768), (768, 1))
    assert_size_stride(primals_287, (30522, ), (1, ))
    assert_size_stride(primals_288, (1, 512), (512, 1))
    assert_size_stride(primals_289, (1, 512), (512, 1))
    assert_size_stride(primals_290, (1, 512), (512, 1))
    assert_size_stride(primals_291, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf624 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0.run(primals_290, primals_13, primals_289, primals_14, primals_288, primals_15, primals_16, primals_17, buf0, buf4, buf5, buf624, 512, 768, grid=grid(512), stream=stream0)
        del primals_13
        del primals_14
        del primals_15
        del primals_17
        # Source Nodes: [embeddings_1, hidden_states], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf6 = aten.native_dropout(buf5, 0.1, True)
        buf7 = buf6[0]
        buf8 = buf6[1]
        del buf6
        buf9 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_18, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del primals_19
        buf10 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_20, (768, 384), (1, 768), 0), out=buf10)
        buf11 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 384), (1, 768), 0), out=buf11)
        buf12 = reinterpret_tensor(buf5, (1, 768, 512), (393216, 512, 1), 0); del buf5  # reuse
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf7, buf12, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_24, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf13, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_25, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf14, (1, 384, 512), (196608, 512, 1))
        buf15 = empty_strided((512, 384), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_attn_layer, conv_kernel_layer], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf14, primals_1, buf9, buf15, 384, 512, grid=grid(384, 512), stream=stream0)
        buf16 = empty((512, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, reinterpret_tensor(primals_26, (384, 54), (1, 384), 0), out=buf16)
        buf23 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf16, primals_27, buf23, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_27
        buf19 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_28, (768, 384), (1, 768), 0), out=buf19)
        buf20 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [conv_out_layer_3], Original ATen: [aten.im2col]
        triton_poi_fused_im2col_4.run(buf20, 1, grid=grid(1), stream=stream0)
        buf21 = empty_strided((9, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [conv_out_layer_3], Original ATen: [aten.im2col]
        triton_poi_fused_im2col_5.run(buf21, 4608, grid=grid(4608), stream=stream0)
        buf22 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf19, primals_29, buf22, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_29
        buf24 = reinterpret_tensor(buf19, (3072, 64, 1), (64, 1, 1), 0); del buf19  # reuse
        # Source Nodes: [conv_out_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (3072, 64, 9), (576, 9, 1), 0), buf23, out=buf24)
        buf25 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf9, buf25, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf26 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf10, primals_21, buf26, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_21
        buf27 = reinterpret_tensor(buf10, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf11, primals_23, buf27, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_23
        buf28 = reinterpret_tensor(buf11, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf25, buf28, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf29 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf26, buf29, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf30 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf27, buf30, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf31 = aten._scaled_dot_product_efficient_attention(buf28, buf29, buf30, None, True, 0.1, scale=0.125)
        buf32 = buf31[0]
        buf33 = buf31[1]
        buf34 = buf31[2]
        buf35 = buf31[3]
        del buf31
        buf36 = reinterpret_tensor(buf30, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf32, buf36, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf37 = reinterpret_tensor(buf12, (512, 768), (768, 1), 0); del buf12  # reuse
        # Source Nodes: [hidden_states_1], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf32, buf24, buf37, 393216, grid=grid(393216), stream=stream0)
        buf38 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, buf37, reinterpret_tensor(primals_30, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf38)
        del primals_31
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
        buf39 = aten.native_dropout(reinterpret_tensor(buf38, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf45 = reinterpret_tensor(buf38, (1, 512, 768), (393216, 768, 1), 0); del buf38  # reuse
        buf46 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf623 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, attention_output, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf40, buf7, primals_32, primals_33, buf45, buf46, buf623, 512, 768, grid=grid(512), stream=stream0)
        buf47 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf46, reinterpret_tensor(primals_34, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del primals_35
        buf48 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf47, buf48, 1572864, grid=grid(1572864), stream=stream0)
        buf49 = reinterpret_tensor(buf40, (512, 768), (768, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf48, reinterpret_tensor(primals_36, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf49)
        del primals_37
        # Source Nodes: [hidden_states_7], Original ATen: [aten.native_dropout]
        buf50 = aten.native_dropout(reinterpret_tensor(buf49, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf56 = reinterpret_tensor(buf49, (1, 512, 768), (393216, 768, 1), 0); del buf49  # reuse
        buf57 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf622 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, attention_output, hidden_states_9, mixed_query_layer_1, transpose_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf51, buf45, primals_32, primals_33, primals_38, primals_39, buf56, buf57, buf61, buf622, 512, 768, grid=grid(512), stream=stream0)
        del primals_33
        buf58 = reinterpret_tensor(buf32, (512, 384), (384, 1), 0); del buf32  # reuse
        # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, buf57, reinterpret_tensor(primals_40, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf58)
        del primals_41
        buf59 = reinterpret_tensor(buf24, (512, 384), (384, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf57, reinterpret_tensor(primals_42, (768, 384), (1, 768), 0), out=buf59)
        buf60 = reinterpret_tensor(buf29, (512, 384), (384, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf57, reinterpret_tensor(primals_44, (768, 384), (1, 768), 0), out=buf60)
        buf62 = reinterpret_tensor(buf51, (1, 768, 512), (393216, 512, 1), 0); del buf51  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf61, buf62, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_46, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf63, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_47, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf64, (1, 384, 512), (196608, 512, 1))
        buf65 = reinterpret_tensor(buf28, (512, 384), (1, 512), 0); del buf28  # reuse
        # Source Nodes: [conv_attn_layer_1, conv_kernel_layer_3], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf64, primals_2, buf58, buf65, 384, 512, grid=grid(384, 512), stream=stream0)
        buf66 = buf16; del buf16  # reuse
        # Source Nodes: [conv_kernel_layer_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf65, reinterpret_tensor(primals_48, (384, 54), (1, 384), 0), out=buf66)
        buf71 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf66, primals_49, buf71, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_49
        buf69 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf57, reinterpret_tensor(primals_50, (768, 384), (1, 768), 0), out=buf69)
        buf70 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf69, primals_51, buf70, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_51
        buf72 = reinterpret_tensor(buf69, (3072, 64, 1), (64, 1, 1), 0); del buf69  # reuse
        # Source Nodes: [conv_out_layer_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (3072, 64, 9), (576, 9, 1), 0), buf71, out=buf72)
        buf73 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf58, buf73, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf74 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf59, primals_43, buf74, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_43
        buf75 = reinterpret_tensor(buf59, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf60, primals_45, buf75, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_45
        buf76 = reinterpret_tensor(buf60, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf73, buf76, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf77 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf74, buf77, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf78 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf75, buf78, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf79 = aten._scaled_dot_product_efficient_attention(buf76, buf77, buf78, None, True, 0.1, scale=0.125)
        buf80 = buf79[0]
        buf81 = buf79[1]
        buf82 = buf79[2]
        buf83 = buf79[3]
        del buf79
        buf84 = reinterpret_tensor(buf78, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf80, buf84, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf85 = reinterpret_tensor(buf62, (512, 768), (768, 1), 0); del buf62  # reuse
        # Source Nodes: [hidden_states_10], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf80, buf72, buf85, 393216, grid=grid(393216), stream=stream0)
        buf86 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, buf85, reinterpret_tensor(primals_52, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
        del primals_53
        # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
        buf87 = aten.native_dropout(reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf88 = buf87[0]
        buf89 = buf87[1]
        del buf87
        buf93 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf94 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf621 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, attention_output_2, hidden_states_13, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf88, buf56, primals_38, primals_39, primals_54, primals_55, buf93, buf94, buf621, 512, 768, grid=grid(512), stream=stream0)
        del primals_39
        buf95 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, buf94, reinterpret_tensor(primals_56, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf95)
        del primals_57
        buf96 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_15, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf95, buf96, 1572864, grid=grid(1572864), stream=stream0)
        buf97 = reinterpret_tensor(buf88, (512, 768), (768, 1), 0); del buf88  # reuse
        # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, buf96, reinterpret_tensor(primals_58, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf97)
        del primals_59
        # Source Nodes: [hidden_states_16], Original ATen: [aten.native_dropout]
        buf98 = aten.native_dropout(reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf104 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        buf105 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf620 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, attention_output_2, hidden_states_18, mixed_query_layer_2, transpose_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf99, buf93, primals_54, primals_55, primals_60, primals_61, buf104, buf105, buf109, buf620, 512, 768, grid=grid(512), stream=stream0)
        del primals_55
        buf106 = reinterpret_tensor(buf80, (512, 384), (384, 1), 0); del buf80  # reuse
        # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_63, buf105, reinterpret_tensor(primals_62, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf106)
        del primals_63
        buf107 = reinterpret_tensor(buf72, (512, 384), (384, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_64, (768, 384), (1, 768), 0), out=buf107)
        buf108 = reinterpret_tensor(buf77, (512, 384), (384, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_66, (768, 384), (1, 768), 0), out=buf108)
        buf110 = reinterpret_tensor(buf99, (1, 768, 512), (393216, 512, 1), 0); del buf99  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf109, buf110, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_68, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf111, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_69, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf112, (1, 384, 512), (196608, 512, 1))
        buf113 = reinterpret_tensor(buf76, (512, 384), (1, 512), 0); del buf76  # reuse
        # Source Nodes: [conv_attn_layer_2, conv_kernel_layer_6], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf112, primals_3, buf106, buf113, 384, 512, grid=grid(384, 512), stream=stream0)
        buf114 = buf66; del buf66  # reuse
        # Source Nodes: [conv_kernel_layer_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf113, reinterpret_tensor(primals_70, (384, 54), (1, 384), 0), out=buf114)
        buf119 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf114, primals_71, buf119, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_71
        buf117 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_72, (768, 384), (1, 768), 0), out=buf117)
        buf118 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf117, primals_73, buf118, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_73
        buf120 = reinterpret_tensor(buf117, (3072, 64, 1), (64, 1, 1), 0); del buf117  # reuse
        # Source Nodes: [conv_out_layer_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf118, (3072, 64, 9), (576, 9, 1), 0), buf119, out=buf120)
        buf121 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf106, buf121, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf122 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf107, primals_65, buf122, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_65
        buf123 = reinterpret_tensor(buf107, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf108, primals_67, buf123, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_67
        buf124 = reinterpret_tensor(buf108, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf121, buf124, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf125 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf122, buf125, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf126 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf123, buf126, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf127 = aten._scaled_dot_product_efficient_attention(buf124, buf125, buf126, None, True, 0.1, scale=0.125)
        buf128 = buf127[0]
        buf129 = buf127[1]
        buf130 = buf127[2]
        buf131 = buf127[3]
        del buf127
        buf132 = reinterpret_tensor(buf126, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf128, buf132, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf133 = reinterpret_tensor(buf110, (512, 768), (768, 1), 0); del buf110  # reuse
        # Source Nodes: [hidden_states_19], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf128, buf120, buf133, 393216, grid=grid(393216), stream=stream0)
        buf134 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, buf133, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del primals_75
        # Source Nodes: [hidden_states_20], Original ATen: [aten.native_dropout]
        buf135 = aten.native_dropout(reinterpret_tensor(buf134, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf141 = reinterpret_tensor(buf134, (1, 512, 768), (393216, 768, 1), 0); del buf134  # reuse
        buf142 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf619 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attention_output_4, hidden_states_18, hidden_states_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf136, buf104, primals_60, primals_61, primals_76, primals_77, buf141, buf142, buf619, 512, 768, grid=grid(512), stream=stream0)
        del primals_61
        buf143 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_79, buf142, reinterpret_tensor(primals_78, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf143)
        del primals_79
        buf144 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf143, buf144, 1572864, grid=grid(1572864), stream=stream0)
        buf145 = reinterpret_tensor(buf136, (512, 768), (768, 1), 0); del buf136  # reuse
        # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf144, reinterpret_tensor(primals_80, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf145)
        del primals_81
        # Source Nodes: [hidden_states_25], Original ATen: [aten.native_dropout]
        buf146 = aten.native_dropout(reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf147 = buf146[0]
        buf148 = buf146[1]
        del buf146
        buf152 = reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0); del buf145  # reuse
        buf153 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf618 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, attention_output_4, hidden_states_27, mixed_query_layer_3, transpose_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf147, buf141, primals_76, primals_77, primals_82, primals_83, buf152, buf153, buf157, buf618, 512, 768, grid=grid(512), stream=stream0)
        del primals_77
        buf154 = reinterpret_tensor(buf128, (512, 384), (384, 1), 0); del buf128  # reuse
        # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, buf153, reinterpret_tensor(primals_84, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf154)
        del primals_85
        buf155 = reinterpret_tensor(buf120, (512, 384), (384, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf153, reinterpret_tensor(primals_86, (768, 384), (1, 768), 0), out=buf155)
        buf156 = reinterpret_tensor(buf125, (512, 384), (384, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf153, reinterpret_tensor(primals_88, (768, 384), (1, 768), 0), out=buf156)
        buf158 = reinterpret_tensor(buf147, (1, 768, 512), (393216, 512, 1), 0); del buf147  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf157, buf158, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_90, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf159, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_91, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf160, (1, 384, 512), (196608, 512, 1))
        buf161 = reinterpret_tensor(buf124, (512, 384), (1, 512), 0); del buf124  # reuse
        # Source Nodes: [conv_attn_layer_3, conv_kernel_layer_9], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf160, primals_4, buf154, buf161, 384, 512, grid=grid(384, 512), stream=stream0)
        buf162 = buf114; del buf114  # reuse
        # Source Nodes: [conv_kernel_layer_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf161, reinterpret_tensor(primals_92, (384, 54), (1, 384), 0), out=buf162)
        buf167 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf162, primals_93, buf167, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_93
        buf165 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf153, reinterpret_tensor(primals_94, (768, 384), (1, 768), 0), out=buf165)
        buf166 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf165, primals_95, buf166, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_95
        buf168 = reinterpret_tensor(buf165, (3072, 64, 1), (64, 1, 1), 0); del buf165  # reuse
        # Source Nodes: [conv_out_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (3072, 64, 9), (576, 9, 1), 0), buf167, out=buf168)
        buf169 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf154, buf169, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf170 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf155, primals_87, buf170, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_87
        buf171 = reinterpret_tensor(buf155, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf156, primals_89, buf171, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_89
        buf172 = reinterpret_tensor(buf156, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf169, buf172, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf173 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf170, buf173, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf174 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf171, buf174, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf175 = aten._scaled_dot_product_efficient_attention(buf172, buf173, buf174, None, True, 0.1, scale=0.125)
        buf176 = buf175[0]
        buf177 = buf175[1]
        buf178 = buf175[2]
        buf179 = buf175[3]
        del buf175
        buf180 = reinterpret_tensor(buf174, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf176, buf180, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf181 = reinterpret_tensor(buf158, (512, 768), (768, 1), 0); del buf158  # reuse
        # Source Nodes: [hidden_states_28], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf176, buf168, buf181, 393216, grid=grid(393216), stream=stream0)
        buf182 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf181, reinterpret_tensor(primals_96, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf182)
        del primals_97
        # Source Nodes: [hidden_states_29], Original ATen: [aten.native_dropout]
        buf183 = aten.native_dropout(reinterpret_tensor(buf182, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf184 = buf183[0]
        buf185 = buf183[1]
        del buf183
        buf189 = reinterpret_tensor(buf182, (1, 512, 768), (393216, 768, 1), 0); del buf182  # reuse
        buf190 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf617 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_6, hidden_states_27, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf184, buf152, primals_82, primals_83, primals_98, primals_99, buf189, buf190, buf617, 512, 768, grid=grid(512), stream=stream0)
        del primals_83
        buf191 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_101, buf190, reinterpret_tensor(primals_100, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf191)
        del primals_101
        buf192 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_33, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf191, buf192, 1572864, grid=grid(1572864), stream=stream0)
        buf193 = reinterpret_tensor(buf184, (512, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_103, buf192, reinterpret_tensor(primals_102, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf193)
        del primals_103
        # Source Nodes: [hidden_states_34], Original ATen: [aten.native_dropout]
        buf194 = aten.native_dropout(reinterpret_tensor(buf193, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf200 = reinterpret_tensor(buf193, (1, 512, 768), (393216, 768, 1), 0); del buf193  # reuse
        buf201 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf616 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, attention_output_6, hidden_states_36, mixed_query_layer_4, transpose_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf195, buf189, primals_98, primals_99, primals_104, primals_105, buf200, buf201, buf205, buf616, 512, 768, grid=grid(512), stream=stream0)
        del primals_99
        buf202 = reinterpret_tensor(buf176, (512, 384), (384, 1), 0); del buf176  # reuse
        # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_107, buf201, reinterpret_tensor(primals_106, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf202)
        del primals_107
        buf203 = reinterpret_tensor(buf168, (512, 384), (384, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_108, (768, 384), (1, 768), 0), out=buf203)
        buf204 = reinterpret_tensor(buf173, (512, 384), (384, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_110, (768, 384), (1, 768), 0), out=buf204)
        buf206 = reinterpret_tensor(buf195, (1, 768, 512), (393216, 512, 1), 0); del buf195  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf205, buf206, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_112, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf207, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_113, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf208, (1, 384, 512), (196608, 512, 1))
        buf209 = reinterpret_tensor(buf172, (512, 384), (1, 512), 0); del buf172  # reuse
        # Source Nodes: [conv_attn_layer_4, conv_kernel_layer_12], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf208, primals_5, buf202, buf209, 384, 512, grid=grid(384, 512), stream=stream0)
        buf210 = buf162; del buf162  # reuse
        # Source Nodes: [conv_kernel_layer_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf209, reinterpret_tensor(primals_114, (384, 54), (1, 384), 0), out=buf210)
        buf215 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf210, primals_115, buf215, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_115
        buf213 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_116, (768, 384), (1, 768), 0), out=buf213)
        buf214 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf213, primals_117, buf214, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_117
        buf216 = reinterpret_tensor(buf213, (3072, 64, 1), (64, 1, 1), 0); del buf213  # reuse
        # Source Nodes: [conv_out_layer_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (3072, 64, 9), (576, 9, 1), 0), buf215, out=buf216)
        buf217 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf202, buf217, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf218 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf203, primals_109, buf218, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_109
        buf219 = reinterpret_tensor(buf203, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf204, primals_111, buf219, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_111
        buf220 = reinterpret_tensor(buf204, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf217, buf220, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf221 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf218, buf221, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf222 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf219, buf222, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf223 = aten._scaled_dot_product_efficient_attention(buf220, buf221, buf222, None, True, 0.1, scale=0.125)
        buf224 = buf223[0]
        buf225 = buf223[1]
        buf226 = buf223[2]
        buf227 = buf223[3]
        del buf223
        buf228 = reinterpret_tensor(buf222, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf224, buf228, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf229 = reinterpret_tensor(buf206, (512, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [hidden_states_37], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf224, buf216, buf229, 393216, grid=grid(393216), stream=stream0)
        buf230 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_119, buf229, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf230)
        del primals_119
        # Source Nodes: [hidden_states_38], Original ATen: [aten.native_dropout]
        buf231 = aten.native_dropout(reinterpret_tensor(buf230, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf232 = buf231[0]
        buf233 = buf231[1]
        del buf231
        buf237 = reinterpret_tensor(buf230, (1, 512, 768), (393216, 768, 1), 0); del buf230  # reuse
        buf238 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf615 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, attention_output_8, hidden_states_36, hidden_states_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf232, buf200, primals_104, primals_105, primals_120, primals_121, buf237, buf238, buf615, 512, 768, grid=grid(512), stream=stream0)
        del primals_105
        buf239 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_123, buf238, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf239)
        del primals_123
        buf240 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf239, buf240, 1572864, grid=grid(1572864), stream=stream0)
        buf241 = reinterpret_tensor(buf232, (512, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf240, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf241)
        del primals_125
        # Source Nodes: [hidden_states_43], Original ATen: [aten.native_dropout]
        buf242 = aten.native_dropout(reinterpret_tensor(buf241, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        buf248 = reinterpret_tensor(buf241, (1, 512, 768), (393216, 768, 1), 0); del buf241  # reuse
        buf249 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf253 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf614 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_16, attention_output_8, hidden_states_45, mixed_query_layer_5, transpose_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf243, buf237, primals_120, primals_121, primals_126, primals_127, buf248, buf249, buf253, buf614, 512, 768, grid=grid(512), stream=stream0)
        del primals_121
        buf250 = reinterpret_tensor(buf224, (512, 384), (384, 1), 0); del buf224  # reuse
        # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf249, reinterpret_tensor(primals_128, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf250)
        del primals_129
        buf251 = reinterpret_tensor(buf216, (512, 384), (384, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, reinterpret_tensor(primals_130, (768, 384), (1, 768), 0), out=buf251)
        buf252 = reinterpret_tensor(buf221, (512, 384), (384, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, reinterpret_tensor(primals_132, (768, 384), (1, 768), 0), out=buf252)
        buf254 = reinterpret_tensor(buf243, (1, 768, 512), (393216, 512, 1), 0); del buf243  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf253, buf254, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_134, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf255, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_135, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf256, (1, 384, 512), (196608, 512, 1))
        buf257 = reinterpret_tensor(buf220, (512, 384), (1, 512), 0); del buf220  # reuse
        # Source Nodes: [conv_attn_layer_5, conv_kernel_layer_15], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf256, primals_6, buf250, buf257, 384, 512, grid=grid(384, 512), stream=stream0)
        buf258 = buf210; del buf210  # reuse
        # Source Nodes: [conv_kernel_layer_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf257, reinterpret_tensor(primals_136, (384, 54), (1, 384), 0), out=buf258)
        buf263 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf258, primals_137, buf263, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_137
        buf261 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, reinterpret_tensor(primals_138, (768, 384), (1, 768), 0), out=buf261)
        buf262 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf261, primals_139, buf262, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_139
        buf264 = reinterpret_tensor(buf261, (3072, 64, 1), (64, 1, 1), 0); del buf261  # reuse
        # Source Nodes: [conv_out_layer_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (3072, 64, 9), (576, 9, 1), 0), buf263, out=buf264)
        buf265 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf250, buf265, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf266 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf251, primals_131, buf266, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_131
        buf267 = reinterpret_tensor(buf251, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf252, primals_133, buf267, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_133
        buf268 = reinterpret_tensor(buf252, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf265, buf268, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf269 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf266, buf269, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf270 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf267, buf270, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf271 = aten._scaled_dot_product_efficient_attention(buf268, buf269, buf270, None, True, 0.1, scale=0.125)
        buf272 = buf271[0]
        buf273 = buf271[1]
        buf274 = buf271[2]
        buf275 = buf271[3]
        del buf271
        buf276 = reinterpret_tensor(buf270, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf272, buf276, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf277 = reinterpret_tensor(buf254, (512, 768), (768, 1), 0); del buf254  # reuse
        # Source Nodes: [hidden_states_46], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf272, buf264, buf277, 393216, grid=grid(393216), stream=stream0)
        buf278 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf277, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf278)
        del primals_141
        # Source Nodes: [hidden_states_47], Original ATen: [aten.native_dropout]
        buf279 = aten.native_dropout(reinterpret_tensor(buf278, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf285 = reinterpret_tensor(buf278, (1, 512, 768), (393216, 768, 1), 0); del buf278  # reuse
        buf286 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf613 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attention_output_10, hidden_states_45, hidden_states_49], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf280, buf248, primals_126, primals_127, primals_142, primals_143, buf285, buf286, buf613, 512, 768, grid=grid(512), stream=stream0)
        del primals_127
        buf287 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf286, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf287)
        del primals_145
        buf288 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf287, buf288, 1572864, grid=grid(1572864), stream=stream0)
        buf289 = reinterpret_tensor(buf280, (512, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf288, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf289)
        del primals_147
        # Source Nodes: [hidden_states_52], Original ATen: [aten.native_dropout]
        buf290 = aten.native_dropout(reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf291 = buf290[0]
        buf292 = buf290[1]
        del buf290
        buf296 = reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0); del buf289  # reuse
        buf297 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf301 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf612 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_19, attention_output_10, hidden_states_54, mixed_query_layer_6, transpose_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf291, buf285, primals_142, primals_143, primals_148, primals_149, buf296, buf297, buf301, buf612, 512, 768, grid=grid(512), stream=stream0)
        del primals_143
        buf298 = reinterpret_tensor(buf272, (512, 384), (384, 1), 0); del buf272  # reuse
        # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_151, buf297, reinterpret_tensor(primals_150, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf298)
        del primals_151
        buf299 = reinterpret_tensor(buf264, (512, 384), (384, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_152, (768, 384), (1, 768), 0), out=buf299)
        buf300 = reinterpret_tensor(buf269, (512, 384), (384, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_154, (768, 384), (1, 768), 0), out=buf300)
        buf302 = reinterpret_tensor(buf291, (1, 768, 512), (393216, 512, 1), 0); del buf291  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf301, buf302, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_156, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf303, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_157, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf304, (1, 384, 512), (196608, 512, 1))
        buf305 = reinterpret_tensor(buf268, (512, 384), (1, 512), 0); del buf268  # reuse
        # Source Nodes: [conv_attn_layer_6, conv_kernel_layer_18], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf304, primals_7, buf298, buf305, 384, 512, grid=grid(384, 512), stream=stream0)
        buf306 = buf258; del buf258  # reuse
        # Source Nodes: [conv_kernel_layer_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf305, reinterpret_tensor(primals_158, (384, 54), (1, 384), 0), out=buf306)
        buf311 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_20], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf306, primals_159, buf311, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_159
        buf309 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_160, (768, 384), (1, 768), 0), out=buf309)
        buf310 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf309, primals_161, buf310, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_161
        buf312 = reinterpret_tensor(buf309, (3072, 64, 1), (64, 1, 1), 0); del buf309  # reuse
        # Source Nodes: [conv_out_layer_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf310, (3072, 64, 9), (576, 9, 1), 0), buf311, out=buf312)
        buf313 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf298, buf313, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf314 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf299, primals_153, buf314, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_153
        buf315 = reinterpret_tensor(buf299, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf300, primals_155, buf315, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_155
        buf316 = reinterpret_tensor(buf300, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf313, buf316, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf317 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf314, buf317, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf318 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf315, buf318, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf319 = aten._scaled_dot_product_efficient_attention(buf316, buf317, buf318, None, True, 0.1, scale=0.125)
        buf320 = buf319[0]
        buf321 = buf319[1]
        buf322 = buf319[2]
        buf323 = buf319[3]
        del buf319
        buf324 = reinterpret_tensor(buf318, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf320, buf324, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf325 = reinterpret_tensor(buf302, (512, 768), (768, 1), 0); del buf302  # reuse
        # Source Nodes: [hidden_states_55], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf320, buf312, buf325, 393216, grid=grid(393216), stream=stream0)
        buf326 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, buf325, reinterpret_tensor(primals_162, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf326)
        del primals_163
        # Source Nodes: [hidden_states_56], Original ATen: [aten.native_dropout]
        buf327 = aten.native_dropout(reinterpret_tensor(buf326, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf333 = reinterpret_tensor(buf326, (1, 512, 768), (393216, 768, 1), 0); del buf326  # reuse
        buf334 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf611 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_12, hidden_states_54, hidden_states_58], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf328, buf296, primals_148, primals_149, primals_164, primals_165, buf333, buf334, buf611, 512, 768, grid=grid(512), stream=stream0)
        del primals_149
        buf335 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, buf334, reinterpret_tensor(primals_166, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf335)
        del primals_167
        buf336 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_60, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf335, buf336, 1572864, grid=grid(1572864), stream=stream0)
        buf337 = reinterpret_tensor(buf328, (512, 768), (768, 1), 0); del buf328  # reuse
        # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, buf336, reinterpret_tensor(primals_168, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf337)
        del primals_169
        # Source Nodes: [hidden_states_61], Original ATen: [aten.native_dropout]
        buf338 = aten.native_dropout(reinterpret_tensor(buf337, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf339 = buf338[0]
        buf340 = buf338[1]
        del buf338
        buf344 = reinterpret_tensor(buf337, (1, 512, 768), (393216, 768, 1), 0); del buf337  # reuse
        buf345 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf610 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, attention_output_12, hidden_states_63, mixed_query_layer_7, transpose_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf339, buf333, primals_164, primals_165, primals_170, primals_171, buf344, buf345, buf349, buf610, 512, 768, grid=grid(512), stream=stream0)
        del primals_165
        buf346 = reinterpret_tensor(buf320, (512, 384), (384, 1), 0); del buf320  # reuse
        # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf345, reinterpret_tensor(primals_172, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf346)
        del primals_173
        buf347 = reinterpret_tensor(buf312, (512, 384), (384, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf345, reinterpret_tensor(primals_174, (768, 384), (1, 768), 0), out=buf347)
        buf348 = reinterpret_tensor(buf317, (512, 384), (384, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf345, reinterpret_tensor(primals_176, (768, 384), (1, 768), 0), out=buf348)
        buf350 = reinterpret_tensor(buf339, (1, 768, 512), (393216, 512, 1), 0); del buf339  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf349, buf350, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_178, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf351, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_179, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf352, (1, 384, 512), (196608, 512, 1))
        buf353 = reinterpret_tensor(buf316, (512, 384), (1, 512), 0); del buf316  # reuse
        # Source Nodes: [conv_attn_layer_7, conv_kernel_layer_21], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf352, primals_8, buf346, buf353, 384, 512, grid=grid(384, 512), stream=stream0)
        buf354 = buf306; del buf306  # reuse
        # Source Nodes: [conv_kernel_layer_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf353, reinterpret_tensor(primals_180, (384, 54), (1, 384), 0), out=buf354)
        buf359 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf354, primals_181, buf359, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_181
        buf357 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf345, reinterpret_tensor(primals_182, (768, 384), (1, 768), 0), out=buf357)
        buf358 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf357, primals_183, buf358, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_183
        buf360 = reinterpret_tensor(buf357, (3072, 64, 1), (64, 1, 1), 0); del buf357  # reuse
        # Source Nodes: [conv_out_layer_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (3072, 64, 9), (576, 9, 1), 0), buf359, out=buf360)
        buf361 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf346, buf361, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf362 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf347, primals_175, buf362, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_175
        buf363 = reinterpret_tensor(buf347, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf348, primals_177, buf363, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_177
        buf364 = reinterpret_tensor(buf348, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf361, buf364, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf365 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf362, buf365, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf366 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf363, buf366, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf367 = aten._scaled_dot_product_efficient_attention(buf364, buf365, buf366, None, True, 0.1, scale=0.125)
        buf368 = buf367[0]
        buf369 = buf367[1]
        buf370 = buf367[2]
        buf371 = buf367[3]
        del buf367
        buf372 = reinterpret_tensor(buf366, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf368, buf372, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf373 = reinterpret_tensor(buf350, (512, 768), (768, 1), 0); del buf350  # reuse
        # Source Nodes: [hidden_states_64], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf368, buf360, buf373, 393216, grid=grid(393216), stream=stream0)
        buf374 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_185, buf373, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf374)
        del primals_185
        # Source Nodes: [hidden_states_65], Original ATen: [aten.native_dropout]
        buf375 = aten.native_dropout(reinterpret_tensor(buf374, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf376 = buf375[0]
        buf377 = buf375[1]
        del buf375
        buf381 = reinterpret_tensor(buf374, (1, 512, 768), (393216, 768, 1), 0); del buf374  # reuse
        buf382 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf609 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attention_output_14, hidden_states_63, hidden_states_67], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf376, buf344, primals_170, primals_171, primals_186, primals_187, buf381, buf382, buf609, 512, 768, grid=grid(512), stream=stream0)
        del primals_171
        buf383 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf382, reinterpret_tensor(primals_188, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf383)
        del primals_189
        buf384 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_69, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf383, buf384, 1572864, grid=grid(1572864), stream=stream0)
        buf385 = reinterpret_tensor(buf376, (512, 768), (768, 1), 0); del buf376  # reuse
        # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_191, buf384, reinterpret_tensor(primals_190, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf385)
        del primals_191
        # Source Nodes: [hidden_states_70], Original ATen: [aten.native_dropout]
        buf386 = aten.native_dropout(reinterpret_tensor(buf385, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf387 = buf386[0]
        buf388 = buf386[1]
        del buf386
        buf392 = reinterpret_tensor(buf385, (1, 512, 768), (393216, 768, 1), 0); del buf385  # reuse
        buf393 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf397 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf608 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, attention_output_14, hidden_states_72, mixed_query_layer_8, transpose_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf387, buf381, primals_186, primals_187, primals_192, primals_193, buf392, buf393, buf397, buf608, 512, 768, grid=grid(512), stream=stream0)
        del primals_187
        buf394 = reinterpret_tensor(buf368, (512, 384), (384, 1), 0); del buf368  # reuse
        # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_195, buf393, reinterpret_tensor(primals_194, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf394)
        del primals_195
        buf395 = reinterpret_tensor(buf360, (512, 384), (384, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_196, (768, 384), (1, 768), 0), out=buf395)
        buf396 = reinterpret_tensor(buf365, (512, 384), (384, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_198, (768, 384), (1, 768), 0), out=buf396)
        buf398 = reinterpret_tensor(buf387, (1, 768, 512), (393216, 512, 1), 0); del buf387  # reuse
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf397, buf398, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_200, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf399, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_201, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf400, (1, 384, 512), (196608, 512, 1))
        buf401 = reinterpret_tensor(buf364, (512, 384), (1, 512), 0); del buf364  # reuse
        # Source Nodes: [conv_attn_layer_8, conv_kernel_layer_24], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf400, primals_9, buf394, buf401, 384, 512, grid=grid(384, 512), stream=stream0)
        buf402 = buf354; del buf354  # reuse
        # Source Nodes: [conv_kernel_layer_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf401, reinterpret_tensor(primals_202, (384, 54), (1, 384), 0), out=buf402)
        buf407 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_26], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf402, primals_203, buf407, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_203
        buf405 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_204, (768, 384), (1, 768), 0), out=buf405)
        buf406 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf405, primals_205, buf406, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_205
        buf408 = reinterpret_tensor(buf405, (3072, 64, 1), (64, 1, 1), 0); del buf405  # reuse
        # Source Nodes: [conv_out_layer_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (3072, 64, 9), (576, 9, 1), 0), buf407, out=buf408)
        buf409 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf394, buf409, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf410 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf395, primals_197, buf410, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_197
        buf411 = reinterpret_tensor(buf395, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf396, primals_199, buf411, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_199
        buf412 = reinterpret_tensor(buf396, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf409, buf412, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf413 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf410, buf413, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf414 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf411, buf414, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf415 = aten._scaled_dot_product_efficient_attention(buf412, buf413, buf414, None, True, 0.1, scale=0.125)
        buf416 = buf415[0]
        buf417 = buf415[1]
        buf418 = buf415[2]
        buf419 = buf415[3]
        del buf415
        buf420 = reinterpret_tensor(buf414, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf416, buf420, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf421 = reinterpret_tensor(buf398, (512, 768), (768, 1), 0); del buf398  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf416, buf408, buf421, 393216, grid=grid(393216), stream=stream0)
        buf422 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_207, buf421, reinterpret_tensor(primals_206, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf422)
        del primals_207
        # Source Nodes: [hidden_states_74], Original ATen: [aten.native_dropout]
        buf423 = aten.native_dropout(reinterpret_tensor(buf422, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf424 = buf423[0]
        buf425 = buf423[1]
        del buf423
        buf429 = reinterpret_tensor(buf422, (1, 512, 768), (393216, 768, 1), 0); del buf422  # reuse
        buf430 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf607 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_16, hidden_states_72, hidden_states_76], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf424, buf392, primals_192, primals_193, primals_208, primals_209, buf429, buf430, buf607, 512, 768, grid=grid(512), stream=stream0)
        del primals_193
        buf431 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_211, buf430, reinterpret_tensor(primals_210, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf431)
        del primals_211
        buf432 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_78, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf431, buf432, 1572864, grid=grid(1572864), stream=stream0)
        buf433 = reinterpret_tensor(buf424, (512, 768), (768, 1), 0); del buf424  # reuse
        # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, buf432, reinterpret_tensor(primals_212, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf433)
        del primals_213
        # Source Nodes: [hidden_states_79], Original ATen: [aten.native_dropout]
        buf434 = aten.native_dropout(reinterpret_tensor(buf433, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf435 = buf434[0]
        buf436 = buf434[1]
        del buf434
        buf440 = reinterpret_tensor(buf433, (1, 512, 768), (393216, 768, 1), 0); del buf433  # reuse
        buf441 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf445 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf606 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, attention_output_16, hidden_states_81, mixed_query_layer_9, transpose_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf435, buf429, primals_208, primals_209, primals_214, primals_215, buf440, buf441, buf445, buf606, 512, 768, grid=grid(512), stream=stream0)
        del primals_209
        buf442 = reinterpret_tensor(buf416, (512, 384), (384, 1), 0); del buf416  # reuse
        # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_217, buf441, reinterpret_tensor(primals_216, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf442)
        del primals_217
        buf443 = reinterpret_tensor(buf408, (512, 384), (384, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf441, reinterpret_tensor(primals_218, (768, 384), (1, 768), 0), out=buf443)
        buf444 = reinterpret_tensor(buf413, (512, 384), (384, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf441, reinterpret_tensor(primals_220, (768, 384), (1, 768), 0), out=buf444)
        buf446 = reinterpret_tensor(buf435, (1, 768, 512), (393216, 512, 1), 0); del buf435  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf445, buf446, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_222, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf447, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_223, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf448, (1, 384, 512), (196608, 512, 1))
        buf449 = reinterpret_tensor(buf412, (512, 384), (1, 512), 0); del buf412  # reuse
        # Source Nodes: [conv_attn_layer_9, conv_kernel_layer_27], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf448, primals_10, buf442, buf449, 384, 512, grid=grid(384, 512), stream=stream0)
        buf450 = buf402; del buf402  # reuse
        # Source Nodes: [conv_kernel_layer_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf449, reinterpret_tensor(primals_224, (384, 54), (1, 384), 0), out=buf450)
        buf455 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf450, primals_225, buf455, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_225
        buf453 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf441, reinterpret_tensor(primals_226, (768, 384), (1, 768), 0), out=buf453)
        buf454 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf453, primals_227, buf454, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_227
        buf456 = reinterpret_tensor(buf453, (3072, 64, 1), (64, 1, 1), 0); del buf453  # reuse
        # Source Nodes: [conv_out_layer_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf454, (3072, 64, 9), (576, 9, 1), 0), buf455, out=buf456)
        buf457 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf442, buf457, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf458 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf443, primals_219, buf458, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_219
        buf459 = reinterpret_tensor(buf443, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf444, primals_221, buf459, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_221
        buf460 = reinterpret_tensor(buf444, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf457, buf460, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf461 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf458, buf461, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf462 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf459, buf462, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf463 = aten._scaled_dot_product_efficient_attention(buf460, buf461, buf462, None, True, 0.1, scale=0.125)
        buf464 = buf463[0]
        buf465 = buf463[1]
        buf466 = buf463[2]
        buf467 = buf463[3]
        del buf463
        buf468 = reinterpret_tensor(buf462, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf464, buf468, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf469 = reinterpret_tensor(buf446, (512, 768), (768, 1), 0); del buf446  # reuse
        # Source Nodes: [hidden_states_82], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf464, buf456, buf469, 393216, grid=grid(393216), stream=stream0)
        buf470 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_229, buf469, reinterpret_tensor(primals_228, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf470)
        del primals_229
        # Source Nodes: [hidden_states_83], Original ATen: [aten.native_dropout]
        buf471 = aten.native_dropout(reinterpret_tensor(buf470, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf472 = buf471[0]
        buf473 = buf471[1]
        del buf471
        buf477 = reinterpret_tensor(buf470, (1, 512, 768), (393216, 768, 1), 0); del buf470  # reuse
        buf478 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf605 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, attention_output_18, hidden_states_81, hidden_states_85], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf472, buf440, primals_214, primals_215, primals_230, primals_231, buf477, buf478, buf605, 512, 768, grid=grid(512), stream=stream0)
        del primals_215
        buf479 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_233, buf478, reinterpret_tensor(primals_232, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf479)
        del primals_233
        buf480 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_87, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf479, buf480, 1572864, grid=grid(1572864), stream=stream0)
        buf481 = reinterpret_tensor(buf472, (512, 768), (768, 1), 0); del buf472  # reuse
        # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_235, buf480, reinterpret_tensor(primals_234, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf481)
        del primals_235
        # Source Nodes: [hidden_states_88], Original ATen: [aten.native_dropout]
        buf482 = aten.native_dropout(reinterpret_tensor(buf481, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf483 = buf482[0]
        buf484 = buf482[1]
        del buf482
        buf488 = reinterpret_tensor(buf481, (1, 512, 768), (393216, 768, 1), 0); del buf481  # reuse
        buf489 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf493 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf604 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, attention_output_18, hidden_states_90, mixed_query_layer_10, transpose_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf483, buf477, primals_230, primals_231, primals_236, primals_237, buf488, buf489, buf493, buf604, 512, 768, grid=grid(512), stream=stream0)
        del primals_231
        buf490 = reinterpret_tensor(buf464, (512, 384), (384, 1), 0); del buf464  # reuse
        # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_239, buf489, reinterpret_tensor(primals_238, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf490)
        del primals_239
        buf491 = reinterpret_tensor(buf456, (512, 384), (384, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf489, reinterpret_tensor(primals_240, (768, 384), (1, 768), 0), out=buf491)
        buf492 = reinterpret_tensor(buf461, (512, 384), (384, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf489, reinterpret_tensor(primals_242, (768, 384), (1, 768), 0), out=buf492)
        buf494 = reinterpret_tensor(buf483, (1, 768, 512), (393216, 512, 1), 0); del buf483  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf493, buf494, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_244, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf495, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_245, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf496, (1, 384, 512), (196608, 512, 1))
        buf497 = reinterpret_tensor(buf460, (512, 384), (1, 512), 0); del buf460  # reuse
        # Source Nodes: [conv_attn_layer_10, conv_kernel_layer_30], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf496, primals_11, buf490, buf497, 384, 512, grid=grid(384, 512), stream=stream0)
        buf498 = buf450; del buf450  # reuse
        # Source Nodes: [conv_kernel_layer_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf497, reinterpret_tensor(primals_246, (384, 54), (1, 384), 0), out=buf498)
        buf503 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_32], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf498, primals_247, buf503, 3072, 9, grid=grid(3072), stream=stream0)
        del primals_247
        buf501 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf489, reinterpret_tensor(primals_248, (768, 384), (1, 768), 0), out=buf501)
        buf502 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf501, primals_249, buf502, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_249
        buf504 = reinterpret_tensor(buf501, (3072, 64, 1), (64, 1, 1), 0); del buf501  # reuse
        # Source Nodes: [conv_out_layer_86], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (3072, 64, 9), (576, 9, 1), 0), buf503, out=buf504)
        buf505 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf490, buf505, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf506 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf491, primals_241, buf506, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_241
        buf507 = reinterpret_tensor(buf491, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf492, primals_243, buf507, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_243
        buf508 = reinterpret_tensor(buf492, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf505, buf508, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf509 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf506, buf509, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf510 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf507, buf510, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf511 = aten._scaled_dot_product_efficient_attention(buf508, buf509, buf510, None, True, 0.1, scale=0.125)
        buf512 = buf511[0]
        buf513 = buf511[1]
        buf514 = buf511[2]
        buf515 = buf511[3]
        del buf511
        buf516 = reinterpret_tensor(buf510, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf512, buf516, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf517 = reinterpret_tensor(buf494, (512, 768), (768, 1), 0); del buf494  # reuse
        # Source Nodes: [hidden_states_91], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf512, buf504, buf517, 393216, grid=grid(393216), stream=stream0)
        buf518 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_251, buf517, reinterpret_tensor(primals_250, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf518)
        del primals_251
        # Source Nodes: [hidden_states_92], Original ATen: [aten.native_dropout]
        buf519 = aten.native_dropout(reinterpret_tensor(buf518, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf520 = buf519[0]
        buf521 = buf519[1]
        del buf519
        buf525 = reinterpret_tensor(buf518, (1, 512, 768), (393216, 768, 1), 0); del buf518  # reuse
        buf526 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf603 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attention_output_20, hidden_states_90, hidden_states_94], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf520, buf488, primals_236, primals_237, primals_252, primals_253, buf525, buf526, buf603, 512, 768, grid=grid(512), stream=stream0)
        del primals_237
        buf527 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_255, buf526, reinterpret_tensor(primals_254, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf527)
        del primals_255
        buf528 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_96, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf527, buf528, 1572864, grid=grid(1572864), stream=stream0)
        buf529 = reinterpret_tensor(buf520, (512, 768), (768, 1), 0); del buf520  # reuse
        # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_257, buf528, reinterpret_tensor(primals_256, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf529)
        del primals_257
        # Source Nodes: [hidden_states_97], Original ATen: [aten.native_dropout]
        buf530 = aten.native_dropout(reinterpret_tensor(buf529, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf531 = buf530[0]
        buf532 = buf530[1]
        del buf530
        buf536 = reinterpret_tensor(buf529, (1, 512, 768), (393216, 768, 1), 0); del buf529  # reuse
        buf537 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf541 = empty_strided((1, 768, 512), (393216, 1, 768), device='cuda', dtype=torch.float32)
        buf602 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, attention_output_20, hidden_states_99, mixed_query_layer_11, transpose_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.transpose, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf531, buf525, primals_252, primals_253, primals_258, primals_259, buf536, buf537, buf541, buf602, 512, 768, grid=grid(512), stream=stream0)
        del primals_253
        buf538 = reinterpret_tensor(buf512, (512, 384), (384, 1), 0); del buf512  # reuse
        # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_261, buf537, reinterpret_tensor(primals_260, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf538)
        del primals_261
        buf539 = reinterpret_tensor(buf504, (512, 384), (384, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf537, reinterpret_tensor(primals_262, (768, 384), (1, 768), 0), out=buf539)
        buf540 = reinterpret_tensor(buf509, (512, 384), (384, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf537, reinterpret_tensor(primals_264, (768, 384), (1, 768), 0), out=buf540)
        buf542 = reinterpret_tensor(buf531, (1, 768, 512), (393216, 512, 1), 0); del buf531  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf541, buf542, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_266, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf543, (1, 768, 512), (393216, 512, 1))
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_267, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf544, (1, 384, 512), (196608, 512, 1))
        buf545 = reinterpret_tensor(buf508, (512, 384), (1, 512), 0); del buf508  # reuse
        # Source Nodes: [conv_attn_layer_11, conv_kernel_layer_33], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_2.run(buf544, primals_12, buf538, buf545, 384, 512, grid=grid(384, 512), stream=stream0)
        buf546 = buf498; del buf498  # reuse
        # Source Nodes: [conv_kernel_layer_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf545, reinterpret_tensor(primals_268, (384, 54), (1, 384), 0), out=buf546)
        buf551 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf546, primals_269, buf551, 3072, 9, grid=grid(3072), stream=stream0)
        del buf546
        del primals_269
        buf549 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf537, reinterpret_tensor(primals_270, (768, 384), (1, 768), 0), out=buf549)
        buf550 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf549, primals_271, buf550, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del primals_271
        buf552 = reinterpret_tensor(buf549, (3072, 64, 1), (64, 1, 1), 0); del buf549  # reuse
        # Source Nodes: [conv_out_layer_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (3072, 64, 9), (576, 9, 1), 0), buf551, out=buf552)
        buf553 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf538, buf553, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf554 = empty_strided((1, 6, 512, 64), (196608, 1, 384, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf539, primals_263, buf554, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_263
        buf555 = reinterpret_tensor(buf539, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(buf540, primals_265, buf555, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del primals_265
        buf556 = reinterpret_tensor(buf540, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf553, buf556, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf557 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf554, buf557, 6, 32768, grid=grid(6, 32768), stream=stream0)
        buf558 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf555, buf558, 6, 32768, grid=grid(6, 32768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf559 = aten._scaled_dot_product_efficient_attention(buf556, buf557, buf558, None, True, 0.1, scale=0.125)
        del buf556
        del buf557
        buf560 = buf559[0]
        buf561 = buf559[1]
        buf562 = buf559[2]
        buf563 = buf559[3]
        del buf559
        buf564 = reinterpret_tensor(buf558, (1, 6, 512, 64), (196608, 1, 384, 6), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf560, buf564, 3072, 64, grid=grid(3072, 64), stream=stream0)
        buf565 = reinterpret_tensor(buf542, (512, 768), (768, 1), 0); del buf542  # reuse
        # Source Nodes: [hidden_states_100], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf560, buf552, buf565, 393216, grid=grid(393216), stream=stream0)
        del buf552
        del buf560
        buf566 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_273, buf565, reinterpret_tensor(primals_272, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf566)
        del primals_273
        # Source Nodes: [hidden_states_101], Original ATen: [aten.native_dropout]
        buf567 = aten.native_dropout(reinterpret_tensor(buf566, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf568 = buf567[0]
        buf569 = buf567[1]
        del buf567
        buf573 = reinterpret_tensor(buf566, (1, 512, 768), (393216, 768, 1), 0); del buf566  # reuse
        buf574 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf601 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_22, hidden_states_103, hidden_states_99], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf568, buf536, primals_258, primals_259, primals_274, primals_275, buf573, buf574, buf601, 512, 768, grid=grid(512), stream=stream0)
        del primals_259
        buf575 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_277, buf574, reinterpret_tensor(primals_276, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf575)
        del primals_277
        buf576 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_105, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf575, buf576, 1572864, grid=grid(1572864), stream=stream0)
        buf577 = reinterpret_tensor(buf568, (512, 768), (768, 1), 0); del buf568  # reuse
        # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_279, buf576, reinterpret_tensor(primals_278, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf577)
        del primals_279
        # Source Nodes: [hidden_states_106], Original ATen: [aten.native_dropout]
        buf578 = aten.native_dropout(reinterpret_tensor(buf577, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf579 = buf578[0]
        buf580 = buf578[1]
        del buf578
        buf584 = reinterpret_tensor(buf577, (1, 512, 768), (393216, 768, 1), 0); del buf577  # reuse
        buf585 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf600 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, attention_output_22, generator_sequence_output, hidden_states_109], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf579, buf573, primals_274, primals_275, primals_280, primals_281, buf584, buf585, buf600, 512, 768, grid=grid(512), stream=stream0)
        del primals_275
        del primals_281
        buf586 = reinterpret_tensor(buf579, (512, 768), (768, 1), 0); del buf579  # reuse
        # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_283, buf585, reinterpret_tensor(primals_282, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf586)
        del primals_283
        buf590 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf591 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf599 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_110, prediction_scores, prediction_scores_1], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15.run(buf586, primals_284, primals_285, buf590, buf591, buf599, 512, 768, grid=grid(512), stream=stream0)
        del primals_285
        buf592 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_287, buf591, reinterpret_tensor(primals_286, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf592)
        del primals_287
        buf595 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_16.run(buf592, buf595, 512, 30522, grid=grid(512), stream=stream0)
        buf598 = empty((), device='cuda', dtype=torch.float32)
        buf597 = empty((), device='cuda', dtype=torch.float32)
        buf625 = buf598; del buf598  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_17.run(buf625, primals_291, buf595, buf597, 1, 512, grid=grid(1), stream=stream0)
        return (buf625, reinterpret_tensor(buf592, (1, 512, 30522), (15627264, 30522, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, primals_288, primals_289, buf4, buf8, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), buf9, reinterpret_tensor(buf7, (1, 768, 512), (393216, 1, 768), 0), buf13, buf14, reinterpret_tensor(primals_26, (384, 54), (1, 384), 0), buf15, buf20, buf21, buf25, buf26, buf27, buf33, buf34, buf35, buf36, buf37, buf41, buf45, buf46, buf47, buf48, buf52, buf56, buf57, buf58, buf61, buf63, buf64, reinterpret_tensor(primals_48, (384, 54), (1, 384), 0), buf65, buf73, buf74, buf75, buf81, buf82, buf83, buf84, buf85, buf89, buf93, buf94, buf95, buf96, buf100, buf104, buf105, buf106, buf109, buf111, buf112, reinterpret_tensor(primals_70, (384, 54), (1, 384), 0), buf113, buf121, buf122, buf123, buf129, buf130, buf131, buf132, buf133, buf137, buf141, buf142, buf143, buf144, buf148, buf152, buf153, buf154, buf157, buf159, buf160, reinterpret_tensor(primals_92, (384, 54), (1, 384), 0), buf161, buf169, buf170, buf171, buf177, buf178, buf179, buf180, buf181, buf185, buf189, buf190, buf191, buf192, buf196, buf200, buf201, buf202, buf205, buf207, buf208, reinterpret_tensor(primals_114, (384, 54), (1, 384), 0), buf209, buf217, buf218, buf219, buf225, buf226, buf227, buf228, buf229, buf233, buf237, buf238, buf239, buf240, buf244, buf248, buf249, buf250, buf253, buf255, buf256, reinterpret_tensor(primals_136, (384, 54), (1, 384), 0), buf257, buf265, buf266, buf267, buf273, buf274, buf275, buf276, buf277, buf281, buf285, buf286, buf287, buf288, buf292, buf296, buf297, buf298, buf301, buf303, buf304, reinterpret_tensor(primals_158, (384, 54), (1, 384), 0), buf305, buf313, buf314, buf315, buf321, buf322, buf323, buf324, buf325, buf329, buf333, buf334, buf335, buf336, buf340, buf344, buf345, buf346, buf349, buf351, buf352, reinterpret_tensor(primals_180, (384, 54), (1, 384), 0), buf353, buf361, buf362, buf363, buf369, buf370, buf371, buf372, buf373, buf377, buf381, buf382, buf383, buf384, buf388, buf392, buf393, buf394, buf397, buf399, buf400, reinterpret_tensor(primals_202, (384, 54), (1, 384), 0), buf401, buf409, buf410, buf411, buf417, buf418, buf419, buf420, buf421, buf425, buf429, buf430, buf431, buf432, buf436, buf440, buf441, buf442, buf445, buf447, buf448, reinterpret_tensor(primals_224, (384, 54), (1, 384), 0), buf449, buf457, buf458, buf459, buf465, buf466, buf467, buf468, buf469, buf473, buf477, buf478, buf479, buf480, buf484, buf488, buf489, buf490, buf493, buf495, buf496, reinterpret_tensor(primals_246, (384, 54), (1, 384), 0), buf497, buf505, buf506, buf507, buf513, buf514, buf515, buf516, buf517, buf521, buf525, buf526, buf527, buf528, buf532, buf536, buf537, buf538, buf541, buf543, buf544, reinterpret_tensor(primals_268, (384, 54), (1, 384), 0), buf545, buf553, buf554, buf555, buf561, buf562, buf563, buf564, buf565, buf569, buf573, buf574, buf575, buf576, buf580, buf584, buf585, buf586, buf590, buf591, buf595, buf597, reinterpret_tensor(primals_286, (30522, 768), (768, 1), 0), buf599, reinterpret_tensor(primals_282, (768, 768), (768, 1), 0), buf600, reinterpret_tensor(primals_278, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_276, (3072, 768), (768, 1), 0), buf601, reinterpret_tensor(primals_272, (768, 768), (768, 1), 0), reinterpret_tensor(buf550, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf551, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_270, (384, 768), (768, 1), 0), buf551, reinterpret_tensor(primals_264, (384, 768), (768, 1), 0), reinterpret_tensor(primals_262, (384, 768), (768, 1), 0), reinterpret_tensor(primals_260, (384, 768), (768, 1), 0), buf602, reinterpret_tensor(primals_256, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_254, (3072, 768), (768, 1), 0), buf603, reinterpret_tensor(primals_250, (768, 768), (768, 1), 0), reinterpret_tensor(buf502, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf503, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_248, (384, 768), (768, 1), 0), buf503, reinterpret_tensor(primals_242, (384, 768), (768, 1), 0), reinterpret_tensor(primals_240, (384, 768), (768, 1), 0), reinterpret_tensor(primals_238, (384, 768), (768, 1), 0), buf604, reinterpret_tensor(primals_234, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_232, (3072, 768), (768, 1), 0), buf605, reinterpret_tensor(primals_228, (768, 768), (768, 1), 0), reinterpret_tensor(buf454, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf455, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_226, (384, 768), (768, 1), 0), buf455, reinterpret_tensor(primals_220, (384, 768), (768, 1), 0), reinterpret_tensor(primals_218, (384, 768), (768, 1), 0), reinterpret_tensor(primals_216, (384, 768), (768, 1), 0), buf606, reinterpret_tensor(primals_212, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_210, (3072, 768), (768, 1), 0), buf607, reinterpret_tensor(primals_206, (768, 768), (768, 1), 0), reinterpret_tensor(buf406, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf407, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_204, (384, 768), (768, 1), 0), buf407, reinterpret_tensor(primals_198, (384, 768), (768, 1), 0), reinterpret_tensor(primals_196, (384, 768), (768, 1), 0), reinterpret_tensor(primals_194, (384, 768), (768, 1), 0), buf608, reinterpret_tensor(primals_190, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_188, (3072, 768), (768, 1), 0), buf609, reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(buf358, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf359, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_182, (384, 768), (768, 1), 0), buf359, reinterpret_tensor(primals_176, (384, 768), (768, 1), 0), reinterpret_tensor(primals_174, (384, 768), (768, 1), 0), reinterpret_tensor(primals_172, (384, 768), (768, 1), 0), buf610, reinterpret_tensor(primals_168, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_166, (3072, 768), (768, 1), 0), buf611, reinterpret_tensor(primals_162, (768, 768), (768, 1), 0), reinterpret_tensor(buf310, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf311, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_160, (384, 768), (768, 1), 0), buf311, reinterpret_tensor(primals_154, (384, 768), (768, 1), 0), reinterpret_tensor(primals_152, (384, 768), (768, 1), 0), reinterpret_tensor(primals_150, (384, 768), (768, 1), 0), buf612, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf613, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(buf262, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf263, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_138, (384, 768), (768, 1), 0), buf263, reinterpret_tensor(primals_132, (384, 768), (768, 1), 0), reinterpret_tensor(primals_130, (384, 768), (768, 1), 0), reinterpret_tensor(primals_128, (384, 768), (768, 1), 0), buf614, reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), buf615, reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), reinterpret_tensor(buf214, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf215, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_116, (384, 768), (768, 1), 0), buf215, reinterpret_tensor(primals_110, (384, 768), (768, 1), 0), reinterpret_tensor(primals_108, (384, 768), (768, 1), 0), reinterpret_tensor(primals_106, (384, 768), (768, 1), 0), buf616, reinterpret_tensor(primals_102, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 768), (768, 1), 0), buf617, reinterpret_tensor(primals_96, (768, 768), (768, 1), 0), reinterpret_tensor(buf166, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf167, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_94, (384, 768), (768, 1), 0), buf167, reinterpret_tensor(primals_88, (384, 768), (768, 1), 0), reinterpret_tensor(primals_86, (384, 768), (768, 1), 0), reinterpret_tensor(primals_84, (384, 768), (768, 1), 0), buf618, reinterpret_tensor(primals_80, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_78, (3072, 768), (768, 1), 0), buf619, reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(buf118, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf119, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_72, (384, 768), (768, 1), 0), buf119, reinterpret_tensor(primals_66, (384, 768), (768, 1), 0), reinterpret_tensor(primals_64, (384, 768), (768, 1), 0), reinterpret_tensor(primals_62, (384, 768), (768, 1), 0), buf620, reinterpret_tensor(primals_58, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_56, (3072, 768), (768, 1), 0), buf621, reinterpret_tensor(primals_52, (768, 768), (768, 1), 0), reinterpret_tensor(buf70, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf71, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_50, (384, 768), (768, 1), 0), buf71, reinterpret_tensor(primals_44, (384, 768), (768, 1), 0), reinterpret_tensor(primals_42, (384, 768), (768, 1), 0), reinterpret_tensor(primals_40, (384, 768), (768, 1), 0), buf622, reinterpret_tensor(primals_36, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_34, (3072, 768), (768, 1), 0), buf623, reinterpret_tensor(primals_30, (768, 768), (768, 1), 0), reinterpret_tensor(buf22, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf23, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_28, (384, 768), (768, 1), 0), buf23, reinterpret_tensor(primals_22, (384, 768), (768, 1), 0), reinterpret_tensor(primals_20, (384, 768), (768, 1), 0), reinterpret_tensor(primals_18, (384, 768), (768, 1), 0), buf624, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
