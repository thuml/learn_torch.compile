
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


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4vimmcaejxfjpyyfbetorv7gfg4whyuqzczzsnnqsekj4v6dxy.py
# Source Nodes: [add, embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# embeddings => add_1
# embeddings_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# position_embeddings => embedding_1
# token_type_embeddings => embedding_2
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
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp43, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdjeu6gylxjpsuhjbuu7cmfphefzv6ibpvpnejd63ozezvulpgq.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57p5edi6yib5cdmjkvca2bekkxp7veth3xeukr46hiltexfxqtz.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csxxmofpi2tyojjaehraba67lnracootztlnai5gdr5hh24koncd.py
# Source Nodes: [conv_attn_layer], Original ATen: [aten.mul]
# conv_attn_layer => mul_3
triton_poi_fused_mul_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x1 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 * tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + (384*y0)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxhviqgixgb5oydn5gvznmiee5ngecf2pgr4dhz4bsea42ndlhx.py
# Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
# conv_kernel_layer_2 => amax, div, exp, sub_2, sum_1
triton_per_fused__softmax_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mr/cmry5k7bysyc6drn6m3igrvilrz7zuvikti3zeo5orr7afmejkqi.py
# Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
# conv_out_layer_5 => clone_2
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = (-4) + x2 + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-1536) + y3 + (384*x2)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x2 + (9*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmiy2iczmxz4vsmyksfsawdxifnhejujaer6ganxhg4nvp325bmg.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex % 768
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (384*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 12, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-384) + x3 + (384*x2)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cviibhwtyuwb2u5fjfazgps2twrqjzewwwi2mfqxwbgwnousyiqf.py
# Source Nodes: [add_3, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_3 => add_9
# attention_output => add_10, add_11, mul_4, mul_5, rsqrt_1, sub_4, var_mean_1
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxvkshc7eh3rivjzluz5qrgkprmmbh4kb4nzqrog3vkuq3rxcdb.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_12, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3abvwsz7qwjmvmwofozzkuc4mo4cl4h5ho3womn2dxqdn3mk6dm.py
# Source Nodes: [hidden_states_110, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm]
# hidden_states_110 => add_148, erf_12, mul_100, mul_101, mul_99
# prediction_scores => add_149, add_150, mul_102, mul_103, rsqrt_25, sub_50, var_mean_25
triton_per_fused_gelu_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kt/cktkd5yzb2tph6spqps7q4l7zhmfrrpcx4qawzujajvuc2tw52wu.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24, exp_24, sub_51, sum_25
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/po/cpozwl24d7qa3mh3lcome5optfur6gagqji45vfq5nafhmjt22xy.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_36, full_default_14, ne_1, ne_2, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1 = args
    args.clear()
    assert_size_stride(arg0_1, (384, 1), (1, 1))
    assert_size_stride(arg1_1, (384, 1), (1, 1))
    assert_size_stride(arg2_1, (384, 1), (1, 1))
    assert_size_stride(arg3_1, (384, 1), (1, 1))
    assert_size_stride(arg4_1, (384, 1), (1, 1))
    assert_size_stride(arg5_1, (384, 1), (1, 1))
    assert_size_stride(arg6_1, (384, 1), (1, 1))
    assert_size_stride(arg7_1, (384, 1), (1, 1))
    assert_size_stride(arg8_1, (384, 1), (1, 1))
    assert_size_stride(arg9_1, (384, 1), (1, 1))
    assert_size_stride(arg10_1, (384, 1), (1, 1))
    assert_size_stride(arg11_1, (384, 1), (1, 1))
    assert_size_stride(arg12_1, (30522, 768), (768, 1))
    assert_size_stride(arg13_1, (512, 768), (768, 1))
    assert_size_stride(arg14_1, (2, 768), (768, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (384, 768), (768, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (384, 768), (768, 1))
    assert_size_stride(arg20_1, (384, ), (1, ))
    assert_size_stride(arg21_1, (384, 768), (768, 1))
    assert_size_stride(arg22_1, (384, ), (1, ))
    assert_size_stride(arg23_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg24_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg25_1, (54, 384), (384, 1))
    assert_size_stride(arg26_1, (54, ), (1, ))
    assert_size_stride(arg27_1, (384, 768), (768, 1))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (768, 768), (768, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (3072, 768), (768, 1))
    assert_size_stride(arg34_1, (3072, ), (1, ))
    assert_size_stride(arg35_1, (768, 3072), (3072, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (384, 768), (768, 1))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, 768), (768, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (384, 768), (768, 1))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg46_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg47_1, (54, 384), (384, 1))
    assert_size_stride(arg48_1, (54, ), (1, ))
    assert_size_stride(arg49_1, (384, 768), (768, 1))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (768, 768), (768, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (3072, 768), (768, 1))
    assert_size_stride(arg56_1, (3072, ), (1, ))
    assert_size_stride(arg57_1, (768, 3072), (3072, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (384, 768), (768, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, 768), (768, 1))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, 768), (768, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg68_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg69_1, (54, 384), (384, 1))
    assert_size_stride(arg70_1, (54, ), (1, ))
    assert_size_stride(arg71_1, (384, 768), (768, 1))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (3072, 768), (768, 1))
    assert_size_stride(arg78_1, (3072, ), (1, ))
    assert_size_stride(arg79_1, (768, 3072), (3072, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (384, 768), (768, 1))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, 768), (768, 1))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, 768), (768, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg90_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg91_1, (54, 384), (384, 1))
    assert_size_stride(arg92_1, (54, ), (1, ))
    assert_size_stride(arg93_1, (384, 768), (768, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (768, 768), (768, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (3072, 768), (768, 1))
    assert_size_stride(arg100_1, (3072, ), (1, ))
    assert_size_stride(arg101_1, (768, 3072), (3072, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (384, 768), (768, 1))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, 768), (768, 1))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, 768), (768, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg112_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg113_1, (54, 384), (384, 1))
    assert_size_stride(arg114_1, (54, ), (1, ))
    assert_size_stride(arg115_1, (384, 768), (768, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (384, 768), (768, 1))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, 768), (768, 1))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, 768), (768, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg134_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg135_1, (54, 384), (384, 1))
    assert_size_stride(arg136_1, (54, ), (1, ))
    assert_size_stride(arg137_1, (384, 768), (768, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (384, 768), (768, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, 768), (768, 1))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, 768), (768, 1))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg156_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg157_1, (54, 384), (384, 1))
    assert_size_stride(arg158_1, (54, ), (1, ))
    assert_size_stride(arg159_1, (384, 768), (768, 1))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (768, 768), (768, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (3072, 768), (768, 1))
    assert_size_stride(arg166_1, (3072, ), (1, ))
    assert_size_stride(arg167_1, (768, 3072), (3072, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (384, 768), (768, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, 768), (768, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, 768), (768, 1))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg178_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg179_1, (54, 384), (384, 1))
    assert_size_stride(arg180_1, (54, ), (1, ))
    assert_size_stride(arg181_1, (384, 768), (768, 1))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (3072, 768), (768, 1))
    assert_size_stride(arg188_1, (3072, ), (1, ))
    assert_size_stride(arg189_1, (768, 3072), (3072, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (384, 768), (768, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, 768), (768, 1))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, 768), (768, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg200_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg201_1, (54, 384), (384, 1))
    assert_size_stride(arg202_1, (54, ), (1, ))
    assert_size_stride(arg203_1, (384, 768), (768, 1))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (768, 768), (768, 1))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (3072, 768), (768, 1))
    assert_size_stride(arg210_1, (3072, ), (1, ))
    assert_size_stride(arg211_1, (768, 3072), (3072, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (384, 768), (768, 1))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, 768), (768, 1))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, 768), (768, 1))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg222_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg223_1, (54, 384), (384, 1))
    assert_size_stride(arg224_1, (54, ), (1, ))
    assert_size_stride(arg225_1, (384, 768), (768, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (768, 768), (768, 1))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (768, ), (1, ))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (3072, 768), (768, 1))
    assert_size_stride(arg232_1, (3072, ), (1, ))
    assert_size_stride(arg233_1, (768, 3072), (3072, 1))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (384, 768), (768, 1))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, 768), (768, 1))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, 768), (768, 1))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg244_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg245_1, (54, 384), (384, 1))
    assert_size_stride(arg246_1, (54, ), (1, ))
    assert_size_stride(arg247_1, (384, 768), (768, 1))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (768, 768), (768, 1))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (3072, 768), (768, 1))
    assert_size_stride(arg254_1, (3072, ), (1, ))
    assert_size_stride(arg255_1, (768, 3072), (3072, 1))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (384, 768), (768, 1))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, 768), (768, 1))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (384, 768), (768, 1))
    assert_size_stride(arg264_1, (384, ), (1, ))
    assert_size_stride(arg265_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg266_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg267_1, (54, 384), (384, 1))
    assert_size_stride(arg268_1, (54, ), (1, ))
    assert_size_stride(arg269_1, (384, 768), (768, 1))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (768, 768), (768, 1))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (3072, 768), (768, 1))
    assert_size_stride(arg276_1, (3072, ), (1, ))
    assert_size_stride(arg277_1, (768, 3072), (3072, 1))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (768, ), (1, ))
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (768, 768), (768, 1))
    assert_size_stride(arg282_1, (768, ), (1, ))
    assert_size_stride(arg283_1, (768, ), (1, ))
    assert_size_stride(arg284_1, (768, ), (1, ))
    assert_size_stride(arg285_1, (30522, 768), (768, 1))
    assert_size_stride(arg286_1, (30522, ), (1, ))
    assert_size_stride(arg287_1, (1, 512), (512, 1))
    assert_size_stride(arg288_1, (1, 512), (512, 1))
    assert_size_stride(arg289_1, (1, 512), (512, 1))
    assert_size_stride(arg290_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg289_1, arg12_1, arg288_1, arg13_1, arg287_1, arg14_1, arg15_1, arg16_1, buf0, buf4, 512, 768, grid=grid(512), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        del arg287_1
        del arg288_1
        del arg289_1
        buf5 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 384), (1, 768), 0), out=buf5)
        del arg17_1
        buf6 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 384), (1, 768), 0), out=buf6)
        del arg19_1
        buf7 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 384), (1, 768), 0), out=buf7)
        del arg21_1
        buf8 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf5, arg18_1, buf8, 196608, grid=grid(196608), stream=stream0)
        buf9 = empty((1, 6, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg20_1, buf9, 196608, grid=grid(196608), stream=stream0)
        del arg20_1
        buf10 = reinterpret_tensor(buf6, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg22_1, buf10, 196608, grid=grid(196608), stream=stream0)
        del arg22_1
        del buf7
        # Source Nodes: [], Original ATen: []
        buf11 = aten._scaled_dot_product_efficient_attention(buf8, buf9, buf10, None, False, scale=0.125)
        del buf10
        buf12 = buf11[0]
        del buf11
        buf16 = reinterpret_tensor(buf9, (512, 384), (384, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 384), (1, 768), 0), out=buf16)
        del arg27_1
        buf17 = reinterpret_tensor(buf0, (1, 768, 512), (393216, 512, 1), 0); del buf0  # reuse
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf4, buf17, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg23_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf18, (1, 768, 512), (393216, 512, 1))
        del arg23_1
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg24_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf19, (1, 384, 512), (196608, 512, 1))
        del arg24_1
        buf20 = reinterpret_tensor(buf5, (1, 512, 384), (196608, 384, 1), 0); del buf5  # reuse
        # Source Nodes: [conv_attn_layer], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf20, buf19, arg0_1, arg18_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg0_1
        del arg18_1
        buf21 = empty((512, 54), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 54), (1, 384), 0), out=buf21)
        del arg25_1
        buf25 = empty_strided((3072, 9, 1), (9, 1, 27648), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf21, arg26_1, buf25, 3072, 9, grid=grid(3072), stream=stream0)
        del arg26_1
        buf24 = empty((1, 512, 384, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf16, arg28_1, buf24, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg28_1
        buf26 = reinterpret_tensor(buf16, (3072, 64, 1), (64, 1, 1), 0); del buf16  # reuse
        # Source Nodes: [conv_kernel_layer_2, conv_out_layer_6], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (3072, 64, 9), (576, 9, 1), 0), buf25, out=buf26)
        buf27 = reinterpret_tensor(buf18, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf18  # reuse
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf12, buf26, buf27, 393216, grid=grid(393216), stream=stream0)
        buf28 = reinterpret_tensor(buf17, (512, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), out=buf28)
        del arg29_1
        buf32 = reinterpret_tensor(buf27, (1, 512, 768), (393216, 768, 1), 0); del buf27  # reuse
        # Source Nodes: [add_3, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf28, arg30_1, buf4, arg31_1, arg32_1, buf32, 512, 768, grid=grid(512), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        buf33 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (512, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 3072), (1, 768), 0), out=buf33)
        del arg33_1
        buf34 = reinterpret_tensor(buf33, (1, 512, 3072), (1572864, 3072, 1), 0); del buf33  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf34, arg34_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg34_1
        buf35 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg35_1, (3072, 768), (1, 3072), 0), out=buf35)
        del arg35_1
        buf39 = reinterpret_tensor(buf28, (1, 512, 768), (393216, 768, 1), 0); del buf28  # reuse
        # Source Nodes: [add_4, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf35, arg36_1, buf32, arg37_1, arg38_1, buf39, 512, 768, grid=grid(512), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        del buf32
        buf40 = reinterpret_tensor(buf26, (512, 384), (384, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 384), (1, 768), 0), out=buf40)
        del arg39_1
        buf41 = reinterpret_tensor(buf12, (512, 384), (384, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 384), (1, 768), 0), out=buf41)
        del arg41_1
        buf42 = reinterpret_tensor(buf20, (512, 384), (384, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 384), (1, 768), 0), out=buf42)
        del arg43_1
        buf43 = reinterpret_tensor(buf19, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf40, arg40_1, buf43, 196608, grid=grid(196608), stream=stream0)
        buf44 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf41, arg42_1, buf44, 196608, grid=grid(196608), stream=stream0)
        del arg42_1
        buf45 = reinterpret_tensor(buf41, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf42, arg44_1, buf45, 196608, grid=grid(196608), stream=stream0)
        del arg44_1
        del buf42
        # Source Nodes: [], Original ATen: []
        buf46 = aten._scaled_dot_product_efficient_attention(buf43, buf44, buf45, None, False, scale=0.125)
        del buf43
        buf47 = buf46[0]
        del buf46
        buf51 = reinterpret_tensor(buf45, (512, 384), (384, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 384), (1, 768), 0), out=buf51)
        del arg49_1
        buf52 = reinterpret_tensor(buf35, (1, 768, 512), (393216, 512, 1), 0); del buf35  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf39, buf52, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg45_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf53, (1, 768, 512), (393216, 512, 1))
        del arg45_1
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg46_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf54, (1, 384, 512), (196608, 512, 1))
        del arg46_1
        buf55 = reinterpret_tensor(buf40, (1, 512, 384), (196608, 384, 1), 0); del buf40  # reuse
        # Source Nodes: [conv_attn_layer_1], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf55, buf54, arg1_1, arg40_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg1_1
        del arg40_1
        buf56 = reinterpret_tensor(buf25, (512, 54), (54, 1), 0); del buf25  # reuse
        # Source Nodes: [conv_kernel_layer_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (512, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 54), (1, 384), 0), out=buf56)
        del arg47_1
        buf60 = reinterpret_tensor(buf21, (3072, 9, 1), (9, 1, 27648), 0); del buf21  # reuse
        # Source Nodes: [conv_kernel_layer_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf56, arg48_1, buf60, 3072, 9, grid=grid(3072), stream=stream0)
        del arg48_1
        buf59 = buf24; del buf24  # reuse
        # Source Nodes: [conv_out_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf51, arg50_1, buf59, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg50_1
        buf61 = reinterpret_tensor(buf51, (3072, 64, 1), (64, 1, 1), 0); del buf51  # reuse
        # Source Nodes: [conv_kernel_layer_5, conv_out_layer_14], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (3072, 64, 9), (576, 9, 1), 0), buf60, out=buf61)
        buf62 = reinterpret_tensor(buf53, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf47, buf61, buf62, 393216, grid=grid(393216), stream=stream0)
        buf63 = reinterpret_tensor(buf52, (512, 768), (768, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (512, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 768), (1, 768), 0), out=buf63)
        del arg51_1
        buf67 = reinterpret_tensor(buf62, (1, 512, 768), (393216, 768, 1), 0); del buf62  # reuse
        # Source Nodes: [add_6, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf63, arg52_1, buf39, arg53_1, arg54_1, buf67, 512, 768, grid=grid(512), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        buf68 = reinterpret_tensor(buf34, (512, 3072), (3072, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 3072), (1, 768), 0), out=buf68)
        del arg55_1
        buf69 = reinterpret_tensor(buf68, (1, 512, 3072), (1572864, 3072, 1), 0); del buf68  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf69, arg56_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg56_1
        buf70 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg57_1, (3072, 768), (1, 3072), 0), out=buf70)
        del arg57_1
        buf74 = buf39; del buf39  # reuse
        # Source Nodes: [add_7, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf70, arg58_1, buf67, arg59_1, arg60_1, buf74, 512, 768, grid=grid(512), stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        del buf67
        buf75 = reinterpret_tensor(buf61, (512, 384), (384, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 384), (1, 768), 0), out=buf75)
        del arg61_1
        buf76 = reinterpret_tensor(buf47, (512, 384), (384, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 384), (1, 768), 0), out=buf76)
        del arg63_1
        buf77 = reinterpret_tensor(buf55, (512, 384), (384, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 768), (768, 1), 0), reinterpret_tensor(arg65_1, (768, 384), (1, 768), 0), out=buf77)
        del arg65_1
        buf78 = reinterpret_tensor(buf54, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf75, arg62_1, buf78, 196608, grid=grid(196608), stream=stream0)
        buf79 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg64_1, buf79, 196608, grid=grid(196608), stream=stream0)
        del arg64_1
        buf80 = reinterpret_tensor(buf76, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf77, arg66_1, buf80, 196608, grid=grid(196608), stream=stream0)
        del arg66_1
        del buf77
        # Source Nodes: [], Original ATen: []
        buf81 = aten._scaled_dot_product_efficient_attention(buf78, buf79, buf80, None, False, scale=0.125)
        del buf78
        buf82 = buf81[0]
        del buf81
        buf86 = reinterpret_tensor(buf80, (512, 384), (384, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 384), (1, 768), 0), out=buf86)
        del arg71_1
        buf87 = reinterpret_tensor(buf70, (1, 768, 512), (393216, 512, 1), 0); del buf70  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf74, buf87, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg67_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf88, (1, 768, 512), (393216, 512, 1))
        del arg67_1
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg68_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf89, (1, 384, 512), (196608, 512, 1))
        del arg68_1
        buf90 = reinterpret_tensor(buf75, (1, 512, 384), (196608, 384, 1), 0); del buf75  # reuse
        # Source Nodes: [conv_attn_layer_2], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf90, buf89, arg2_1, arg62_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg2_1
        del arg62_1
        buf91 = reinterpret_tensor(buf60, (512, 54), (54, 1), 0); del buf60  # reuse
        # Source Nodes: [conv_kernel_layer_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 384), (384, 1), 0), reinterpret_tensor(arg69_1, (384, 54), (1, 384), 0), out=buf91)
        del arg69_1
        buf95 = reinterpret_tensor(buf56, (3072, 9, 1), (9, 1, 27648), 0); del buf56  # reuse
        # Source Nodes: [conv_kernel_layer_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf91, arg70_1, buf95, 3072, 9, grid=grid(3072), stream=stream0)
        del arg70_1
        buf94 = buf59; del buf59  # reuse
        # Source Nodes: [conv_out_layer_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf86, arg72_1, buf94, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg72_1
        buf96 = reinterpret_tensor(buf86, (3072, 64, 1), (64, 1, 1), 0); del buf86  # reuse
        # Source Nodes: [conv_kernel_layer_8, conv_out_layer_22], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (3072, 64, 9), (576, 9, 1), 0), buf95, out=buf96)
        buf97 = reinterpret_tensor(buf88, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf88  # reuse
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf82, buf96, buf97, 393216, grid=grid(393216), stream=stream0)
        buf98 = reinterpret_tensor(buf87, (512, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf98)
        del arg73_1
        buf102 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        # Source Nodes: [add_9, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf98, arg74_1, buf74, arg75_1, arg76_1, buf102, 512, 768, grid=grid(512), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        buf103 = reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 3072), (1, 768), 0), out=buf103)
        del arg77_1
        buf104 = reinterpret_tensor(buf103, (1, 512, 3072), (1572864, 3072, 1), 0); del buf103  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf104, arg78_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg78_1
        buf105 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg79_1, (3072, 768), (1, 3072), 0), out=buf105)
        del arg79_1
        buf109 = buf74; del buf74  # reuse
        # Source Nodes: [add_10, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf105, arg80_1, buf102, arg81_1, arg82_1, buf109, 512, 768, grid=grid(512), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        del buf102
        buf110 = reinterpret_tensor(buf96, (512, 384), (384, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 384), (1, 768), 0), out=buf110)
        del arg83_1
        buf111 = reinterpret_tensor(buf82, (512, 384), (384, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 384), (1, 768), 0), out=buf111)
        del arg85_1
        buf112 = reinterpret_tensor(buf90, (512, 384), (384, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 384), (1, 768), 0), out=buf112)
        del arg87_1
        buf113 = reinterpret_tensor(buf89, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf110, arg84_1, buf113, 196608, grid=grid(196608), stream=stream0)
        buf114 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf111, arg86_1, buf114, 196608, grid=grid(196608), stream=stream0)
        del arg86_1
        buf115 = reinterpret_tensor(buf111, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf112, arg88_1, buf115, 196608, grid=grid(196608), stream=stream0)
        del arg88_1
        del buf112
        # Source Nodes: [], Original ATen: []
        buf116 = aten._scaled_dot_product_efficient_attention(buf113, buf114, buf115, None, False, scale=0.125)
        del buf113
        buf117 = buf116[0]
        del buf116
        buf121 = reinterpret_tensor(buf115, (512, 384), (384, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 384), (1, 768), 0), out=buf121)
        del arg93_1
        buf122 = reinterpret_tensor(buf105, (1, 768, 512), (393216, 512, 1), 0); del buf105  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf109, buf122, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg89_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf123, (1, 768, 512), (393216, 512, 1))
        del arg89_1
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg90_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf124, (1, 384, 512), (196608, 512, 1))
        del arg90_1
        buf125 = reinterpret_tensor(buf110, (1, 512, 384), (196608, 384, 1), 0); del buf110  # reuse
        # Source Nodes: [conv_attn_layer_3], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf125, buf124, arg3_1, arg84_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg3_1
        del arg84_1
        buf126 = reinterpret_tensor(buf95, (512, 54), (54, 1), 0); del buf95  # reuse
        # Source Nodes: [conv_kernel_layer_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 54), (1, 384), 0), out=buf126)
        del arg91_1
        buf130 = reinterpret_tensor(buf91, (3072, 9, 1), (9, 1, 27648), 0); del buf91  # reuse
        # Source Nodes: [conv_kernel_layer_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf126, arg92_1, buf130, 3072, 9, grid=grid(3072), stream=stream0)
        del arg92_1
        buf129 = buf94; del buf94  # reuse
        # Source Nodes: [conv_out_layer_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf121, arg94_1, buf129, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg94_1
        buf131 = reinterpret_tensor(buf121, (3072, 64, 1), (64, 1, 1), 0); del buf121  # reuse
        # Source Nodes: [conv_kernel_layer_11, conv_out_layer_30], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (3072, 64, 9), (576, 9, 1), 0), buf130, out=buf131)
        buf132 = reinterpret_tensor(buf123, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf123  # reuse
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf117, buf131, buf132, 393216, grid=grid(393216), stream=stream0)
        buf133 = reinterpret_tensor(buf122, (512, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (512, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), out=buf133)
        del arg95_1
        buf137 = reinterpret_tensor(buf132, (1, 512, 768), (393216, 768, 1), 0); del buf132  # reuse
        # Source Nodes: [add_12, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf133, arg96_1, buf109, arg97_1, arg98_1, buf137, 512, 768, grid=grid(512), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf138 = reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (512, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), out=buf138)
        del arg99_1
        buf139 = reinterpret_tensor(buf138, (1, 512, 3072), (1572864, 3072, 1), 0); del buf138  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf139, arg100_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg100_1
        buf140 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), out=buf140)
        del arg101_1
        buf144 = buf109; del buf109  # reuse
        # Source Nodes: [add_13, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf140, arg102_1, buf137, arg103_1, arg104_1, buf144, 512, 768, grid=grid(512), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del buf137
        buf145 = reinterpret_tensor(buf131, (512, 384), (384, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 384), (1, 768), 0), out=buf145)
        del arg105_1
        buf146 = reinterpret_tensor(buf117, (512, 384), (384, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 384), (1, 768), 0), out=buf146)
        del arg107_1
        buf147 = reinterpret_tensor(buf125, (512, 384), (384, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 384), (1, 768), 0), out=buf147)
        del arg109_1
        buf148 = reinterpret_tensor(buf124, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf145, arg106_1, buf148, 196608, grid=grid(196608), stream=stream0)
        buf149 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf146, arg108_1, buf149, 196608, grid=grid(196608), stream=stream0)
        del arg108_1
        buf150 = reinterpret_tensor(buf146, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf147, arg110_1, buf150, 196608, grid=grid(196608), stream=stream0)
        del arg110_1
        del buf147
        # Source Nodes: [], Original ATen: []
        buf151 = aten._scaled_dot_product_efficient_attention(buf148, buf149, buf150, None, False, scale=0.125)
        del buf148
        buf152 = buf151[0]
        del buf151
        buf156 = reinterpret_tensor(buf150, (512, 384), (384, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 384), (1, 768), 0), out=buf156)
        del arg115_1
        buf157 = reinterpret_tensor(buf140, (1, 768, 512), (393216, 512, 1), 0); del buf140  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf144, buf157, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg111_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf158, (1, 768, 512), (393216, 512, 1))
        del arg111_1
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg112_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf159, (1, 384, 512), (196608, 512, 1))
        del arg112_1
        buf160 = reinterpret_tensor(buf145, (1, 512, 384), (196608, 384, 1), 0); del buf145  # reuse
        # Source Nodes: [conv_attn_layer_4], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf160, buf159, arg4_1, arg106_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg106_1
        del arg4_1
        buf161 = reinterpret_tensor(buf130, (512, 54), (54, 1), 0); del buf130  # reuse
        # Source Nodes: [conv_kernel_layer_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 54), (1, 384), 0), out=buf161)
        del arg113_1
        buf165 = reinterpret_tensor(buf126, (3072, 9, 1), (9, 1, 27648), 0); del buf126  # reuse
        # Source Nodes: [conv_kernel_layer_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf161, arg114_1, buf165, 3072, 9, grid=grid(3072), stream=stream0)
        del arg114_1
        buf164 = buf129; del buf129  # reuse
        # Source Nodes: [conv_out_layer_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf156, arg116_1, buf164, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg116_1
        buf166 = reinterpret_tensor(buf156, (3072, 64, 1), (64, 1, 1), 0); del buf156  # reuse
        # Source Nodes: [conv_kernel_layer_14, conv_out_layer_38], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (3072, 64, 9), (576, 9, 1), 0), buf165, out=buf166)
        buf167 = reinterpret_tensor(buf158, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf158  # reuse
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf152, buf166, buf167, 393216, grid=grid(393216), stream=stream0)
        buf168 = reinterpret_tensor(buf157, (512, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (512, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf168)
        del arg117_1
        buf172 = reinterpret_tensor(buf167, (1, 512, 768), (393216, 768, 1), 0); del buf167  # reuse
        # Source Nodes: [add_15, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf168, arg118_1, buf144, arg119_1, arg120_1, buf172, 512, 768, grid=grid(512), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        buf173 = reinterpret_tensor(buf139, (512, 3072), (3072, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), out=buf173)
        del arg121_1
        buf174 = reinterpret_tensor(buf173, (1, 512, 3072), (1572864, 3072, 1), 0); del buf173  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf174, arg122_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg122_1
        buf175 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), out=buf175)
        del arg123_1
        buf179 = buf144; del buf144  # reuse
        # Source Nodes: [add_16, hidden_states_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf175, arg124_1, buf172, arg125_1, arg126_1, buf179, 512, 768, grid=grid(512), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        del buf172
        buf180 = reinterpret_tensor(buf166, (512, 384), (384, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 384), (1, 768), 0), out=buf180)
        del arg127_1
        buf181 = reinterpret_tensor(buf152, (512, 384), (384, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), reinterpret_tensor(arg129_1, (768, 384), (1, 768), 0), out=buf181)
        del arg129_1
        buf182 = reinterpret_tensor(buf160, (512, 384), (384, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 384), (1, 768), 0), out=buf182)
        del arg131_1
        buf183 = reinterpret_tensor(buf159, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf180, arg128_1, buf183, 196608, grid=grid(196608), stream=stream0)
        buf184 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf181, arg130_1, buf184, 196608, grid=grid(196608), stream=stream0)
        del arg130_1
        buf185 = reinterpret_tensor(buf181, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf182, arg132_1, buf185, 196608, grid=grid(196608), stream=stream0)
        del arg132_1
        del buf182
        # Source Nodes: [], Original ATen: []
        buf186 = aten._scaled_dot_product_efficient_attention(buf183, buf184, buf185, None, False, scale=0.125)
        del buf183
        buf187 = buf186[0]
        del buf186
        buf191 = reinterpret_tensor(buf185, (512, 384), (384, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 384), (1, 768), 0), out=buf191)
        del arg137_1
        buf192 = reinterpret_tensor(buf175, (1, 768, 512), (393216, 512, 1), 0); del buf175  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf179, buf192, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg133_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf193, (1, 768, 512), (393216, 512, 1))
        del arg133_1
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg134_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf194, (1, 384, 512), (196608, 512, 1))
        del arg134_1
        buf195 = reinterpret_tensor(buf180, (1, 512, 384), (196608, 384, 1), 0); del buf180  # reuse
        # Source Nodes: [conv_attn_layer_5], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf195, buf194, arg5_1, arg128_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg128_1
        del arg5_1
        buf196 = reinterpret_tensor(buf165, (512, 54), (54, 1), 0); del buf165  # reuse
        # Source Nodes: [conv_kernel_layer_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 384), (384, 1), 0), reinterpret_tensor(arg135_1, (384, 54), (1, 384), 0), out=buf196)
        del arg135_1
        buf200 = reinterpret_tensor(buf161, (3072, 9, 1), (9, 1, 27648), 0); del buf161  # reuse
        # Source Nodes: [conv_kernel_layer_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf196, arg136_1, buf200, 3072, 9, grid=grid(3072), stream=stream0)
        del arg136_1
        buf199 = buf164; del buf164  # reuse
        # Source Nodes: [conv_out_layer_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf191, arg138_1, buf199, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg138_1
        buf201 = reinterpret_tensor(buf191, (3072, 64, 1), (64, 1, 1), 0); del buf191  # reuse
        # Source Nodes: [conv_kernel_layer_17, conv_out_layer_46], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (3072, 64, 9), (576, 9, 1), 0), buf200, out=buf201)
        buf202 = reinterpret_tensor(buf193, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf193  # reuse
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf187, buf201, buf202, 393216, grid=grid(393216), stream=stream0)
        buf203 = reinterpret_tensor(buf192, (512, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf203)
        del arg139_1
        buf207 = reinterpret_tensor(buf202, (1, 512, 768), (393216, 768, 1), 0); del buf202  # reuse
        # Source Nodes: [add_18, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf203, arg140_1, buf179, arg141_1, arg142_1, buf207, 512, 768, grid=grid(512), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        buf208 = reinterpret_tensor(buf174, (512, 3072), (3072, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), out=buf208)
        del arg143_1
        buf209 = reinterpret_tensor(buf208, (1, 512, 3072), (1572864, 3072, 1), 0); del buf208  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf209, arg144_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg144_1
        buf210 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), out=buf210)
        del arg145_1
        buf214 = buf179; del buf179  # reuse
        # Source Nodes: [add_19, hidden_states_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf210, arg146_1, buf207, arg147_1, arg148_1, buf214, 512, 768, grid=grid(512), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        del buf207
        buf215 = reinterpret_tensor(buf201, (512, 384), (384, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 384), (1, 768), 0), out=buf215)
        del arg149_1
        buf216 = reinterpret_tensor(buf187, (512, 384), (384, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 384), (1, 768), 0), out=buf216)
        del arg151_1
        buf217 = reinterpret_tensor(buf195, (512, 384), (384, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 384), (1, 768), 0), out=buf217)
        del arg153_1
        buf218 = reinterpret_tensor(buf194, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf215, arg150_1, buf218, 196608, grid=grid(196608), stream=stream0)
        buf219 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf216, arg152_1, buf219, 196608, grid=grid(196608), stream=stream0)
        del arg152_1
        buf220 = reinterpret_tensor(buf216, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf217, arg154_1, buf220, 196608, grid=grid(196608), stream=stream0)
        del arg154_1
        del buf217
        # Source Nodes: [], Original ATen: []
        buf221 = aten._scaled_dot_product_efficient_attention(buf218, buf219, buf220, None, False, scale=0.125)
        del buf218
        buf222 = buf221[0]
        del buf221
        buf226 = reinterpret_tensor(buf220, (512, 384), (384, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 384), (1, 768), 0), out=buf226)
        del arg159_1
        buf227 = reinterpret_tensor(buf210, (1, 768, 512), (393216, 512, 1), 0); del buf210  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf214, buf227, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, arg155_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf228, (1, 768, 512), (393216, 512, 1))
        del arg155_1
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, arg156_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf229, (1, 384, 512), (196608, 512, 1))
        del arg156_1
        buf230 = reinterpret_tensor(buf215, (1, 512, 384), (196608, 384, 1), 0); del buf215  # reuse
        # Source Nodes: [conv_attn_layer_6], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf230, buf229, arg6_1, arg150_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg150_1
        del arg6_1
        buf231 = reinterpret_tensor(buf200, (512, 54), (54, 1), 0); del buf200  # reuse
        # Source Nodes: [conv_kernel_layer_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 384), (384, 1), 0), reinterpret_tensor(arg157_1, (384, 54), (1, 384), 0), out=buf231)
        del arg157_1
        buf235 = reinterpret_tensor(buf196, (3072, 9, 1), (9, 1, 27648), 0); del buf196  # reuse
        # Source Nodes: [conv_kernel_layer_20], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf231, arg158_1, buf235, 3072, 9, grid=grid(3072), stream=stream0)
        del arg158_1
        buf234 = buf199; del buf199  # reuse
        # Source Nodes: [conv_out_layer_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf226, arg160_1, buf234, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg160_1
        buf236 = reinterpret_tensor(buf226, (3072, 64, 1), (64, 1, 1), 0); del buf226  # reuse
        # Source Nodes: [conv_kernel_layer_20, conv_out_layer_54], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf234, (3072, 64, 9), (576, 9, 1), 0), buf235, out=buf236)
        buf237 = reinterpret_tensor(buf228, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf222, buf236, buf237, 393216, grid=grid(393216), stream=stream0)
        buf238 = reinterpret_tensor(buf227, (512, 768), (768, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 768), (1, 768), 0), out=buf238)
        del arg161_1
        buf242 = reinterpret_tensor(buf237, (1, 512, 768), (393216, 768, 1), 0); del buf237  # reuse
        # Source Nodes: [add_21, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf238, arg162_1, buf214, arg163_1, arg164_1, buf242, 512, 768, grid=grid(512), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        buf243 = reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 3072), (1, 768), 0), out=buf243)
        del arg165_1
        buf244 = reinterpret_tensor(buf243, (1, 512, 3072), (1572864, 3072, 1), 0); del buf243  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf244, arg166_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg166_1
        buf245 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg167_1, (3072, 768), (1, 3072), 0), out=buf245)
        del arg167_1
        buf249 = buf214; del buf214  # reuse
        # Source Nodes: [add_22, hidden_states_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf245, arg168_1, buf242, arg169_1, arg170_1, buf249, 512, 768, grid=grid(512), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del buf242
        buf250 = reinterpret_tensor(buf236, (512, 384), (384, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 384), (1, 768), 0), out=buf250)
        del arg171_1
        buf251 = reinterpret_tensor(buf222, (512, 384), (384, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 384), (1, 768), 0), out=buf251)
        del arg173_1
        buf252 = reinterpret_tensor(buf230, (512, 384), (384, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 384), (1, 768), 0), out=buf252)
        del arg175_1
        buf253 = reinterpret_tensor(buf229, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf250, arg172_1, buf253, 196608, grid=grid(196608), stream=stream0)
        buf254 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf251, arg174_1, buf254, 196608, grid=grid(196608), stream=stream0)
        del arg174_1
        buf255 = reinterpret_tensor(buf251, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf252, arg176_1, buf255, 196608, grid=grid(196608), stream=stream0)
        del arg176_1
        del buf252
        # Source Nodes: [], Original ATen: []
        buf256 = aten._scaled_dot_product_efficient_attention(buf253, buf254, buf255, None, False, scale=0.125)
        del buf253
        buf257 = buf256[0]
        del buf256
        buf261 = reinterpret_tensor(buf255, (512, 384), (384, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 384), (1, 768), 0), out=buf261)
        del arg181_1
        buf262 = reinterpret_tensor(buf245, (1, 768, 512), (393216, 512, 1), 0); del buf245  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf249, buf262, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg177_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf263, (1, 768, 512), (393216, 512, 1))
        del arg177_1
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, arg178_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf264, (1, 384, 512), (196608, 512, 1))
        del arg178_1
        buf265 = reinterpret_tensor(buf250, (1, 512, 384), (196608, 384, 1), 0); del buf250  # reuse
        # Source Nodes: [conv_attn_layer_7], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf265, buf264, arg7_1, arg172_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg172_1
        del arg7_1
        buf266 = reinterpret_tensor(buf235, (512, 54), (54, 1), 0); del buf235  # reuse
        # Source Nodes: [conv_kernel_layer_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (512, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 54), (1, 384), 0), out=buf266)
        del arg179_1
        buf270 = reinterpret_tensor(buf231, (3072, 9, 1), (9, 1, 27648), 0); del buf231  # reuse
        # Source Nodes: [conv_kernel_layer_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf266, arg180_1, buf270, 3072, 9, grid=grid(3072), stream=stream0)
        del arg180_1
        buf269 = buf234; del buf234  # reuse
        # Source Nodes: [conv_out_layer_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf261, arg182_1, buf269, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg182_1
        buf271 = reinterpret_tensor(buf261, (3072, 64, 1), (64, 1, 1), 0); del buf261  # reuse
        # Source Nodes: [conv_kernel_layer_23, conv_out_layer_62], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf269, (3072, 64, 9), (576, 9, 1), 0), buf270, out=buf271)
        buf272 = reinterpret_tensor(buf263, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf263  # reuse
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf257, buf271, buf272, 393216, grid=grid(393216), stream=stream0)
        buf273 = reinterpret_tensor(buf262, (512, 768), (768, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (512, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), out=buf273)
        del arg183_1
        buf277 = reinterpret_tensor(buf272, (1, 512, 768), (393216, 768, 1), 0); del buf272  # reuse
        # Source Nodes: [add_24, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf273, arg184_1, buf249, arg185_1, arg186_1, buf277, 512, 768, grid=grid(512), stream=stream0)
        del arg184_1
        del arg185_1
        del arg186_1
        buf278 = reinterpret_tensor(buf244, (512, 3072), (3072, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (512, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 3072), (1, 768), 0), out=buf278)
        del arg187_1
        buf279 = reinterpret_tensor(buf278, (1, 512, 3072), (1572864, 3072, 1), 0); del buf278  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf279, arg188_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg188_1
        buf280 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg189_1, (3072, 768), (1, 3072), 0), out=buf280)
        del arg189_1
        buf284 = buf249; del buf249  # reuse
        # Source Nodes: [add_25, hidden_states_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf280, arg190_1, buf277, arg191_1, arg192_1, buf284, 512, 768, grid=grid(512), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        del buf277
        buf285 = reinterpret_tensor(buf271, (512, 384), (384, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 384), (1, 768), 0), out=buf285)
        del arg193_1
        buf286 = reinterpret_tensor(buf257, (512, 384), (384, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 384), (1, 768), 0), out=buf286)
        del arg195_1
        buf287 = reinterpret_tensor(buf265, (512, 384), (384, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 384), (1, 768), 0), out=buf287)
        del arg197_1
        buf288 = reinterpret_tensor(buf264, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf285, arg194_1, buf288, 196608, grid=grid(196608), stream=stream0)
        buf289 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf286, arg196_1, buf289, 196608, grid=grid(196608), stream=stream0)
        del arg196_1
        buf290 = reinterpret_tensor(buf286, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf287, arg198_1, buf290, 196608, grid=grid(196608), stream=stream0)
        del arg198_1
        del buf287
        # Source Nodes: [], Original ATen: []
        buf291 = aten._scaled_dot_product_efficient_attention(buf288, buf289, buf290, None, False, scale=0.125)
        del buf288
        buf292 = buf291[0]
        del buf291
        buf296 = reinterpret_tensor(buf290, (512, 384), (384, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 768), (768, 1), 0), reinterpret_tensor(arg203_1, (768, 384), (1, 768), 0), out=buf296)
        del arg203_1
        buf297 = reinterpret_tensor(buf280, (1, 768, 512), (393216, 512, 1), 0); del buf280  # reuse
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf284, buf297, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, arg199_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf298, (1, 768, 512), (393216, 512, 1))
        del arg199_1
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, arg200_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf299, (1, 384, 512), (196608, 512, 1))
        del arg200_1
        buf300 = reinterpret_tensor(buf285, (1, 512, 384), (196608, 384, 1), 0); del buf285  # reuse
        # Source Nodes: [conv_attn_layer_8], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf300, buf299, arg8_1, arg194_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg194_1
        del arg8_1
        buf301 = reinterpret_tensor(buf270, (512, 54), (54, 1), 0); del buf270  # reuse
        # Source Nodes: [conv_kernel_layer_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (512, 384), (384, 1), 0), reinterpret_tensor(arg201_1, (384, 54), (1, 384), 0), out=buf301)
        del arg201_1
        buf305 = reinterpret_tensor(buf266, (3072, 9, 1), (9, 1, 27648), 0); del buf266  # reuse
        # Source Nodes: [conv_kernel_layer_26], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf301, arg202_1, buf305, 3072, 9, grid=grid(3072), stream=stream0)
        del arg202_1
        buf304 = buf269; del buf269  # reuse
        # Source Nodes: [conv_out_layer_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf296, arg204_1, buf304, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg204_1
        buf306 = reinterpret_tensor(buf296, (3072, 64, 1), (64, 1, 1), 0); del buf296  # reuse
        # Source Nodes: [conv_kernel_layer_26, conv_out_layer_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (3072, 64, 9), (576, 9, 1), 0), buf305, out=buf306)
        buf307 = reinterpret_tensor(buf298, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf292, buf306, buf307, 393216, grid=grid(393216), stream=stream0)
        buf308 = reinterpret_tensor(buf297, (512, 768), (768, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (512, 768), (768, 1), 0), reinterpret_tensor(arg205_1, (768, 768), (1, 768), 0), out=buf308)
        del arg205_1
        buf312 = reinterpret_tensor(buf307, (1, 512, 768), (393216, 768, 1), 0); del buf307  # reuse
        # Source Nodes: [add_27, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf308, arg206_1, buf284, arg207_1, arg208_1, buf312, 512, 768, grid=grid(512), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        buf313 = reinterpret_tensor(buf279, (512, 3072), (3072, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (512, 768), (768, 1), 0), reinterpret_tensor(arg209_1, (768, 3072), (1, 768), 0), out=buf313)
        del arg209_1
        buf314 = reinterpret_tensor(buf313, (1, 512, 3072), (1572864, 3072, 1), 0); del buf313  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf314, arg210_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg210_1
        buf315 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf314, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg211_1, (3072, 768), (1, 3072), 0), out=buf315)
        del arg211_1
        buf319 = buf284; del buf284  # reuse
        # Source Nodes: [add_28, hidden_states_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf315, arg212_1, buf312, arg213_1, arg214_1, buf319, 512, 768, grid=grid(512), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del buf312
        buf320 = reinterpret_tensor(buf306, (512, 384), (384, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (512, 768), (768, 1), 0), reinterpret_tensor(arg215_1, (768, 384), (1, 768), 0), out=buf320)
        del arg215_1
        buf321 = reinterpret_tensor(buf292, (512, 384), (384, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (512, 768), (768, 1), 0), reinterpret_tensor(arg217_1, (768, 384), (1, 768), 0), out=buf321)
        del arg217_1
        buf322 = reinterpret_tensor(buf300, (512, 384), (384, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (512, 768), (768, 1), 0), reinterpret_tensor(arg219_1, (768, 384), (1, 768), 0), out=buf322)
        del arg219_1
        buf323 = reinterpret_tensor(buf299, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf320, arg216_1, buf323, 196608, grid=grid(196608), stream=stream0)
        buf324 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf321, arg218_1, buf324, 196608, grid=grid(196608), stream=stream0)
        del arg218_1
        buf325 = reinterpret_tensor(buf321, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf322, arg220_1, buf325, 196608, grid=grid(196608), stream=stream0)
        del arg220_1
        del buf322
        # Source Nodes: [], Original ATen: []
        buf326 = aten._scaled_dot_product_efficient_attention(buf323, buf324, buf325, None, False, scale=0.125)
        del buf323
        buf327 = buf326[0]
        del buf326
        buf331 = reinterpret_tensor(buf325, (512, 384), (384, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (512, 768), (768, 1), 0), reinterpret_tensor(arg225_1, (768, 384), (1, 768), 0), out=buf331)
        del arg225_1
        buf332 = reinterpret_tensor(buf315, (1, 768, 512), (393216, 512, 1), 0); del buf315  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf319, buf332, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, arg221_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf333, (1, 768, 512), (393216, 512, 1))
        del arg221_1
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, arg222_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf334, (1, 384, 512), (196608, 512, 1))
        del arg222_1
        buf335 = reinterpret_tensor(buf320, (1, 512, 384), (196608, 384, 1), 0); del buf320  # reuse
        # Source Nodes: [conv_attn_layer_9], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf335, buf334, arg9_1, arg216_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg216_1
        del arg9_1
        buf336 = reinterpret_tensor(buf305, (512, 54), (54, 1), 0); del buf305  # reuse
        # Source Nodes: [conv_kernel_layer_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (512, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 54), (1, 384), 0), out=buf336)
        del arg223_1
        buf340 = reinterpret_tensor(buf301, (3072, 9, 1), (9, 1, 27648), 0); del buf301  # reuse
        # Source Nodes: [conv_kernel_layer_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf336, arg224_1, buf340, 3072, 9, grid=grid(3072), stream=stream0)
        del arg224_1
        buf339 = buf304; del buf304  # reuse
        # Source Nodes: [conv_out_layer_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf331, arg226_1, buf339, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg226_1
        buf341 = reinterpret_tensor(buf331, (3072, 64, 1), (64, 1, 1), 0); del buf331  # reuse
        # Source Nodes: [conv_kernel_layer_29, conv_out_layer_78], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf339, (3072, 64, 9), (576, 9, 1), 0), buf340, out=buf341)
        buf342 = reinterpret_tensor(buf333, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf327, buf341, buf342, 393216, grid=grid(393216), stream=stream0)
        buf343 = reinterpret_tensor(buf332, (512, 768), (768, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (512, 768), (768, 1), 0), reinterpret_tensor(arg227_1, (768, 768), (1, 768), 0), out=buf343)
        del arg227_1
        buf347 = reinterpret_tensor(buf342, (1, 512, 768), (393216, 768, 1), 0); del buf342  # reuse
        # Source Nodes: [add_30, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf343, arg228_1, buf319, arg229_1, arg230_1, buf347, 512, 768, grid=grid(512), stream=stream0)
        del arg228_1
        del arg229_1
        del arg230_1
        buf348 = reinterpret_tensor(buf314, (512, 3072), (3072, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf347, (512, 768), (768, 1), 0), reinterpret_tensor(arg231_1, (768, 3072), (1, 768), 0), out=buf348)
        del arg231_1
        buf349 = reinterpret_tensor(buf348, (1, 512, 3072), (1572864, 3072, 1), 0); del buf348  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf349, arg232_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg232_1
        buf350 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg233_1, (3072, 768), (1, 3072), 0), out=buf350)
        del arg233_1
        buf354 = buf319; del buf319  # reuse
        # Source Nodes: [add_31, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf350, arg234_1, buf347, arg235_1, arg236_1, buf354, 512, 768, grid=grid(512), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del buf347
        buf355 = reinterpret_tensor(buf341, (512, 384), (384, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 768), (768, 1), 0), reinterpret_tensor(arg237_1, (768, 384), (1, 768), 0), out=buf355)
        del arg237_1
        buf356 = reinterpret_tensor(buf327, (512, 384), (384, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 768), (768, 1), 0), reinterpret_tensor(arg239_1, (768, 384), (1, 768), 0), out=buf356)
        del arg239_1
        buf357 = reinterpret_tensor(buf335, (512, 384), (384, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 384), (1, 768), 0), out=buf357)
        del arg241_1
        buf358 = reinterpret_tensor(buf334, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf355, arg238_1, buf358, 196608, grid=grid(196608), stream=stream0)
        buf359 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf356, arg240_1, buf359, 196608, grid=grid(196608), stream=stream0)
        del arg240_1
        buf360 = reinterpret_tensor(buf356, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf357, arg242_1, buf360, 196608, grid=grid(196608), stream=stream0)
        del arg242_1
        del buf357
        # Source Nodes: [], Original ATen: []
        buf361 = aten._scaled_dot_product_efficient_attention(buf358, buf359, buf360, None, False, scale=0.125)
        del buf358
        buf362 = buf361[0]
        del buf361
        buf366 = reinterpret_tensor(buf360, (512, 384), (384, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 768), (768, 1), 0), reinterpret_tensor(arg247_1, (768, 384), (1, 768), 0), out=buf366)
        del arg247_1
        buf367 = reinterpret_tensor(buf350, (1, 768, 512), (393216, 512, 1), 0); del buf350  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf354, buf367, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, arg243_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf368, (1, 768, 512), (393216, 512, 1))
        del arg243_1
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, arg244_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf369, (1, 384, 512), (196608, 512, 1))
        del arg244_1
        buf370 = reinterpret_tensor(buf355, (1, 512, 384), (196608, 384, 1), 0); del buf355  # reuse
        # Source Nodes: [conv_attn_layer_10], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf370, buf369, arg10_1, arg238_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg10_1
        del arg238_1
        buf371 = reinterpret_tensor(buf340, (512, 54), (54, 1), 0); del buf340  # reuse
        # Source Nodes: [conv_kernel_layer_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (512, 384), (384, 1), 0), reinterpret_tensor(arg245_1, (384, 54), (1, 384), 0), out=buf371)
        del arg245_1
        buf375 = reinterpret_tensor(buf336, (3072, 9, 1), (9, 1, 27648), 0); del buf336  # reuse
        # Source Nodes: [conv_kernel_layer_32], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf371, arg246_1, buf375, 3072, 9, grid=grid(3072), stream=stream0)
        del arg246_1
        buf374 = buf339; del buf339  # reuse
        # Source Nodes: [conv_out_layer_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf366, arg248_1, buf374, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg248_1
        buf376 = reinterpret_tensor(buf366, (3072, 64, 1), (64, 1, 1), 0); del buf366  # reuse
        # Source Nodes: [conv_kernel_layer_32, conv_out_layer_86], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (3072, 64, 9), (576, 9, 1), 0), buf375, out=buf376)
        buf377 = reinterpret_tensor(buf368, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf368  # reuse
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf362, buf376, buf377, 393216, grid=grid(393216), stream=stream0)
        buf378 = reinterpret_tensor(buf367, (512, 768), (768, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (512, 768), (768, 1), 0), reinterpret_tensor(arg249_1, (768, 768), (1, 768), 0), out=buf378)
        del arg249_1
        buf382 = reinterpret_tensor(buf377, (1, 512, 768), (393216, 768, 1), 0); del buf377  # reuse
        # Source Nodes: [add_33, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf378, arg250_1, buf354, arg251_1, arg252_1, buf382, 512, 768, grid=grid(512), stream=stream0)
        del arg250_1
        del arg251_1
        del arg252_1
        buf383 = reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (512, 768), (768, 1), 0), reinterpret_tensor(arg253_1, (768, 3072), (1, 768), 0), out=buf383)
        del arg253_1
        buf384 = reinterpret_tensor(buf383, (1, 512, 3072), (1572864, 3072, 1), 0); del buf383  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf384, arg254_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg254_1
        buf385 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf384, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg255_1, (3072, 768), (1, 3072), 0), out=buf385)
        del arg255_1
        buf389 = buf354; del buf354  # reuse
        # Source Nodes: [add_34, hidden_states_99], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf385, arg256_1, buf382, arg257_1, arg258_1, buf389, 512, 768, grid=grid(512), stream=stream0)
        del arg256_1
        del arg257_1
        del arg258_1
        del buf382
        buf390 = reinterpret_tensor(buf376, (512, 384), (384, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 768), (768, 1), 0), reinterpret_tensor(arg259_1, (768, 384), (1, 768), 0), out=buf390)
        del arg259_1
        buf391 = reinterpret_tensor(buf362, (512, 384), (384, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 768), (768, 1), 0), reinterpret_tensor(arg261_1, (768, 384), (1, 768), 0), out=buf391)
        del arg261_1
        buf392 = reinterpret_tensor(buf370, (512, 384), (384, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 768), (768, 1), 0), reinterpret_tensor(arg263_1, (768, 384), (1, 768), 0), out=buf392)
        del arg263_1
        buf393 = reinterpret_tensor(buf369, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf390, arg260_1, buf393, 196608, grid=grid(196608), stream=stream0)
        buf394 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf391, arg262_1, buf394, 196608, grid=grid(196608), stream=stream0)
        del arg262_1
        buf395 = reinterpret_tensor(buf391, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf392, arg264_1, buf395, 196608, grid=grid(196608), stream=stream0)
        del arg264_1
        del buf392
        # Source Nodes: [], Original ATen: []
        buf396 = aten._scaled_dot_product_efficient_attention(buf393, buf394, buf395, None, False, scale=0.125)
        del buf393
        del buf394
        buf397 = buf396[0]
        del buf396
        buf401 = reinterpret_tensor(buf395, (512, 384), (384, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 768), (768, 1), 0), reinterpret_tensor(arg269_1, (768, 384), (1, 768), 0), out=buf401)
        del arg269_1
        buf402 = reinterpret_tensor(buf385, (1, 768, 512), (393216, 512, 1), 0); del buf385  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf389, buf402, 768, 512, grid=grid(768, 512), stream=stream0)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, arg265_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf403, (1, 768, 512), (393216, 512, 1))
        del arg265_1
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, arg266_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf404, (1, 384, 512), (196608, 512, 1))
        del arg266_1
        buf405 = reinterpret_tensor(buf390, (1, 512, 384), (196608, 384, 1), 0); del buf390  # reuse
        # Source Nodes: [conv_attn_layer_11], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf405, buf404, arg11_1, arg260_1, 512, 384, grid=grid(512, 384), stream=stream0)
        del arg11_1
        del arg260_1
        del buf404
        buf406 = reinterpret_tensor(buf375, (512, 54), (54, 1), 0); del buf375  # reuse
        # Source Nodes: [conv_kernel_layer_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 384), (384, 1), 0), reinterpret_tensor(arg267_1, (384, 54), (1, 384), 0), out=buf406)
        del arg267_1
        del buf405
        buf410 = reinterpret_tensor(buf371, (3072, 9, 1), (9, 1, 27648), 0); del buf371  # reuse
        # Source Nodes: [conv_kernel_layer_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf406, arg268_1, buf410, 3072, 9, grid=grid(3072), stream=stream0)
        del arg268_1
        del buf406
        buf409 = buf374; del buf374  # reuse
        # Source Nodes: [conv_out_layer_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf401, arg270_1, buf409, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del arg270_1
        buf411 = reinterpret_tensor(buf401, (3072, 64, 1), (64, 1, 1), 0); del buf401  # reuse
        # Source Nodes: [conv_kernel_layer_35, conv_out_layer_94], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (3072, 64, 9), (576, 9, 1), 0), buf410, out=buf411)
        del buf409
        del buf410
        buf412 = reinterpret_tensor(buf403, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf403  # reuse
        # Source Nodes: [cat_12], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf397, buf411, buf412, 393216, grid=grid(393216), stream=stream0)
        del buf397
        del buf411
        buf413 = reinterpret_tensor(buf402, (512, 768), (768, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 768), (768, 1), 0), reinterpret_tensor(arg271_1, (768, 768), (1, 768), 0), out=buf413)
        del arg271_1
        buf417 = reinterpret_tensor(buf412, (1, 512, 768), (393216, 768, 1), 0); del buf412  # reuse
        # Source Nodes: [add_36, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf413, arg272_1, buf389, arg273_1, arg274_1, buf417, 512, 768, grid=grid(512), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        buf418 = reinterpret_tensor(buf384, (512, 3072), (3072, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf417, (512, 768), (768, 1), 0), reinterpret_tensor(arg275_1, (768, 3072), (1, 768), 0), out=buf418)
        del arg275_1
        buf419 = reinterpret_tensor(buf418, (1, 512, 3072), (1572864, 3072, 1), 0); del buf418  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf419, arg276_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg276_1
        buf420 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg277_1, (3072, 768), (1, 3072), 0), out=buf420)
        del arg277_1
        del buf419
        buf424 = buf389; del buf389  # reuse
        # Source Nodes: [add_37, generator_sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf420, arg278_1, buf417, arg279_1, arg280_1, buf424, 512, 768, grid=grid(512), stream=stream0)
        del arg278_1
        del arg279_1
        del arg280_1
        del buf417
        buf425 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf424, (512, 768), (768, 1), 0), reinterpret_tensor(arg281_1, (768, 768), (1, 768), 0), out=buf425)
        del arg281_1
        buf429 = buf424; del buf424  # reuse
        # Source Nodes: [hidden_states_110, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_9.run(buf425, arg282_1, arg283_1, arg284_1, buf429, 512, 768, grid=grid(512), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del buf425
        buf430 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg286_1, reinterpret_tensor(buf429, (512, 768), (768, 1), 0), reinterpret_tensor(arg285_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf430)
        del arg285_1
        del arg286_1
        del buf429
        buf431 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        buf432 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf430, buf431, buf432, 512, 30522, grid=grid(512), stream=stream0)
        buf433 = empty((), device='cuda', dtype=torch.float32)
        buf435 = buf433; del buf433  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_11.run(buf435, arg290_1, buf430, buf431, buf432, 1, 512, grid=grid(1), stream=stream0)
        del arg290_1
        return (buf435, reinterpret_tensor(buf430, (1, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg288_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg289_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg290_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
