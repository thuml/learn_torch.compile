
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabhuquyhgzsb34pxyb7fxfu5pfxtmnnox54m7m5ukahwvnqbtiv.py
# Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
# token_type_ids => full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlwwn2pg4v2nttvm7hbyccbqcf23v4fxh3lhzeuoox3e7tqvoci.py
# Source Nodes: [cat_1, inputs_embeds_2], Original ATen: [aten.cat, aten.view]
# cat_1 => cat
# inputs_embeds_2 => view
triton_poi_fused_cat_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 127, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (1 + x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp9 + 30522
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 30522)) | ~tmp8, "index out of bounds: 0 <= tmp12 < 30522")
    tmp13 = tl.load(in_ptr1 + (x0 + (128*tmp12)), tmp8, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (x1), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 + 30522
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 30522)) | ~tmp21, "index out of bounds: 0 <= tmp25 < 30522")
    tmp26 = tl.load(in_ptr1 + ((-128) + x0 + (128*tmp25)), tmp21, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp21, tmp26, tmp27)
    tmp29 = tmp0 >= tmp19
    tmp30 = tl.full([1], 384, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = (-1) + x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp33 & tmp29
    tmp35 = tl.load(in_ptr0 + ((-1) + x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + 30522
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tl.device_assert(((0 <= tmp38) & (tmp38 < 30522)) | ~tmp34, "index out of bounds: 0 <= tmp38 < 30522")
    tmp39 = tl.load(in_ptr1 + ((-256) + x0 + (128*tmp38)), tmp34, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp34, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp29, tmp41, tmp42)
    tmp44 = tl.where(tmp21, tmp28, tmp43)
    tmp45 = tl.where(tmp4, tmp17, tmp44)
    tl.store(out_ptr0 + (x2), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crf5rsqtfop7qxd247mw7rjhedmq5vox26ffhp2t5w5jcvuyeeux.py
# Source Nodes: [add, embeddings, embeddings_1, layer_input, mul_1, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.mul, aten.view]
# add => add
# embeddings => add_1
# embeddings_1 => add_2
# layer_input => view_2
# mul_1 => mul_1
# position_embeddings => embedding_1
# token_type_embeddings => embedding_2
triton_poi_fused_add_embedding_mul_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_view_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + 512
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert((0 <= tmp6) & (tmp6 < 512), "index out of bounds: 0 <= tmp6 < 512")
    tmp7 = tl.load(in_ptr2 + (x0 + (512*tmp6)), None)
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crciac3foghcwdniznfbv73b6hbjanayd33ohfb7rsa54jbco4ob.py
# Source Nodes: [key_tensor, mixed_query_layer, mul_3], Original ATen: [aten.add, aten.mul, aten.view]
# key_tensor => add_4
# mixed_query_layer => view_6
# mul_3 => mul_3
triton_poi_fused_add_mul_view_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnebinx72rzt6bilmf6qw7mjvltiqedecn2ojeuvgfdd5bdsimi.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (128*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcfixllntcolgeu7jjopxajkv3eijjw2odi7np4gdbarka3pdxn.py
# Source Nodes: [layer_outputs], Original ATen: [aten.view]
# layer_outputs => view_22
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c267bxtsijwqdmfth5oy4wrrlax2lyxukosuffcidoxjqv2z3vtj.py
# Source Nodes: [add_6, attention_output, layer_input_4, mul_2, mul_4], Original ATen: [aten.add, aten.mul]
# add_6 => add_6
# attention_output => add_7
# layer_input_4 => add_3
# mul_2 => mul_2
# mul_4 => mul_4
triton_poi_fused_add_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwqjercuebixt34zsvqjieazpo4h74kna3chvcr3ngwkalsr3jg.py
# Source Nodes: [intermediate_output, layer_outputs_2], Original ATen: [aten.relu, aten.view]
# intermediate_output => relu
# layer_outputs_2 => view_26
triton_poi_fused_relu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4jrn2tktnke3tvwzdtmazk2sjai6tgra7ich4z6gvnoiqj3u4f.py
# Source Nodes: [add_8, attention_output_1, hidden_states_2, mul_5], Original ATen: [aten.add, aten.mul, aten.view]
# add_8 => add_8
# attention_output_1 => add_9
# hidden_states_2 => view_28
# mul_5 => mul_5
triton_poi_fused_add_mul_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cuginuxjtapst7hnylk7caqktkl75iecjqtr5bpw2qzrsybl2ge6.py
# Source Nodes: [add_10, add_8, attention_output_1, attention_output_2, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
# add_10 => add_10
# add_8 => add_8
# attention_output_1 => add_9
# attention_output_2 => add_11
# mul_5 => mul_5
# mul_6 => mul_6
triton_poi_fused_add_mul_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhczlnm4a57mqhgorfzsnvrob5icfok7bsos2uhwscvbzua3oq4.py
# Source Nodes: [add_16, embeddings_1, layer_input_5, mul_1, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul, aten.view]
# add_16 => add_16
# embeddings_1 => add_2
# layer_input_5 => view_42
# mul_1 => mul_1
# mul_9 => mul_9
# value_tensor_1 => add_17
triton_poi_fused_add_mul_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd253liyeugzufxctnpof7rqf5k3npu4c62k5im732dhk7alyol7.py
# Source Nodes: [add_31, mul_17, mul_9, value_tensor_1, value_tensor_2], Original ATen: [aten.add, aten.mul]
# add_31 => add_31
# mul_17 => mul_17
# mul_9 => mul_9
# value_tensor_1 => add_17
# value_tensor_2 => add_32
triton_poi_fused_add_mul_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3hp5a3c54dhe6ugzdthf5ddpsjzcvt2dgsjvygyx2dggkyfvav3.py
# Source Nodes: [add_46, layer_input_15, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul, aten.view]
# add_46 => add_46
# layer_input_15 => view_122
# mul_25 => mul_25
# value_tensor_3 => add_47
triton_poi_fused_add_mul_view_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cut5vn5lbqh64nl3whsvavyercf5t6e3tto5rkb3sohhwupkors5.py
# Source Nodes: [add_46, add_61, mul_25, mul_33, value_tensor_3, value_tensor_4], Original ATen: [aten.add, aten.mul]
# add_46 => add_46
# add_61 => add_61
# mul_25 => mul_25
# mul_33 => mul_33
# value_tensor_3 => add_47
# value_tensor_4 => add_62
triton_poi_fused_add_mul_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23obnjrkonuvhe46kuhexyjb6torkp4cg6t6fkwnzl33zz5jpag.py
# Source Nodes: [intermediate_output_90, layer_outputs_316], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
# intermediate_output_90 => relu_90
# layer_outputs_316 => view_914
triton_poi_fused_relu_threshold_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_view_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cje32nbesvvd2pdqz6rlf4xn62xby3xky4aaygbsmg5ykpqsg36i.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_49
# start_loss => amax_24, exp_24, log, sub_25, sub_26, sum_25
triton_per_fused__log_softmax_clone_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/r2/cr257bbw7yhxgagpk5ptq5ayrnxyc3h2e7jv7v5tr3de2lsdqk76.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_50
# end_loss => amax_25, exp_25, log_1, sub_27, sub_28, sum_28
triton_per_fused__log_softmax_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_16', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpopq5lk44mexmdxafwkrya2qqp6fzbfjxzfg6otx2kbv7mpcaa.py
# Source Nodes: [add_363, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_forward]
# add_363 => add_363
# end_loss => convert_element_type_1, div_49, ne_3, neg_1, sum_29, sum_30, where_3
# end_positions => clamp_max_1, clamp_min_1
# loss => div_50
# start_loss => convert_element_type, div_48, full_default_3, ne, neg, sum_26, sum_27, where_1
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_nll_loss_forward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_forward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqwgzdrzlkqwhchx6q7avj6kji7nktr5yz6fox63qui5mbc34h7.py
# Source Nodes: [start_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# start_loss => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 128, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp7 = tl.where(tmp6, tmp5, tmp2)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhwl3n3uie4zgezxp23ptt5ub35s64swjvdtwv3dcpxg6ubu6xl.py
# Source Nodes: [intermediate_output_89], Original ATen: [aten.relu, aten.threshold_backward]
# intermediate_output_89 => relu_89
triton_poi_fused_relu_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (512, ), (1, ))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, ), (1, ))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (128, ), (1, ))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (128, ), (1, ))
    assert_size_stride(primals_241, (512, ), (1, ))
    assert_size_stride(primals_242, (512, ), (1, ))
    assert_size_stride(primals_243, (128, ), (1, ))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (128, ), (1, ))
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (128, ), (1, ))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, ), (1, ))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (128, ), (1, ))
    assert_size_stride(primals_277, (128, ), (1, ))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, ), (1, ))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (128, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (128, ), (1, ))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (128, ), (1, ))
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (128, ), (1, ))
    assert_size_stride(primals_308, (128, ), (1, ))
    assert_size_stride(primals_309, (128, ), (1, ))
    assert_size_stride(primals_310, (128, ), (1, ))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (128, ), (1, ))
    assert_size_stride(primals_318, (128, ), (1, ))
    assert_size_stride(primals_319, (128, ), (1, ))
    assert_size_stride(primals_320, (128, ), (1, ))
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_326, (128, ), (1, ))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (128, ), (1, ))
    assert_size_stride(primals_337, (512, ), (1, ))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (128, ), (1, ))
    assert_size_stride(primals_340, (128, ), (1, ))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (128, ), (1, ))
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (128, ), (1, ))
    assert_size_stride(primals_351, (128, ), (1, ))
    assert_size_stride(primals_352, (128, ), (1, ))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (128, ), (1, ))
    assert_size_stride(primals_356, (128, ), (1, ))
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (128, ), (1, ))
    assert_size_stride(primals_360, (128, ), (1, ))
    assert_size_stride(primals_361, (128, ), (1, ))
    assert_size_stride(primals_362, (128, ), (1, ))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (128, ), (1, ))
    assert_size_stride(primals_368, (128, ), (1, ))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (128, ), (1, ))
    assert_size_stride(primals_374, (128, ), (1, ))
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (128, ), (1, ))
    assert_size_stride(primals_377, (128, ), (1, ))
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (128, ), (1, ))
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_386, (512, ), (1, ))
    assert_size_stride(primals_387, (30522, 128), (128, 1))
    assert_size_stride(primals_388, (512, 384), (384, 1))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (512, 512), (512, 1))
    assert_size_stride(primals_391, (2, 512), (512, 1))
    assert_size_stride(primals_392, (128, 512), (512, 1))
    assert_size_stride(primals_393, (128, ), (1, ))
    assert_size_stride(primals_394, (128, 512), (512, 1))
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (128, 128), (128, 1))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (128, 128), (128, 1))
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (128, 512), (512, 1))
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (128, 128), (128, 1))
    assert_size_stride(primals_403, (128, ), (1, ))
    assert_size_stride(primals_404, (512, 128), (128, 1))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (128, 512), (512, 1))
    assert_size_stride(primals_407, (128, ), (1, ))
    assert_size_stride(primals_408, (512, 128), (128, 1))
    assert_size_stride(primals_409, (512, ), (1, ))
    assert_size_stride(primals_410, (128, 512), (512, 1))
    assert_size_stride(primals_411, (128, ), (1, ))
    assert_size_stride(primals_412, (512, 128), (128, 1))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_414, (128, 512), (512, 1))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (512, 128), (128, 1))
    assert_size_stride(primals_417, (512, ), (1, ))
    assert_size_stride(primals_418, (128, 512), (512, 1))
    assert_size_stride(primals_419, (128, ), (1, ))
    assert_size_stride(primals_420, (512, 128), (128, 1))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (128, 512), (512, 1))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, 512), (512, 1))
    assert_size_stride(primals_425, (128, ), (1, ))
    assert_size_stride(primals_426, (128, 128), (128, 1))
    assert_size_stride(primals_427, (128, ), (1, ))
    assert_size_stride(primals_428, (128, 128), (128, 1))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, 512), (512, 1))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, 128), (128, 1))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (512, 128), (128, 1))
    assert_size_stride(primals_435, (512, ), (1, ))
    assert_size_stride(primals_436, (128, 512), (512, 1))
    assert_size_stride(primals_437, (128, ), (1, ))
    assert_size_stride(primals_438, (512, 128), (128, 1))
    assert_size_stride(primals_439, (512, ), (1, ))
    assert_size_stride(primals_440, (128, 512), (512, 1))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (512, 128), (128, 1))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_444, (128, 512), (512, 1))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (512, 128), (128, 1))
    assert_size_stride(primals_447, (512, ), (1, ))
    assert_size_stride(primals_448, (128, 512), (512, 1))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (512, 128), (128, 1))
    assert_size_stride(primals_451, (512, ), (1, ))
    assert_size_stride(primals_452, (128, 512), (512, 1))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, 512), (512, 1))
    assert_size_stride(primals_455, (128, ), (1, ))
    assert_size_stride(primals_456, (128, 128), (128, 1))
    assert_size_stride(primals_457, (128, ), (1, ))
    assert_size_stride(primals_458, (128, 128), (128, 1))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, 512), (512, 1))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (128, 128), (128, 1))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (512, 128), (128, 1))
    assert_size_stride(primals_465, (512, ), (1, ))
    assert_size_stride(primals_466, (128, 512), (512, 1))
    assert_size_stride(primals_467, (128, ), (1, ))
    assert_size_stride(primals_468, (512, 128), (128, 1))
    assert_size_stride(primals_469, (512, ), (1, ))
    assert_size_stride(primals_470, (128, 512), (512, 1))
    assert_size_stride(primals_471, (128, ), (1, ))
    assert_size_stride(primals_472, (512, 128), (128, 1))
    assert_size_stride(primals_473, (512, ), (1, ))
    assert_size_stride(primals_474, (128, 512), (512, 1))
    assert_size_stride(primals_475, (128, ), (1, ))
    assert_size_stride(primals_476, (512, 128), (128, 1))
    assert_size_stride(primals_477, (512, ), (1, ))
    assert_size_stride(primals_478, (128, 512), (512, 1))
    assert_size_stride(primals_479, (128, ), (1, ))
    assert_size_stride(primals_480, (512, 128), (128, 1))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (128, 512), (512, 1))
    assert_size_stride(primals_483, (128, ), (1, ))
    assert_size_stride(primals_484, (128, 512), (512, 1))
    assert_size_stride(primals_485, (128, ), (1, ))
    assert_size_stride(primals_486, (128, 128), (128, 1))
    assert_size_stride(primals_487, (128, ), (1, ))
    assert_size_stride(primals_488, (128, 128), (128, 1))
    assert_size_stride(primals_489, (128, ), (1, ))
    assert_size_stride(primals_490, (128, 512), (512, 1))
    assert_size_stride(primals_491, (128, ), (1, ))
    assert_size_stride(primals_492, (128, 128), (128, 1))
    assert_size_stride(primals_493, (128, ), (1, ))
    assert_size_stride(primals_494, (512, 128), (128, 1))
    assert_size_stride(primals_495, (512, ), (1, ))
    assert_size_stride(primals_496, (128, 512), (512, 1))
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (512, 128), (128, 1))
    assert_size_stride(primals_499, (512, ), (1, ))
    assert_size_stride(primals_500, (128, 512), (512, 1))
    assert_size_stride(primals_501, (128, ), (1, ))
    assert_size_stride(primals_502, (512, 128), (128, 1))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (128, 512), (512, 1))
    assert_size_stride(primals_505, (128, ), (1, ))
    assert_size_stride(primals_506, (512, 128), (128, 1))
    assert_size_stride(primals_507, (512, ), (1, ))
    assert_size_stride(primals_508, (128, 512), (512, 1))
    assert_size_stride(primals_509, (128, ), (1, ))
    assert_size_stride(primals_510, (512, 128), (128, 1))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_512, (128, 512), (512, 1))
    assert_size_stride(primals_513, (128, ), (1, ))
    assert_size_stride(primals_514, (128, 512), (512, 1))
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, 128), (128, 1))
    assert_size_stride(primals_517, (128, ), (1, ))
    assert_size_stride(primals_518, (128, 128), (128, 1))
    assert_size_stride(primals_519, (128, ), (1, ))
    assert_size_stride(primals_520, (128, 512), (512, 1))
    assert_size_stride(primals_521, (128, ), (1, ))
    assert_size_stride(primals_522, (128, 128), (128, 1))
    assert_size_stride(primals_523, (128, ), (1, ))
    assert_size_stride(primals_524, (512, 128), (128, 1))
    assert_size_stride(primals_525, (512, ), (1, ))
    assert_size_stride(primals_526, (128, 512), (512, 1))
    assert_size_stride(primals_527, (128, ), (1, ))
    assert_size_stride(primals_528, (512, 128), (128, 1))
    assert_size_stride(primals_529, (512, ), (1, ))
    assert_size_stride(primals_530, (128, 512), (512, 1))
    assert_size_stride(primals_531, (128, ), (1, ))
    assert_size_stride(primals_532, (512, 128), (128, 1))
    assert_size_stride(primals_533, (512, ), (1, ))
    assert_size_stride(primals_534, (128, 512), (512, 1))
    assert_size_stride(primals_535, (128, ), (1, ))
    assert_size_stride(primals_536, (512, 128), (128, 1))
    assert_size_stride(primals_537, (512, ), (1, ))
    assert_size_stride(primals_538, (128, 512), (512, 1))
    assert_size_stride(primals_539, (128, ), (1, ))
    assert_size_stride(primals_540, (512, 128), (128, 1))
    assert_size_stride(primals_541, (512, ), (1, ))
    assert_size_stride(primals_542, (128, 512), (512, 1))
    assert_size_stride(primals_543, (128, ), (1, ))
    assert_size_stride(primals_544, (128, 512), (512, 1))
    assert_size_stride(primals_545, (128, ), (1, ))
    assert_size_stride(primals_546, (128, 128), (128, 1))
    assert_size_stride(primals_547, (128, ), (1, ))
    assert_size_stride(primals_548, (128, 128), (128, 1))
    assert_size_stride(primals_549, (128, ), (1, ))
    assert_size_stride(primals_550, (128, 512), (512, 1))
    assert_size_stride(primals_551, (128, ), (1, ))
    assert_size_stride(primals_552, (128, 128), (128, 1))
    assert_size_stride(primals_553, (128, ), (1, ))
    assert_size_stride(primals_554, (512, 128), (128, 1))
    assert_size_stride(primals_555, (512, ), (1, ))
    assert_size_stride(primals_556, (128, 512), (512, 1))
    assert_size_stride(primals_557, (128, ), (1, ))
    assert_size_stride(primals_558, (512, 128), (128, 1))
    assert_size_stride(primals_559, (512, ), (1, ))
    assert_size_stride(primals_560, (128, 512), (512, 1))
    assert_size_stride(primals_561, (128, ), (1, ))
    assert_size_stride(primals_562, (512, 128), (128, 1))
    assert_size_stride(primals_563, (512, ), (1, ))
    assert_size_stride(primals_564, (128, 512), (512, 1))
    assert_size_stride(primals_565, (128, ), (1, ))
    assert_size_stride(primals_566, (512, 128), (128, 1))
    assert_size_stride(primals_567, (512, ), (1, ))
    assert_size_stride(primals_568, (128, 512), (512, 1))
    assert_size_stride(primals_569, (128, ), (1, ))
    assert_size_stride(primals_570, (512, 128), (128, 1))
    assert_size_stride(primals_571, (512, ), (1, ))
    assert_size_stride(primals_572, (128, 512), (512, 1))
    assert_size_stride(primals_573, (128, ), (1, ))
    assert_size_stride(primals_574, (128, 512), (512, 1))
    assert_size_stride(primals_575, (128, ), (1, ))
    assert_size_stride(primals_576, (128, 128), (128, 1))
    assert_size_stride(primals_577, (128, ), (1, ))
    assert_size_stride(primals_578, (128, 128), (128, 1))
    assert_size_stride(primals_579, (128, ), (1, ))
    assert_size_stride(primals_580, (128, 512), (512, 1))
    assert_size_stride(primals_581, (128, ), (1, ))
    assert_size_stride(primals_582, (128, 128), (128, 1))
    assert_size_stride(primals_583, (128, ), (1, ))
    assert_size_stride(primals_584, (512, 128), (128, 1))
    assert_size_stride(primals_585, (512, ), (1, ))
    assert_size_stride(primals_586, (128, 512), (512, 1))
    assert_size_stride(primals_587, (128, ), (1, ))
    assert_size_stride(primals_588, (512, 128), (128, 1))
    assert_size_stride(primals_589, (512, ), (1, ))
    assert_size_stride(primals_590, (128, 512), (512, 1))
    assert_size_stride(primals_591, (128, ), (1, ))
    assert_size_stride(primals_592, (512, 128), (128, 1))
    assert_size_stride(primals_593, (512, ), (1, ))
    assert_size_stride(primals_594, (128, 512), (512, 1))
    assert_size_stride(primals_595, (128, ), (1, ))
    assert_size_stride(primals_596, (512, 128), (128, 1))
    assert_size_stride(primals_597, (512, ), (1, ))
    assert_size_stride(primals_598, (128, 512), (512, 1))
    assert_size_stride(primals_599, (128, ), (1, ))
    assert_size_stride(primals_600, (512, 128), (128, 1))
    assert_size_stride(primals_601, (512, ), (1, ))
    assert_size_stride(primals_602, (128, 512), (512, 1))
    assert_size_stride(primals_603, (128, ), (1, ))
    assert_size_stride(primals_604, (128, 512), (512, 1))
    assert_size_stride(primals_605, (128, ), (1, ))
    assert_size_stride(primals_606, (128, 128), (128, 1))
    assert_size_stride(primals_607, (128, ), (1, ))
    assert_size_stride(primals_608, (128, 128), (128, 1))
    assert_size_stride(primals_609, (128, ), (1, ))
    assert_size_stride(primals_610, (128, 512), (512, 1))
    assert_size_stride(primals_611, (128, ), (1, ))
    assert_size_stride(primals_612, (128, 128), (128, 1))
    assert_size_stride(primals_613, (128, ), (1, ))
    assert_size_stride(primals_614, (512, 128), (128, 1))
    assert_size_stride(primals_615, (512, ), (1, ))
    assert_size_stride(primals_616, (128, 512), (512, 1))
    assert_size_stride(primals_617, (128, ), (1, ))
    assert_size_stride(primals_618, (512, 128), (128, 1))
    assert_size_stride(primals_619, (512, ), (1, ))
    assert_size_stride(primals_620, (128, 512), (512, 1))
    assert_size_stride(primals_621, (128, ), (1, ))
    assert_size_stride(primals_622, (512, 128), (128, 1))
    assert_size_stride(primals_623, (512, ), (1, ))
    assert_size_stride(primals_624, (128, 512), (512, 1))
    assert_size_stride(primals_625, (128, ), (1, ))
    assert_size_stride(primals_626, (512, 128), (128, 1))
    assert_size_stride(primals_627, (512, ), (1, ))
    assert_size_stride(primals_628, (128, 512), (512, 1))
    assert_size_stride(primals_629, (128, ), (1, ))
    assert_size_stride(primals_630, (512, 128), (128, 1))
    assert_size_stride(primals_631, (512, ), (1, ))
    assert_size_stride(primals_632, (128, 512), (512, 1))
    assert_size_stride(primals_633, (128, ), (1, ))
    assert_size_stride(primals_634, (128, 512), (512, 1))
    assert_size_stride(primals_635, (128, ), (1, ))
    assert_size_stride(primals_636, (128, 128), (128, 1))
    assert_size_stride(primals_637, (128, ), (1, ))
    assert_size_stride(primals_638, (128, 128), (128, 1))
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (128, 512), (512, 1))
    assert_size_stride(primals_641, (128, ), (1, ))
    assert_size_stride(primals_642, (128, 128), (128, 1))
    assert_size_stride(primals_643, (128, ), (1, ))
    assert_size_stride(primals_644, (512, 128), (128, 1))
    assert_size_stride(primals_645, (512, ), (1, ))
    assert_size_stride(primals_646, (128, 512), (512, 1))
    assert_size_stride(primals_647, (128, ), (1, ))
    assert_size_stride(primals_648, (512, 128), (128, 1))
    assert_size_stride(primals_649, (512, ), (1, ))
    assert_size_stride(primals_650, (128, 512), (512, 1))
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (512, 128), (128, 1))
    assert_size_stride(primals_653, (512, ), (1, ))
    assert_size_stride(primals_654, (128, 512), (512, 1))
    assert_size_stride(primals_655, (128, ), (1, ))
    assert_size_stride(primals_656, (512, 128), (128, 1))
    assert_size_stride(primals_657, (512, ), (1, ))
    assert_size_stride(primals_658, (128, 512), (512, 1))
    assert_size_stride(primals_659, (128, ), (1, ))
    assert_size_stride(primals_660, (512, 128), (128, 1))
    assert_size_stride(primals_661, (512, ), (1, ))
    assert_size_stride(primals_662, (128, 512), (512, 1))
    assert_size_stride(primals_663, (128, ), (1, ))
    assert_size_stride(primals_664, (128, 512), (512, 1))
    assert_size_stride(primals_665, (128, ), (1, ))
    assert_size_stride(primals_666, (128, 128), (128, 1))
    assert_size_stride(primals_667, (128, ), (1, ))
    assert_size_stride(primals_668, (128, 128), (128, 1))
    assert_size_stride(primals_669, (128, ), (1, ))
    assert_size_stride(primals_670, (128, 512), (512, 1))
    assert_size_stride(primals_671, (128, ), (1, ))
    assert_size_stride(primals_672, (128, 128), (128, 1))
    assert_size_stride(primals_673, (128, ), (1, ))
    assert_size_stride(primals_674, (512, 128), (128, 1))
    assert_size_stride(primals_675, (512, ), (1, ))
    assert_size_stride(primals_676, (128, 512), (512, 1))
    assert_size_stride(primals_677, (128, ), (1, ))
    assert_size_stride(primals_678, (512, 128), (128, 1))
    assert_size_stride(primals_679, (512, ), (1, ))
    assert_size_stride(primals_680, (128, 512), (512, 1))
    assert_size_stride(primals_681, (128, ), (1, ))
    assert_size_stride(primals_682, (512, 128), (128, 1))
    assert_size_stride(primals_683, (512, ), (1, ))
    assert_size_stride(primals_684, (128, 512), (512, 1))
    assert_size_stride(primals_685, (128, ), (1, ))
    assert_size_stride(primals_686, (512, 128), (128, 1))
    assert_size_stride(primals_687, (512, ), (1, ))
    assert_size_stride(primals_688, (128, 512), (512, 1))
    assert_size_stride(primals_689, (128, ), (1, ))
    assert_size_stride(primals_690, (512, 128), (128, 1))
    assert_size_stride(primals_691, (512, ), (1, ))
    assert_size_stride(primals_692, (128, 512), (512, 1))
    assert_size_stride(primals_693, (128, ), (1, ))
    assert_size_stride(primals_694, (128, 512), (512, 1))
    assert_size_stride(primals_695, (128, ), (1, ))
    assert_size_stride(primals_696, (128, 128), (128, 1))
    assert_size_stride(primals_697, (128, ), (1, ))
    assert_size_stride(primals_698, (128, 128), (128, 1))
    assert_size_stride(primals_699, (128, ), (1, ))
    assert_size_stride(primals_700, (128, 512), (512, 1))
    assert_size_stride(primals_701, (128, ), (1, ))
    assert_size_stride(primals_702, (128, 128), (128, 1))
    assert_size_stride(primals_703, (128, ), (1, ))
    assert_size_stride(primals_704, (512, 128), (128, 1))
    assert_size_stride(primals_705, (512, ), (1, ))
    assert_size_stride(primals_706, (128, 512), (512, 1))
    assert_size_stride(primals_707, (128, ), (1, ))
    assert_size_stride(primals_708, (512, 128), (128, 1))
    assert_size_stride(primals_709, (512, ), (1, ))
    assert_size_stride(primals_710, (128, 512), (512, 1))
    assert_size_stride(primals_711, (128, ), (1, ))
    assert_size_stride(primals_712, (512, 128), (128, 1))
    assert_size_stride(primals_713, (512, ), (1, ))
    assert_size_stride(primals_714, (128, 512), (512, 1))
    assert_size_stride(primals_715, (128, ), (1, ))
    assert_size_stride(primals_716, (512, 128), (128, 1))
    assert_size_stride(primals_717, (512, ), (1, ))
    assert_size_stride(primals_718, (128, 512), (512, 1))
    assert_size_stride(primals_719, (128, ), (1, ))
    assert_size_stride(primals_720, (512, 128), (128, 1))
    assert_size_stride(primals_721, (512, ), (1, ))
    assert_size_stride(primals_722, (128, 512), (512, 1))
    assert_size_stride(primals_723, (128, ), (1, ))
    assert_size_stride(primals_724, (128, 512), (512, 1))
    assert_size_stride(primals_725, (128, ), (1, ))
    assert_size_stride(primals_726, (128, 128), (128, 1))
    assert_size_stride(primals_727, (128, ), (1, ))
    assert_size_stride(primals_728, (128, 128), (128, 1))
    assert_size_stride(primals_729, (128, ), (1, ))
    assert_size_stride(primals_730, (128, 512), (512, 1))
    assert_size_stride(primals_731, (128, ), (1, ))
    assert_size_stride(primals_732, (128, 128), (128, 1))
    assert_size_stride(primals_733, (128, ), (1, ))
    assert_size_stride(primals_734, (512, 128), (128, 1))
    assert_size_stride(primals_735, (512, ), (1, ))
    assert_size_stride(primals_736, (128, 512), (512, 1))
    assert_size_stride(primals_737, (128, ), (1, ))
    assert_size_stride(primals_738, (512, 128), (128, 1))
    assert_size_stride(primals_739, (512, ), (1, ))
    assert_size_stride(primals_740, (128, 512), (512, 1))
    assert_size_stride(primals_741, (128, ), (1, ))
    assert_size_stride(primals_742, (512, 128), (128, 1))
    assert_size_stride(primals_743, (512, ), (1, ))
    assert_size_stride(primals_744, (128, 512), (512, 1))
    assert_size_stride(primals_745, (128, ), (1, ))
    assert_size_stride(primals_746, (512, 128), (128, 1))
    assert_size_stride(primals_747, (512, ), (1, ))
    assert_size_stride(primals_748, (128, 512), (512, 1))
    assert_size_stride(primals_749, (128, ), (1, ))
    assert_size_stride(primals_750, (512, 128), (128, 1))
    assert_size_stride(primals_751, (512, ), (1, ))
    assert_size_stride(primals_752, (128, 512), (512, 1))
    assert_size_stride(primals_753, (128, ), (1, ))
    assert_size_stride(primals_754, (128, 512), (512, 1))
    assert_size_stride(primals_755, (128, ), (1, ))
    assert_size_stride(primals_756, (128, 128), (128, 1))
    assert_size_stride(primals_757, (128, ), (1, ))
    assert_size_stride(primals_758, (128, 128), (128, 1))
    assert_size_stride(primals_759, (128, ), (1, ))
    assert_size_stride(primals_760, (128, 512), (512, 1))
    assert_size_stride(primals_761, (128, ), (1, ))
    assert_size_stride(primals_762, (128, 128), (128, 1))
    assert_size_stride(primals_763, (128, ), (1, ))
    assert_size_stride(primals_764, (512, 128), (128, 1))
    assert_size_stride(primals_765, (512, ), (1, ))
    assert_size_stride(primals_766, (128, 512), (512, 1))
    assert_size_stride(primals_767, (128, ), (1, ))
    assert_size_stride(primals_768, (512, 128), (128, 1))
    assert_size_stride(primals_769, (512, ), (1, ))
    assert_size_stride(primals_770, (128, 512), (512, 1))
    assert_size_stride(primals_771, (128, ), (1, ))
    assert_size_stride(primals_772, (512, 128), (128, 1))
    assert_size_stride(primals_773, (512, ), (1, ))
    assert_size_stride(primals_774, (128, 512), (512, 1))
    assert_size_stride(primals_775, (128, ), (1, ))
    assert_size_stride(primals_776, (512, 128), (128, 1))
    assert_size_stride(primals_777, (512, ), (1, ))
    assert_size_stride(primals_778, (128, 512), (512, 1))
    assert_size_stride(primals_779, (128, ), (1, ))
    assert_size_stride(primals_780, (512, 128), (128, 1))
    assert_size_stride(primals_781, (512, ), (1, ))
    assert_size_stride(primals_782, (128, 512), (512, 1))
    assert_size_stride(primals_783, (128, ), (1, ))
    assert_size_stride(primals_784, (128, 512), (512, 1))
    assert_size_stride(primals_785, (128, ), (1, ))
    assert_size_stride(primals_786, (128, 128), (128, 1))
    assert_size_stride(primals_787, (128, ), (1, ))
    assert_size_stride(primals_788, (128, 128), (128, 1))
    assert_size_stride(primals_789, (128, ), (1, ))
    assert_size_stride(primals_790, (128, 512), (512, 1))
    assert_size_stride(primals_791, (128, ), (1, ))
    assert_size_stride(primals_792, (128, 128), (128, 1))
    assert_size_stride(primals_793, (128, ), (1, ))
    assert_size_stride(primals_794, (512, 128), (128, 1))
    assert_size_stride(primals_795, (512, ), (1, ))
    assert_size_stride(primals_796, (128, 512), (512, 1))
    assert_size_stride(primals_797, (128, ), (1, ))
    assert_size_stride(primals_798, (512, 128), (128, 1))
    assert_size_stride(primals_799, (512, ), (1, ))
    assert_size_stride(primals_800, (128, 512), (512, 1))
    assert_size_stride(primals_801, (128, ), (1, ))
    assert_size_stride(primals_802, (512, 128), (128, 1))
    assert_size_stride(primals_803, (512, ), (1, ))
    assert_size_stride(primals_804, (128, 512), (512, 1))
    assert_size_stride(primals_805, (128, ), (1, ))
    assert_size_stride(primals_806, (512, 128), (128, 1))
    assert_size_stride(primals_807, (512, ), (1, ))
    assert_size_stride(primals_808, (128, 512), (512, 1))
    assert_size_stride(primals_809, (128, ), (1, ))
    assert_size_stride(primals_810, (512, 128), (128, 1))
    assert_size_stride(primals_811, (512, ), (1, ))
    assert_size_stride(primals_812, (128, 512), (512, 1))
    assert_size_stride(primals_813, (128, ), (1, ))
    assert_size_stride(primals_814, (128, 512), (512, 1))
    assert_size_stride(primals_815, (128, ), (1, ))
    assert_size_stride(primals_816, (128, 128), (128, 1))
    assert_size_stride(primals_817, (128, ), (1, ))
    assert_size_stride(primals_818, (128, 128), (128, 1))
    assert_size_stride(primals_819, (128, ), (1, ))
    assert_size_stride(primals_820, (128, 512), (512, 1))
    assert_size_stride(primals_821, (128, ), (1, ))
    assert_size_stride(primals_822, (128, 128), (128, 1))
    assert_size_stride(primals_823, (128, ), (1, ))
    assert_size_stride(primals_824, (512, 128), (128, 1))
    assert_size_stride(primals_825, (512, ), (1, ))
    assert_size_stride(primals_826, (128, 512), (512, 1))
    assert_size_stride(primals_827, (128, ), (1, ))
    assert_size_stride(primals_828, (512, 128), (128, 1))
    assert_size_stride(primals_829, (512, ), (1, ))
    assert_size_stride(primals_830, (128, 512), (512, 1))
    assert_size_stride(primals_831, (128, ), (1, ))
    assert_size_stride(primals_832, (512, 128), (128, 1))
    assert_size_stride(primals_833, (512, ), (1, ))
    assert_size_stride(primals_834, (128, 512), (512, 1))
    assert_size_stride(primals_835, (128, ), (1, ))
    assert_size_stride(primals_836, (512, 128), (128, 1))
    assert_size_stride(primals_837, (512, ), (1, ))
    assert_size_stride(primals_838, (128, 512), (512, 1))
    assert_size_stride(primals_839, (128, ), (1, ))
    assert_size_stride(primals_840, (512, 128), (128, 1))
    assert_size_stride(primals_841, (512, ), (1, ))
    assert_size_stride(primals_842, (128, 512), (512, 1))
    assert_size_stride(primals_843, (128, ), (1, ))
    assert_size_stride(primals_844, (128, 512), (512, 1))
    assert_size_stride(primals_845, (128, ), (1, ))
    assert_size_stride(primals_846, (128, 128), (128, 1))
    assert_size_stride(primals_847, (128, ), (1, ))
    assert_size_stride(primals_848, (128, 128), (128, 1))
    assert_size_stride(primals_849, (128, ), (1, ))
    assert_size_stride(primals_850, (128, 512), (512, 1))
    assert_size_stride(primals_851, (128, ), (1, ))
    assert_size_stride(primals_852, (128, 128), (128, 1))
    assert_size_stride(primals_853, (128, ), (1, ))
    assert_size_stride(primals_854, (512, 128), (128, 1))
    assert_size_stride(primals_855, (512, ), (1, ))
    assert_size_stride(primals_856, (128, 512), (512, 1))
    assert_size_stride(primals_857, (128, ), (1, ))
    assert_size_stride(primals_858, (512, 128), (128, 1))
    assert_size_stride(primals_859, (512, ), (1, ))
    assert_size_stride(primals_860, (128, 512), (512, 1))
    assert_size_stride(primals_861, (128, ), (1, ))
    assert_size_stride(primals_862, (512, 128), (128, 1))
    assert_size_stride(primals_863, (512, ), (1, ))
    assert_size_stride(primals_864, (128, 512), (512, 1))
    assert_size_stride(primals_865, (128, ), (1, ))
    assert_size_stride(primals_866, (512, 128), (128, 1))
    assert_size_stride(primals_867, (512, ), (1, ))
    assert_size_stride(primals_868, (128, 512), (512, 1))
    assert_size_stride(primals_869, (128, ), (1, ))
    assert_size_stride(primals_870, (512, 128), (128, 1))
    assert_size_stride(primals_871, (512, ), (1, ))
    assert_size_stride(primals_872, (128, 512), (512, 1))
    assert_size_stride(primals_873, (128, ), (1, ))
    assert_size_stride(primals_874, (128, 512), (512, 1))
    assert_size_stride(primals_875, (128, ), (1, ))
    assert_size_stride(primals_876, (128, 128), (128, 1))
    assert_size_stride(primals_877, (128, ), (1, ))
    assert_size_stride(primals_878, (128, 128), (128, 1))
    assert_size_stride(primals_879, (128, ), (1, ))
    assert_size_stride(primals_880, (128, 512), (512, 1))
    assert_size_stride(primals_881, (128, ), (1, ))
    assert_size_stride(primals_882, (128, 128), (128, 1))
    assert_size_stride(primals_883, (128, ), (1, ))
    assert_size_stride(primals_884, (512, 128), (128, 1))
    assert_size_stride(primals_885, (512, ), (1, ))
    assert_size_stride(primals_886, (128, 512), (512, 1))
    assert_size_stride(primals_887, (128, ), (1, ))
    assert_size_stride(primals_888, (512, 128), (128, 1))
    assert_size_stride(primals_889, (512, ), (1, ))
    assert_size_stride(primals_890, (128, 512), (512, 1))
    assert_size_stride(primals_891, (128, ), (1, ))
    assert_size_stride(primals_892, (512, 128), (128, 1))
    assert_size_stride(primals_893, (512, ), (1, ))
    assert_size_stride(primals_894, (128, 512), (512, 1))
    assert_size_stride(primals_895, (128, ), (1, ))
    assert_size_stride(primals_896, (512, 128), (128, 1))
    assert_size_stride(primals_897, (512, ), (1, ))
    assert_size_stride(primals_898, (128, 512), (512, 1))
    assert_size_stride(primals_899, (128, ), (1, ))
    assert_size_stride(primals_900, (512, 128), (128, 1))
    assert_size_stride(primals_901, (512, ), (1, ))
    assert_size_stride(primals_902, (128, 512), (512, 1))
    assert_size_stride(primals_903, (128, ), (1, ))
    assert_size_stride(primals_904, (128, 512), (512, 1))
    assert_size_stride(primals_905, (128, ), (1, ))
    assert_size_stride(primals_906, (128, 128), (128, 1))
    assert_size_stride(primals_907, (128, ), (1, ))
    assert_size_stride(primals_908, (128, 128), (128, 1))
    assert_size_stride(primals_909, (128, ), (1, ))
    assert_size_stride(primals_910, (128, 512), (512, 1))
    assert_size_stride(primals_911, (128, ), (1, ))
    assert_size_stride(primals_912, (128, 128), (128, 1))
    assert_size_stride(primals_913, (128, ), (1, ))
    assert_size_stride(primals_914, (512, 128), (128, 1))
    assert_size_stride(primals_915, (512, ), (1, ))
    assert_size_stride(primals_916, (128, 512), (512, 1))
    assert_size_stride(primals_917, (128, ), (1, ))
    assert_size_stride(primals_918, (512, 128), (128, 1))
    assert_size_stride(primals_919, (512, ), (1, ))
    assert_size_stride(primals_920, (128, 512), (512, 1))
    assert_size_stride(primals_921, (128, ), (1, ))
    assert_size_stride(primals_922, (512, 128), (128, 1))
    assert_size_stride(primals_923, (512, ), (1, ))
    assert_size_stride(primals_924, (128, 512), (512, 1))
    assert_size_stride(primals_925, (128, ), (1, ))
    assert_size_stride(primals_926, (512, 128), (128, 1))
    assert_size_stride(primals_927, (512, ), (1, ))
    assert_size_stride(primals_928, (128, 512), (512, 1))
    assert_size_stride(primals_929, (128, ), (1, ))
    assert_size_stride(primals_930, (512, 128), (128, 1))
    assert_size_stride(primals_931, (512, ), (1, ))
    assert_size_stride(primals_932, (128, 512), (512, 1))
    assert_size_stride(primals_933, (128, ), (1, ))
    assert_size_stride(primals_934, (128, 512), (512, 1))
    assert_size_stride(primals_935, (128, ), (1, ))
    assert_size_stride(primals_936, (128, 128), (128, 1))
    assert_size_stride(primals_937, (128, ), (1, ))
    assert_size_stride(primals_938, (128, 128), (128, 1))
    assert_size_stride(primals_939, (128, ), (1, ))
    assert_size_stride(primals_940, (128, 512), (512, 1))
    assert_size_stride(primals_941, (128, ), (1, ))
    assert_size_stride(primals_942, (128, 128), (128, 1))
    assert_size_stride(primals_943, (128, ), (1, ))
    assert_size_stride(primals_944, (512, 128), (128, 1))
    assert_size_stride(primals_945, (512, ), (1, ))
    assert_size_stride(primals_946, (128, 512), (512, 1))
    assert_size_stride(primals_947, (128, ), (1, ))
    assert_size_stride(primals_948, (512, 128), (128, 1))
    assert_size_stride(primals_949, (512, ), (1, ))
    assert_size_stride(primals_950, (128, 512), (512, 1))
    assert_size_stride(primals_951, (128, ), (1, ))
    assert_size_stride(primals_952, (512, 128), (128, 1))
    assert_size_stride(primals_953, (512, ), (1, ))
    assert_size_stride(primals_954, (128, 512), (512, 1))
    assert_size_stride(primals_955, (128, ), (1, ))
    assert_size_stride(primals_956, (512, 128), (128, 1))
    assert_size_stride(primals_957, (512, ), (1, ))
    assert_size_stride(primals_958, (128, 512), (512, 1))
    assert_size_stride(primals_959, (128, ), (1, ))
    assert_size_stride(primals_960, (512, 128), (128, 1))
    assert_size_stride(primals_961, (512, ), (1, ))
    assert_size_stride(primals_962, (128, 512), (512, 1))
    assert_size_stride(primals_963, (128, ), (1, ))
    assert_size_stride(primals_964, (128, 512), (512, 1))
    assert_size_stride(primals_965, (128, ), (1, ))
    assert_size_stride(primals_966, (128, 128), (128, 1))
    assert_size_stride(primals_967, (128, ), (1, ))
    assert_size_stride(primals_968, (128, 128), (128, 1))
    assert_size_stride(primals_969, (128, ), (1, ))
    assert_size_stride(primals_970, (128, 512), (512, 1))
    assert_size_stride(primals_971, (128, ), (1, ))
    assert_size_stride(primals_972, (128, 128), (128, 1))
    assert_size_stride(primals_973, (128, ), (1, ))
    assert_size_stride(primals_974, (512, 128), (128, 1))
    assert_size_stride(primals_975, (512, ), (1, ))
    assert_size_stride(primals_976, (128, 512), (512, 1))
    assert_size_stride(primals_977, (128, ), (1, ))
    assert_size_stride(primals_978, (512, 128), (128, 1))
    assert_size_stride(primals_979, (512, ), (1, ))
    assert_size_stride(primals_980, (128, 512), (512, 1))
    assert_size_stride(primals_981, (128, ), (1, ))
    assert_size_stride(primals_982, (512, 128), (128, 1))
    assert_size_stride(primals_983, (512, ), (1, ))
    assert_size_stride(primals_984, (128, 512), (512, 1))
    assert_size_stride(primals_985, (128, ), (1, ))
    assert_size_stride(primals_986, (512, 128), (128, 1))
    assert_size_stride(primals_987, (512, ), (1, ))
    assert_size_stride(primals_988, (128, 512), (512, 1))
    assert_size_stride(primals_989, (128, ), (1, ))
    assert_size_stride(primals_990, (512, 128), (128, 1))
    assert_size_stride(primals_991, (512, ), (1, ))
    assert_size_stride(primals_992, (128, 512), (512, 1))
    assert_size_stride(primals_993, (128, ), (1, ))
    assert_size_stride(primals_994, (128, 512), (512, 1))
    assert_size_stride(primals_995, (128, ), (1, ))
    assert_size_stride(primals_996, (128, 128), (128, 1))
    assert_size_stride(primals_997, (128, ), (1, ))
    assert_size_stride(primals_998, (128, 128), (128, 1))
    assert_size_stride(primals_999, (128, ), (1, ))
    assert_size_stride(primals_1000, (128, 512), (512, 1))
    assert_size_stride(primals_1001, (128, ), (1, ))
    assert_size_stride(primals_1002, (128, 128), (128, 1))
    assert_size_stride(primals_1003, (128, ), (1, ))
    assert_size_stride(primals_1004, (512, 128), (128, 1))
    assert_size_stride(primals_1005, (512, ), (1, ))
    assert_size_stride(primals_1006, (128, 512), (512, 1))
    assert_size_stride(primals_1007, (128, ), (1, ))
    assert_size_stride(primals_1008, (512, 128), (128, 1))
    assert_size_stride(primals_1009, (512, ), (1, ))
    assert_size_stride(primals_1010, (128, 512), (512, 1))
    assert_size_stride(primals_1011, (128, ), (1, ))
    assert_size_stride(primals_1012, (512, 128), (128, 1))
    assert_size_stride(primals_1013, (512, ), (1, ))
    assert_size_stride(primals_1014, (128, 512), (512, 1))
    assert_size_stride(primals_1015, (128, ), (1, ))
    assert_size_stride(primals_1016, (512, 128), (128, 1))
    assert_size_stride(primals_1017, (512, ), (1, ))
    assert_size_stride(primals_1018, (128, 512), (512, 1))
    assert_size_stride(primals_1019, (128, ), (1, ))
    assert_size_stride(primals_1020, (512, 128), (128, 1))
    assert_size_stride(primals_1021, (512, ), (1, ))
    assert_size_stride(primals_1022, (128, 512), (512, 1))
    assert_size_stride(primals_1023, (128, ), (1, ))
    assert_size_stride(primals_1024, (128, 512), (512, 1))
    assert_size_stride(primals_1025, (128, ), (1, ))
    assert_size_stride(primals_1026, (128, 128), (128, 1))
    assert_size_stride(primals_1027, (128, ), (1, ))
    assert_size_stride(primals_1028, (128, 128), (128, 1))
    assert_size_stride(primals_1029, (128, ), (1, ))
    assert_size_stride(primals_1030, (128, 512), (512, 1))
    assert_size_stride(primals_1031, (128, ), (1, ))
    assert_size_stride(primals_1032, (128, 128), (128, 1))
    assert_size_stride(primals_1033, (128, ), (1, ))
    assert_size_stride(primals_1034, (512, 128), (128, 1))
    assert_size_stride(primals_1035, (512, ), (1, ))
    assert_size_stride(primals_1036, (128, 512), (512, 1))
    assert_size_stride(primals_1037, (128, ), (1, ))
    assert_size_stride(primals_1038, (512, 128), (128, 1))
    assert_size_stride(primals_1039, (512, ), (1, ))
    assert_size_stride(primals_1040, (128, 512), (512, 1))
    assert_size_stride(primals_1041, (128, ), (1, ))
    assert_size_stride(primals_1042, (512, 128), (128, 1))
    assert_size_stride(primals_1043, (512, ), (1, ))
    assert_size_stride(primals_1044, (128, 512), (512, 1))
    assert_size_stride(primals_1045, (128, ), (1, ))
    assert_size_stride(primals_1046, (512, 128), (128, 1))
    assert_size_stride(primals_1047, (512, ), (1, ))
    assert_size_stride(primals_1048, (128, 512), (512, 1))
    assert_size_stride(primals_1049, (128, ), (1, ))
    assert_size_stride(primals_1050, (512, 128), (128, 1))
    assert_size_stride(primals_1051, (512, ), (1, ))
    assert_size_stride(primals_1052, (128, 512), (512, 1))
    assert_size_stride(primals_1053, (128, ), (1, ))
    assert_size_stride(primals_1054, (128, 512), (512, 1))
    assert_size_stride(primals_1055, (128, ), (1, ))
    assert_size_stride(primals_1056, (128, 128), (128, 1))
    assert_size_stride(primals_1057, (128, ), (1, ))
    assert_size_stride(primals_1058, (128, 128), (128, 1))
    assert_size_stride(primals_1059, (128, ), (1, ))
    assert_size_stride(primals_1060, (128, 512), (512, 1))
    assert_size_stride(primals_1061, (128, ), (1, ))
    assert_size_stride(primals_1062, (128, 128), (128, 1))
    assert_size_stride(primals_1063, (128, ), (1, ))
    assert_size_stride(primals_1064, (512, 128), (128, 1))
    assert_size_stride(primals_1065, (512, ), (1, ))
    assert_size_stride(primals_1066, (128, 512), (512, 1))
    assert_size_stride(primals_1067, (128, ), (1, ))
    assert_size_stride(primals_1068, (512, 128), (128, 1))
    assert_size_stride(primals_1069, (512, ), (1, ))
    assert_size_stride(primals_1070, (128, 512), (512, 1))
    assert_size_stride(primals_1071, (128, ), (1, ))
    assert_size_stride(primals_1072, (512, 128), (128, 1))
    assert_size_stride(primals_1073, (512, ), (1, ))
    assert_size_stride(primals_1074, (128, 512), (512, 1))
    assert_size_stride(primals_1075, (128, ), (1, ))
    assert_size_stride(primals_1076, (512, 128), (128, 1))
    assert_size_stride(primals_1077, (512, ), (1, ))
    assert_size_stride(primals_1078, (128, 512), (512, 1))
    assert_size_stride(primals_1079, (128, ), (1, ))
    assert_size_stride(primals_1080, (512, 128), (128, 1))
    assert_size_stride(primals_1081, (512, ), (1, ))
    assert_size_stride(primals_1082, (128, 512), (512, 1))
    assert_size_stride(primals_1083, (128, ), (1, ))
    assert_size_stride(primals_1084, (128, 512), (512, 1))
    assert_size_stride(primals_1085, (128, ), (1, ))
    assert_size_stride(primals_1086, (128, 128), (128, 1))
    assert_size_stride(primals_1087, (128, ), (1, ))
    assert_size_stride(primals_1088, (128, 128), (128, 1))
    assert_size_stride(primals_1089, (128, ), (1, ))
    assert_size_stride(primals_1090, (128, 512), (512, 1))
    assert_size_stride(primals_1091, (128, ), (1, ))
    assert_size_stride(primals_1092, (128, 128), (128, 1))
    assert_size_stride(primals_1093, (128, ), (1, ))
    assert_size_stride(primals_1094, (512, 128), (128, 1))
    assert_size_stride(primals_1095, (512, ), (1, ))
    assert_size_stride(primals_1096, (128, 512), (512, 1))
    assert_size_stride(primals_1097, (128, ), (1, ))
    assert_size_stride(primals_1098, (512, 128), (128, 1))
    assert_size_stride(primals_1099, (512, ), (1, ))
    assert_size_stride(primals_1100, (128, 512), (512, 1))
    assert_size_stride(primals_1101, (128, ), (1, ))
    assert_size_stride(primals_1102, (512, 128), (128, 1))
    assert_size_stride(primals_1103, (512, ), (1, ))
    assert_size_stride(primals_1104, (128, 512), (512, 1))
    assert_size_stride(primals_1105, (128, ), (1, ))
    assert_size_stride(primals_1106, (512, 128), (128, 1))
    assert_size_stride(primals_1107, (512, ), (1, ))
    assert_size_stride(primals_1108, (128, 512), (512, 1))
    assert_size_stride(primals_1109, (128, ), (1, ))
    assert_size_stride(primals_1110, (512, 128), (128, 1))
    assert_size_stride(primals_1111, (512, ), (1, ))
    assert_size_stride(primals_1112, (2, 512), (512, 1))
    assert_size_stride(primals_1113, (2, ), (1, ))
    assert_size_stride(primals_1114, (1, 512), (512, 1))
    assert_size_stride(primals_1115, (1, 128), (128, 1))
    assert_size_stride(primals_1116, (1, ), (1, ))
    assert_size_stride(primals_1117, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 128, grid=grid(128), stream=stream0)
        buf1 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, inputs_embeds_2], Original ATen: [aten.cat, aten.view]
        triton_poi_fused_cat_view_1.run(primals_1115, primals_387, buf1, 49152, grid=grid(49152), stream=stream0)
        del primals_387
        buf2 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf1, reinterpret_tensor(primals_388, (384, 512), (1, 384), 0), out=buf2)
        buf3 = reinterpret_tensor(buf2, (1, 128, 512), (65536, 512, 1), 0); del buf2  # reuse
        buf4 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, layer_input, mul_1, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.mul, aten.view]
        triton_poi_fused_add_embedding_mul_view_2.run(buf3, primals_389, primals_1114, primals_390, primals_391, primals_1, primals_2, buf4, 65536, grid=grid(65536), stream=stream0)
        del primals_389
        del primals_390
        del primals_391
        buf5 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_393, buf4, reinterpret_tensor(primals_392, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf5)
        del primals_393
        buf6 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_395, buf4, reinterpret_tensor(primals_394, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf6)
        del primals_395
        buf7 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor, mixed_query_layer, mul_3], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf6, primals_5, primals_6, buf7, 16384, grid=grid(16384), stream=stream0)
        del primals_6
        buf8 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, reinterpret_tensor(primals_396, (128, 128), (1, 128), 0), out=buf8)
        buf9 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, reinterpret_tensor(primals_398, (128, 128), (1, 128), 0), out=buf9)
        buf10 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_400, (512, 128), (1, 512), 0), out=buf10)
        buf11 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf8, primals_397, buf11, 16384, grid=grid(16384), stream=stream0)
        del primals_397
        buf12 = reinterpret_tensor(buf8, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf9, primals_399, buf12, 16384, grid=grid(16384), stream=stream0)
        del primals_399
        buf13 = reinterpret_tensor(buf9, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf10, primals_401, buf13, 16384, grid=grid(16384), stream=stream0)
        del primals_401
        # Source Nodes: [], Original ATen: []
        buf14 = aten._scaled_dot_product_efficient_attention(buf11, buf12, buf13, None, True, 0.1, scale=0.17677669529663687)
        buf15 = buf14[0]
        buf16 = buf14[1]
        buf17 = buf14[2]
        buf18 = buf14[3]
        del buf14
        buf19 = buf10; del buf10  # reuse
        # Source Nodes: [layer_outputs], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf15, buf19, 16384, grid=grid(16384), stream=stream0)
        buf20 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_403, buf19, reinterpret_tensor(primals_402, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf20)
        del primals_403
        buf21 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, attention_output, layer_input_4, mul_2, mul_4], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf20, buf5, primals_3, primals_4, primals_7, primals_8, buf21, 16384, grid=grid(16384), stream=stream0)
        buf22 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (128, 128), (128, 1), 0), reinterpret_tensor(primals_404, (128, 512), (1, 128), 0), out=buf22)
        buf23 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output, layer_outputs_2], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf22, primals_405, buf23, 65536, grid=grid(65536), stream=stream0)
        buf24 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_407, buf23, reinterpret_tensor(primals_406, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf24)
        del primals_407
        buf25 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attention_output_1, hidden_states_2, mul_5], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf24, buf21, primals_9, primals_10, buf25, 16384, grid=grid(16384), stream=stream0)
        buf26 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf25, reinterpret_tensor(primals_408, (128, 512), (1, 128), 0), out=buf26)
        buf27 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_1, layer_outputs_5], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf26, primals_409, buf27, 65536, grid=grid(65536), stream=stream0)
        buf28 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_411, buf27, reinterpret_tensor(primals_410, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf28)
        del primals_411
        buf29 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_8, attention_output_1, attention_output_2, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf28, buf24, buf21, primals_9, primals_10, primals_11, primals_12, buf29, 16384, grid=grid(16384), stream=stream0)
        buf30 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (128, 128), (128, 1), 0), reinterpret_tensor(primals_412, (128, 512), (1, 128), 0), out=buf30)
        buf31 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_2, layer_outputs_8], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf30, primals_413, buf31, 65536, grid=grid(65536), stream=stream0)
        buf32 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_415, buf31, reinterpret_tensor(primals_414, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf32)
        del primals_415
        buf33 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_3, hidden_states_6, mul_7], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf32, buf29, primals_13, primals_14, buf33, 16384, grid=grid(16384), stream=stream0)
        buf34 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_416, (128, 512), (1, 128), 0), out=buf34)
        buf35 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_3, layer_output], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf34, primals_417, buf35, 65536, grid=grid(65536), stream=stream0)
        buf36 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_419, buf35, reinterpret_tensor(primals_418, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf36)
        del primals_419
        buf37 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, add_14, attention_output_3, layer_output_1, layer_outputs_11, mul_7, mul_8], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf36, buf32, buf29, primals_13, primals_14, primals_15, primals_16, buf37, 16384, grid=grid(16384), stream=stream0)
        del primals_16
        buf38 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf37, reinterpret_tensor(primals_420, (128, 512), (1, 128), 0), out=buf38)
        buf39 = reinterpret_tensor(buf38, (1, 128, 512), (65536, 512, 1), 0); del buf38  # reuse
        buf40 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_16, embeddings_1, layer_input_5, mul_1, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_10.run(buf39, primals_421, buf3, primals_1, primals_2, primals_17, primals_18, buf40, 65536, grid=grid(65536), stream=stream0)
        del primals_2
        del primals_421
        buf41 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_423, buf40, reinterpret_tensor(primals_422, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf41)
        del primals_423
        buf42 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_425, buf40, reinterpret_tensor(primals_424, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf42)
        del primals_425
        buf43 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_1, mixed_query_layer_1, mul_11], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf42, primals_21, primals_22, buf43, 16384, grid=grid(16384), stream=stream0)
        del primals_22
        buf44 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf43, reinterpret_tensor(primals_426, (128, 128), (1, 128), 0), out=buf44)
        buf45 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf43, reinterpret_tensor(primals_428, (128, 128), (1, 128), 0), out=buf45)
        buf46 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_430, (512, 128), (1, 512), 0), out=buf46)
        buf47 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf44, primals_427, buf47, 16384, grid=grid(16384), stream=stream0)
        del primals_427
        buf48 = reinterpret_tensor(buf44, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf45, primals_429, buf48, 16384, grid=grid(16384), stream=stream0)
        del primals_429
        buf49 = reinterpret_tensor(buf45, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf46, primals_431, buf49, 16384, grid=grid(16384), stream=stream0)
        del primals_431
        # Source Nodes: [], Original ATen: []
        buf50 = aten._scaled_dot_product_efficient_attention(buf47, buf48, buf49, None, True, 0.1, scale=0.17677669529663687)
        buf51 = buf50[0]
        buf52 = buf50[1]
        buf53 = buf50[2]
        buf54 = buf50[3]
        del buf50
        buf55 = buf46; del buf46  # reuse
        # Source Nodes: [layer_outputs_14], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf51, buf55, 16384, grid=grid(16384), stream=stream0)
        buf56 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_433, buf55, reinterpret_tensor(primals_432, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf56)
        del primals_433
        buf57 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_5, layer_input_9, mul_10, mul_12], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf56, buf41, primals_19, primals_20, primals_23, primals_24, buf57, 16384, grid=grid(16384), stream=stream0)
        buf58 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (128, 128), (128, 1), 0), reinterpret_tensor(primals_434, (128, 512), (1, 128), 0), out=buf58)
        buf59 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_4, layer_outputs_16], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf58, primals_435, buf59, 65536, grid=grid(65536), stream=stream0)
        buf60 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_437, buf59, reinterpret_tensor(primals_436, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf60)
        del primals_437
        buf61 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attention_output_6, hidden_states_11, mul_13], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf60, buf57, primals_25, primals_26, buf61, 16384, grid=grid(16384), stream=stream0)
        buf62 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_438, (128, 512), (1, 128), 0), out=buf62)
        buf63 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_5, layer_outputs_19], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf62, primals_439, buf63, 65536, grid=grid(65536), stream=stream0)
        buf64 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_441, buf63, reinterpret_tensor(primals_440, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf64)
        del primals_441
        buf65 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, add_25, attention_output_6, attention_output_7, mul_13, mul_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf64, buf60, buf57, primals_25, primals_26, primals_27, primals_28, buf65, 16384, grid=grid(16384), stream=stream0)
        buf66 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (128, 128), (128, 1), 0), reinterpret_tensor(primals_442, (128, 512), (1, 128), 0), out=buf66)
        buf67 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_6, layer_outputs_22], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf66, primals_443, buf67, 65536, grid=grid(65536), stream=stream0)
        buf68 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_445, buf67, reinterpret_tensor(primals_444, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf68)
        del primals_445
        buf69 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_8, hidden_states_15, mul_15], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf68, buf65, primals_29, primals_30, buf69, 16384, grid=grid(16384), stream=stream0)
        buf70 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf69, reinterpret_tensor(primals_446, (128, 512), (1, 128), 0), out=buf70)
        buf71 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_7, layer_output_4], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf70, primals_447, buf71, 65536, grid=grid(65536), stream=stream0)
        buf72 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_449, buf71, reinterpret_tensor(primals_448, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf72)
        del primals_449
        buf73 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, add_29, attention_output_8, layer_output_5, layer_outputs_25, mul_15, mul_16], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf72, buf68, buf65, primals_29, primals_30, primals_31, primals_32, buf73, 16384, grid=grid(16384), stream=stream0)
        del primals_32
        buf74 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_451, buf73, reinterpret_tensor(primals_450, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf74)
        del primals_451
        buf75 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, mul_17, mul_9, value_tensor_1, value_tensor_2], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_11.run(buf74, buf39, primals_17, primals_18, primals_33, primals_34, buf75, 65536, grid=grid(65536), stream=stream0)
        buf76 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_453, reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(primals_452, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf76)
        del primals_453
        buf77 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_455, reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(primals_454, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf77)
        del primals_455
        buf78 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_2, mixed_query_layer_2, mul_19], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf77, primals_37, primals_38, buf78, 16384, grid=grid(16384), stream=stream0)
        del primals_38
        buf79 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf78, reinterpret_tensor(primals_456, (128, 128), (1, 128), 0), out=buf79)
        buf80 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf78, reinterpret_tensor(primals_458, (128, 128), (1, 128), 0), out=buf80)
        buf81 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (128, 512), (512, 1), 0), reinterpret_tensor(primals_460, (512, 128), (1, 512), 0), out=buf81)
        buf82 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf79, primals_457, buf82, 16384, grid=grid(16384), stream=stream0)
        del primals_457
        buf83 = reinterpret_tensor(buf79, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf80, primals_459, buf83, 16384, grid=grid(16384), stream=stream0)
        del primals_459
        buf84 = reinterpret_tensor(buf80, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf81, primals_461, buf84, 16384, grid=grid(16384), stream=stream0)
        del primals_461
        # Source Nodes: [], Original ATen: []
        buf85 = aten._scaled_dot_product_efficient_attention(buf82, buf83, buf84, None, True, 0.1, scale=0.17677669529663687)
        buf86 = buf85[0]
        buf87 = buf85[1]
        buf88 = buf85[2]
        buf89 = buf85[3]
        del buf85
        buf90 = buf81; del buf81  # reuse
        # Source Nodes: [layer_outputs_28], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf86, buf90, 16384, grid=grid(16384), stream=stream0)
        buf91 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_463, buf90, reinterpret_tensor(primals_462, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf91)
        del primals_463
        buf92 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_10, layer_input_14, mul_18, mul_20], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf91, buf76, primals_35, primals_36, primals_39, primals_40, buf92, 16384, grid=grid(16384), stream=stream0)
        buf93 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (128, 128), (128, 1), 0), reinterpret_tensor(primals_464, (128, 512), (1, 128), 0), out=buf93)
        buf94 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_8, layer_outputs_30], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf93, primals_465, buf94, 65536, grid=grid(65536), stream=stream0)
        buf95 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_467, buf94, reinterpret_tensor(primals_466, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf95)
        del primals_467
        buf96 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, attention_output_11, hidden_states_20, mul_21], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf95, buf92, primals_41, primals_42, buf96, 16384, grid=grid(16384), stream=stream0)
        buf97 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf96, reinterpret_tensor(primals_468, (128, 512), (1, 128), 0), out=buf97)
        buf98 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_9, layer_outputs_33], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf97, primals_469, buf98, 65536, grid=grid(65536), stream=stream0)
        buf99 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_471, buf98, reinterpret_tensor(primals_470, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf99)
        del primals_471
        buf100 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_40, attention_output_11, attention_output_12, mul_21, mul_22], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf99, buf95, buf92, primals_41, primals_42, primals_43, primals_44, buf100, 16384, grid=grid(16384), stream=stream0)
        buf101 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (128, 128), (128, 1), 0), reinterpret_tensor(primals_472, (128, 512), (1, 128), 0), out=buf101)
        buf102 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_10, layer_outputs_36], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf101, primals_473, buf102, 65536, grid=grid(65536), stream=stream0)
        buf103 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_475, buf102, reinterpret_tensor(primals_474, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf103)
        del primals_475
        buf104 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, attention_output_13, hidden_states_24, mul_23], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf103, buf100, primals_45, primals_46, buf104, 16384, grid=grid(16384), stream=stream0)
        buf105 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_476, (128, 512), (1, 128), 0), out=buf105)
        buf106 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_11, layer_output_8], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf105, primals_477, buf106, 65536, grid=grid(65536), stream=stream0)
        buf107 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_479, buf106, reinterpret_tensor(primals_478, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf107)
        del primals_479
        buf108 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_44, attention_output_13, layer_output_9, layer_outputs_39, mul_23, mul_24], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf107, buf103, buf100, primals_45, primals_46, primals_47, primals_48, buf108, 16384, grid=grid(16384), stream=stream0)
        del primals_48
        buf109 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_481, buf108, reinterpret_tensor(primals_480, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf109)
        del primals_481
        buf110 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, layer_input_15, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf109, buf75, primals_49, primals_50, buf110, 65536, grid=grid(65536), stream=stream0)
        buf111 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_483, buf110, reinterpret_tensor(primals_482, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf111)
        del primals_483
        buf112 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_485, buf110, reinterpret_tensor(primals_484, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf112)
        del primals_485
        buf113 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_3, mixed_query_layer_3, mul_27], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf112, primals_53, primals_54, buf113, 16384, grid=grid(16384), stream=stream0)
        del primals_54
        buf114 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_486, (128, 128), (1, 128), 0), out=buf114)
        buf115 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_488, (128, 128), (1, 128), 0), out=buf115)
        buf116 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_490, (512, 128), (1, 512), 0), out=buf116)
        buf117 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf114, primals_487, buf117, 16384, grid=grid(16384), stream=stream0)
        del primals_487
        buf118 = reinterpret_tensor(buf114, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf115, primals_489, buf118, 16384, grid=grid(16384), stream=stream0)
        del primals_489
        buf119 = reinterpret_tensor(buf115, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf116, primals_491, buf119, 16384, grid=grid(16384), stream=stream0)
        del primals_491
        # Source Nodes: [], Original ATen: []
        buf120 = aten._scaled_dot_product_efficient_attention(buf117, buf118, buf119, None, True, 0.1, scale=0.17677669529663687)
        buf121 = buf120[0]
        buf122 = buf120[1]
        buf123 = buf120[2]
        buf124 = buf120[3]
        del buf120
        buf125 = buf116; del buf116  # reuse
        # Source Nodes: [layer_outputs_42], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf121, buf125, 16384, grid=grid(16384), stream=stream0)
        buf126 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_493, buf125, reinterpret_tensor(primals_492, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf126)
        del primals_493
        buf127 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_51, attention_output_15, layer_input_19, mul_26, mul_28], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf126, buf111, primals_51, primals_52, primals_55, primals_56, buf127, 16384, grid=grid(16384), stream=stream0)
        buf128 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (128, 128), (128, 1), 0), reinterpret_tensor(primals_494, (128, 512), (1, 128), 0), out=buf128)
        buf129 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_12, layer_outputs_44], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf128, primals_495, buf129, 65536, grid=grid(65536), stream=stream0)
        buf130 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_497, buf129, reinterpret_tensor(primals_496, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf130)
        del primals_497
        buf131 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, attention_output_16, hidden_states_29, mul_29], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf130, buf127, primals_57, primals_58, buf131, 16384, grid=grid(16384), stream=stream0)
        buf132 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf131, reinterpret_tensor(primals_498, (128, 512), (1, 128), 0), out=buf132)
        buf133 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_13, layer_outputs_47], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf132, primals_499, buf133, 65536, grid=grid(65536), stream=stream0)
        buf134 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_501, buf133, reinterpret_tensor(primals_500, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf134)
        del primals_501
        buf135 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, add_55, attention_output_16, attention_output_17, mul_29, mul_30], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf134, buf130, buf127, primals_57, primals_58, primals_59, primals_60, buf135, 16384, grid=grid(16384), stream=stream0)
        buf136 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 128), (128, 1), 0), reinterpret_tensor(primals_502, (128, 512), (1, 128), 0), out=buf136)
        buf137 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_14, layer_outputs_50], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf136, primals_503, buf137, 65536, grid=grid(65536), stream=stream0)
        buf138 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_505, buf137, reinterpret_tensor(primals_504, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf138)
        del primals_505
        buf139 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_57, attention_output_18, hidden_states_33, mul_31], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf138, buf135, primals_61, primals_62, buf139, 16384, grid=grid(16384), stream=stream0)
        buf140 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_506, (128, 512), (1, 128), 0), out=buf140)
        buf141 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_15, layer_output_12], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf140, primals_507, buf141, 65536, grid=grid(65536), stream=stream0)
        buf142 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_509, buf141, reinterpret_tensor(primals_508, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf142)
        del primals_509
        buf143 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_57, add_59, attention_output_18, layer_output_13, layer_outputs_53, mul_31, mul_32], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf142, buf138, buf135, primals_61, primals_62, primals_63, primals_64, buf143, 16384, grid=grid(16384), stream=stream0)
        del primals_64
        buf144 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_511, buf143, reinterpret_tensor(primals_510, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf144)
        del primals_511
        buf145 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, add_61, mul_25, mul_33, value_tensor_3, value_tensor_4], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf144, buf109, buf75, primals_49, primals_50, primals_65, primals_66, buf145, 65536, grid=grid(65536), stream=stream0)
        buf146 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_513, reinterpret_tensor(buf145, (128, 512), (512, 1), 0), reinterpret_tensor(primals_512, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf146)
        del primals_513
        buf147 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_515, reinterpret_tensor(buf145, (128, 512), (512, 1), 0), reinterpret_tensor(primals_514, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf147)
        del primals_515
        buf148 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_4, mixed_query_layer_4, mul_35], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf147, primals_69, primals_70, buf148, 16384, grid=grid(16384), stream=stream0)
        del primals_70
        buf149 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf148, reinterpret_tensor(primals_516, (128, 128), (1, 128), 0), out=buf149)
        buf150 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf148, reinterpret_tensor(primals_518, (128, 128), (1, 128), 0), out=buf150)
        buf151 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (128, 512), (512, 1), 0), reinterpret_tensor(primals_520, (512, 128), (1, 512), 0), out=buf151)
        buf152 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf149, primals_517, buf152, 16384, grid=grid(16384), stream=stream0)
        del primals_517
        buf153 = reinterpret_tensor(buf149, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf150, primals_519, buf153, 16384, grid=grid(16384), stream=stream0)
        del primals_519
        buf154 = reinterpret_tensor(buf150, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf151, primals_521, buf154, 16384, grid=grid(16384), stream=stream0)
        del primals_521
        # Source Nodes: [], Original ATen: []
        buf155 = aten._scaled_dot_product_efficient_attention(buf152, buf153, buf154, None, True, 0.1, scale=0.17677669529663687)
        buf156 = buf155[0]
        buf157 = buf155[1]
        buf158 = buf155[2]
        buf159 = buf155[3]
        del buf155
        buf160 = buf151; del buf151  # reuse
        # Source Nodes: [layer_outputs_56], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf156, buf160, 16384, grid=grid(16384), stream=stream0)
        buf161 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_523, buf160, reinterpret_tensor(primals_522, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf161)
        del primals_523
        buf162 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, attention_output_20, layer_input_24, mul_34, mul_36], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf161, buf146, primals_67, primals_68, primals_71, primals_72, buf162, 16384, grid=grid(16384), stream=stream0)
        buf163 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (128, 128), (128, 1), 0), reinterpret_tensor(primals_524, (128, 512), (1, 128), 0), out=buf163)
        buf164 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_16, layer_outputs_58], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf163, primals_525, buf164, 65536, grid=grid(65536), stream=stream0)
        buf165 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_527, buf164, reinterpret_tensor(primals_526, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf165)
        del primals_527
        buf166 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, attention_output_21, hidden_states_38, mul_37], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf165, buf162, primals_73, primals_74, buf166, 16384, grid=grid(16384), stream=stream0)
        buf167 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf166, reinterpret_tensor(primals_528, (128, 512), (1, 128), 0), out=buf167)
        buf168 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_17, layer_outputs_61], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf167, primals_529, buf168, 65536, grid=grid(65536), stream=stream0)
        buf169 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_531, buf168, reinterpret_tensor(primals_530, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf169)
        del primals_531
        buf170 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, add_70, attention_output_21, attention_output_22, mul_37, mul_38], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf169, buf165, buf162, primals_73, primals_74, primals_75, primals_76, buf170, 16384, grid=grid(16384), stream=stream0)
        buf171 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (128, 128), (128, 1), 0), reinterpret_tensor(primals_532, (128, 512), (1, 128), 0), out=buf171)
        buf172 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_18, layer_outputs_64], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf171, primals_533, buf172, 65536, grid=grid(65536), stream=stream0)
        buf173 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_535, buf172, reinterpret_tensor(primals_534, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf173)
        del primals_535
        buf174 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_72, attention_output_23, hidden_states_42, mul_39], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf173, buf170, primals_77, primals_78, buf174, 16384, grid=grid(16384), stream=stream0)
        buf175 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf174, reinterpret_tensor(primals_536, (128, 512), (1, 128), 0), out=buf175)
        buf176 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_19, layer_output_16], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf175, primals_537, buf176, 65536, grid=grid(65536), stream=stream0)
        buf177 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_539, buf176, reinterpret_tensor(primals_538, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf177)
        del primals_539
        buf178 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_72, add_74, attention_output_23, layer_output_17, layer_outputs_67, mul_39, mul_40], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf177, buf173, buf170, primals_77, primals_78, primals_79, primals_80, buf178, 16384, grid=grid(16384), stream=stream0)
        del primals_80
        buf179 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_541, buf178, reinterpret_tensor(primals_540, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf179)
        del primals_541
        buf180 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_76, layer_input_25, mul_41, value_tensor_5], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf179, buf145, primals_81, primals_82, buf180, 65536, grid=grid(65536), stream=stream0)
        buf181 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_543, buf180, reinterpret_tensor(primals_542, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf181)
        del primals_543
        buf182 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_545, buf180, reinterpret_tensor(primals_544, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf182)
        del primals_545
        buf183 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_5, mixed_query_layer_5, mul_43], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf182, primals_85, primals_86, buf183, 16384, grid=grid(16384), stream=stream0)
        del primals_86
        buf184 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_546, (128, 128), (1, 128), 0), out=buf184)
        buf185 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_548, (128, 128), (1, 128), 0), out=buf185)
        buf186 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf180, reinterpret_tensor(primals_550, (512, 128), (1, 512), 0), out=buf186)
        buf187 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf184, primals_547, buf187, 16384, grid=grid(16384), stream=stream0)
        del primals_547
        buf188 = reinterpret_tensor(buf184, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf185, primals_549, buf188, 16384, grid=grid(16384), stream=stream0)
        del primals_549
        buf189 = reinterpret_tensor(buf185, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf186, primals_551, buf189, 16384, grid=grid(16384), stream=stream0)
        del primals_551
        # Source Nodes: [], Original ATen: []
        buf190 = aten._scaled_dot_product_efficient_attention(buf187, buf188, buf189, None, True, 0.1, scale=0.17677669529663687)
        buf191 = buf190[0]
        buf192 = buf190[1]
        buf193 = buf190[2]
        buf194 = buf190[3]
        del buf190
        buf195 = buf186; del buf186  # reuse
        # Source Nodes: [layer_outputs_70], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf191, buf195, 16384, grid=grid(16384), stream=stream0)
        buf196 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_553, buf195, reinterpret_tensor(primals_552, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf196)
        del primals_553
        buf197 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_81, attention_output_25, layer_input_29, mul_42, mul_44], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf196, buf181, primals_83, primals_84, primals_87, primals_88, buf197, 16384, grid=grid(16384), stream=stream0)
        buf198 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (128, 128), (128, 1), 0), reinterpret_tensor(primals_554, (128, 512), (1, 128), 0), out=buf198)
        buf199 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_20, layer_outputs_72], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf198, primals_555, buf199, 65536, grid=grid(65536), stream=stream0)
        buf200 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_557, buf199, reinterpret_tensor(primals_556, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf200)
        del primals_557
        buf201 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_83, attention_output_26, hidden_states_47, mul_45], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf200, buf197, primals_89, primals_90, buf201, 16384, grid=grid(16384), stream=stream0)
        buf202 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_558, (128, 512), (1, 128), 0), out=buf202)
        buf203 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_21, layer_outputs_75], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf202, primals_559, buf203, 65536, grid=grid(65536), stream=stream0)
        buf204 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_561, buf203, reinterpret_tensor(primals_560, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf204)
        del primals_561
        buf205 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_83, add_85, attention_output_26, attention_output_27, mul_45, mul_46], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf204, buf200, buf197, primals_89, primals_90, primals_91, primals_92, buf205, 16384, grid=grid(16384), stream=stream0)
        buf206 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (128, 128), (128, 1), 0), reinterpret_tensor(primals_562, (128, 512), (1, 128), 0), out=buf206)
        buf207 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_22, layer_outputs_78], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf206, primals_563, buf207, 65536, grid=grid(65536), stream=stream0)
        buf208 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_565, buf207, reinterpret_tensor(primals_564, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf208)
        del primals_565
        buf209 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_87, attention_output_28, hidden_states_51, mul_47], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf208, buf205, primals_93, primals_94, buf209, 16384, grid=grid(16384), stream=stream0)
        buf210 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf209, reinterpret_tensor(primals_566, (128, 512), (1, 128), 0), out=buf210)
        buf211 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_23, layer_output_20], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf210, primals_567, buf211, 65536, grid=grid(65536), stream=stream0)
        buf212 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_569, buf211, reinterpret_tensor(primals_568, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf212)
        del primals_569
        buf213 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_87, add_89, attention_output_28, layer_output_21, layer_outputs_81, mul_47, mul_48], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf212, buf208, buf205, primals_93, primals_94, primals_95, primals_96, buf213, 16384, grid=grid(16384), stream=stream0)
        del primals_96
        buf214 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_571, buf213, reinterpret_tensor(primals_570, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf214)
        del primals_571
        buf215 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_76, add_91, mul_41, mul_49, value_tensor_5, value_tensor_6], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf214, buf179, buf145, primals_81, primals_82, primals_97, primals_98, buf215, 65536, grid=grid(65536), stream=stream0)
        buf216 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_573, reinterpret_tensor(buf215, (128, 512), (512, 1), 0), reinterpret_tensor(primals_572, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf216)
        del primals_573
        buf217 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_575, reinterpret_tensor(buf215, (128, 512), (512, 1), 0), reinterpret_tensor(primals_574, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf217)
        del primals_575
        buf218 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_6, mixed_query_layer_6, mul_51], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf217, primals_101, primals_102, buf218, 16384, grid=grid(16384), stream=stream0)
        del primals_102
        buf219 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf218, reinterpret_tensor(primals_576, (128, 128), (1, 128), 0), out=buf219)
        buf220 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf218, reinterpret_tensor(primals_578, (128, 128), (1, 128), 0), out=buf220)
        buf221 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 512), (512, 1), 0), reinterpret_tensor(primals_580, (512, 128), (1, 512), 0), out=buf221)
        buf222 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf219, primals_577, buf222, 16384, grid=grid(16384), stream=stream0)
        del primals_577
        buf223 = reinterpret_tensor(buf219, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf220, primals_579, buf223, 16384, grid=grid(16384), stream=stream0)
        del primals_579
        buf224 = reinterpret_tensor(buf220, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf221, primals_581, buf224, 16384, grid=grid(16384), stream=stream0)
        del primals_581
        # Source Nodes: [], Original ATen: []
        buf225 = aten._scaled_dot_product_efficient_attention(buf222, buf223, buf224, None, True, 0.1, scale=0.17677669529663687)
        buf226 = buf225[0]
        buf227 = buf225[1]
        buf228 = buf225[2]
        buf229 = buf225[3]
        del buf225
        buf230 = buf221; del buf221  # reuse
        # Source Nodes: [layer_outputs_84], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf226, buf230, 16384, grid=grid(16384), stream=stream0)
        buf231 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_583, buf230, reinterpret_tensor(primals_582, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf231)
        del primals_583
        buf232 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_96, attention_output_30, layer_input_34, mul_50, mul_52], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf231, buf216, primals_99, primals_100, primals_103, primals_104, buf232, 16384, grid=grid(16384), stream=stream0)
        buf233 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 128), (128, 1), 0), reinterpret_tensor(primals_584, (128, 512), (1, 128), 0), out=buf233)
        buf234 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_24, layer_outputs_86], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf233, primals_585, buf234, 65536, grid=grid(65536), stream=stream0)
        buf235 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_587, buf234, reinterpret_tensor(primals_586, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf235)
        del primals_587
        buf236 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_98, attention_output_31, hidden_states_56, mul_53], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf235, buf232, primals_105, primals_106, buf236, 16384, grid=grid(16384), stream=stream0)
        buf237 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_588, (128, 512), (1, 128), 0), out=buf237)
        buf238 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_25, layer_outputs_89], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf237, primals_589, buf238, 65536, grid=grid(65536), stream=stream0)
        buf239 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_591, buf238, reinterpret_tensor(primals_590, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf239)
        del primals_591
        buf240 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_100, add_98, attention_output_31, attention_output_32, mul_53, mul_54], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf239, buf235, buf232, primals_105, primals_106, primals_107, primals_108, buf240, 16384, grid=grid(16384), stream=stream0)
        buf241 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 128), (128, 1), 0), reinterpret_tensor(primals_592, (128, 512), (1, 128), 0), out=buf241)
        buf242 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_26, layer_outputs_92], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf241, primals_593, buf242, 65536, grid=grid(65536), stream=stream0)
        buf243 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_595, buf242, reinterpret_tensor(primals_594, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf243)
        del primals_595
        buf244 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_102, attention_output_33, hidden_states_60, mul_55], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf243, buf240, primals_109, primals_110, buf244, 16384, grid=grid(16384), stream=stream0)
        buf245 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf244, reinterpret_tensor(primals_596, (128, 512), (1, 128), 0), out=buf245)
        buf246 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_27, layer_output_24], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf245, primals_597, buf246, 65536, grid=grid(65536), stream=stream0)
        buf247 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_599, buf246, reinterpret_tensor(primals_598, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf247)
        del primals_599
        buf248 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_102, add_104, attention_output_33, layer_output_25, layer_outputs_95, mul_55, mul_56], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf247, buf243, buf240, primals_109, primals_110, primals_111, primals_112, buf248, 16384, grid=grid(16384), stream=stream0)
        del primals_112
        buf249 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_601, buf248, reinterpret_tensor(primals_600, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf249)
        del primals_601
        buf250 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_106, layer_input_35, mul_57, value_tensor_7], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf249, buf215, primals_113, primals_114, buf250, 65536, grid=grid(65536), stream=stream0)
        buf251 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_603, buf250, reinterpret_tensor(primals_602, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf251)
        del primals_603
        buf252 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_605, buf250, reinterpret_tensor(primals_604, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf252)
        del primals_605
        buf253 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_7, mixed_query_layer_7, mul_59], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf252, primals_117, primals_118, buf253, 16384, grid=grid(16384), stream=stream0)
        del primals_118
        buf254 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf253, reinterpret_tensor(primals_606, (128, 128), (1, 128), 0), out=buf254)
        buf255 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf253, reinterpret_tensor(primals_608, (128, 128), (1, 128), 0), out=buf255)
        buf256 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_610, (512, 128), (1, 512), 0), out=buf256)
        buf257 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf254, primals_607, buf257, 16384, grid=grid(16384), stream=stream0)
        del primals_607
        buf258 = reinterpret_tensor(buf254, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf255, primals_609, buf258, 16384, grid=grid(16384), stream=stream0)
        del primals_609
        buf259 = reinterpret_tensor(buf255, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf256, primals_611, buf259, 16384, grid=grid(16384), stream=stream0)
        del primals_611
        # Source Nodes: [], Original ATen: []
        buf260 = aten._scaled_dot_product_efficient_attention(buf257, buf258, buf259, None, True, 0.1, scale=0.17677669529663687)
        buf261 = buf260[0]
        buf262 = buf260[1]
        buf263 = buf260[2]
        buf264 = buf260[3]
        del buf260
        buf265 = buf256; del buf256  # reuse
        # Source Nodes: [layer_outputs_98], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf261, buf265, 16384, grid=grid(16384), stream=stream0)
        buf266 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_613, buf265, reinterpret_tensor(primals_612, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf266)
        del primals_613
        buf267 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_111, attention_output_35, layer_input_39, mul_58, mul_60], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf266, buf251, primals_115, primals_116, primals_119, primals_120, buf267, 16384, grid=grid(16384), stream=stream0)
        buf268 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (128, 128), (128, 1), 0), reinterpret_tensor(primals_614, (128, 512), (1, 128), 0), out=buf268)
        buf269 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_28, layer_outputs_100], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf268, primals_615, buf269, 65536, grid=grid(65536), stream=stream0)
        buf270 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_617, buf269, reinterpret_tensor(primals_616, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf270)
        del primals_617
        buf271 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_113, attention_output_36, hidden_states_65, mul_61], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf270, buf267, primals_121, primals_122, buf271, 16384, grid=grid(16384), stream=stream0)
        buf272 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_618, (128, 512), (1, 128), 0), out=buf272)
        buf273 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_29, layer_outputs_103], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf272, primals_619, buf273, 65536, grid=grid(65536), stream=stream0)
        buf274 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_621, buf273, reinterpret_tensor(primals_620, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf274)
        del primals_621
        buf275 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_113, add_115, attention_output_36, attention_output_37, mul_61, mul_62], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf274, buf270, buf267, primals_121, primals_122, primals_123, primals_124, buf275, 16384, grid=grid(16384), stream=stream0)
        buf276 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf275, (128, 128), (128, 1), 0), reinterpret_tensor(primals_622, (128, 512), (1, 128), 0), out=buf276)
        buf277 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_30, layer_outputs_106], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf276, primals_623, buf277, 65536, grid=grid(65536), stream=stream0)
        buf278 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_106], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_625, buf277, reinterpret_tensor(primals_624, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf278)
        del primals_625
        buf279 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_117, attention_output_38, hidden_states_69, mul_63], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf278, buf275, primals_125, primals_126, buf279, 16384, grid=grid(16384), stream=stream0)
        buf280 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf279, reinterpret_tensor(primals_626, (128, 512), (1, 128), 0), out=buf280)
        buf281 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_31, layer_output_28], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf280, primals_627, buf281, 65536, grid=grid(65536), stream=stream0)
        buf282 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_629, buf281, reinterpret_tensor(primals_628, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf282)
        del primals_629
        buf283 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_117, add_119, attention_output_38, layer_output_29, layer_outputs_109, mul_63, mul_64], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf282, buf278, buf275, primals_125, primals_126, primals_127, primals_128, buf283, 16384, grid=grid(16384), stream=stream0)
        del primals_128
        buf284 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_631, buf283, reinterpret_tensor(primals_630, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf284)
        del primals_631
        buf285 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_106, add_121, mul_57, mul_65, value_tensor_7, value_tensor_8], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf284, buf249, buf215, primals_113, primals_114, primals_129, primals_130, buf285, 65536, grid=grid(65536), stream=stream0)
        buf286 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_633, reinterpret_tensor(buf285, (128, 512), (512, 1), 0), reinterpret_tensor(primals_632, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf286)
        del primals_633
        buf287 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_635, reinterpret_tensor(buf285, (128, 512), (512, 1), 0), reinterpret_tensor(primals_634, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf287)
        del primals_635
        buf288 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_8, mixed_query_layer_8, mul_67], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf287, primals_133, primals_134, buf288, 16384, grid=grid(16384), stream=stream0)
        del primals_134
        buf289 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf288, reinterpret_tensor(primals_636, (128, 128), (1, 128), 0), out=buf289)
        buf290 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf288, reinterpret_tensor(primals_638, (128, 128), (1, 128), 0), out=buf290)
        buf291 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (128, 512), (512, 1), 0), reinterpret_tensor(primals_640, (512, 128), (1, 512), 0), out=buf291)
        buf292 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf289, primals_637, buf292, 16384, grid=grid(16384), stream=stream0)
        del primals_637
        buf293 = reinterpret_tensor(buf289, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf290, primals_639, buf293, 16384, grid=grid(16384), stream=stream0)
        del primals_639
        buf294 = reinterpret_tensor(buf290, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf291, primals_641, buf294, 16384, grid=grid(16384), stream=stream0)
        del primals_641
        # Source Nodes: [], Original ATen: []
        buf295 = aten._scaled_dot_product_efficient_attention(buf292, buf293, buf294, None, True, 0.1, scale=0.17677669529663687)
        buf296 = buf295[0]
        buf297 = buf295[1]
        buf298 = buf295[2]
        buf299 = buf295[3]
        del buf295
        buf300 = buf291; del buf291  # reuse
        # Source Nodes: [layer_outputs_112], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf296, buf300, 16384, grid=grid(16384), stream=stream0)
        buf301 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_112], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_643, buf300, reinterpret_tensor(primals_642, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf301)
        del primals_643
        buf302 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, attention_output_40, layer_input_44, mul_66, mul_68], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf301, buf286, primals_131, primals_132, primals_135, primals_136, buf302, 16384, grid=grid(16384), stream=stream0)
        buf303 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (128, 128), (128, 1), 0), reinterpret_tensor(primals_644, (128, 512), (1, 128), 0), out=buf303)
        buf304 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_32, layer_outputs_114], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf303, primals_645, buf304, 65536, grid=grid(65536), stream=stream0)
        buf305 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_647, buf304, reinterpret_tensor(primals_646, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf305)
        del primals_647
        buf306 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_128, attention_output_41, hidden_states_74, mul_69], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf305, buf302, primals_137, primals_138, buf306, 16384, grid=grid(16384), stream=stream0)
        buf307 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf306, reinterpret_tensor(primals_648, (128, 512), (1, 128), 0), out=buf307)
        buf308 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_33, layer_outputs_117], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf307, primals_649, buf308, 65536, grid=grid(65536), stream=stream0)
        buf309 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_651, buf308, reinterpret_tensor(primals_650, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf309)
        del primals_651
        buf310 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_128, add_130, attention_output_41, attention_output_42, mul_69, mul_70], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf309, buf305, buf302, primals_137, primals_138, primals_139, primals_140, buf310, 16384, grid=grid(16384), stream=stream0)
        buf311 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (128, 1), 0), reinterpret_tensor(primals_652, (128, 512), (1, 128), 0), out=buf311)
        buf312 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_34, layer_outputs_120], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf311, primals_653, buf312, 65536, grid=grid(65536), stream=stream0)
        buf313 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_120], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_655, buf312, reinterpret_tensor(primals_654, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf313)
        del primals_655
        buf314 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_132, attention_output_43, hidden_states_78, mul_71], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf313, buf310, primals_141, primals_142, buf314, 16384, grid=grid(16384), stream=stream0)
        buf315 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf314, reinterpret_tensor(primals_656, (128, 512), (1, 128), 0), out=buf315)
        buf316 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_35, layer_output_32], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf315, primals_657, buf316, 65536, grid=grid(65536), stream=stream0)
        buf317 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_659, buf316, reinterpret_tensor(primals_658, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf317)
        del primals_659
        buf318 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_132, add_134, attention_output_43, layer_output_33, layer_outputs_123, mul_71, mul_72], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf317, buf313, buf310, primals_141, primals_142, primals_143, primals_144, buf318, 16384, grid=grid(16384), stream=stream0)
        del primals_144
        buf319 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_123], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_661, buf318, reinterpret_tensor(primals_660, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf319)
        del primals_661
        buf320 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_136, layer_input_45, mul_73, value_tensor_9], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf319, buf285, primals_145, primals_146, buf320, 65536, grid=grid(65536), stream=stream0)
        buf321 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_663, buf320, reinterpret_tensor(primals_662, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf321)
        del primals_663
        buf322 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_665, buf320, reinterpret_tensor(primals_664, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf322)
        del primals_665
        buf323 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_9, mixed_query_layer_9, mul_75], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf322, primals_149, primals_150, buf323, 16384, grid=grid(16384), stream=stream0)
        del primals_150
        buf324 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf323, reinterpret_tensor(primals_666, (128, 128), (1, 128), 0), out=buf324)
        buf325 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf323, reinterpret_tensor(primals_668, (128, 128), (1, 128), 0), out=buf325)
        buf326 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf320, reinterpret_tensor(primals_670, (512, 128), (1, 512), 0), out=buf326)
        buf327 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf324, primals_667, buf327, 16384, grid=grid(16384), stream=stream0)
        del primals_667
        buf328 = reinterpret_tensor(buf324, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf325, primals_669, buf328, 16384, grid=grid(16384), stream=stream0)
        del primals_669
        buf329 = reinterpret_tensor(buf325, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf326, primals_671, buf329, 16384, grid=grid(16384), stream=stream0)
        del primals_671
        # Source Nodes: [], Original ATen: []
        buf330 = aten._scaled_dot_product_efficient_attention(buf327, buf328, buf329, None, True, 0.1, scale=0.17677669529663687)
        buf331 = buf330[0]
        buf332 = buf330[1]
        buf333 = buf330[2]
        buf334 = buf330[3]
        del buf330
        buf335 = buf326; del buf326  # reuse
        # Source Nodes: [layer_outputs_126], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf331, buf335, 16384, grid=grid(16384), stream=stream0)
        buf336 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_126], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_673, buf335, reinterpret_tensor(primals_672, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf336)
        del primals_673
        buf337 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_141, attention_output_45, layer_input_49, mul_74, mul_76], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf336, buf321, primals_147, primals_148, primals_151, primals_152, buf337, 16384, grid=grid(16384), stream=stream0)
        buf338 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (128, 128), (128, 1), 0), reinterpret_tensor(primals_674, (128, 512), (1, 128), 0), out=buf338)
        buf339 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_36, layer_outputs_128], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf338, primals_675, buf339, 65536, grid=grid(65536), stream=stream0)
        buf340 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_128], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_677, buf339, reinterpret_tensor(primals_676, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf340)
        del primals_677
        buf341 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_143, attention_output_46, hidden_states_83, mul_77], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf340, buf337, primals_153, primals_154, buf341, 16384, grid=grid(16384), stream=stream0)
        buf342 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf341, reinterpret_tensor(primals_678, (128, 512), (1, 128), 0), out=buf342)
        buf343 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_37, layer_outputs_131], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf342, primals_679, buf343, 65536, grid=grid(65536), stream=stream0)
        buf344 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_131], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_681, buf343, reinterpret_tensor(primals_680, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf344)
        del primals_681
        buf345 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_143, add_145, attention_output_46, attention_output_47, mul_77, mul_78], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf344, buf340, buf337, primals_153, primals_154, primals_155, primals_156, buf345, 16384, grid=grid(16384), stream=stream0)
        buf346 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (128, 128), (128, 1), 0), reinterpret_tensor(primals_682, (128, 512), (1, 128), 0), out=buf346)
        buf347 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_38, layer_outputs_134], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf346, primals_683, buf347, 65536, grid=grid(65536), stream=stream0)
        buf348 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_685, buf347, reinterpret_tensor(primals_684, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf348)
        del primals_685
        buf349 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_147, attention_output_48, hidden_states_87, mul_79], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf348, buf345, primals_157, primals_158, buf349, 16384, grid=grid(16384), stream=stream0)
        buf350 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_686, (128, 512), (1, 128), 0), out=buf350)
        buf351 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_39, layer_output_36], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf350, primals_687, buf351, 65536, grid=grid(65536), stream=stream0)
        buf352 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_689, buf351, reinterpret_tensor(primals_688, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf352)
        del primals_689
        buf353 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_147, add_149, attention_output_48, layer_output_37, layer_outputs_137, mul_79, mul_80], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf352, buf348, buf345, primals_157, primals_158, primals_159, primals_160, buf353, 16384, grid=grid(16384), stream=stream0)
        del primals_160
        buf354 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_691, buf353, reinterpret_tensor(primals_690, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf354)
        del primals_691
        buf355 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_136, add_151, mul_73, mul_81, value_tensor_10, value_tensor_9], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf354, buf319, buf285, primals_145, primals_146, primals_161, primals_162, buf355, 65536, grid=grid(65536), stream=stream0)
        buf356 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_693, reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(primals_692, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf356)
        del primals_693
        buf357 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_695, reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(primals_694, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf357)
        del primals_695
        buf358 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_10, mixed_query_layer_10, mul_83], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf357, primals_165, primals_166, buf358, 16384, grid=grid(16384), stream=stream0)
        del primals_166
        buf359 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf358, reinterpret_tensor(primals_696, (128, 128), (1, 128), 0), out=buf359)
        buf360 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf358, reinterpret_tensor(primals_698, (128, 128), (1, 128), 0), out=buf360)
        buf361 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(primals_700, (512, 128), (1, 512), 0), out=buf361)
        buf362 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf359, primals_697, buf362, 16384, grid=grid(16384), stream=stream0)
        del primals_697
        buf363 = reinterpret_tensor(buf359, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf360, primals_699, buf363, 16384, grid=grid(16384), stream=stream0)
        del primals_699
        buf364 = reinterpret_tensor(buf360, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf361, primals_701, buf364, 16384, grid=grid(16384), stream=stream0)
        del primals_701
        # Source Nodes: [], Original ATen: []
        buf365 = aten._scaled_dot_product_efficient_attention(buf362, buf363, buf364, None, True, 0.1, scale=0.17677669529663687)
        buf366 = buf365[0]
        buf367 = buf365[1]
        buf368 = buf365[2]
        buf369 = buf365[3]
        del buf365
        buf370 = buf361; del buf361  # reuse
        # Source Nodes: [layer_outputs_140], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf366, buf370, 16384, grid=grid(16384), stream=stream0)
        buf371 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_140], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_703, buf370, reinterpret_tensor(primals_702, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf371)
        del primals_703
        buf372 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_156, attention_output_50, layer_input_54, mul_82, mul_84], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf371, buf356, primals_163, primals_164, primals_167, primals_168, buf372, 16384, grid=grid(16384), stream=stream0)
        buf373 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (128, 128), (128, 1), 0), reinterpret_tensor(primals_704, (128, 512), (1, 128), 0), out=buf373)
        buf374 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_40, layer_outputs_142], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf373, primals_705, buf374, 65536, grid=grid(65536), stream=stream0)
        buf375 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_142], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_707, buf374, reinterpret_tensor(primals_706, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf375)
        del primals_707
        buf376 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_158, attention_output_51, hidden_states_92, mul_85], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf375, buf372, primals_169, primals_170, buf376, 16384, grid=grid(16384), stream=stream0)
        buf377 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf376, reinterpret_tensor(primals_708, (128, 512), (1, 128), 0), out=buf377)
        buf378 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_41, layer_outputs_145], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf377, primals_709, buf378, 65536, grid=grid(65536), stream=stream0)
        buf379 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_145], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_711, buf378, reinterpret_tensor(primals_710, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf379)
        del primals_711
        buf380 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_158, add_160, attention_output_51, attention_output_52, mul_85, mul_86], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf379, buf375, buf372, primals_169, primals_170, primals_171, primals_172, buf380, 16384, grid=grid(16384), stream=stream0)
        buf381 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (128, 128), (128, 1), 0), reinterpret_tensor(primals_712, (128, 512), (1, 128), 0), out=buf381)
        buf382 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_42, layer_outputs_148], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf381, primals_713, buf382, 65536, grid=grid(65536), stream=stream0)
        buf383 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_715, buf382, reinterpret_tensor(primals_714, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf383)
        del primals_715
        buf384 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_162, attention_output_53, hidden_states_96, mul_87], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf383, buf380, primals_173, primals_174, buf384, 16384, grid=grid(16384), stream=stream0)
        buf385 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf384, reinterpret_tensor(primals_716, (128, 512), (1, 128), 0), out=buf385)
        buf386 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_43, layer_output_40], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf385, primals_717, buf386, 65536, grid=grid(65536), stream=stream0)
        buf387 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_719, buf386, reinterpret_tensor(primals_718, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf387)
        del primals_719
        buf388 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_162, add_164, attention_output_53, layer_output_41, layer_outputs_151, mul_87, mul_88], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf387, buf383, buf380, primals_173, primals_174, primals_175, primals_176, buf388, 16384, grid=grid(16384), stream=stream0)
        del primals_176
        buf389 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_721, buf388, reinterpret_tensor(primals_720, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf389)
        del primals_721
        buf390 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_166, layer_input_55, mul_89, value_tensor_11], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf389, buf355, primals_177, primals_178, buf390, 65536, grid=grid(65536), stream=stream0)
        buf391 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_723, buf390, reinterpret_tensor(primals_722, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf391)
        del primals_723
        buf392 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_725, buf390, reinterpret_tensor(primals_724, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf392)
        del primals_725
        buf393 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_11, mixed_query_layer_11, mul_91], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf392, primals_181, primals_182, buf393, 16384, grid=grid(16384), stream=stream0)
        del primals_182
        buf394 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_726, (128, 128), (1, 128), 0), out=buf394)
        buf395 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_728, (128, 128), (1, 128), 0), out=buf395)
        buf396 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf390, reinterpret_tensor(primals_730, (512, 128), (1, 512), 0), out=buf396)
        buf397 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf394, primals_727, buf397, 16384, grid=grid(16384), stream=stream0)
        del primals_727
        buf398 = reinterpret_tensor(buf394, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf395, primals_729, buf398, 16384, grid=grid(16384), stream=stream0)
        del primals_729
        buf399 = reinterpret_tensor(buf395, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf396, primals_731, buf399, 16384, grid=grid(16384), stream=stream0)
        del primals_731
        # Source Nodes: [], Original ATen: []
        buf400 = aten._scaled_dot_product_efficient_attention(buf397, buf398, buf399, None, True, 0.1, scale=0.17677669529663687)
        buf401 = buf400[0]
        buf402 = buf400[1]
        buf403 = buf400[2]
        buf404 = buf400[3]
        del buf400
        buf405 = buf396; del buf396  # reuse
        # Source Nodes: [layer_outputs_154], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf401, buf405, 16384, grid=grid(16384), stream=stream0)
        buf406 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_154], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_733, buf405, reinterpret_tensor(primals_732, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf406)
        del primals_733
        buf407 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_171, attention_output_55, layer_input_59, mul_90, mul_92], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf406, buf391, primals_179, primals_180, primals_183, primals_184, buf407, 16384, grid=grid(16384), stream=stream0)
        buf408 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf407, (128, 128), (128, 1), 0), reinterpret_tensor(primals_734, (128, 512), (1, 128), 0), out=buf408)
        buf409 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_44, layer_outputs_156], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf408, primals_735, buf409, 65536, grid=grid(65536), stream=stream0)
        buf410 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_737, buf409, reinterpret_tensor(primals_736, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf410)
        del primals_737
        buf411 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_173, attention_output_56, hidden_states_101, mul_93], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf410, buf407, primals_185, primals_186, buf411, 16384, grid=grid(16384), stream=stream0)
        buf412 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf411, reinterpret_tensor(primals_738, (128, 512), (1, 128), 0), out=buf412)
        buf413 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_45, layer_outputs_159], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf412, primals_739, buf413, 65536, grid=grid(65536), stream=stream0)
        buf414 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_741, buf413, reinterpret_tensor(primals_740, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf414)
        del primals_741
        buf415 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_173, add_175, attention_output_56, attention_output_57, mul_93, mul_94], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf414, buf410, buf407, primals_185, primals_186, primals_187, primals_188, buf415, 16384, grid=grid(16384), stream=stream0)
        buf416 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf415, (128, 128), (128, 1), 0), reinterpret_tensor(primals_742, (128, 512), (1, 128), 0), out=buf416)
        buf417 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_46, layer_outputs_162], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf416, primals_743, buf417, 65536, grid=grid(65536), stream=stream0)
        buf418 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_745, buf417, reinterpret_tensor(primals_744, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf418)
        del primals_745
        buf419 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_177, attention_output_58, hidden_states_105, mul_95], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf418, buf415, primals_189, primals_190, buf419, 16384, grid=grid(16384), stream=stream0)
        buf420 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf419, reinterpret_tensor(primals_746, (128, 512), (1, 128), 0), out=buf420)
        buf421 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_47, layer_output_44], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf420, primals_747, buf421, 65536, grid=grid(65536), stream=stream0)
        buf422 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_749, buf421, reinterpret_tensor(primals_748, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf422)
        del primals_749
        buf423 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_177, add_179, attention_output_58, layer_output_45, layer_outputs_165, mul_95, mul_96], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf422, buf418, buf415, primals_189, primals_190, primals_191, primals_192, buf423, 16384, grid=grid(16384), stream=stream0)
        del primals_192
        buf424 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_165], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_751, buf423, reinterpret_tensor(primals_750, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf424)
        del primals_751
        buf425 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_166, add_181, mul_89, mul_97, value_tensor_11, value_tensor_12], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf424, buf389, buf355, primals_177, primals_178, primals_193, primals_194, buf425, 65536, grid=grid(65536), stream=stream0)
        buf426 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_753, reinterpret_tensor(buf425, (128, 512), (512, 1), 0), reinterpret_tensor(primals_752, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf426)
        del primals_753
        buf427 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_755, reinterpret_tensor(buf425, (128, 512), (512, 1), 0), reinterpret_tensor(primals_754, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf427)
        del primals_755
        buf428 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_12, mixed_query_layer_12, mul_99], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf427, primals_197, primals_198, buf428, 16384, grid=grid(16384), stream=stream0)
        del primals_198
        buf429 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf428, reinterpret_tensor(primals_756, (128, 128), (1, 128), 0), out=buf429)
        buf430 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf428, reinterpret_tensor(primals_758, (128, 128), (1, 128), 0), out=buf430)
        buf431 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf425, (128, 512), (512, 1), 0), reinterpret_tensor(primals_760, (512, 128), (1, 512), 0), out=buf431)
        buf432 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf429, primals_757, buf432, 16384, grid=grid(16384), stream=stream0)
        del primals_757
        buf433 = reinterpret_tensor(buf429, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf430, primals_759, buf433, 16384, grid=grid(16384), stream=stream0)
        del primals_759
        buf434 = reinterpret_tensor(buf430, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf431, primals_761, buf434, 16384, grid=grid(16384), stream=stream0)
        del primals_761
        # Source Nodes: [], Original ATen: []
        buf435 = aten._scaled_dot_product_efficient_attention(buf432, buf433, buf434, None, True, 0.1, scale=0.17677669529663687)
        buf436 = buf435[0]
        buf437 = buf435[1]
        buf438 = buf435[2]
        buf439 = buf435[3]
        del buf435
        buf440 = buf431; del buf431  # reuse
        # Source Nodes: [layer_outputs_168], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf436, buf440, 16384, grid=grid(16384), stream=stream0)
        buf441 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_168], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_763, buf440, reinterpret_tensor(primals_762, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf441)
        del primals_763
        buf442 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_186, attention_output_60, layer_input_64, mul_100, mul_98], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf441, buf426, primals_195, primals_196, primals_199, primals_200, buf442, 16384, grid=grid(16384), stream=stream0)
        buf443 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf442, (128, 128), (128, 1), 0), reinterpret_tensor(primals_764, (128, 512), (1, 128), 0), out=buf443)
        buf444 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_48, layer_outputs_170], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf443, primals_765, buf444, 65536, grid=grid(65536), stream=stream0)
        buf445 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_170], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_767, buf444, reinterpret_tensor(primals_766, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf445)
        del primals_767
        buf446 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_188, attention_output_61, hidden_states_110, mul_101], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf445, buf442, primals_201, primals_202, buf446, 16384, grid=grid(16384), stream=stream0)
        buf447 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf446, reinterpret_tensor(primals_768, (128, 512), (1, 128), 0), out=buf447)
        buf448 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_49, layer_outputs_173], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf447, primals_769, buf448, 65536, grid=grid(65536), stream=stream0)
        buf449 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_771, buf448, reinterpret_tensor(primals_770, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf449)
        del primals_771
        buf450 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_188, add_190, attention_output_61, attention_output_62, mul_101, mul_102], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf449, buf445, buf442, primals_201, primals_202, primals_203, primals_204, buf450, 16384, grid=grid(16384), stream=stream0)
        buf451 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (128, 128), (128, 1), 0), reinterpret_tensor(primals_772, (128, 512), (1, 128), 0), out=buf451)
        buf452 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_50, layer_outputs_176], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf451, primals_773, buf452, 65536, grid=grid(65536), stream=stream0)
        buf453 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_176], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_775, buf452, reinterpret_tensor(primals_774, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf453)
        del primals_775
        buf454 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_192, attention_output_63, hidden_states_114, mul_103], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf453, buf450, primals_205, primals_206, buf454, 16384, grid=grid(16384), stream=stream0)
        buf455 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf454, reinterpret_tensor(primals_776, (128, 512), (1, 128), 0), out=buf455)
        buf456 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_51, layer_output_48], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf455, primals_777, buf456, 65536, grid=grid(65536), stream=stream0)
        buf457 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_779, buf456, reinterpret_tensor(primals_778, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf457)
        del primals_779
        buf458 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_192, add_194, attention_output_63, layer_output_49, layer_outputs_179, mul_103, mul_104], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf457, buf453, buf450, primals_205, primals_206, primals_207, primals_208, buf458, 16384, grid=grid(16384), stream=stream0)
        del primals_208
        buf459 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_179], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_781, buf458, reinterpret_tensor(primals_780, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf459)
        del primals_781
        buf460 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_196, layer_input_65, mul_105, value_tensor_13], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf459, buf425, primals_209, primals_210, buf460, 65536, grid=grid(65536), stream=stream0)
        buf461 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_783, buf460, reinterpret_tensor(primals_782, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf461)
        del primals_783
        buf462 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_785, buf460, reinterpret_tensor(primals_784, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf462)
        del primals_785
        buf463 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_13, mixed_query_layer_13, mul_107], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf462, primals_213, primals_214, buf463, 16384, grid=grid(16384), stream=stream0)
        del primals_214
        buf464 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf463, reinterpret_tensor(primals_786, (128, 128), (1, 128), 0), out=buf464)
        buf465 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf463, reinterpret_tensor(primals_788, (128, 128), (1, 128), 0), out=buf465)
        buf466 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf460, reinterpret_tensor(primals_790, (512, 128), (1, 512), 0), out=buf466)
        buf467 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf464, primals_787, buf467, 16384, grid=grid(16384), stream=stream0)
        del primals_787
        buf468 = reinterpret_tensor(buf464, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf465, primals_789, buf468, 16384, grid=grid(16384), stream=stream0)
        del primals_789
        buf469 = reinterpret_tensor(buf465, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf466, primals_791, buf469, 16384, grid=grid(16384), stream=stream0)
        del primals_791
        # Source Nodes: [], Original ATen: []
        buf470 = aten._scaled_dot_product_efficient_attention(buf467, buf468, buf469, None, True, 0.1, scale=0.17677669529663687)
        buf471 = buf470[0]
        buf472 = buf470[1]
        buf473 = buf470[2]
        buf474 = buf470[3]
        del buf470
        buf475 = buf466; del buf466  # reuse
        # Source Nodes: [layer_outputs_182], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf471, buf475, 16384, grid=grid(16384), stream=stream0)
        buf476 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_182], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_793, buf475, reinterpret_tensor(primals_792, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf476)
        del primals_793
        buf477 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_201, attention_output_65, layer_input_69, mul_106, mul_108], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf476, buf461, primals_211, primals_212, primals_215, primals_216, buf477, 16384, grid=grid(16384), stream=stream0)
        buf478 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf477, (128, 128), (128, 1), 0), reinterpret_tensor(primals_794, (128, 512), (1, 128), 0), out=buf478)
        buf479 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_52, layer_outputs_184], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf478, primals_795, buf479, 65536, grid=grid(65536), stream=stream0)
        buf480 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_184], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_797, buf479, reinterpret_tensor(primals_796, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf480)
        del primals_797
        buf481 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_203, attention_output_66, hidden_states_119, mul_109], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf480, buf477, primals_217, primals_218, buf481, 16384, grid=grid(16384), stream=stream0)
        buf482 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf481, reinterpret_tensor(primals_798, (128, 512), (1, 128), 0), out=buf482)
        buf483 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_53, layer_outputs_187], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf482, primals_799, buf483, 65536, grid=grid(65536), stream=stream0)
        buf484 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_187], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_801, buf483, reinterpret_tensor(primals_800, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf484)
        del primals_801
        buf485 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_203, add_205, attention_output_66, attention_output_67, mul_109, mul_110], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf484, buf480, buf477, primals_217, primals_218, primals_219, primals_220, buf485, 16384, grid=grid(16384), stream=stream0)
        buf486 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf485, (128, 128), (128, 1), 0), reinterpret_tensor(primals_802, (128, 512), (1, 128), 0), out=buf486)
        buf487 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_54, layer_outputs_190], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf486, primals_803, buf487, 65536, grid=grid(65536), stream=stream0)
        buf488 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_190], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_805, buf487, reinterpret_tensor(primals_804, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf488)
        del primals_805
        buf489 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_207, attention_output_68, hidden_states_123, mul_111], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf488, buf485, primals_221, primals_222, buf489, 16384, grid=grid(16384), stream=stream0)
        buf490 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf489, reinterpret_tensor(primals_806, (128, 512), (1, 128), 0), out=buf490)
        buf491 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_55, layer_output_52], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf490, primals_807, buf491, 65536, grid=grid(65536), stream=stream0)
        buf492 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_809, buf491, reinterpret_tensor(primals_808, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf492)
        del primals_809
        buf493 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_207, add_209, attention_output_68, layer_output_53, layer_outputs_193, mul_111, mul_112], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf492, buf488, buf485, primals_221, primals_222, primals_223, primals_224, buf493, 16384, grid=grid(16384), stream=stream0)
        del primals_224
        buf494 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_193], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_811, buf493, reinterpret_tensor(primals_810, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf494)
        del primals_811
        buf495 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_196, add_211, mul_105, mul_113, value_tensor_13, value_tensor_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf494, buf459, buf425, primals_209, primals_210, primals_225, primals_226, buf495, 65536, grid=grid(65536), stream=stream0)
        buf496 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_813, reinterpret_tensor(buf495, (128, 512), (512, 1), 0), reinterpret_tensor(primals_812, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf496)
        del primals_813
        buf497 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_815, reinterpret_tensor(buf495, (128, 512), (512, 1), 0), reinterpret_tensor(primals_814, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf497)
        del primals_815
        buf498 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_14, mixed_query_layer_14, mul_115], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf497, primals_229, primals_230, buf498, 16384, grid=grid(16384), stream=stream0)
        del primals_230
        buf499 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf498, reinterpret_tensor(primals_816, (128, 128), (1, 128), 0), out=buf499)
        buf500 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf498, reinterpret_tensor(primals_818, (128, 128), (1, 128), 0), out=buf500)
        buf501 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (128, 512), (512, 1), 0), reinterpret_tensor(primals_820, (512, 128), (1, 512), 0), out=buf501)
        buf502 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf499, primals_817, buf502, 16384, grid=grid(16384), stream=stream0)
        del primals_817
        buf503 = reinterpret_tensor(buf499, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf500, primals_819, buf503, 16384, grid=grid(16384), stream=stream0)
        del primals_819
        buf504 = reinterpret_tensor(buf500, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf501, primals_821, buf504, 16384, grid=grid(16384), stream=stream0)
        del primals_821
        # Source Nodes: [], Original ATen: []
        buf505 = aten._scaled_dot_product_efficient_attention(buf502, buf503, buf504, None, True, 0.1, scale=0.17677669529663687)
        buf506 = buf505[0]
        buf507 = buf505[1]
        buf508 = buf505[2]
        buf509 = buf505[3]
        del buf505
        buf510 = buf501; del buf501  # reuse
        # Source Nodes: [layer_outputs_196], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf506, buf510, 16384, grid=grid(16384), stream=stream0)
        buf511 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_196], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_823, buf510, reinterpret_tensor(primals_822, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf511)
        del primals_823
        buf512 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_216, attention_output_70, layer_input_74, mul_114, mul_116], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf511, buf496, primals_227, primals_228, primals_231, primals_232, buf512, 16384, grid=grid(16384), stream=stream0)
        buf513 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (128, 128), (128, 1), 0), reinterpret_tensor(primals_824, (128, 512), (1, 128), 0), out=buf513)
        buf514 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_56, layer_outputs_198], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf513, primals_825, buf514, 65536, grid=grid(65536), stream=stream0)
        buf515 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_198], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_827, buf514, reinterpret_tensor(primals_826, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf515)
        del primals_827
        buf516 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_218, attention_output_71, hidden_states_128, mul_117], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf515, buf512, primals_233, primals_234, buf516, 16384, grid=grid(16384), stream=stream0)
        buf517 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf516, reinterpret_tensor(primals_828, (128, 512), (1, 128), 0), out=buf517)
        buf518 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_57, layer_outputs_201], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf517, primals_829, buf518, 65536, grid=grid(65536), stream=stream0)
        buf519 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_201], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_831, buf518, reinterpret_tensor(primals_830, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf519)
        del primals_831
        buf520 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_218, add_220, attention_output_71, attention_output_72, mul_117, mul_118], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf519, buf515, buf512, primals_233, primals_234, primals_235, primals_236, buf520, 16384, grid=grid(16384), stream=stream0)
        buf521 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (128, 128), (128, 1), 0), reinterpret_tensor(primals_832, (128, 512), (1, 128), 0), out=buf521)
        buf522 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_58, layer_outputs_204], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf521, primals_833, buf522, 65536, grid=grid(65536), stream=stream0)
        buf523 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_204], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_835, buf522, reinterpret_tensor(primals_834, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf523)
        del primals_835
        buf524 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_222, attention_output_73, hidden_states_132, mul_119], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf523, buf520, primals_237, primals_238, buf524, 16384, grid=grid(16384), stream=stream0)
        buf525 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf524, reinterpret_tensor(primals_836, (128, 512), (1, 128), 0), out=buf525)
        buf526 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_59, layer_output_56], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf525, primals_837, buf526, 65536, grid=grid(65536), stream=stream0)
        buf527 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_839, buf526, reinterpret_tensor(primals_838, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf527)
        del primals_839
        buf528 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_222, add_224, attention_output_73, layer_output_57, layer_outputs_207, mul_119, mul_120], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf527, buf523, buf520, primals_237, primals_238, primals_239, primals_240, buf528, 16384, grid=grid(16384), stream=stream0)
        del primals_240
        buf529 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_207], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_841, buf528, reinterpret_tensor(primals_840, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf529)
        del primals_841
        buf530 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_226, layer_input_75, mul_121, value_tensor_15], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf529, buf495, primals_241, primals_242, buf530, 65536, grid=grid(65536), stream=stream0)
        buf531 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_843, buf530, reinterpret_tensor(primals_842, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf531)
        del primals_843
        buf532 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_845, buf530, reinterpret_tensor(primals_844, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf532)
        del primals_845
        buf533 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_15, mixed_query_layer_15, mul_123], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf532, primals_245, primals_246, buf533, 16384, grid=grid(16384), stream=stream0)
        del primals_246
        buf534 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf533, reinterpret_tensor(primals_846, (128, 128), (1, 128), 0), out=buf534)
        buf535 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf533, reinterpret_tensor(primals_848, (128, 128), (1, 128), 0), out=buf535)
        buf536 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf530, reinterpret_tensor(primals_850, (512, 128), (1, 512), 0), out=buf536)
        buf537 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf534, primals_847, buf537, 16384, grid=grid(16384), stream=stream0)
        del primals_847
        buf538 = reinterpret_tensor(buf534, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf535, primals_849, buf538, 16384, grid=grid(16384), stream=stream0)
        del primals_849
        buf539 = reinterpret_tensor(buf535, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf536, primals_851, buf539, 16384, grid=grid(16384), stream=stream0)
        del primals_851
        # Source Nodes: [], Original ATen: []
        buf540 = aten._scaled_dot_product_efficient_attention(buf537, buf538, buf539, None, True, 0.1, scale=0.17677669529663687)
        buf541 = buf540[0]
        buf542 = buf540[1]
        buf543 = buf540[2]
        buf544 = buf540[3]
        del buf540
        buf545 = buf536; del buf536  # reuse
        # Source Nodes: [layer_outputs_210], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf541, buf545, 16384, grid=grid(16384), stream=stream0)
        buf546 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_210], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_853, buf545, reinterpret_tensor(primals_852, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf546)
        del primals_853
        buf547 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_231, attention_output_75, layer_input_79, mul_122, mul_124], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf546, buf531, primals_243, primals_244, primals_247, primals_248, buf547, 16384, grid=grid(16384), stream=stream0)
        buf548 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf547, (128, 128), (128, 1), 0), reinterpret_tensor(primals_854, (128, 512), (1, 128), 0), out=buf548)
        buf549 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_60, layer_outputs_212], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf548, primals_855, buf549, 65536, grid=grid(65536), stream=stream0)
        buf550 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_212], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_857, buf549, reinterpret_tensor(primals_856, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf550)
        del primals_857
        buf551 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_233, attention_output_76, hidden_states_137, mul_125], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf550, buf547, primals_249, primals_250, buf551, 16384, grid=grid(16384), stream=stream0)
        buf552 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf551, reinterpret_tensor(primals_858, (128, 512), (1, 128), 0), out=buf552)
        buf553 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_61, layer_outputs_215], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf552, primals_859, buf553, 65536, grid=grid(65536), stream=stream0)
        buf554 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_215], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_861, buf553, reinterpret_tensor(primals_860, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf554)
        del primals_861
        buf555 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_233, add_235, attention_output_76, attention_output_77, mul_125, mul_126], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf554, buf550, buf547, primals_249, primals_250, primals_251, primals_252, buf555, 16384, grid=grid(16384), stream=stream0)
        buf556 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (128, 128), (128, 1), 0), reinterpret_tensor(primals_862, (128, 512), (1, 128), 0), out=buf556)
        buf557 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_62, layer_outputs_218], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf556, primals_863, buf557, 65536, grid=grid(65536), stream=stream0)
        buf558 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_218], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_865, buf557, reinterpret_tensor(primals_864, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf558)
        del primals_865
        buf559 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_237, attention_output_78, hidden_states_141, mul_127], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf558, buf555, primals_253, primals_254, buf559, 16384, grid=grid(16384), stream=stream0)
        buf560 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf559, reinterpret_tensor(primals_866, (128, 512), (1, 128), 0), out=buf560)
        buf561 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_63, layer_output_60], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf560, primals_867, buf561, 65536, grid=grid(65536), stream=stream0)
        buf562 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_869, buf561, reinterpret_tensor(primals_868, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf562)
        del primals_869
        buf563 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_237, add_239, attention_output_78, layer_output_61, layer_outputs_221, mul_127, mul_128], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf562, buf558, buf555, primals_253, primals_254, primals_255, primals_256, buf563, 16384, grid=grid(16384), stream=stream0)
        del primals_256
        buf564 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_221], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_871, buf563, reinterpret_tensor(primals_870, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf564)
        del primals_871
        buf565 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_226, add_241, mul_121, mul_129, value_tensor_15, value_tensor_16], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf564, buf529, buf495, primals_241, primals_242, primals_257, primals_258, buf565, 65536, grid=grid(65536), stream=stream0)
        buf566 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_873, reinterpret_tensor(buf565, (128, 512), (512, 1), 0), reinterpret_tensor(primals_872, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf566)
        del primals_873
        buf567 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_875, reinterpret_tensor(buf565, (128, 512), (512, 1), 0), reinterpret_tensor(primals_874, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf567)
        del primals_875
        buf568 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_16, mixed_query_layer_16, mul_131], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf567, primals_261, primals_262, buf568, 16384, grid=grid(16384), stream=stream0)
        del primals_262
        buf569 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf568, reinterpret_tensor(primals_876, (128, 128), (1, 128), 0), out=buf569)
        buf570 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf568, reinterpret_tensor(primals_878, (128, 128), (1, 128), 0), out=buf570)
        buf571 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (128, 512), (512, 1), 0), reinterpret_tensor(primals_880, (512, 128), (1, 512), 0), out=buf571)
        buf572 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf569, primals_877, buf572, 16384, grid=grid(16384), stream=stream0)
        del primals_877
        buf573 = reinterpret_tensor(buf569, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf570, primals_879, buf573, 16384, grid=grid(16384), stream=stream0)
        del primals_879
        buf574 = reinterpret_tensor(buf570, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf571, primals_881, buf574, 16384, grid=grid(16384), stream=stream0)
        del primals_881
        # Source Nodes: [], Original ATen: []
        buf575 = aten._scaled_dot_product_efficient_attention(buf572, buf573, buf574, None, True, 0.1, scale=0.17677669529663687)
        buf576 = buf575[0]
        buf577 = buf575[1]
        buf578 = buf575[2]
        buf579 = buf575[3]
        del buf575
        buf580 = buf571; del buf571  # reuse
        # Source Nodes: [layer_outputs_224], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf576, buf580, 16384, grid=grid(16384), stream=stream0)
        buf581 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_224], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_883, buf580, reinterpret_tensor(primals_882, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf581)
        del primals_883
        buf582 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_246, attention_output_80, layer_input_84, mul_130, mul_132], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf581, buf566, primals_259, primals_260, primals_263, primals_264, buf582, 16384, grid=grid(16384), stream=stream0)
        buf583 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (128, 128), (128, 1), 0), reinterpret_tensor(primals_884, (128, 512), (1, 128), 0), out=buf583)
        buf584 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_64, layer_outputs_226], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf583, primals_885, buf584, 65536, grid=grid(65536), stream=stream0)
        buf585 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_226], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_887, buf584, reinterpret_tensor(primals_886, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf585)
        del primals_887
        buf586 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_248, attention_output_81, hidden_states_146, mul_133], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf585, buf582, primals_265, primals_266, buf586, 16384, grid=grid(16384), stream=stream0)
        buf587 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf586, reinterpret_tensor(primals_888, (128, 512), (1, 128), 0), out=buf587)
        buf588 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_65, layer_outputs_229], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf587, primals_889, buf588, 65536, grid=grid(65536), stream=stream0)
        buf589 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_229], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_891, buf588, reinterpret_tensor(primals_890, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf589)
        del primals_891
        buf590 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_248, add_250, attention_output_81, attention_output_82, mul_133, mul_134], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf589, buf585, buf582, primals_265, primals_266, primals_267, primals_268, buf590, 16384, grid=grid(16384), stream=stream0)
        buf591 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf590, (128, 128), (128, 1), 0), reinterpret_tensor(primals_892, (128, 512), (1, 128), 0), out=buf591)
        buf592 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_66, layer_outputs_232], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf591, primals_893, buf592, 65536, grid=grid(65536), stream=stream0)
        buf593 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_232], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_895, buf592, reinterpret_tensor(primals_894, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf593)
        del primals_895
        buf594 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_252, attention_output_83, hidden_states_150, mul_135], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf593, buf590, primals_269, primals_270, buf594, 16384, grid=grid(16384), stream=stream0)
        buf595 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf594, reinterpret_tensor(primals_896, (128, 512), (1, 128), 0), out=buf595)
        buf596 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_67, layer_output_64], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf595, primals_897, buf596, 65536, grid=grid(65536), stream=stream0)
        buf597 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_899, buf596, reinterpret_tensor(primals_898, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf597)
        del primals_899
        buf598 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_252, add_254, attention_output_83, layer_output_65, layer_outputs_235, mul_135, mul_136], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf597, buf593, buf590, primals_269, primals_270, primals_271, primals_272, buf598, 16384, grid=grid(16384), stream=stream0)
        del primals_272
        buf599 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_235], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_901, buf598, reinterpret_tensor(primals_900, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf599)
        del primals_901
        buf600 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_256, layer_input_85, mul_137, value_tensor_17], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf599, buf565, primals_273, primals_274, buf600, 65536, grid=grid(65536), stream=stream0)
        buf601 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_85], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_903, buf600, reinterpret_tensor(primals_902, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf601)
        del primals_903
        buf602 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_87], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_905, buf600, reinterpret_tensor(primals_904, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf602)
        del primals_905
        buf603 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_17, mixed_query_layer_17, mul_139], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf602, primals_277, primals_278, buf603, 16384, grid=grid(16384), stream=stream0)
        del primals_278
        buf604 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf603, reinterpret_tensor(primals_906, (128, 128), (1, 128), 0), out=buf604)
        buf605 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf603, reinterpret_tensor(primals_908, (128, 128), (1, 128), 0), out=buf605)
        buf606 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf600, reinterpret_tensor(primals_910, (512, 128), (1, 512), 0), out=buf606)
        buf607 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf604, primals_907, buf607, 16384, grid=grid(16384), stream=stream0)
        del primals_907
        buf608 = reinterpret_tensor(buf604, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf605, primals_909, buf608, 16384, grid=grid(16384), stream=stream0)
        del primals_909
        buf609 = reinterpret_tensor(buf605, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf606, primals_911, buf609, 16384, grid=grid(16384), stream=stream0)
        del primals_911
        # Source Nodes: [], Original ATen: []
        buf610 = aten._scaled_dot_product_efficient_attention(buf607, buf608, buf609, None, True, 0.1, scale=0.17677669529663687)
        buf611 = buf610[0]
        buf612 = buf610[1]
        buf613 = buf610[2]
        buf614 = buf610[3]
        del buf610
        buf615 = buf606; del buf606  # reuse
        # Source Nodes: [layer_outputs_238], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf611, buf615, 16384, grid=grid(16384), stream=stream0)
        buf616 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_238], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_913, buf615, reinterpret_tensor(primals_912, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf616)
        del primals_913
        buf617 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_261, attention_output_85, layer_input_89, mul_138, mul_140], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf616, buf601, primals_275, primals_276, primals_279, primals_280, buf617, 16384, grid=grid(16384), stream=stream0)
        buf618 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf617, (128, 128), (128, 1), 0), reinterpret_tensor(primals_914, (128, 512), (1, 128), 0), out=buf618)
        buf619 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_68, layer_outputs_240], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf618, primals_915, buf619, 65536, grid=grid(65536), stream=stream0)
        buf620 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_240], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_917, buf619, reinterpret_tensor(primals_916, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf620)
        del primals_917
        buf621 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_263, attention_output_86, hidden_states_155, mul_141], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf620, buf617, primals_281, primals_282, buf621, 16384, grid=grid(16384), stream=stream0)
        buf622 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf621, reinterpret_tensor(primals_918, (128, 512), (1, 128), 0), out=buf622)
        buf623 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_69, layer_outputs_243], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf622, primals_919, buf623, 65536, grid=grid(65536), stream=stream0)
        buf624 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_243], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_921, buf623, reinterpret_tensor(primals_920, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf624)
        del primals_921
        buf625 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_263, add_265, attention_output_86, attention_output_87, mul_141, mul_142], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf624, buf620, buf617, primals_281, primals_282, primals_283, primals_284, buf625, 16384, grid=grid(16384), stream=stream0)
        buf626 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf625, (128, 128), (128, 1), 0), reinterpret_tensor(primals_922, (128, 512), (1, 128), 0), out=buf626)
        buf627 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_70, layer_outputs_246], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf626, primals_923, buf627, 65536, grid=grid(65536), stream=stream0)
        buf628 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_246], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_925, buf627, reinterpret_tensor(primals_924, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf628)
        del primals_925
        buf629 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_267, attention_output_88, hidden_states_159, mul_143], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf628, buf625, primals_285, primals_286, buf629, 16384, grid=grid(16384), stream=stream0)
        buf630 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf629, reinterpret_tensor(primals_926, (128, 512), (1, 128), 0), out=buf630)
        buf631 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_71, layer_output_68], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf630, primals_927, buf631, 65536, grid=grid(65536), stream=stream0)
        buf632 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_929, buf631, reinterpret_tensor(primals_928, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf632)
        del primals_929
        buf633 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_267, add_269, attention_output_88, layer_output_69, layer_outputs_249, mul_143, mul_144], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf632, buf628, buf625, primals_285, primals_286, primals_287, primals_288, buf633, 16384, grid=grid(16384), stream=stream0)
        del primals_288
        buf634 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_249], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_931, buf633, reinterpret_tensor(primals_930, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf634)
        del primals_931
        buf635 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_256, add_271, mul_137, mul_145, value_tensor_17, value_tensor_18], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf634, buf599, buf565, primals_273, primals_274, primals_289, primals_290, buf635, 65536, grid=grid(65536), stream=stream0)
        buf636 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_933, reinterpret_tensor(buf635, (128, 512), (512, 1), 0), reinterpret_tensor(primals_932, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf636)
        del primals_933
        buf637 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_935, reinterpret_tensor(buf635, (128, 512), (512, 1), 0), reinterpret_tensor(primals_934, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf637)
        del primals_935
        buf638 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_18, mixed_query_layer_18, mul_147], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf637, primals_293, primals_294, buf638, 16384, grid=grid(16384), stream=stream0)
        del primals_294
        buf639 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf638, reinterpret_tensor(primals_936, (128, 128), (1, 128), 0), out=buf639)
        buf640 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf638, reinterpret_tensor(primals_938, (128, 128), (1, 128), 0), out=buf640)
        buf641 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf635, (128, 512), (512, 1), 0), reinterpret_tensor(primals_940, (512, 128), (1, 512), 0), out=buf641)
        buf642 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf639, primals_937, buf642, 16384, grid=grid(16384), stream=stream0)
        del primals_937
        buf643 = reinterpret_tensor(buf639, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf639  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf640, primals_939, buf643, 16384, grid=grid(16384), stream=stream0)
        del primals_939
        buf644 = reinterpret_tensor(buf640, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf641, primals_941, buf644, 16384, grid=grid(16384), stream=stream0)
        del primals_941
        # Source Nodes: [], Original ATen: []
        buf645 = aten._scaled_dot_product_efficient_attention(buf642, buf643, buf644, None, True, 0.1, scale=0.17677669529663687)
        buf646 = buf645[0]
        buf647 = buf645[1]
        buf648 = buf645[2]
        buf649 = buf645[3]
        del buf645
        buf650 = buf641; del buf641  # reuse
        # Source Nodes: [layer_outputs_252], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf646, buf650, 16384, grid=grid(16384), stream=stream0)
        buf651 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_252], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_943, buf650, reinterpret_tensor(primals_942, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf651)
        del primals_943
        buf652 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_276, attention_output_90, layer_input_94, mul_146, mul_148], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf651, buf636, primals_291, primals_292, primals_295, primals_296, buf652, 16384, grid=grid(16384), stream=stream0)
        buf653 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf652, (128, 128), (128, 1), 0), reinterpret_tensor(primals_944, (128, 512), (1, 128), 0), out=buf653)
        buf654 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_72, layer_outputs_254], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf653, primals_945, buf654, 65536, grid=grid(65536), stream=stream0)
        buf655 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_254], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_947, buf654, reinterpret_tensor(primals_946, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf655)
        del primals_947
        buf656 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_278, attention_output_91, hidden_states_164, mul_149], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf655, buf652, primals_297, primals_298, buf656, 16384, grid=grid(16384), stream=stream0)
        buf657 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf656, reinterpret_tensor(primals_948, (128, 512), (1, 128), 0), out=buf657)
        buf658 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_73, layer_outputs_257], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf657, primals_949, buf658, 65536, grid=grid(65536), stream=stream0)
        buf659 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_257], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_951, buf658, reinterpret_tensor(primals_950, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf659)
        del primals_951
        buf660 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_278, add_280, attention_output_91, attention_output_92, mul_149, mul_150], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf659, buf655, buf652, primals_297, primals_298, primals_299, primals_300, buf660, 16384, grid=grid(16384), stream=stream0)
        buf661 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf660, (128, 128), (128, 1), 0), reinterpret_tensor(primals_952, (128, 512), (1, 128), 0), out=buf661)
        buf662 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_74, layer_outputs_260], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf661, primals_953, buf662, 65536, grid=grid(65536), stream=stream0)
        buf663 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_260], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_955, buf662, reinterpret_tensor(primals_954, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf663)
        del primals_955
        buf664 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_282, attention_output_93, hidden_states_168, mul_151], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf663, buf660, primals_301, primals_302, buf664, 16384, grid=grid(16384), stream=stream0)
        buf665 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf664, reinterpret_tensor(primals_956, (128, 512), (1, 128), 0), out=buf665)
        buf666 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_75, layer_output_72], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf665, primals_957, buf666, 65536, grid=grid(65536), stream=stream0)
        buf667 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_959, buf666, reinterpret_tensor(primals_958, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf667)
        del primals_959
        buf668 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_282, add_284, attention_output_93, layer_output_73, layer_outputs_263, mul_151, mul_152], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf667, buf663, buf660, primals_301, primals_302, primals_303, primals_304, buf668, 16384, grid=grid(16384), stream=stream0)
        del primals_304
        buf669 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_263], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_961, buf668, reinterpret_tensor(primals_960, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf669)
        del primals_961
        buf670 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_286, layer_input_95, mul_153, value_tensor_19], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf669, buf635, primals_305, primals_306, buf670, 65536, grid=grid(65536), stream=stream0)
        buf671 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_963, buf670, reinterpret_tensor(primals_962, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf671)
        del primals_963
        buf672 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_97], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_965, buf670, reinterpret_tensor(primals_964, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf672)
        del primals_965
        buf673 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_19, mixed_query_layer_19, mul_155], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf672, primals_309, primals_310, buf673, 16384, grid=grid(16384), stream=stream0)
        del primals_310
        buf674 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf673, reinterpret_tensor(primals_966, (128, 128), (1, 128), 0), out=buf674)
        buf675 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf673, reinterpret_tensor(primals_968, (128, 128), (1, 128), 0), out=buf675)
        buf676 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf670, reinterpret_tensor(primals_970, (512, 128), (1, 512), 0), out=buf676)
        buf677 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf674, primals_967, buf677, 16384, grid=grid(16384), stream=stream0)
        del primals_967
        buf678 = reinterpret_tensor(buf674, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf675, primals_969, buf678, 16384, grid=grid(16384), stream=stream0)
        del primals_969
        buf679 = reinterpret_tensor(buf675, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf676, primals_971, buf679, 16384, grid=grid(16384), stream=stream0)
        del primals_971
        # Source Nodes: [], Original ATen: []
        buf680 = aten._scaled_dot_product_efficient_attention(buf677, buf678, buf679, None, True, 0.1, scale=0.17677669529663687)
        buf681 = buf680[0]
        buf682 = buf680[1]
        buf683 = buf680[2]
        buf684 = buf680[3]
        del buf680
        buf685 = buf676; del buf676  # reuse
        # Source Nodes: [layer_outputs_266], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf681, buf685, 16384, grid=grid(16384), stream=stream0)
        buf686 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_266], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_973, buf685, reinterpret_tensor(primals_972, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf686)
        del primals_973
        buf687 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_291, attention_output_95, layer_input_99, mul_154, mul_156], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf686, buf671, primals_307, primals_308, primals_311, primals_312, buf687, 16384, grid=grid(16384), stream=stream0)
        buf688 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf687, (128, 128), (128, 1), 0), reinterpret_tensor(primals_974, (128, 512), (1, 128), 0), out=buf688)
        buf689 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_76, layer_outputs_268], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf688, primals_975, buf689, 65536, grid=grid(65536), stream=stream0)
        buf690 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_268], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_977, buf689, reinterpret_tensor(primals_976, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf690)
        del primals_977
        buf691 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_293, attention_output_96, hidden_states_173, mul_157], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf690, buf687, primals_313, primals_314, buf691, 16384, grid=grid(16384), stream=stream0)
        buf692 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf691, reinterpret_tensor(primals_978, (128, 512), (1, 128), 0), out=buf692)
        buf693 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_77, layer_outputs_271], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf692, primals_979, buf693, 65536, grid=grid(65536), stream=stream0)
        buf694 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_271], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_981, buf693, reinterpret_tensor(primals_980, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf694)
        del primals_981
        buf695 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_293, add_295, attention_output_96, attention_output_97, mul_157, mul_158], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf694, buf690, buf687, primals_313, primals_314, primals_315, primals_316, buf695, 16384, grid=grid(16384), stream=stream0)
        buf696 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf695, (128, 128), (128, 1), 0), reinterpret_tensor(primals_982, (128, 512), (1, 128), 0), out=buf696)
        buf697 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_78, layer_outputs_274], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf696, primals_983, buf697, 65536, grid=grid(65536), stream=stream0)
        buf698 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_274], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_985, buf697, reinterpret_tensor(primals_984, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf698)
        del primals_985
        buf699 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_297, attention_output_98, hidden_states_177, mul_159], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf698, buf695, primals_317, primals_318, buf699, 16384, grid=grid(16384), stream=stream0)
        buf700 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf699, reinterpret_tensor(primals_986, (128, 512), (1, 128), 0), out=buf700)
        buf701 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_79, layer_output_76], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf700, primals_987, buf701, 65536, grid=grid(65536), stream=stream0)
        buf702 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_989, buf701, reinterpret_tensor(primals_988, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf702)
        del primals_989
        buf703 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_297, add_299, attention_output_98, layer_output_77, layer_outputs_277, mul_159, mul_160], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf702, buf698, buf695, primals_317, primals_318, primals_319, primals_320, buf703, 16384, grid=grid(16384), stream=stream0)
        del primals_320
        buf704 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_277], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_991, buf703, reinterpret_tensor(primals_990, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf704)
        del primals_991
        buf705 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_286, add_301, mul_153, mul_161, value_tensor_19, value_tensor_20], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf704, buf669, buf635, primals_305, primals_306, primals_321, primals_322, buf705, 65536, grid=grid(65536), stream=stream0)
        buf706 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_993, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(primals_992, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf706)
        del primals_993
        buf707 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_995, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(primals_994, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf707)
        del primals_995
        buf708 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_20, mixed_query_layer_20, mul_163], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf707, primals_325, primals_326, buf708, 16384, grid=grid(16384), stream=stream0)
        del primals_326
        buf709 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf708, reinterpret_tensor(primals_996, (128, 128), (1, 128), 0), out=buf709)
        buf710 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf708, reinterpret_tensor(primals_998, (128, 128), (1, 128), 0), out=buf710)
        buf711 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1000, (512, 128), (1, 512), 0), out=buf711)
        buf712 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf709, primals_997, buf712, 16384, grid=grid(16384), stream=stream0)
        del primals_997
        buf713 = reinterpret_tensor(buf709, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf710, primals_999, buf713, 16384, grid=grid(16384), stream=stream0)
        del primals_999
        buf714 = reinterpret_tensor(buf710, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf711, primals_1001, buf714, 16384, grid=grid(16384), stream=stream0)
        del primals_1001
        # Source Nodes: [], Original ATen: []
        buf715 = aten._scaled_dot_product_efficient_attention(buf712, buf713, buf714, None, True, 0.1, scale=0.17677669529663687)
        buf716 = buf715[0]
        buf717 = buf715[1]
        buf718 = buf715[2]
        buf719 = buf715[3]
        del buf715
        buf720 = buf711; del buf711  # reuse
        # Source Nodes: [layer_outputs_280], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf716, buf720, 16384, grid=grid(16384), stream=stream0)
        buf721 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_280], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1003, buf720, reinterpret_tensor(primals_1002, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf721)
        del primals_1003
        buf722 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_306, attention_output_100, layer_input_104, mul_162, mul_164], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf721, buf706, primals_323, primals_324, primals_327, primals_328, buf722, 16384, grid=grid(16384), stream=stream0)
        buf723 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf722, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1004, (128, 512), (1, 128), 0), out=buf723)
        buf724 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_80, layer_outputs_282], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf723, primals_1005, buf724, 65536, grid=grid(65536), stream=stream0)
        buf725 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_282], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1007, buf724, reinterpret_tensor(primals_1006, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf725)
        del primals_1007
        buf726 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_308, attention_output_101, hidden_states_182, mul_165], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf725, buf722, primals_329, primals_330, buf726, 16384, grid=grid(16384), stream=stream0)
        buf727 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf726, reinterpret_tensor(primals_1008, (128, 512), (1, 128), 0), out=buf727)
        buf728 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_81, layer_outputs_285], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf727, primals_1009, buf728, 65536, grid=grid(65536), stream=stream0)
        buf729 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_285], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1011, buf728, reinterpret_tensor(primals_1010, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf729)
        del primals_1011
        buf730 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_308, add_310, attention_output_101, attention_output_102, mul_165, mul_166], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf729, buf725, buf722, primals_329, primals_330, primals_331, primals_332, buf730, 16384, grid=grid(16384), stream=stream0)
        buf731 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf730, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1012, (128, 512), (1, 128), 0), out=buf731)
        buf732 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_82, layer_outputs_288], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf731, primals_1013, buf732, 65536, grid=grid(65536), stream=stream0)
        buf733 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_288], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1015, buf732, reinterpret_tensor(primals_1014, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf733)
        del primals_1015
        buf734 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_312, attention_output_103, hidden_states_186, mul_167], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf733, buf730, primals_333, primals_334, buf734, 16384, grid=grid(16384), stream=stream0)
        buf735 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf734, reinterpret_tensor(primals_1016, (128, 512), (1, 128), 0), out=buf735)
        buf736 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_83, layer_output_80], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf735, primals_1017, buf736, 65536, grid=grid(65536), stream=stream0)
        buf737 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1019, buf736, reinterpret_tensor(primals_1018, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf737)
        del primals_1019
        buf738 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_312, add_314, attention_output_103, layer_output_81, layer_outputs_291, mul_167, mul_168], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf737, buf733, buf730, primals_333, primals_334, primals_335, primals_336, buf738, 16384, grid=grid(16384), stream=stream0)
        del primals_336
        buf739 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_291], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1021, buf738, reinterpret_tensor(primals_1020, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf739)
        del primals_1021
        buf740 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_316, layer_input_105, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf739, buf705, primals_337, primals_338, buf740, 65536, grid=grid(65536), stream=stream0)
        buf741 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_105], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1023, buf740, reinterpret_tensor(primals_1022, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf741)
        del primals_1023
        buf742 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1025, buf740, reinterpret_tensor(primals_1024, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf742)
        del primals_1025
        buf743 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_21, mixed_query_layer_21, mul_171], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf742, primals_341, primals_342, buf743, 16384, grid=grid(16384), stream=stream0)
        del primals_342
        buf744 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf743, reinterpret_tensor(primals_1026, (128, 128), (1, 128), 0), out=buf744)
        buf745 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf743, reinterpret_tensor(primals_1028, (128, 128), (1, 128), 0), out=buf745)
        buf746 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf740, reinterpret_tensor(primals_1030, (512, 128), (1, 512), 0), out=buf746)
        buf747 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf744, primals_1027, buf747, 16384, grid=grid(16384), stream=stream0)
        del primals_1027
        buf748 = reinterpret_tensor(buf744, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf745, primals_1029, buf748, 16384, grid=grid(16384), stream=stream0)
        del primals_1029
        buf749 = reinterpret_tensor(buf745, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf746, primals_1031, buf749, 16384, grid=grid(16384), stream=stream0)
        del primals_1031
        # Source Nodes: [], Original ATen: []
        buf750 = aten._scaled_dot_product_efficient_attention(buf747, buf748, buf749, None, True, 0.1, scale=0.17677669529663687)
        buf751 = buf750[0]
        buf752 = buf750[1]
        buf753 = buf750[2]
        buf754 = buf750[3]
        del buf750
        buf755 = buf746; del buf746  # reuse
        # Source Nodes: [layer_outputs_294], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf751, buf755, 16384, grid=grid(16384), stream=stream0)
        buf756 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_294], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1033, buf755, reinterpret_tensor(primals_1032, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf756)
        del primals_1033
        buf757 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_321, attention_output_105, layer_input_109, mul_170, mul_172], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf756, buf741, primals_339, primals_340, primals_343, primals_344, buf757, 16384, grid=grid(16384), stream=stream0)
        buf758 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf757, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1034, (128, 512), (1, 128), 0), out=buf758)
        buf759 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_84, layer_outputs_296], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf758, primals_1035, buf759, 65536, grid=grid(65536), stream=stream0)
        buf760 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_296], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1037, buf759, reinterpret_tensor(primals_1036, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf760)
        del primals_1037
        buf761 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_323, attention_output_106, hidden_states_191, mul_173], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf760, buf757, primals_345, primals_346, buf761, 16384, grid=grid(16384), stream=stream0)
        buf762 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf761, reinterpret_tensor(primals_1038, (128, 512), (1, 128), 0), out=buf762)
        buf763 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_85, layer_outputs_299], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf762, primals_1039, buf763, 65536, grid=grid(65536), stream=stream0)
        buf764 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_299], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1041, buf763, reinterpret_tensor(primals_1040, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf764)
        del primals_1041
        buf765 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_323, add_325, attention_output_106, attention_output_107, mul_173, mul_174], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf764, buf760, buf757, primals_345, primals_346, primals_347, primals_348, buf765, 16384, grid=grid(16384), stream=stream0)
        buf766 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf765, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1042, (128, 512), (1, 128), 0), out=buf766)
        buf767 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_86, layer_outputs_302], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf766, primals_1043, buf767, 65536, grid=grid(65536), stream=stream0)
        buf768 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_302], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1045, buf767, reinterpret_tensor(primals_1044, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf768)
        del primals_1045
        buf769 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_327, attention_output_108, hidden_states_195, mul_175], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf768, buf765, primals_349, primals_350, buf769, 16384, grid=grid(16384), stream=stream0)
        buf770 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf769, reinterpret_tensor(primals_1046, (128, 512), (1, 128), 0), out=buf770)
        buf771 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_87, layer_output_84], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf770, primals_1047, buf771, 65536, grid=grid(65536), stream=stream0)
        buf772 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1049, buf771, reinterpret_tensor(primals_1048, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf772)
        del primals_1049
        buf773 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_327, add_329, attention_output_108, layer_output_85, layer_outputs_305, mul_175, mul_176], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf772, buf768, buf765, primals_349, primals_350, primals_351, primals_352, buf773, 16384, grid=grid(16384), stream=stream0)
        del primals_352
        buf774 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_305], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1051, buf773, reinterpret_tensor(primals_1050, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf774)
        del primals_1051
        buf775 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_316, add_331, mul_169, mul_177, value_tensor_21, value_tensor_22], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(buf774, buf739, buf705, primals_337, primals_338, primals_353, primals_354, buf775, 65536, grid=grid(65536), stream=stream0)
        buf776 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1053, reinterpret_tensor(buf775, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1052, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf776)
        del primals_1053
        buf777 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_112], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1055, reinterpret_tensor(buf775, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1054, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf777)
        del primals_1055
        buf778 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_22, mixed_query_layer_22, mul_179], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf777, primals_357, primals_358, buf778, 16384, grid=grid(16384), stream=stream0)
        del primals_358
        buf779 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf778, reinterpret_tensor(primals_1056, (128, 128), (1, 128), 0), out=buf779)
        buf780 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf778, reinterpret_tensor(primals_1058, (128, 128), (1, 128), 0), out=buf780)
        buf781 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1060, (512, 128), (1, 512), 0), out=buf781)
        buf782 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf779, primals_1057, buf782, 16384, grid=grid(16384), stream=stream0)
        del primals_1057
        buf783 = reinterpret_tensor(buf779, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf779  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf780, primals_1059, buf783, 16384, grid=grid(16384), stream=stream0)
        del primals_1059
        buf784 = reinterpret_tensor(buf780, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf781, primals_1061, buf784, 16384, grid=grid(16384), stream=stream0)
        del primals_1061
        # Source Nodes: [], Original ATen: []
        buf785 = aten._scaled_dot_product_efficient_attention(buf782, buf783, buf784, None, True, 0.1, scale=0.17677669529663687)
        buf786 = buf785[0]
        buf787 = buf785[1]
        buf788 = buf785[2]
        buf789 = buf785[3]
        del buf785
        buf790 = buf781; del buf781  # reuse
        # Source Nodes: [layer_outputs_308], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf786, buf790, 16384, grid=grid(16384), stream=stream0)
        buf791 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_308], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1063, buf790, reinterpret_tensor(primals_1062, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf791)
        del primals_1063
        buf792 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_336, attention_output_110, layer_input_114, mul_178, mul_180], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf791, buf776, primals_355, primals_356, primals_359, primals_360, buf792, 16384, grid=grid(16384), stream=stream0)
        buf793 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf792, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1064, (128, 512), (1, 128), 0), out=buf793)
        buf794 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_88, layer_outputs_310], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf793, primals_1065, buf794, 65536, grid=grid(65536), stream=stream0)
        buf795 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_310], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1067, buf794, reinterpret_tensor(primals_1066, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf795)
        del primals_1067
        buf796 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_338, attention_output_111, hidden_states_200, mul_181], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf795, buf792, primals_361, primals_362, buf796, 16384, grid=grid(16384), stream=stream0)
        buf797 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf796, reinterpret_tensor(primals_1068, (128, 512), (1, 128), 0), out=buf797)
        buf798 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [intermediate_output_89, layer_outputs_313], Original ATen: [aten.relu, aten.view]
        triton_poi_fused_relu_view_7.run(buf797, primals_1069, buf798, 65536, grid=grid(65536), stream=stream0)
        buf799 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_313], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1071, buf798, reinterpret_tensor(primals_1070, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf799)
        del primals_1071
        buf800 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_338, add_340, attention_output_111, attention_output_112, mul_181, mul_182], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf799, buf795, buf792, primals_361, primals_362, primals_363, primals_364, buf800, 16384, grid=grid(16384), stream=stream0)
        buf801 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf800, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1072, (128, 512), (1, 128), 0), out=buf801)
        buf802 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf866 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_90, layer_outputs_316], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf801, primals_1073, buf802, buf866, 65536, grid=grid(65536), stream=stream0)
        del primals_1073
        buf803 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_316], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1075, buf802, reinterpret_tensor(primals_1074, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf803)
        del primals_1075
        buf804 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_342, attention_output_113, hidden_states_204, mul_183], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf803, buf800, primals_365, primals_366, buf804, 16384, grid=grid(16384), stream=stream0)
        buf805 = buf801; del buf801  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf804, reinterpret_tensor(primals_1076, (128, 512), (1, 128), 0), out=buf805)
        buf806 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf865 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_91, layer_output_88], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf805, primals_1077, buf806, buf865, 65536, grid=grid(65536), stream=stream0)
        del primals_1077
        buf807 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1079, buf806, reinterpret_tensor(primals_1078, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf807)
        del primals_1079
        buf808 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_342, add_344, attention_output_113, layer_output_89, layer_outputs_319, mul_183, mul_184], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf807, buf803, buf800, primals_365, primals_366, primals_367, primals_368, buf808, 16384, grid=grid(16384), stream=stream0)
        del primals_368
        buf809 = buf805; del buf805  # reuse
        # Source Nodes: [layer_outputs_319], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1081, buf808, reinterpret_tensor(primals_1080, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf809)
        del primals_1081
        buf810 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_346, layer_input_115, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_12.run(buf809, buf775, primals_369, primals_370, buf810, 65536, grid=grid(65536), stream=stream0)
        buf811 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_115], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1083, buf810, reinterpret_tensor(primals_1082, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf811)
        del primals_1083
        buf812 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_input_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1085, buf810, reinterpret_tensor(primals_1084, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf812)
        del primals_1085
        buf813 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_tensor_23, mixed_query_layer_23, mul_187], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_3.run(buf812, primals_373, primals_374, buf813, 16384, grid=grid(16384), stream=stream0)
        del primals_374
        buf814 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf813, reinterpret_tensor(primals_1086, (128, 128), (1, 128), 0), out=buf814)
        buf815 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf813, reinterpret_tensor(primals_1088, (128, 128), (1, 128), 0), out=buf815)
        buf816 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf810, reinterpret_tensor(primals_1090, (512, 128), (1, 512), 0), out=buf816)
        buf817 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf814, primals_1087, buf817, 16384, grid=grid(16384), stream=stream0)
        del primals_1087
        buf818 = reinterpret_tensor(buf814, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf814  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf815, primals_1089, buf818, 16384, grid=grid(16384), stream=stream0)
        del primals_1089
        buf819 = reinterpret_tensor(buf815, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf815  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf816, primals_1091, buf819, 16384, grid=grid(16384), stream=stream0)
        del primals_1091
        # Source Nodes: [], Original ATen: []
        buf820 = aten._scaled_dot_product_efficient_attention(buf817, buf818, buf819, None, True, 0.1, scale=0.17677669529663687)
        buf821 = buf820[0]
        buf822 = buf820[1]
        buf823 = buf820[2]
        buf824 = buf820[3]
        del buf820
        buf825 = buf816; del buf816  # reuse
        # Source Nodes: [layer_outputs_322], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf821, buf825, 16384, grid=grid(16384), stream=stream0)
        buf826 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_322], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1093, buf825, reinterpret_tensor(primals_1092, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf826)
        del primals_1093
        buf827 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_351, attention_output_115, layer_input_119, mul_186, mul_188], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf826, buf811, primals_371, primals_372, primals_375, primals_376, buf827, 16384, grid=grid(16384), stream=stream0)
        buf828 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf827, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1094, (128, 512), (1, 128), 0), out=buf828)
        buf829 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf864 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_92, layer_outputs_324], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf828, primals_1095, buf829, buf864, 65536, grid=grid(65536), stream=stream0)
        del primals_1095
        buf830 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_324], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1097, buf829, reinterpret_tensor(primals_1096, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf830)
        del primals_1097
        buf831 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_353, attention_output_116, hidden_states_209, mul_189], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf830, buf827, primals_377, primals_378, buf831, 16384, grid=grid(16384), stream=stream0)
        buf832 = buf828; del buf828  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf831, reinterpret_tensor(primals_1098, (128, 512), (1, 128), 0), out=buf832)
        buf833 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf863 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_93, layer_outputs_327], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf832, primals_1099, buf833, buf863, 65536, grid=grid(65536), stream=stream0)
        del primals_1099
        buf834 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_327], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1101, buf833, reinterpret_tensor(primals_1100, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf834)
        del primals_1101
        buf835 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_353, add_355, attention_output_116, attention_output_117, mul_189, mul_190], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf834, buf830, buf827, primals_377, primals_378, primals_379, primals_380, buf835, 16384, grid=grid(16384), stream=stream0)
        buf836 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf835, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1102, (128, 512), (1, 128), 0), out=buf836)
        buf837 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf862 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_94, layer_outputs_330], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf836, primals_1103, buf837, buf862, 65536, grid=grid(65536), stream=stream0)
        del primals_1103
        buf838 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_outputs_330], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1105, buf837, reinterpret_tensor(primals_1104, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf838)
        del primals_1105
        buf839 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_357, attention_output_118, hidden_states_213, mul_191], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_view_8.run(buf838, buf835, primals_381, primals_382, buf839, 16384, grid=grid(16384), stream=stream0)
        buf840 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf839, reinterpret_tensor(primals_1106, (128, 512), (1, 128), 0), out=buf840)
        buf841 = empty((128, 512), device='cuda', dtype=torch.float32)
        buf861 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_95, layer_output_92], Original ATen: [aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_relu_threshold_backward_view_14.run(buf840, primals_1107, buf841, buf861, 65536, grid=grid(65536), stream=stream0)
        del primals_1107
        buf842 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer_output_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1109, buf841, reinterpret_tensor(primals_1108, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf842)
        del primals_1109
        buf843 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_357, add_359, attention_output_118, layer_output_93, layer_outputs_333, mul_191, mul_192], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_9.run(buf842, buf838, buf835, primals_381, primals_382, primals_383, primals_384, buf843, 16384, grid=grid(16384), stream=stream0)
        del primals_384
        buf844 = buf840; del buf840  # reuse
        # Source Nodes: [layer_outputs_333], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1111, buf843, reinterpret_tensor(primals_1110, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf844)
        del primals_1111
        buf845 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_346, add_361, logits, mul_185, mul_193, sequence_output, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.view]
        triton_poi_fused_add_mul_13.run(buf844, buf809, buf775, primals_369, primals_370, primals_385, primals_386, buf845, 65536, grid=grid(65536), stream=stream0)
        del primals_386
        buf846 = empty((128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf845, reinterpret_tensor(primals_1112, (512, 2), (1, 512), 0), out=buf846)
        buf847 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf851 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_15.run(buf846, primals_1113, buf847, buf851, 1, 128, grid=grid(1), stream=stream0)
        buf848 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf855 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_16.run(buf846, primals_1113, buf848, buf855, 1, 128, grid=grid(1), stream=stream0)
        del buf846
        del primals_1113
        buf852 = empty((1, ), device='cuda', dtype=torch.bool)
        buf856 = empty((1, ), device='cuda', dtype=torch.bool)
        buf957 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_363, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_forward_17.run(primals_1116, primals_1117, buf851, buf855, buf852, buf856, buf957, 1, grid=grid(1), stream=stream0)
        buf857 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf858 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_18.run(primals_1117, buf857, buf858, 1, grid=grid(1), stream=stream0)
        del primals_1117
        buf859 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf860 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_18.run(primals_1116, buf859, buf860, 1, grid=grid(1), stream=stream0)
        del primals_1116
        buf867 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_89], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf797, primals_1069, buf867, 65536, grid=grid(65536), stream=stream0)
        del buf797
        del primals_1069
        buf868 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_88], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf793, primals_1065, buf868, 65536, grid=grid(65536), stream=stream0)
        del buf793
        del primals_1065
        buf869 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_87], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf770, primals_1047, buf869, 65536, grid=grid(65536), stream=stream0)
        del buf770
        del primals_1047
        buf870 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_86], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf766, primals_1043, buf870, 65536, grid=grid(65536), stream=stream0)
        del buf766
        del primals_1043
        buf871 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_85], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf762, primals_1039, buf871, 65536, grid=grid(65536), stream=stream0)
        del buf762
        del primals_1039
        buf872 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_84], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf758, primals_1035, buf872, 65536, grid=grid(65536), stream=stream0)
        del buf758
        del primals_1035
        buf873 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_83], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf735, primals_1017, buf873, 65536, grid=grid(65536), stream=stream0)
        del buf735
        del primals_1017
        buf874 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_82], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf731, primals_1013, buf874, 65536, grid=grid(65536), stream=stream0)
        del buf731
        del primals_1013
        buf875 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_81], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf727, primals_1009, buf875, 65536, grid=grid(65536), stream=stream0)
        del buf727
        del primals_1009
        buf876 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_80], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf723, primals_1005, buf876, 65536, grid=grid(65536), stream=stream0)
        del buf723
        del primals_1005
        buf877 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_79], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf700, primals_987, buf877, 65536, grid=grid(65536), stream=stream0)
        del buf700
        del primals_987
        buf878 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_78], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf696, primals_983, buf878, 65536, grid=grid(65536), stream=stream0)
        del buf696
        del primals_983
        buf879 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_77], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf692, primals_979, buf879, 65536, grid=grid(65536), stream=stream0)
        del buf692
        del primals_979
        buf880 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_76], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf688, primals_975, buf880, 65536, grid=grid(65536), stream=stream0)
        del buf688
        del primals_975
        buf881 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_75], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf665, primals_957, buf881, 65536, grid=grid(65536), stream=stream0)
        del buf665
        del primals_957
        buf882 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_74], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf661, primals_953, buf882, 65536, grid=grid(65536), stream=stream0)
        del buf661
        del primals_953
        buf883 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_73], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf657, primals_949, buf883, 65536, grid=grid(65536), stream=stream0)
        del buf657
        del primals_949
        buf884 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_72], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf653, primals_945, buf884, 65536, grid=grid(65536), stream=stream0)
        del buf653
        del primals_945
        buf885 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_71], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf630, primals_927, buf885, 65536, grid=grid(65536), stream=stream0)
        del buf630
        del primals_927
        buf886 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_70], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf626, primals_923, buf886, 65536, grid=grid(65536), stream=stream0)
        del buf626
        del primals_923
        buf887 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_69], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf622, primals_919, buf887, 65536, grid=grid(65536), stream=stream0)
        del buf622
        del primals_919
        buf888 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_68], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf618, primals_915, buf888, 65536, grid=grid(65536), stream=stream0)
        del buf618
        del primals_915
        buf889 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_67], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf595, primals_897, buf889, 65536, grid=grid(65536), stream=stream0)
        del buf595
        del primals_897
        buf890 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_66], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf591, primals_893, buf890, 65536, grid=grid(65536), stream=stream0)
        del buf591
        del primals_893
        buf891 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_65], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf587, primals_889, buf891, 65536, grid=grid(65536), stream=stream0)
        del buf587
        del primals_889
        buf892 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_64], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf583, primals_885, buf892, 65536, grid=grid(65536), stream=stream0)
        del buf583
        del primals_885
        buf893 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_63], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf560, primals_867, buf893, 65536, grid=grid(65536), stream=stream0)
        del buf560
        del primals_867
        buf894 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_62], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf556, primals_863, buf894, 65536, grid=grid(65536), stream=stream0)
        del buf556
        del primals_863
        buf895 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_61], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf552, primals_859, buf895, 65536, grid=grid(65536), stream=stream0)
        del buf552
        del primals_859
        buf896 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_60], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf548, primals_855, buf896, 65536, grid=grid(65536), stream=stream0)
        del buf548
        del primals_855
        buf897 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_59], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf525, primals_837, buf897, 65536, grid=grid(65536), stream=stream0)
        del buf525
        del primals_837
        buf898 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_58], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf521, primals_833, buf898, 65536, grid=grid(65536), stream=stream0)
        del buf521
        del primals_833
        buf899 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_57], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf517, primals_829, buf899, 65536, grid=grid(65536), stream=stream0)
        del buf517
        del primals_829
        buf900 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_56], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf513, primals_825, buf900, 65536, grid=grid(65536), stream=stream0)
        del buf513
        del primals_825
        buf901 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_55], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf490, primals_807, buf901, 65536, grid=grid(65536), stream=stream0)
        del buf490
        del primals_807
        buf902 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_54], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf486, primals_803, buf902, 65536, grid=grid(65536), stream=stream0)
        del buf486
        del primals_803
        buf903 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_53], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf482, primals_799, buf903, 65536, grid=grid(65536), stream=stream0)
        del buf482
        del primals_799
        buf904 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_52], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf478, primals_795, buf904, 65536, grid=grid(65536), stream=stream0)
        del buf478
        del primals_795
        buf905 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_51], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf455, primals_777, buf905, 65536, grid=grid(65536), stream=stream0)
        del buf455
        del primals_777
        buf906 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_50], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf451, primals_773, buf906, 65536, grid=grid(65536), stream=stream0)
        del buf451
        del primals_773
        buf907 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_49], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf447, primals_769, buf907, 65536, grid=grid(65536), stream=stream0)
        del buf447
        del primals_769
        buf908 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_48], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf443, primals_765, buf908, 65536, grid=grid(65536), stream=stream0)
        del buf443
        del primals_765
        buf909 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_47], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf420, primals_747, buf909, 65536, grid=grid(65536), stream=stream0)
        del buf420
        del primals_747
        buf910 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_46], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf416, primals_743, buf910, 65536, grid=grid(65536), stream=stream0)
        del buf416
        del primals_743
        buf911 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_45], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf412, primals_739, buf911, 65536, grid=grid(65536), stream=stream0)
        del buf412
        del primals_739
        buf912 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_44], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf408, primals_735, buf912, 65536, grid=grid(65536), stream=stream0)
        del buf408
        del primals_735
        buf913 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_43], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf385, primals_717, buf913, 65536, grid=grid(65536), stream=stream0)
        del buf385
        del primals_717
        buf914 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_42], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf381, primals_713, buf914, 65536, grid=grid(65536), stream=stream0)
        del buf381
        del primals_713
        buf915 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_41], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf377, primals_709, buf915, 65536, grid=grid(65536), stream=stream0)
        del buf377
        del primals_709
        buf916 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_40], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf373, primals_705, buf916, 65536, grid=grid(65536), stream=stream0)
        del buf373
        del primals_705
        buf917 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_39], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf350, primals_687, buf917, 65536, grid=grid(65536), stream=stream0)
        del buf350
        del primals_687
        buf918 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_38], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf346, primals_683, buf918, 65536, grid=grid(65536), stream=stream0)
        del buf346
        del primals_683
        buf919 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_37], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf342, primals_679, buf919, 65536, grid=grid(65536), stream=stream0)
        del buf342
        del primals_679
        buf920 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_36], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf338, primals_675, buf920, 65536, grid=grid(65536), stream=stream0)
        del buf338
        del primals_675
        buf921 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_35], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf315, primals_657, buf921, 65536, grid=grid(65536), stream=stream0)
        del buf315
        del primals_657
        buf922 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_34], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf311, primals_653, buf922, 65536, grid=grid(65536), stream=stream0)
        del buf311
        del primals_653
        buf923 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_33], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf307, primals_649, buf923, 65536, grid=grid(65536), stream=stream0)
        del buf307
        del primals_649
        buf924 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_32], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf303, primals_645, buf924, 65536, grid=grid(65536), stream=stream0)
        del buf303
        del primals_645
        buf925 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_31], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf280, primals_627, buf925, 65536, grid=grid(65536), stream=stream0)
        del buf280
        del primals_627
        buf926 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_30], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf276, primals_623, buf926, 65536, grid=grid(65536), stream=stream0)
        del buf276
        del primals_623
        buf927 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_29], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf272, primals_619, buf927, 65536, grid=grid(65536), stream=stream0)
        del buf272
        del primals_619
        buf928 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_28], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf268, primals_615, buf928, 65536, grid=grid(65536), stream=stream0)
        del buf268
        del primals_615
        buf929 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_27], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf245, primals_597, buf929, 65536, grid=grid(65536), stream=stream0)
        del buf245
        del primals_597
        buf930 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_26], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf241, primals_593, buf930, 65536, grid=grid(65536), stream=stream0)
        del buf241
        del primals_593
        buf931 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_25], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf237, primals_589, buf931, 65536, grid=grid(65536), stream=stream0)
        del buf237
        del primals_589
        buf932 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_24], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf233, primals_585, buf932, 65536, grid=grid(65536), stream=stream0)
        del buf233
        del primals_585
        buf933 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_23], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf210, primals_567, buf933, 65536, grid=grid(65536), stream=stream0)
        del buf210
        del primals_567
        buf934 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_22], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf206, primals_563, buf934, 65536, grid=grid(65536), stream=stream0)
        del buf206
        del primals_563
        buf935 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_21], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf202, primals_559, buf935, 65536, grid=grid(65536), stream=stream0)
        del buf202
        del primals_559
        buf936 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_20], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf198, primals_555, buf936, 65536, grid=grid(65536), stream=stream0)
        del buf198
        del primals_555
        buf937 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_19], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf175, primals_537, buf937, 65536, grid=grid(65536), stream=stream0)
        del buf175
        del primals_537
        buf938 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_18], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf171, primals_533, buf938, 65536, grid=grid(65536), stream=stream0)
        del buf171
        del primals_533
        buf939 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_17], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf167, primals_529, buf939, 65536, grid=grid(65536), stream=stream0)
        del buf167
        del primals_529
        buf940 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_16], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf163, primals_525, buf940, 65536, grid=grid(65536), stream=stream0)
        del buf163
        del primals_525
        buf941 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_15], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf140, primals_507, buf941, 65536, grid=grid(65536), stream=stream0)
        del buf140
        del primals_507
        buf942 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_14], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf136, primals_503, buf942, 65536, grid=grid(65536), stream=stream0)
        del buf136
        del primals_503
        buf943 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_13], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf132, primals_499, buf943, 65536, grid=grid(65536), stream=stream0)
        del buf132
        del primals_499
        buf944 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_12], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf128, primals_495, buf944, 65536, grid=grid(65536), stream=stream0)
        del buf128
        del primals_495
        buf945 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf105, primals_477, buf945, 65536, grid=grid(65536), stream=stream0)
        del buf105
        del primals_477
        buf946 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf101, primals_473, buf946, 65536, grid=grid(65536), stream=stream0)
        del buf101
        del primals_473
        buf947 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf97, primals_469, buf947, 65536, grid=grid(65536), stream=stream0)
        del buf97
        del primals_469
        buf948 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf93, primals_465, buf948, 65536, grid=grid(65536), stream=stream0)
        del buf93
        del primals_465
        buf949 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf70, primals_447, buf949, 65536, grid=grid(65536), stream=stream0)
        del buf70
        del primals_447
        buf950 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf66, primals_443, buf950, 65536, grid=grid(65536), stream=stream0)
        del buf66
        del primals_443
        buf951 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf62, primals_439, buf951, 65536, grid=grid(65536), stream=stream0)
        del buf62
        del primals_439
        buf952 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf58, primals_435, buf952, 65536, grid=grid(65536), stream=stream0)
        del buf58
        del primals_435
        buf953 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf34, primals_417, buf953, 65536, grid=grid(65536), stream=stream0)
        del buf34
        del primals_417
        buf954 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf30, primals_413, buf954, 65536, grid=grid(65536), stream=stream0)
        del buf30
        del primals_413
        buf955 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf26, primals_409, buf955, 65536, grid=grid(65536), stream=stream0)
        del buf26
        del primals_409
        buf956 = empty((1, 128, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [intermediate_output], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_19.run(buf22, primals_405, buf956, 65536, grid=grid(65536), stream=stream0)
        del buf22
        del primals_405
        return (buf957, buf847, buf848, primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1115, buf0, reinterpret_tensor(primals_1114, (1, 128), (512, 1), 0), buf1, buf3, buf4, buf5, buf6, buf7, buf11, buf12, buf13, buf16, buf17, buf18, buf15, buf19, buf20, reinterpret_tensor(buf21, (128, 128), (128, 1), 0), buf23, buf24, buf25, buf27, buf28, reinterpret_tensor(buf29, (128, 128), (128, 1), 0), buf31, buf32, buf33, buf35, buf36, buf37, buf39, buf40, buf41, buf42, buf43, buf47, buf48, buf49, buf52, buf53, buf54, buf51, buf55, buf56, reinterpret_tensor(buf57, (128, 128), (128, 1), 0), buf59, buf60, buf61, buf63, buf64, reinterpret_tensor(buf65, (128, 128), (128, 1), 0), buf67, buf68, buf69, buf71, buf72, buf73, buf74, reinterpret_tensor(buf75, (128, 512), (512, 1), 0), buf76, buf77, buf78, buf82, buf83, buf84, buf87, buf88, buf89, buf86, buf90, buf91, reinterpret_tensor(buf92, (128, 128), (128, 1), 0), buf94, buf95, buf96, buf98, buf99, reinterpret_tensor(buf100, (128, 128), (128, 1), 0), buf102, buf103, buf104, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf117, buf118, buf119, buf122, buf123, buf124, buf121, buf125, buf126, reinterpret_tensor(buf127, (128, 128), (128, 1), 0), buf129, buf130, buf131, buf133, buf134, reinterpret_tensor(buf135, (128, 128), (128, 1), 0), buf137, buf138, buf139, buf141, buf142, buf143, buf144, reinterpret_tensor(buf145, (128, 512), (512, 1), 0), buf146, buf147, buf148, buf152, buf153, buf154, buf157, buf158, buf159, buf156, buf160, buf161, reinterpret_tensor(buf162, (128, 128), (128, 1), 0), buf164, buf165, buf166, buf168, buf169, reinterpret_tensor(buf170, (128, 128), (128, 1), 0), buf172, buf173, buf174, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf187, buf188, buf189, buf192, buf193, buf194, buf191, buf195, buf196, reinterpret_tensor(buf197, (128, 128), (128, 1), 0), buf199, buf200, buf201, buf203, buf204, reinterpret_tensor(buf205, (128, 128), (128, 1), 0), buf207, buf208, buf209, buf211, buf212, buf213, buf214, reinterpret_tensor(buf215, (128, 512), (512, 1), 0), buf216, buf217, buf218, buf222, buf223, buf224, buf227, buf228, buf229, buf226, buf230, buf231, reinterpret_tensor(buf232, (128, 128), (128, 1), 0), buf234, buf235, buf236, buf238, buf239, reinterpret_tensor(buf240, (128, 128), (128, 1), 0), buf242, buf243, buf244, buf246, buf247, buf248, buf249, buf250, buf251, buf252, buf253, buf257, buf258, buf259, buf262, buf263, buf264, buf261, buf265, buf266, reinterpret_tensor(buf267, (128, 128), (128, 1), 0), buf269, buf270, buf271, buf273, buf274, reinterpret_tensor(buf275, (128, 128), (128, 1), 0), buf277, buf278, buf279, buf281, buf282, buf283, buf284, reinterpret_tensor(buf285, (128, 512), (512, 1), 0), buf286, buf287, buf288, buf292, buf293, buf294, buf297, buf298, buf299, buf296, buf300, buf301, reinterpret_tensor(buf302, (128, 128), (128, 1), 0), buf304, buf305, buf306, buf308, buf309, reinterpret_tensor(buf310, (128, 128), (128, 1), 0), buf312, buf313, buf314, buf316, buf317, buf318, buf319, buf320, buf321, buf322, buf323, buf327, buf328, buf329, buf332, buf333, buf334, buf331, buf335, buf336, reinterpret_tensor(buf337, (128, 128), (128, 1), 0), buf339, buf340, buf341, buf343, buf344, reinterpret_tensor(buf345, (128, 128), (128, 1), 0), buf347, buf348, buf349, buf351, buf352, buf353, buf354, reinterpret_tensor(buf355, (128, 512), (512, 1), 0), buf356, buf357, buf358, buf362, buf363, buf364, buf367, buf368, buf369, buf366, buf370, buf371, reinterpret_tensor(buf372, (128, 128), (128, 1), 0), buf374, buf375, buf376, buf378, buf379, reinterpret_tensor(buf380, (128, 128), (128, 1), 0), buf382, buf383, buf384, buf386, buf387, buf388, buf389, buf390, buf391, buf392, buf393, buf397, buf398, buf399, buf402, buf403, buf404, buf401, buf405, buf406, reinterpret_tensor(buf407, (128, 128), (128, 1), 0), buf409, buf410, buf411, buf413, buf414, reinterpret_tensor(buf415, (128, 128), (128, 1), 0), buf417, buf418, buf419, buf421, buf422, buf423, buf424, reinterpret_tensor(buf425, (128, 512), (512, 1), 0), buf426, buf427, buf428, buf432, buf433, buf434, buf437, buf438, buf439, buf436, buf440, buf441, reinterpret_tensor(buf442, (128, 128), (128, 1), 0), buf444, buf445, buf446, buf448, buf449, reinterpret_tensor(buf450, (128, 128), (128, 1), 0), buf452, buf453, buf454, buf456, buf457, buf458, buf459, buf460, buf461, buf462, buf463, buf467, buf468, buf469, buf472, buf473, buf474, buf471, buf475, buf476, reinterpret_tensor(buf477, (128, 128), (128, 1), 0), buf479, buf480, buf481, buf483, buf484, reinterpret_tensor(buf485, (128, 128), (128, 1), 0), buf487, buf488, buf489, buf491, buf492, buf493, buf494, reinterpret_tensor(buf495, (128, 512), (512, 1), 0), buf496, buf497, buf498, buf502, buf503, buf504, buf507, buf508, buf509, buf506, buf510, buf511, reinterpret_tensor(buf512, (128, 128), (128, 1), 0), buf514, buf515, buf516, buf518, buf519, reinterpret_tensor(buf520, (128, 128), (128, 1), 0), buf522, buf523, buf524, buf526, buf527, buf528, buf529, buf530, buf531, buf532, buf533, buf537, buf538, buf539, buf542, buf543, buf544, buf541, buf545, buf546, reinterpret_tensor(buf547, (128, 128), (128, 1), 0), buf549, buf550, buf551, buf553, buf554, reinterpret_tensor(buf555, (128, 128), (128, 1), 0), buf557, buf558, buf559, buf561, buf562, buf563, buf564, reinterpret_tensor(buf565, (128, 512), (512, 1), 0), buf566, buf567, buf568, buf572, buf573, buf574, buf577, buf578, buf579, buf576, buf580, buf581, reinterpret_tensor(buf582, (128, 128), (128, 1), 0), buf584, buf585, buf586, buf588, buf589, reinterpret_tensor(buf590, (128, 128), (128, 1), 0), buf592, buf593, buf594, buf596, buf597, buf598, buf599, buf600, buf601, buf602, buf603, buf607, buf608, buf609, buf612, buf613, buf614, buf611, buf615, buf616, reinterpret_tensor(buf617, (128, 128), (128, 1), 0), buf619, buf620, buf621, buf623, buf624, reinterpret_tensor(buf625, (128, 128), (128, 1), 0), buf627, buf628, buf629, buf631, buf632, buf633, buf634, reinterpret_tensor(buf635, (128, 512), (512, 1), 0), buf636, buf637, buf638, buf642, buf643, buf644, buf647, buf648, buf649, buf646, buf650, buf651, reinterpret_tensor(buf652, (128, 128), (128, 1), 0), buf654, buf655, buf656, buf658, buf659, reinterpret_tensor(buf660, (128, 128), (128, 1), 0), buf662, buf663, buf664, buf666, buf667, buf668, buf669, buf670, buf671, buf672, buf673, buf677, buf678, buf679, buf682, buf683, buf684, buf681, buf685, buf686, reinterpret_tensor(buf687, (128, 128), (128, 1), 0), buf689, buf690, buf691, buf693, buf694, reinterpret_tensor(buf695, (128, 128), (128, 1), 0), buf697, buf698, buf699, buf701, buf702, buf703, buf704, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), buf706, buf707, buf708, buf712, buf713, buf714, buf717, buf718, buf719, buf716, buf720, buf721, reinterpret_tensor(buf722, (128, 128), (128, 1), 0), buf724, buf725, buf726, buf728, buf729, reinterpret_tensor(buf730, (128, 128), (128, 1), 0), buf732, buf733, buf734, buf736, buf737, buf738, buf739, buf740, buf741, buf742, buf743, buf747, buf748, buf749, buf752, buf753, buf754, buf751, buf755, buf756, reinterpret_tensor(buf757, (128, 128), (128, 1), 0), buf759, buf760, buf761, buf763, buf764, reinterpret_tensor(buf765, (128, 128), (128, 1), 0), buf767, buf768, buf769, buf771, buf772, buf773, buf774, reinterpret_tensor(buf775, (128, 512), (512, 1), 0), buf776, buf777, buf778, buf782, buf783, buf784, buf787, buf788, buf789, buf786, buf790, buf791, reinterpret_tensor(buf792, (128, 128), (128, 1), 0), buf794, buf795, buf796, buf798, buf799, reinterpret_tensor(buf800, (128, 128), (128, 1), 0), buf802, buf803, buf804, buf806, buf807, buf808, buf809, buf810, buf811, buf812, buf813, buf817, buf818, buf819, buf822, buf823, buf824, buf821, buf825, buf826, reinterpret_tensor(buf827, (128, 128), (128, 1), 0), buf829, buf830, buf831, buf833, buf834, reinterpret_tensor(buf835, (128, 128), (128, 1), 0), buf837, buf838, buf839, buf841, buf842, buf843, buf844, buf845, buf851, buf852, buf855, buf856, buf857, buf858, buf859, buf860, reinterpret_tensor(primals_1112, (2, 512), (512, 1), 0), reinterpret_tensor(primals_1110, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1108, (128, 512), (512, 1), 0), buf861, reinterpret_tensor(primals_1106, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1104, (128, 512), (512, 1), 0), buf862, reinterpret_tensor(primals_1102, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1100, (128, 512), (512, 1), 0), buf863, reinterpret_tensor(primals_1098, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1096, (128, 512), (512, 1), 0), buf864, reinterpret_tensor(primals_1094, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1092, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1090, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1088, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1086, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1084, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1082, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1080, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1078, (128, 512), (512, 1), 0), buf865, reinterpret_tensor(primals_1076, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1074, (128, 512), (512, 1), 0), buf866, reinterpret_tensor(primals_1072, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1070, (128, 512), (512, 1), 0), buf867, reinterpret_tensor(primals_1068, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1066, (128, 512), (512, 1), 0), buf868, reinterpret_tensor(primals_1064, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1062, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1060, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1058, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1056, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1054, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1052, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1050, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1048, (128, 512), (512, 1), 0), buf869, reinterpret_tensor(primals_1046, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1044, (128, 512), (512, 1), 0), buf870, reinterpret_tensor(primals_1042, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1040, (128, 512), (512, 1), 0), buf871, reinterpret_tensor(primals_1038, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1036, (128, 512), (512, 1), 0), buf872, reinterpret_tensor(primals_1034, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1032, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1030, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1028, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1026, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1024, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1022, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1020, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1018, (128, 512), (512, 1), 0), buf873, reinterpret_tensor(primals_1016, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1014, (128, 512), (512, 1), 0), buf874, reinterpret_tensor(primals_1012, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1010, (128, 512), (512, 1), 0), buf875, reinterpret_tensor(primals_1008, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1006, (128, 512), (512, 1), 0), buf876, reinterpret_tensor(primals_1004, (512, 128), (128, 1), 0), reinterpret_tensor(primals_1002, (128, 128), (128, 1), 0), reinterpret_tensor(primals_1000, (128, 512), (512, 1), 0), reinterpret_tensor(primals_998, (128, 128), (128, 1), 0), reinterpret_tensor(primals_996, (128, 128), (128, 1), 0), reinterpret_tensor(primals_994, (128, 512), (512, 1), 0), reinterpret_tensor(primals_992, (128, 512), (512, 1), 0), reinterpret_tensor(primals_990, (512, 128), (128, 1), 0), reinterpret_tensor(primals_988, (128, 512), (512, 1), 0), buf877, reinterpret_tensor(primals_986, (512, 128), (128, 1), 0), reinterpret_tensor(primals_984, (128, 512), (512, 1), 0), buf878, reinterpret_tensor(primals_982, (512, 128), (128, 1), 0), reinterpret_tensor(primals_980, (128, 512), (512, 1), 0), buf879, reinterpret_tensor(primals_978, (512, 128), (128, 1), 0), reinterpret_tensor(primals_976, (128, 512), (512, 1), 0), buf880, reinterpret_tensor(primals_974, (512, 128), (128, 1), 0), reinterpret_tensor(primals_972, (128, 128), (128, 1), 0), reinterpret_tensor(primals_970, (128, 512), (512, 1), 0), reinterpret_tensor(primals_968, (128, 128), (128, 1), 0), reinterpret_tensor(primals_966, (128, 128), (128, 1), 0), reinterpret_tensor(primals_964, (128, 512), (512, 1), 0), reinterpret_tensor(primals_962, (128, 512), (512, 1), 0), reinterpret_tensor(primals_960, (512, 128), (128, 1), 0), reinterpret_tensor(primals_958, (128, 512), (512, 1), 0), buf881, reinterpret_tensor(primals_956, (512, 128), (128, 1), 0), reinterpret_tensor(primals_954, (128, 512), (512, 1), 0), buf882, reinterpret_tensor(primals_952, (512, 128), (128, 1), 0), reinterpret_tensor(primals_950, (128, 512), (512, 1), 0), buf883, reinterpret_tensor(primals_948, (512, 128), (128, 1), 0), reinterpret_tensor(primals_946, (128, 512), (512, 1), 0), buf884, reinterpret_tensor(primals_944, (512, 128), (128, 1), 0), reinterpret_tensor(primals_942, (128, 128), (128, 1), 0), reinterpret_tensor(primals_940, (128, 512), (512, 1), 0), reinterpret_tensor(primals_938, (128, 128), (128, 1), 0), reinterpret_tensor(primals_936, (128, 128), (128, 1), 0), reinterpret_tensor(primals_934, (128, 512), (512, 1), 0), reinterpret_tensor(primals_932, (128, 512), (512, 1), 0), reinterpret_tensor(primals_930, (512, 128), (128, 1), 0), reinterpret_tensor(primals_928, (128, 512), (512, 1), 0), buf885, reinterpret_tensor(primals_926, (512, 128), (128, 1), 0), reinterpret_tensor(primals_924, (128, 512), (512, 1), 0), buf886, reinterpret_tensor(primals_922, (512, 128), (128, 1), 0), reinterpret_tensor(primals_920, (128, 512), (512, 1), 0), buf887, reinterpret_tensor(primals_918, (512, 128), (128, 1), 0), reinterpret_tensor(primals_916, (128, 512), (512, 1), 0), buf888, reinterpret_tensor(primals_914, (512, 128), (128, 1), 0), reinterpret_tensor(primals_912, (128, 128), (128, 1), 0), reinterpret_tensor(primals_910, (128, 512), (512, 1), 0), reinterpret_tensor(primals_908, (128, 128), (128, 1), 0), reinterpret_tensor(primals_906, (128, 128), (128, 1), 0), reinterpret_tensor(primals_904, (128, 512), (512, 1), 0), reinterpret_tensor(primals_902, (128, 512), (512, 1), 0), reinterpret_tensor(primals_900, (512, 128), (128, 1), 0), reinterpret_tensor(primals_898, (128, 512), (512, 1), 0), buf889, reinterpret_tensor(primals_896, (512, 128), (128, 1), 0), reinterpret_tensor(primals_894, (128, 512), (512, 1), 0), buf890, reinterpret_tensor(primals_892, (512, 128), (128, 1), 0), reinterpret_tensor(primals_890, (128, 512), (512, 1), 0), buf891, reinterpret_tensor(primals_888, (512, 128), (128, 1), 0), reinterpret_tensor(primals_886, (128, 512), (512, 1), 0), buf892, reinterpret_tensor(primals_884, (512, 128), (128, 1), 0), reinterpret_tensor(primals_882, (128, 128), (128, 1), 0), reinterpret_tensor(primals_880, (128, 512), (512, 1), 0), reinterpret_tensor(primals_878, (128, 128), (128, 1), 0), reinterpret_tensor(primals_876, (128, 128), (128, 1), 0), reinterpret_tensor(primals_874, (128, 512), (512, 1), 0), reinterpret_tensor(primals_872, (128, 512), (512, 1), 0), reinterpret_tensor(primals_870, (512, 128), (128, 1), 0), reinterpret_tensor(primals_868, (128, 512), (512, 1), 0), buf893, reinterpret_tensor(primals_866, (512, 128), (128, 1), 0), reinterpret_tensor(primals_864, (128, 512), (512, 1), 0), buf894, reinterpret_tensor(primals_862, (512, 128), (128, 1), 0), reinterpret_tensor(primals_860, (128, 512), (512, 1), 0), buf895, reinterpret_tensor(primals_858, (512, 128), (128, 1), 0), reinterpret_tensor(primals_856, (128, 512), (512, 1), 0), buf896, reinterpret_tensor(primals_854, (512, 128), (128, 1), 0), reinterpret_tensor(primals_852, (128, 128), (128, 1), 0), reinterpret_tensor(primals_850, (128, 512), (512, 1), 0), reinterpret_tensor(primals_848, (128, 128), (128, 1), 0), reinterpret_tensor(primals_846, (128, 128), (128, 1), 0), reinterpret_tensor(primals_844, (128, 512), (512, 1), 0), reinterpret_tensor(primals_842, (128, 512), (512, 1), 0), reinterpret_tensor(primals_840, (512, 128), (128, 1), 0), reinterpret_tensor(primals_838, (128, 512), (512, 1), 0), buf897, reinterpret_tensor(primals_836, (512, 128), (128, 1), 0), reinterpret_tensor(primals_834, (128, 512), (512, 1), 0), buf898, reinterpret_tensor(primals_832, (512, 128), (128, 1), 0), reinterpret_tensor(primals_830, (128, 512), (512, 1), 0), buf899, reinterpret_tensor(primals_828, (512, 128), (128, 1), 0), reinterpret_tensor(primals_826, (128, 512), (512, 1), 0), buf900, reinterpret_tensor(primals_824, (512, 128), (128, 1), 0), reinterpret_tensor(primals_822, (128, 128), (128, 1), 0), reinterpret_tensor(primals_820, (128, 512), (512, 1), 0), reinterpret_tensor(primals_818, (128, 128), (128, 1), 0), reinterpret_tensor(primals_816, (128, 128), (128, 1), 0), reinterpret_tensor(primals_814, (128, 512), (512, 1), 0), reinterpret_tensor(primals_812, (128, 512), (512, 1), 0), reinterpret_tensor(primals_810, (512, 128), (128, 1), 0), reinterpret_tensor(primals_808, (128, 512), (512, 1), 0), buf901, reinterpret_tensor(primals_806, (512, 128), (128, 1), 0), reinterpret_tensor(primals_804, (128, 512), (512, 1), 0), buf902, reinterpret_tensor(primals_802, (512, 128), (128, 1), 0), reinterpret_tensor(primals_800, (128, 512), (512, 1), 0), buf903, reinterpret_tensor(primals_798, (512, 128), (128, 1), 0), reinterpret_tensor(primals_796, (128, 512), (512, 1), 0), buf904, reinterpret_tensor(primals_794, (512, 128), (128, 1), 0), reinterpret_tensor(primals_792, (128, 128), (128, 1), 0), reinterpret_tensor(primals_790, (128, 512), (512, 1), 0), reinterpret_tensor(primals_788, (128, 128), (128, 1), 0), reinterpret_tensor(primals_786, (128, 128), (128, 1), 0), reinterpret_tensor(primals_784, (128, 512), (512, 1), 0), reinterpret_tensor(primals_782, (128, 512), (512, 1), 0), reinterpret_tensor(primals_780, (512, 128), (128, 1), 0), reinterpret_tensor(primals_778, (128, 512), (512, 1), 0), buf905, reinterpret_tensor(primals_776, (512, 128), (128, 1), 0), reinterpret_tensor(primals_774, (128, 512), (512, 1), 0), buf906, reinterpret_tensor(primals_772, (512, 128), (128, 1), 0), reinterpret_tensor(primals_770, (128, 512), (512, 1), 0), buf907, reinterpret_tensor(primals_768, (512, 128), (128, 1), 0), reinterpret_tensor(primals_766, (128, 512), (512, 1), 0), buf908, reinterpret_tensor(primals_764, (512, 128), (128, 1), 0), reinterpret_tensor(primals_762, (128, 128), (128, 1), 0), reinterpret_tensor(primals_760, (128, 512), (512, 1), 0), reinterpret_tensor(primals_758, (128, 128), (128, 1), 0), reinterpret_tensor(primals_756, (128, 128), (128, 1), 0), reinterpret_tensor(primals_754, (128, 512), (512, 1), 0), reinterpret_tensor(primals_752, (128, 512), (512, 1), 0), reinterpret_tensor(primals_750, (512, 128), (128, 1), 0), reinterpret_tensor(primals_748, (128, 512), (512, 1), 0), buf909, reinterpret_tensor(primals_746, (512, 128), (128, 1), 0), reinterpret_tensor(primals_744, (128, 512), (512, 1), 0), buf910, reinterpret_tensor(primals_742, (512, 128), (128, 1), 0), reinterpret_tensor(primals_740, (128, 512), (512, 1), 0), buf911, reinterpret_tensor(primals_738, (512, 128), (128, 1), 0), reinterpret_tensor(primals_736, (128, 512), (512, 1), 0), buf912, reinterpret_tensor(primals_734, (512, 128), (128, 1), 0), reinterpret_tensor(primals_732, (128, 128), (128, 1), 0), reinterpret_tensor(primals_730, (128, 512), (512, 1), 0), reinterpret_tensor(primals_728, (128, 128), (128, 1), 0), reinterpret_tensor(primals_726, (128, 128), (128, 1), 0), reinterpret_tensor(primals_724, (128, 512), (512, 1), 0), reinterpret_tensor(primals_722, (128, 512), (512, 1), 0), reinterpret_tensor(primals_720, (512, 128), (128, 1), 0), reinterpret_tensor(primals_718, (128, 512), (512, 1), 0), buf913, reinterpret_tensor(primals_716, (512, 128), (128, 1), 0), reinterpret_tensor(primals_714, (128, 512), (512, 1), 0), buf914, reinterpret_tensor(primals_712, (512, 128), (128, 1), 0), reinterpret_tensor(primals_710, (128, 512), (512, 1), 0), buf915, reinterpret_tensor(primals_708, (512, 128), (128, 1), 0), reinterpret_tensor(primals_706, (128, 512), (512, 1), 0), buf916, reinterpret_tensor(primals_704, (512, 128), (128, 1), 0), reinterpret_tensor(primals_702, (128, 128), (128, 1), 0), reinterpret_tensor(primals_700, (128, 512), (512, 1), 0), reinterpret_tensor(primals_698, (128, 128), (128, 1), 0), reinterpret_tensor(primals_696, (128, 128), (128, 1), 0), reinterpret_tensor(primals_694, (128, 512), (512, 1), 0), reinterpret_tensor(primals_692, (128, 512), (512, 1), 0), reinterpret_tensor(primals_690, (512, 128), (128, 1), 0), reinterpret_tensor(primals_688, (128, 512), (512, 1), 0), buf917, reinterpret_tensor(primals_686, (512, 128), (128, 1), 0), reinterpret_tensor(primals_684, (128, 512), (512, 1), 0), buf918, reinterpret_tensor(primals_682, (512, 128), (128, 1), 0), reinterpret_tensor(primals_680, (128, 512), (512, 1), 0), buf919, reinterpret_tensor(primals_678, (512, 128), (128, 1), 0), reinterpret_tensor(primals_676, (128, 512), (512, 1), 0), buf920, reinterpret_tensor(primals_674, (512, 128), (128, 1), 0), reinterpret_tensor(primals_672, (128, 128), (128, 1), 0), reinterpret_tensor(primals_670, (128, 512), (512, 1), 0), reinterpret_tensor(primals_668, (128, 128), (128, 1), 0), reinterpret_tensor(primals_666, (128, 128), (128, 1), 0), reinterpret_tensor(primals_664, (128, 512), (512, 1), 0), reinterpret_tensor(primals_662, (128, 512), (512, 1), 0), reinterpret_tensor(primals_660, (512, 128), (128, 1), 0), reinterpret_tensor(primals_658, (128, 512), (512, 1), 0), buf921, reinterpret_tensor(primals_656, (512, 128), (128, 1), 0), reinterpret_tensor(primals_654, (128, 512), (512, 1), 0), buf922, reinterpret_tensor(primals_652, (512, 128), (128, 1), 0), reinterpret_tensor(primals_650, (128, 512), (512, 1), 0), buf923, reinterpret_tensor(primals_648, (512, 128), (128, 1), 0), reinterpret_tensor(primals_646, (128, 512), (512, 1), 0), buf924, reinterpret_tensor(primals_644, (512, 128), (128, 1), 0), reinterpret_tensor(primals_642, (128, 128), (128, 1), 0), reinterpret_tensor(primals_640, (128, 512), (512, 1), 0), reinterpret_tensor(primals_638, (128, 128), (128, 1), 0), reinterpret_tensor(primals_636, (128, 128), (128, 1), 0), reinterpret_tensor(primals_634, (128, 512), (512, 1), 0), reinterpret_tensor(primals_632, (128, 512), (512, 1), 0), reinterpret_tensor(primals_630, (512, 128), (128, 1), 0), reinterpret_tensor(primals_628, (128, 512), (512, 1), 0), buf925, reinterpret_tensor(primals_626, (512, 128), (128, 1), 0), reinterpret_tensor(primals_624, (128, 512), (512, 1), 0), buf926, reinterpret_tensor(primals_622, (512, 128), (128, 1), 0), reinterpret_tensor(primals_620, (128, 512), (512, 1), 0), buf927, reinterpret_tensor(primals_618, (512, 128), (128, 1), 0), reinterpret_tensor(primals_616, (128, 512), (512, 1), 0), buf928, reinterpret_tensor(primals_614, (512, 128), (128, 1), 0), reinterpret_tensor(primals_612, (128, 128), (128, 1), 0), reinterpret_tensor(primals_610, (128, 512), (512, 1), 0), reinterpret_tensor(primals_608, (128, 128), (128, 1), 0), reinterpret_tensor(primals_606, (128, 128), (128, 1), 0), reinterpret_tensor(primals_604, (128, 512), (512, 1), 0), reinterpret_tensor(primals_602, (128, 512), (512, 1), 0), reinterpret_tensor(primals_600, (512, 128), (128, 1), 0), reinterpret_tensor(primals_598, (128, 512), (512, 1), 0), buf929, reinterpret_tensor(primals_596, (512, 128), (128, 1), 0), reinterpret_tensor(primals_594, (128, 512), (512, 1), 0), buf930, reinterpret_tensor(primals_592, (512, 128), (128, 1), 0), reinterpret_tensor(primals_590, (128, 512), (512, 1), 0), buf931, reinterpret_tensor(primals_588, (512, 128), (128, 1), 0), reinterpret_tensor(primals_586, (128, 512), (512, 1), 0), buf932, reinterpret_tensor(primals_584, (512, 128), (128, 1), 0), reinterpret_tensor(primals_582, (128, 128), (128, 1), 0), reinterpret_tensor(primals_580, (128, 512), (512, 1), 0), reinterpret_tensor(primals_578, (128, 128), (128, 1), 0), reinterpret_tensor(primals_576, (128, 128), (128, 1), 0), reinterpret_tensor(primals_574, (128, 512), (512, 1), 0), reinterpret_tensor(primals_572, (128, 512), (512, 1), 0), reinterpret_tensor(primals_570, (512, 128), (128, 1), 0), reinterpret_tensor(primals_568, (128, 512), (512, 1), 0), buf933, reinterpret_tensor(primals_566, (512, 128), (128, 1), 0), reinterpret_tensor(primals_564, (128, 512), (512, 1), 0), buf934, reinterpret_tensor(primals_562, (512, 128), (128, 1), 0), reinterpret_tensor(primals_560, (128, 512), (512, 1), 0), buf935, reinterpret_tensor(primals_558, (512, 128), (128, 1), 0), reinterpret_tensor(primals_556, (128, 512), (512, 1), 0), buf936, reinterpret_tensor(primals_554, (512, 128), (128, 1), 0), reinterpret_tensor(primals_552, (128, 128), (128, 1), 0), reinterpret_tensor(primals_550, (128, 512), (512, 1), 0), reinterpret_tensor(primals_548, (128, 128), (128, 1), 0), reinterpret_tensor(primals_546, (128, 128), (128, 1), 0), reinterpret_tensor(primals_544, (128, 512), (512, 1), 0), reinterpret_tensor(primals_542, (128, 512), (512, 1), 0), reinterpret_tensor(primals_540, (512, 128), (128, 1), 0), reinterpret_tensor(primals_538, (128, 512), (512, 1), 0), buf937, reinterpret_tensor(primals_536, (512, 128), (128, 1), 0), reinterpret_tensor(primals_534, (128, 512), (512, 1), 0), buf938, reinterpret_tensor(primals_532, (512, 128), (128, 1), 0), reinterpret_tensor(primals_530, (128, 512), (512, 1), 0), buf939, reinterpret_tensor(primals_528, (512, 128), (128, 1), 0), reinterpret_tensor(primals_526, (128, 512), (512, 1), 0), buf940, reinterpret_tensor(primals_524, (512, 128), (128, 1), 0), reinterpret_tensor(primals_522, (128, 128), (128, 1), 0), reinterpret_tensor(primals_520, (128, 512), (512, 1), 0), reinterpret_tensor(primals_518, (128, 128), (128, 1), 0), reinterpret_tensor(primals_516, (128, 128), (128, 1), 0), reinterpret_tensor(primals_514, (128, 512), (512, 1), 0), reinterpret_tensor(primals_512, (128, 512), (512, 1), 0), reinterpret_tensor(primals_510, (512, 128), (128, 1), 0), reinterpret_tensor(primals_508, (128, 512), (512, 1), 0), buf941, reinterpret_tensor(primals_506, (512, 128), (128, 1), 0), reinterpret_tensor(primals_504, (128, 512), (512, 1), 0), buf942, reinterpret_tensor(primals_502, (512, 128), (128, 1), 0), reinterpret_tensor(primals_500, (128, 512), (512, 1), 0), buf943, reinterpret_tensor(primals_498, (512, 128), (128, 1), 0), reinterpret_tensor(primals_496, (128, 512), (512, 1), 0), buf944, reinterpret_tensor(primals_494, (512, 128), (128, 1), 0), reinterpret_tensor(primals_492, (128, 128), (128, 1), 0), reinterpret_tensor(primals_490, (128, 512), (512, 1), 0), reinterpret_tensor(primals_488, (128, 128), (128, 1), 0), reinterpret_tensor(primals_486, (128, 128), (128, 1), 0), reinterpret_tensor(primals_484, (128, 512), (512, 1), 0), reinterpret_tensor(primals_482, (128, 512), (512, 1), 0), reinterpret_tensor(primals_480, (512, 128), (128, 1), 0), reinterpret_tensor(primals_478, (128, 512), (512, 1), 0), buf945, reinterpret_tensor(primals_476, (512, 128), (128, 1), 0), reinterpret_tensor(primals_474, (128, 512), (512, 1), 0), buf946, reinterpret_tensor(primals_472, (512, 128), (128, 1), 0), reinterpret_tensor(primals_470, (128, 512), (512, 1), 0), buf947, reinterpret_tensor(primals_468, (512, 128), (128, 1), 0), reinterpret_tensor(primals_466, (128, 512), (512, 1), 0), buf948, reinterpret_tensor(primals_464, (512, 128), (128, 1), 0), reinterpret_tensor(primals_462, (128, 128), (128, 1), 0), reinterpret_tensor(primals_460, (128, 512), (512, 1), 0), reinterpret_tensor(primals_458, (128, 128), (128, 1), 0), reinterpret_tensor(primals_456, (128, 128), (128, 1), 0), reinterpret_tensor(primals_454, (128, 512), (512, 1), 0), reinterpret_tensor(primals_452, (128, 512), (512, 1), 0), reinterpret_tensor(primals_450, (512, 128), (128, 1), 0), reinterpret_tensor(primals_448, (128, 512), (512, 1), 0), buf949, reinterpret_tensor(primals_446, (512, 128), (128, 1), 0), reinterpret_tensor(primals_444, (128, 512), (512, 1), 0), buf950, reinterpret_tensor(primals_442, (512, 128), (128, 1), 0), reinterpret_tensor(primals_440, (128, 512), (512, 1), 0), buf951, reinterpret_tensor(primals_438, (512, 128), (128, 1), 0), reinterpret_tensor(primals_436, (128, 512), (512, 1), 0), buf952, reinterpret_tensor(primals_434, (512, 128), (128, 1), 0), reinterpret_tensor(primals_432, (128, 128), (128, 1), 0), reinterpret_tensor(primals_430, (128, 512), (512, 1), 0), reinterpret_tensor(primals_428, (128, 128), (128, 1), 0), reinterpret_tensor(primals_426, (128, 128), (128, 1), 0), reinterpret_tensor(primals_424, (128, 512), (512, 1), 0), reinterpret_tensor(primals_422, (128, 512), (512, 1), 0), reinterpret_tensor(primals_420, (512, 128), (128, 1), 0), reinterpret_tensor(primals_418, (128, 512), (512, 1), 0), buf953, reinterpret_tensor(primals_416, (512, 128), (128, 1), 0), reinterpret_tensor(primals_414, (128, 512), (512, 1), 0), buf954, reinterpret_tensor(primals_412, (512, 128), (128, 1), 0), reinterpret_tensor(primals_410, (128, 512), (512, 1), 0), buf955, reinterpret_tensor(primals_408, (512, 128), (128, 1), 0), reinterpret_tensor(primals_406, (128, 512), (512, 1), 0), buf956, reinterpret_tensor(primals_404, (512, 128), (128, 1), 0), reinterpret_tensor(primals_402, (128, 128), (128, 1), 0), reinterpret_tensor(primals_400, (128, 512), (512, 1), 0), reinterpret_tensor(primals_398, (128, 128), (128, 1), 0), reinterpret_tensor(primals_396, (128, 128), (128, 1), 0), reinterpret_tensor(primals_394, (128, 512), (512, 1), 0), reinterpret_tensor(primals_392, (128, 512), (512, 1), 0), reinterpret_tensor(primals_388, (512, 384), (384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1104 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1106 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1108 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1110 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1112 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_1113 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1114 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_1115 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_1116 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_1117 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForQuestionAnswering', benchmark_compiled_module)
