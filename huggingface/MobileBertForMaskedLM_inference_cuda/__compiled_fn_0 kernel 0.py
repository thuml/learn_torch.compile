
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgd27lerldxir4nf7wz2qny5gl2wi6muz7rzu7fq56ttmmsvrsza.py
# Source Nodes: [cat_3], Original ATen: [aten.cat]
# cat_3 => cat
triton_poi_fused_cat_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': []},
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/bo/cbormbqagahggjr57lzup3h4hq5zcpqvnz2n46ubfbg3uds3t5w4.py
# Source Nodes: [add, embeddings, embeddings_1, mul_1, position_embeddings, token_type_embeddings, token_type_ids], Original ATen: [aten.add, aten.embedding, aten.mul, aten.zeros]
# add => add
# embeddings => add_1
# embeddings_1 => add_2
# mul_1 => mul_1
# position_embeddings => embedding_1
# token_type_embeddings => embedding_2
# token_type_ids => full_default
triton_poi_fused_add_embedding_mul_zeros_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_zeros_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3chumbghavrqgpdgbzy35hp7vwdaiva5ueknaym3izxakbxovpc.py
# Source Nodes: [key_tensor, mul_3], Original ATen: [aten.add, aten.mul]
# key_tensor => add_4
# mul_3 => mul_3
triton_poi_fused_add_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxa5n7spm7ud3hnxweouj3unllmtxieku3tqig2rvtegfmybwii.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wm/cwme3vtajdsprhnlicujtush2j2j5heghaahhdxyad54tizjx5kd.py
# Source Nodes: [add_6, attention_output, layer_input_4, mul_2, mul_4], Original ATen: [aten.add, aten.mul]
# add_6 => add_6
# attention_output => add_7
# layer_input_4 => add_3
# mul_2 => mul_2
# mul_4 => mul_4
triton_poi_fused_add_mul_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvo3bhlgunglvzx2of4apem6epl3c5pjjak7sahgnlls6nvpaz2.py
# Source Nodes: [intermediate_output], Original ATen: [aten.relu]
# intermediate_output => relu
triton_poi_fused_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5joqbakwjczhsiwfrbctxe634kj7rxlzckoi25atpluyyhutl4.py
# Source Nodes: [add_8, attention_output_1, mul_5], Original ATen: [aten.add, aten.mul]
# add_8 => add_8
# attention_output_1 => add_9
# mul_5 => mul_5
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4p3w2fgss6pf5tjgj2knn3hpvy42zsjuhtl5balgbjjk4euyv5.py
# Source Nodes: [add_16, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul]
# add_16 => add_16
# mul_9 => mul_9
# value_tensor_1 => add_17
triton_poi_fused_add_mul_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqprp4ble3vko77m4vlix4mjshoauyxx7p2oyzcwlkzan5g7kek.py
# Source Nodes: [add_44, layer_output_9, mul_24], Original ATen: [aten.add, aten.mul]
# add_44 => add_44
# layer_output_9 => add_45
# mul_24 => mul_24
triton_poi_fused_add_mul_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmc3as6vgokd5tcvjeqjolizojb5ikdefblsxrghijjtlza7zqy.py
# Source Nodes: [add_46, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul]
# add_46 => add_46
# mul_25 => mul_25
# value_tensor_3 => add_47
triton_poi_fused_add_mul_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3hujxgtraacfz2jbz3db5xgj3hdgvfqzdykqpqmown2qudzakc.py
# Source Nodes: [hidden_states_217, hidden_states_219], Original ATen: [aten.native_layer_norm, aten.relu]
# hidden_states_217 => relu_96
# hidden_states_219 => add_363, add_364, mul_194, mul_195, rsqrt, sub_25, var_mean
triton_per_fused_native_layer_norm_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_relu_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 512, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 512.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-12
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/con3xk2irrlreze65q5qswehioqtuzqmzavaxiytlce6jgeoxq2a.py
# Source Nodes: [cat_2], Original ATen: [aten.cat]
# cat_2 => cat_1
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 30522
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (128*x1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 512, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-3906816) + x1 + (30522*y0)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x1 + (30522*y0)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehvmfbfvrbme4k74raenujkelvi47nag4uiwsc2wgu3osmx5p7a.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_24
triton_red_fused__log_softmax_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r2 + (7631*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, float("-inf"), tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7k2nch5ul5cnraipne4xdswtmnkcobmemuc4frjurmai4z3yss5.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_24
triton_per_fused__log_softmax_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdbmxbyftrmq6gixf3afbkt7obmav7mnjkrrtw24vh5eqwstgfp.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_24, sub_26, sum_25
triton_red_fused__log_softmax_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r2 + (7631*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 - tmp6
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvalhuigelrg2mwl3tos7fxklmqrrdbm7gnbxs5qzbrmqywriop.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_24, sub_26, sum_25
triton_per_fused__log_softmax_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/ckn4lr3jxdkdve3buklvintejluhcamkyrqjqvz7ri2oz7pb5lbn.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_48, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 30522
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30522), "index out of bounds: 0 <= tmp7 < 30522")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (30522*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (tmp7), None, eviction_policy='evict_last')
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = tl.log(tmp13)
    tmp15 = tmp12 - tmp14
    tmp16 = -tmp15
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2.to(tl.int64)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp22 / tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctozklciadc7f52ojuzjcouvmfuwhtocrei5tqupp2eojz4czu3j.py
# Source Nodes: [prediction_scores], Original ATen: [aten.view]
# prediction_scores => view_967
triton_poi_fused_view_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30522
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (128, ), (1, ))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, ), (1, ))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, ), (1, ))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, ), (1, ))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (128, ), (1, ))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (128, ), (1, ))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (128, ), (1, ))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (128, ), (1, ))
    assert_size_stride(arg213_1, (128, ), (1, ))
    assert_size_stride(arg214_1, (128, ), (1, ))
    assert_size_stride(arg215_1, (128, ), (1, ))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (128, ), (1, ))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (128, ), (1, ))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (128, ), (1, ))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (128, ), (1, ))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, ), (1, ))
    assert_size_stride(arg242_1, (128, ), (1, ))
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, ), (1, ))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, ), (1, ))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, ), (1, ))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (128, ), (1, ))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (128, ), (1, ))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (128, ), (1, ))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (128, ), (1, ))
    assert_size_stride(arg276_1, (128, ), (1, ))
    assert_size_stride(arg277_1, (128, ), (1, ))
    assert_size_stride(arg278_1, (128, ), (1, ))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (128, ), (1, ))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (128, ), (1, ))
    assert_size_stride(arg284_1, (128, ), (1, ))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (128, ), (1, ))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (128, ), (1, ))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (128, ), (1, ))
    assert_size_stride(arg294_1, (128, ), (1, ))
    assert_size_stride(arg295_1, (128, ), (1, ))
    assert_size_stride(arg296_1, (128, ), (1, ))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (128, ), (1, ))
    assert_size_stride(arg300_1, (128, ), (1, ))
    assert_size_stride(arg301_1, (128, ), (1, ))
    assert_size_stride(arg302_1, (128, ), (1, ))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (128, ), (1, ))
    assert_size_stride(arg307_1, (128, ), (1, ))
    assert_size_stride(arg308_1, (128, ), (1, ))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, ), (1, ))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (128, ), (1, ))
    assert_size_stride(arg314_1, (128, ), (1, ))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (128, ), (1, ))
    assert_size_stride(arg317_1, (128, ), (1, ))
    assert_size_stride(arg318_1, (128, ), (1, ))
    assert_size_stride(arg319_1, (128, ), (1, ))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (128, ), (1, ))
    assert_size_stride(arg324_1, (128, ), (1, ))
    assert_size_stride(arg325_1, (128, ), (1, ))
    assert_size_stride(arg326_1, (128, ), (1, ))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (128, ), (1, ))
    assert_size_stride(arg330_1, (128, ), (1, ))
    assert_size_stride(arg331_1, (128, ), (1, ))
    assert_size_stride(arg332_1, (128, ), (1, ))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (128, ), (1, ))
    assert_size_stride(arg342_1, (128, ), (1, ))
    assert_size_stride(arg343_1, (128, ), (1, ))
    assert_size_stride(arg344_1, (128, ), (1, ))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (128, ), (1, ))
    assert_size_stride(arg348_1, (128, ), (1, ))
    assert_size_stride(arg349_1, (128, ), (1, ))
    assert_size_stride(arg350_1, (128, ), (1, ))
    assert_size_stride(arg351_1, (128, ), (1, ))
    assert_size_stride(arg352_1, (512, ), (1, ))
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (128, ), (1, ))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (128, ), (1, ))
    assert_size_stride(arg360_1, (128, ), (1, ))
    assert_size_stride(arg361_1, (128, ), (1, ))
    assert_size_stride(arg362_1, (128, ), (1, ))
    assert_size_stride(arg363_1, (128, ), (1, ))
    assert_size_stride(arg364_1, (128, ), (1, ))
    assert_size_stride(arg365_1, (128, ), (1, ))
    assert_size_stride(arg366_1, (128, ), (1, ))
    assert_size_stride(arg367_1, (128, ), (1, ))
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (512, ), (1, ))
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (128, ), (1, ))
    assert_size_stride(arg373_1, (128, ), (1, ))
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (128, ), (1, ))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (128, ), (1, ))
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (128, ), (1, ))
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (512, ), (1, ))
    assert_size_stride(arg386_1, (30522, 128), (128, 1))
    assert_size_stride(arg387_1, (384, 30522), (30522, 1))
    assert_size_stride(arg388_1, (30522, ), (1, ))
    assert_size_stride(arg389_1, (30522, 128), (128, 1))
    assert_size_stride(arg390_1, (512, 384), (384, 1))
    assert_size_stride(arg391_1, (512, ), (1, ))
    assert_size_stride(arg392_1, (512, 512), (512, 1))
    assert_size_stride(arg393_1, (2, 512), (512, 1))
    assert_size_stride(arg394_1, (128, 512), (512, 1))
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (128, 512), (512, 1))
    assert_size_stride(arg397_1, (128, ), (1, ))
    assert_size_stride(arg398_1, (128, 128), (128, 1))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (128, 128), (128, 1))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (128, 512), (512, 1))
    assert_size_stride(arg403_1, (128, ), (1, ))
    assert_size_stride(arg404_1, (128, 128), (128, 1))
    assert_size_stride(arg405_1, (128, ), (1, ))
    assert_size_stride(arg406_1, (512, 128), (128, 1))
    assert_size_stride(arg407_1, (512, ), (1, ))
    assert_size_stride(arg408_1, (128, 512), (512, 1))
    assert_size_stride(arg409_1, (128, ), (1, ))
    assert_size_stride(arg410_1, (512, 128), (128, 1))
    assert_size_stride(arg411_1, (512, ), (1, ))
    assert_size_stride(arg412_1, (128, 512), (512, 1))
    assert_size_stride(arg413_1, (128, ), (1, ))
    assert_size_stride(arg414_1, (512, 128), (128, 1))
    assert_size_stride(arg415_1, (512, ), (1, ))
    assert_size_stride(arg416_1, (128, 512), (512, 1))
    assert_size_stride(arg417_1, (128, ), (1, ))
    assert_size_stride(arg418_1, (512, 128), (128, 1))
    assert_size_stride(arg419_1, (512, ), (1, ))
    assert_size_stride(arg420_1, (128, 512), (512, 1))
    assert_size_stride(arg421_1, (128, ), (1, ))
    assert_size_stride(arg422_1, (512, 128), (128, 1))
    assert_size_stride(arg423_1, (512, ), (1, ))
    assert_size_stride(arg424_1, (128, 512), (512, 1))
    assert_size_stride(arg425_1, (128, ), (1, ))
    assert_size_stride(arg426_1, (128, 512), (512, 1))
    assert_size_stride(arg427_1, (128, ), (1, ))
    assert_size_stride(arg428_1, (128, 128), (128, 1))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (128, 128), (128, 1))
    assert_size_stride(arg431_1, (128, ), (1, ))
    assert_size_stride(arg432_1, (128, 512), (512, 1))
    assert_size_stride(arg433_1, (128, ), (1, ))
    assert_size_stride(arg434_1, (128, 128), (128, 1))
    assert_size_stride(arg435_1, (128, ), (1, ))
    assert_size_stride(arg436_1, (512, 128), (128, 1))
    assert_size_stride(arg437_1, (512, ), (1, ))
    assert_size_stride(arg438_1, (128, 512), (512, 1))
    assert_size_stride(arg439_1, (128, ), (1, ))
    assert_size_stride(arg440_1, (512, 128), (128, 1))
    assert_size_stride(arg441_1, (512, ), (1, ))
    assert_size_stride(arg442_1, (128, 512), (512, 1))
    assert_size_stride(arg443_1, (128, ), (1, ))
    assert_size_stride(arg444_1, (512, 128), (128, 1))
    assert_size_stride(arg445_1, (512, ), (1, ))
    assert_size_stride(arg446_1, (128, 512), (512, 1))
    assert_size_stride(arg447_1, (128, ), (1, ))
    assert_size_stride(arg448_1, (512, 128), (128, 1))
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (128, 512), (512, 1))
    assert_size_stride(arg451_1, (128, ), (1, ))
    assert_size_stride(arg452_1, (512, 128), (128, 1))
    assert_size_stride(arg453_1, (512, ), (1, ))
    assert_size_stride(arg454_1, (128, 512), (512, 1))
    assert_size_stride(arg455_1, (128, ), (1, ))
    assert_size_stride(arg456_1, (128, 512), (512, 1))
    assert_size_stride(arg457_1, (128, ), (1, ))
    assert_size_stride(arg458_1, (128, 128), (128, 1))
    assert_size_stride(arg459_1, (128, ), (1, ))
    assert_size_stride(arg460_1, (128, 128), (128, 1))
    assert_size_stride(arg461_1, (128, ), (1, ))
    assert_size_stride(arg462_1, (128, 512), (512, 1))
    assert_size_stride(arg463_1, (128, ), (1, ))
    assert_size_stride(arg464_1, (128, 128), (128, 1))
    assert_size_stride(arg465_1, (128, ), (1, ))
    assert_size_stride(arg466_1, (512, 128), (128, 1))
    assert_size_stride(arg467_1, (512, ), (1, ))
    assert_size_stride(arg468_1, (128, 512), (512, 1))
    assert_size_stride(arg469_1, (128, ), (1, ))
    assert_size_stride(arg470_1, (512, 128), (128, 1))
    assert_size_stride(arg471_1, (512, ), (1, ))
    assert_size_stride(arg472_1, (128, 512), (512, 1))
    assert_size_stride(arg473_1, (128, ), (1, ))
    assert_size_stride(arg474_1, (512, 128), (128, 1))
    assert_size_stride(arg475_1, (512, ), (1, ))
    assert_size_stride(arg476_1, (128, 512), (512, 1))
    assert_size_stride(arg477_1, (128, ), (1, ))
    assert_size_stride(arg478_1, (512, 128), (128, 1))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (128, 512), (512, 1))
    assert_size_stride(arg481_1, (128, ), (1, ))
    assert_size_stride(arg482_1, (512, 128), (128, 1))
    assert_size_stride(arg483_1, (512, ), (1, ))
    assert_size_stride(arg484_1, (128, 512), (512, 1))
    assert_size_stride(arg485_1, (128, ), (1, ))
    assert_size_stride(arg486_1, (128, 512), (512, 1))
    assert_size_stride(arg487_1, (128, ), (1, ))
    assert_size_stride(arg488_1, (128, 128), (128, 1))
    assert_size_stride(arg489_1, (128, ), (1, ))
    assert_size_stride(arg490_1, (128, 128), (128, 1))
    assert_size_stride(arg491_1, (128, ), (1, ))
    assert_size_stride(arg492_1, (128, 512), (512, 1))
    assert_size_stride(arg493_1, (128, ), (1, ))
    assert_size_stride(arg494_1, (128, 128), (128, 1))
    assert_size_stride(arg495_1, (128, ), (1, ))
    assert_size_stride(arg496_1, (512, 128), (128, 1))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (128, 512), (512, 1))
    assert_size_stride(arg499_1, (128, ), (1, ))
    assert_size_stride(arg500_1, (512, 128), (128, 1))
    assert_size_stride(arg501_1, (512, ), (1, ))
    assert_size_stride(arg502_1, (128, 512), (512, 1))
    assert_size_stride(arg503_1, (128, ), (1, ))
    assert_size_stride(arg504_1, (512, 128), (128, 1))
    assert_size_stride(arg505_1, (512, ), (1, ))
    assert_size_stride(arg506_1, (128, 512), (512, 1))
    assert_size_stride(arg507_1, (128, ), (1, ))
    assert_size_stride(arg508_1, (512, 128), (128, 1))
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (128, 512), (512, 1))
    assert_size_stride(arg511_1, (128, ), (1, ))
    assert_size_stride(arg512_1, (512, 128), (128, 1))
    assert_size_stride(arg513_1, (512, ), (1, ))
    assert_size_stride(arg514_1, (128, 512), (512, 1))
    assert_size_stride(arg515_1, (128, ), (1, ))
    assert_size_stride(arg516_1, (128, 512), (512, 1))
    assert_size_stride(arg517_1, (128, ), (1, ))
    assert_size_stride(arg518_1, (128, 128), (128, 1))
    assert_size_stride(arg519_1, (128, ), (1, ))
    assert_size_stride(arg520_1, (128, 128), (128, 1))
    assert_size_stride(arg521_1, (128, ), (1, ))
    assert_size_stride(arg522_1, (128, 512), (512, 1))
    assert_size_stride(arg523_1, (128, ), (1, ))
    assert_size_stride(arg524_1, (128, 128), (128, 1))
    assert_size_stride(arg525_1, (128, ), (1, ))
    assert_size_stride(arg526_1, (512, 128), (128, 1))
    assert_size_stride(arg527_1, (512, ), (1, ))
    assert_size_stride(arg528_1, (128, 512), (512, 1))
    assert_size_stride(arg529_1, (128, ), (1, ))
    assert_size_stride(arg530_1, (512, 128), (128, 1))
    assert_size_stride(arg531_1, (512, ), (1, ))
    assert_size_stride(arg532_1, (128, 512), (512, 1))
    assert_size_stride(arg533_1, (128, ), (1, ))
    assert_size_stride(arg534_1, (512, 128), (128, 1))
    assert_size_stride(arg535_1, (512, ), (1, ))
    assert_size_stride(arg536_1, (128, 512), (512, 1))
    assert_size_stride(arg537_1, (128, ), (1, ))
    assert_size_stride(arg538_1, (512, 128), (128, 1))
    assert_size_stride(arg539_1, (512, ), (1, ))
    assert_size_stride(arg540_1, (128, 512), (512, 1))
    assert_size_stride(arg541_1, (128, ), (1, ))
    assert_size_stride(arg542_1, (512, 128), (128, 1))
    assert_size_stride(arg543_1, (512, ), (1, ))
    assert_size_stride(arg544_1, (128, 512), (512, 1))
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (128, 512), (512, 1))
    assert_size_stride(arg547_1, (128, ), (1, ))
    assert_size_stride(arg548_1, (128, 128), (128, 1))
    assert_size_stride(arg549_1, (128, ), (1, ))
    assert_size_stride(arg550_1, (128, 128), (128, 1))
    assert_size_stride(arg551_1, (128, ), (1, ))
    assert_size_stride(arg552_1, (128, 512), (512, 1))
    assert_size_stride(arg553_1, (128, ), (1, ))
    assert_size_stride(arg554_1, (128, 128), (128, 1))
    assert_size_stride(arg555_1, (128, ), (1, ))
    assert_size_stride(arg556_1, (512, 128), (128, 1))
    assert_size_stride(arg557_1, (512, ), (1, ))
    assert_size_stride(arg558_1, (128, 512), (512, 1))
    assert_size_stride(arg559_1, (128, ), (1, ))
    assert_size_stride(arg560_1, (512, 128), (128, 1))
    assert_size_stride(arg561_1, (512, ), (1, ))
    assert_size_stride(arg562_1, (128, 512), (512, 1))
    assert_size_stride(arg563_1, (128, ), (1, ))
    assert_size_stride(arg564_1, (512, 128), (128, 1))
    assert_size_stride(arg565_1, (512, ), (1, ))
    assert_size_stride(arg566_1, (128, 512), (512, 1))
    assert_size_stride(arg567_1, (128, ), (1, ))
    assert_size_stride(arg568_1, (512, 128), (128, 1))
    assert_size_stride(arg569_1, (512, ), (1, ))
    assert_size_stride(arg570_1, (128, 512), (512, 1))
    assert_size_stride(arg571_1, (128, ), (1, ))
    assert_size_stride(arg572_1, (512, 128), (128, 1))
    assert_size_stride(arg573_1, (512, ), (1, ))
    assert_size_stride(arg574_1, (128, 512), (512, 1))
    assert_size_stride(arg575_1, (128, ), (1, ))
    assert_size_stride(arg576_1, (128, 512), (512, 1))
    assert_size_stride(arg577_1, (128, ), (1, ))
    assert_size_stride(arg578_1, (128, 128), (128, 1))
    assert_size_stride(arg579_1, (128, ), (1, ))
    assert_size_stride(arg580_1, (128, 128), (128, 1))
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (128, 512), (512, 1))
    assert_size_stride(arg583_1, (128, ), (1, ))
    assert_size_stride(arg584_1, (128, 128), (128, 1))
    assert_size_stride(arg585_1, (128, ), (1, ))
    assert_size_stride(arg586_1, (512, 128), (128, 1))
    assert_size_stride(arg587_1, (512, ), (1, ))
    assert_size_stride(arg588_1, (128, 512), (512, 1))
    assert_size_stride(arg589_1, (128, ), (1, ))
    assert_size_stride(arg590_1, (512, 128), (128, 1))
    assert_size_stride(arg591_1, (512, ), (1, ))
    assert_size_stride(arg592_1, (128, 512), (512, 1))
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (512, 128), (128, 1))
    assert_size_stride(arg595_1, (512, ), (1, ))
    assert_size_stride(arg596_1, (128, 512), (512, 1))
    assert_size_stride(arg597_1, (128, ), (1, ))
    assert_size_stride(arg598_1, (512, 128), (128, 1))
    assert_size_stride(arg599_1, (512, ), (1, ))
    assert_size_stride(arg600_1, (128, 512), (512, 1))
    assert_size_stride(arg601_1, (128, ), (1, ))
    assert_size_stride(arg602_1, (512, 128), (128, 1))
    assert_size_stride(arg603_1, (512, ), (1, ))
    assert_size_stride(arg604_1, (128, 512), (512, 1))
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (128, 512), (512, 1))
    assert_size_stride(arg607_1, (128, ), (1, ))
    assert_size_stride(arg608_1, (128, 128), (128, 1))
    assert_size_stride(arg609_1, (128, ), (1, ))
    assert_size_stride(arg610_1, (128, 128), (128, 1))
    assert_size_stride(arg611_1, (128, ), (1, ))
    assert_size_stride(arg612_1, (128, 512), (512, 1))
    assert_size_stride(arg613_1, (128, ), (1, ))
    assert_size_stride(arg614_1, (128, 128), (128, 1))
    assert_size_stride(arg615_1, (128, ), (1, ))
    assert_size_stride(arg616_1, (512, 128), (128, 1))
    assert_size_stride(arg617_1, (512, ), (1, ))
    assert_size_stride(arg618_1, (128, 512), (512, 1))
    assert_size_stride(arg619_1, (128, ), (1, ))
    assert_size_stride(arg620_1, (512, 128), (128, 1))
    assert_size_stride(arg621_1, (512, ), (1, ))
    assert_size_stride(arg622_1, (128, 512), (512, 1))
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (512, 128), (128, 1))
    assert_size_stride(arg625_1, (512, ), (1, ))
    assert_size_stride(arg626_1, (128, 512), (512, 1))
    assert_size_stride(arg627_1, (128, ), (1, ))
    assert_size_stride(arg628_1, (512, 128), (128, 1))
    assert_size_stride(arg629_1, (512, ), (1, ))
    assert_size_stride(arg630_1, (128, 512), (512, 1))
    assert_size_stride(arg631_1, (128, ), (1, ))
    assert_size_stride(arg632_1, (512, 128), (128, 1))
    assert_size_stride(arg633_1, (512, ), (1, ))
    assert_size_stride(arg634_1, (128, 512), (512, 1))
    assert_size_stride(arg635_1, (128, ), (1, ))
    assert_size_stride(arg636_1, (128, 512), (512, 1))
    assert_size_stride(arg637_1, (128, ), (1, ))
    assert_size_stride(arg638_1, (128, 128), (128, 1))
    assert_size_stride(arg639_1, (128, ), (1, ))
    assert_size_stride(arg640_1, (128, 128), (128, 1))
    assert_size_stride(arg641_1, (128, ), (1, ))
    assert_size_stride(arg642_1, (128, 512), (512, 1))
    assert_size_stride(arg643_1, (128, ), (1, ))
    assert_size_stride(arg644_1, (128, 128), (128, 1))
    assert_size_stride(arg645_1, (128, ), (1, ))
    assert_size_stride(arg646_1, (512, 128), (128, 1))
    assert_size_stride(arg647_1, (512, ), (1, ))
    assert_size_stride(arg648_1, (128, 512), (512, 1))
    assert_size_stride(arg649_1, (128, ), (1, ))
    assert_size_stride(arg650_1, (512, 128), (128, 1))
    assert_size_stride(arg651_1, (512, ), (1, ))
    assert_size_stride(arg652_1, (128, 512), (512, 1))
    assert_size_stride(arg653_1, (128, ), (1, ))
    assert_size_stride(arg654_1, (512, 128), (128, 1))
    assert_size_stride(arg655_1, (512, ), (1, ))
    assert_size_stride(arg656_1, (128, 512), (512, 1))
    assert_size_stride(arg657_1, (128, ), (1, ))
    assert_size_stride(arg658_1, (512, 128), (128, 1))
    assert_size_stride(arg659_1, (512, ), (1, ))
    assert_size_stride(arg660_1, (128, 512), (512, 1))
    assert_size_stride(arg661_1, (128, ), (1, ))
    assert_size_stride(arg662_1, (512, 128), (128, 1))
    assert_size_stride(arg663_1, (512, ), (1, ))
    assert_size_stride(arg664_1, (128, 512), (512, 1))
    assert_size_stride(arg665_1, (128, ), (1, ))
    assert_size_stride(arg666_1, (128, 512), (512, 1))
    assert_size_stride(arg667_1, (128, ), (1, ))
    assert_size_stride(arg668_1, (128, 128), (128, 1))
    assert_size_stride(arg669_1, (128, ), (1, ))
    assert_size_stride(arg670_1, (128, 128), (128, 1))
    assert_size_stride(arg671_1, (128, ), (1, ))
    assert_size_stride(arg672_1, (128, 512), (512, 1))
    assert_size_stride(arg673_1, (128, ), (1, ))
    assert_size_stride(arg674_1, (128, 128), (128, 1))
    assert_size_stride(arg675_1, (128, ), (1, ))
    assert_size_stride(arg676_1, (512, 128), (128, 1))
    assert_size_stride(arg677_1, (512, ), (1, ))
    assert_size_stride(arg678_1, (128, 512), (512, 1))
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (512, 128), (128, 1))
    assert_size_stride(arg681_1, (512, ), (1, ))
    assert_size_stride(arg682_1, (128, 512), (512, 1))
    assert_size_stride(arg683_1, (128, ), (1, ))
    assert_size_stride(arg684_1, (512, 128), (128, 1))
    assert_size_stride(arg685_1, (512, ), (1, ))
    assert_size_stride(arg686_1, (128, 512), (512, 1))
    assert_size_stride(arg687_1, (128, ), (1, ))
    assert_size_stride(arg688_1, (512, 128), (128, 1))
    assert_size_stride(arg689_1, (512, ), (1, ))
    assert_size_stride(arg690_1, (128, 512), (512, 1))
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (512, 128), (128, 1))
    assert_size_stride(arg693_1, (512, ), (1, ))
    assert_size_stride(arg694_1, (128, 512), (512, 1))
    assert_size_stride(arg695_1, (128, ), (1, ))
    assert_size_stride(arg696_1, (128, 512), (512, 1))
    assert_size_stride(arg697_1, (128, ), (1, ))
    assert_size_stride(arg698_1, (128, 128), (128, 1))
    assert_size_stride(arg699_1, (128, ), (1, ))
    assert_size_stride(arg700_1, (128, 128), (128, 1))
    assert_size_stride(arg701_1, (128, ), (1, ))
    assert_size_stride(arg702_1, (128, 512), (512, 1))
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (128, 128), (128, 1))
    assert_size_stride(arg705_1, (128, ), (1, ))
    assert_size_stride(arg706_1, (512, 128), (128, 1))
    assert_size_stride(arg707_1, (512, ), (1, ))
    assert_size_stride(arg708_1, (128, 512), (512, 1))
    assert_size_stride(arg709_1, (128, ), (1, ))
    assert_size_stride(arg710_1, (512, 128), (128, 1))
    assert_size_stride(arg711_1, (512, ), (1, ))
    assert_size_stride(arg712_1, (128, 512), (512, 1))
    assert_size_stride(arg713_1, (128, ), (1, ))
    assert_size_stride(arg714_1, (512, 128), (128, 1))
    assert_size_stride(arg715_1, (512, ), (1, ))
    assert_size_stride(arg716_1, (128, 512), (512, 1))
    assert_size_stride(arg717_1, (128, ), (1, ))
    assert_size_stride(arg718_1, (512, 128), (128, 1))
    assert_size_stride(arg719_1, (512, ), (1, ))
    assert_size_stride(arg720_1, (128, 512), (512, 1))
    assert_size_stride(arg721_1, (128, ), (1, ))
    assert_size_stride(arg722_1, (512, 128), (128, 1))
    assert_size_stride(arg723_1, (512, ), (1, ))
    assert_size_stride(arg724_1, (128, 512), (512, 1))
    assert_size_stride(arg725_1, (128, ), (1, ))
    assert_size_stride(arg726_1, (128, 512), (512, 1))
    assert_size_stride(arg727_1, (128, ), (1, ))
    assert_size_stride(arg728_1, (128, 128), (128, 1))
    assert_size_stride(arg729_1, (128, ), (1, ))
    assert_size_stride(arg730_1, (128, 128), (128, 1))
    assert_size_stride(arg731_1, (128, ), (1, ))
    assert_size_stride(arg732_1, (128, 512), (512, 1))
    assert_size_stride(arg733_1, (128, ), (1, ))
    assert_size_stride(arg734_1, (128, 128), (128, 1))
    assert_size_stride(arg735_1, (128, ), (1, ))
    assert_size_stride(arg736_1, (512, 128), (128, 1))
    assert_size_stride(arg737_1, (512, ), (1, ))
    assert_size_stride(arg738_1, (128, 512), (512, 1))
    assert_size_stride(arg739_1, (128, ), (1, ))
    assert_size_stride(arg740_1, (512, 128), (128, 1))
    assert_size_stride(arg741_1, (512, ), (1, ))
    assert_size_stride(arg742_1, (128, 512), (512, 1))
    assert_size_stride(arg743_1, (128, ), (1, ))
    assert_size_stride(arg744_1, (512, 128), (128, 1))
    assert_size_stride(arg745_1, (512, ), (1, ))
    assert_size_stride(arg746_1, (128, 512), (512, 1))
    assert_size_stride(arg747_1, (128, ), (1, ))
    assert_size_stride(arg748_1, (512, 128), (128, 1))
    assert_size_stride(arg749_1, (512, ), (1, ))
    assert_size_stride(arg750_1, (128, 512), (512, 1))
    assert_size_stride(arg751_1, (128, ), (1, ))
    assert_size_stride(arg752_1, (512, 128), (128, 1))
    assert_size_stride(arg753_1, (512, ), (1, ))
    assert_size_stride(arg754_1, (128, 512), (512, 1))
    assert_size_stride(arg755_1, (128, ), (1, ))
    assert_size_stride(arg756_1, (128, 512), (512, 1))
    assert_size_stride(arg757_1, (128, ), (1, ))
    assert_size_stride(arg758_1, (128, 128), (128, 1))
    assert_size_stride(arg759_1, (128, ), (1, ))
    assert_size_stride(arg760_1, (128, 128), (128, 1))
    assert_size_stride(arg761_1, (128, ), (1, ))
    assert_size_stride(arg762_1, (128, 512), (512, 1))
    assert_size_stride(arg763_1, (128, ), (1, ))
    assert_size_stride(arg764_1, (128, 128), (128, 1))
    assert_size_stride(arg765_1, (128, ), (1, ))
    assert_size_stride(arg766_1, (512, 128), (128, 1))
    assert_size_stride(arg767_1, (512, ), (1, ))
    assert_size_stride(arg768_1, (128, 512), (512, 1))
    assert_size_stride(arg769_1, (128, ), (1, ))
    assert_size_stride(arg770_1, (512, 128), (128, 1))
    assert_size_stride(arg771_1, (512, ), (1, ))
    assert_size_stride(arg772_1, (128, 512), (512, 1))
    assert_size_stride(arg773_1, (128, ), (1, ))
    assert_size_stride(arg774_1, (512, 128), (128, 1))
    assert_size_stride(arg775_1, (512, ), (1, ))
    assert_size_stride(arg776_1, (128, 512), (512, 1))
    assert_size_stride(arg777_1, (128, ), (1, ))
    assert_size_stride(arg778_1, (512, 128), (128, 1))
    assert_size_stride(arg779_1, (512, ), (1, ))
    assert_size_stride(arg780_1, (128, 512), (512, 1))
    assert_size_stride(arg781_1, (128, ), (1, ))
    assert_size_stride(arg782_1, (512, 128), (128, 1))
    assert_size_stride(arg783_1, (512, ), (1, ))
    assert_size_stride(arg784_1, (128, 512), (512, 1))
    assert_size_stride(arg785_1, (128, ), (1, ))
    assert_size_stride(arg786_1, (128, 512), (512, 1))
    assert_size_stride(arg787_1, (128, ), (1, ))
    assert_size_stride(arg788_1, (128, 128), (128, 1))
    assert_size_stride(arg789_1, (128, ), (1, ))
    assert_size_stride(arg790_1, (128, 128), (128, 1))
    assert_size_stride(arg791_1, (128, ), (1, ))
    assert_size_stride(arg792_1, (128, 512), (512, 1))
    assert_size_stride(arg793_1, (128, ), (1, ))
    assert_size_stride(arg794_1, (128, 128), (128, 1))
    assert_size_stride(arg795_1, (128, ), (1, ))
    assert_size_stride(arg796_1, (512, 128), (128, 1))
    assert_size_stride(arg797_1, (512, ), (1, ))
    assert_size_stride(arg798_1, (128, 512), (512, 1))
    assert_size_stride(arg799_1, (128, ), (1, ))
    assert_size_stride(arg800_1, (512, 128), (128, 1))
    assert_size_stride(arg801_1, (512, ), (1, ))
    assert_size_stride(arg802_1, (128, 512), (512, 1))
    assert_size_stride(arg803_1, (128, ), (1, ))
    assert_size_stride(arg804_1, (512, 128), (128, 1))
    assert_size_stride(arg805_1, (512, ), (1, ))
    assert_size_stride(arg806_1, (128, 512), (512, 1))
    assert_size_stride(arg807_1, (128, ), (1, ))
    assert_size_stride(arg808_1, (512, 128), (128, 1))
    assert_size_stride(arg809_1, (512, ), (1, ))
    assert_size_stride(arg810_1, (128, 512), (512, 1))
    assert_size_stride(arg811_1, (128, ), (1, ))
    assert_size_stride(arg812_1, (512, 128), (128, 1))
    assert_size_stride(arg813_1, (512, ), (1, ))
    assert_size_stride(arg814_1, (128, 512), (512, 1))
    assert_size_stride(arg815_1, (128, ), (1, ))
    assert_size_stride(arg816_1, (128, 512), (512, 1))
    assert_size_stride(arg817_1, (128, ), (1, ))
    assert_size_stride(arg818_1, (128, 128), (128, 1))
    assert_size_stride(arg819_1, (128, ), (1, ))
    assert_size_stride(arg820_1, (128, 128), (128, 1))
    assert_size_stride(arg821_1, (128, ), (1, ))
    assert_size_stride(arg822_1, (128, 512), (512, 1))
    assert_size_stride(arg823_1, (128, ), (1, ))
    assert_size_stride(arg824_1, (128, 128), (128, 1))
    assert_size_stride(arg825_1, (128, ), (1, ))
    assert_size_stride(arg826_1, (512, 128), (128, 1))
    assert_size_stride(arg827_1, (512, ), (1, ))
    assert_size_stride(arg828_1, (128, 512), (512, 1))
    assert_size_stride(arg829_1, (128, ), (1, ))
    assert_size_stride(arg830_1, (512, 128), (128, 1))
    assert_size_stride(arg831_1, (512, ), (1, ))
    assert_size_stride(arg832_1, (128, 512), (512, 1))
    assert_size_stride(arg833_1, (128, ), (1, ))
    assert_size_stride(arg834_1, (512, 128), (128, 1))
    assert_size_stride(arg835_1, (512, ), (1, ))
    assert_size_stride(arg836_1, (128, 512), (512, 1))
    assert_size_stride(arg837_1, (128, ), (1, ))
    assert_size_stride(arg838_1, (512, 128), (128, 1))
    assert_size_stride(arg839_1, (512, ), (1, ))
    assert_size_stride(arg840_1, (128, 512), (512, 1))
    assert_size_stride(arg841_1, (128, ), (1, ))
    assert_size_stride(arg842_1, (512, 128), (128, 1))
    assert_size_stride(arg843_1, (512, ), (1, ))
    assert_size_stride(arg844_1, (128, 512), (512, 1))
    assert_size_stride(arg845_1, (128, ), (1, ))
    assert_size_stride(arg846_1, (128, 512), (512, 1))
    assert_size_stride(arg847_1, (128, ), (1, ))
    assert_size_stride(arg848_1, (128, 128), (128, 1))
    assert_size_stride(arg849_1, (128, ), (1, ))
    assert_size_stride(arg850_1, (128, 128), (128, 1))
    assert_size_stride(arg851_1, (128, ), (1, ))
    assert_size_stride(arg852_1, (128, 512), (512, 1))
    assert_size_stride(arg853_1, (128, ), (1, ))
    assert_size_stride(arg854_1, (128, 128), (128, 1))
    assert_size_stride(arg855_1, (128, ), (1, ))
    assert_size_stride(arg856_1, (512, 128), (128, 1))
    assert_size_stride(arg857_1, (512, ), (1, ))
    assert_size_stride(arg858_1, (128, 512), (512, 1))
    assert_size_stride(arg859_1, (128, ), (1, ))
    assert_size_stride(arg860_1, (512, 128), (128, 1))
    assert_size_stride(arg861_1, (512, ), (1, ))
    assert_size_stride(arg862_1, (128, 512), (512, 1))
    assert_size_stride(arg863_1, (128, ), (1, ))
    assert_size_stride(arg864_1, (512, 128), (128, 1))
    assert_size_stride(arg865_1, (512, ), (1, ))
    assert_size_stride(arg866_1, (128, 512), (512, 1))
    assert_size_stride(arg867_1, (128, ), (1, ))
    assert_size_stride(arg868_1, (512, 128), (128, 1))
    assert_size_stride(arg869_1, (512, ), (1, ))
    assert_size_stride(arg870_1, (128, 512), (512, 1))
    assert_size_stride(arg871_1, (128, ), (1, ))
    assert_size_stride(arg872_1, (512, 128), (128, 1))
    assert_size_stride(arg873_1, (512, ), (1, ))
    assert_size_stride(arg874_1, (128, 512), (512, 1))
    assert_size_stride(arg875_1, (128, ), (1, ))
    assert_size_stride(arg876_1, (128, 512), (512, 1))
    assert_size_stride(arg877_1, (128, ), (1, ))
    assert_size_stride(arg878_1, (128, 128), (128, 1))
    assert_size_stride(arg879_1, (128, ), (1, ))
    assert_size_stride(arg880_1, (128, 128), (128, 1))
    assert_size_stride(arg881_1, (128, ), (1, ))
    assert_size_stride(arg882_1, (128, 512), (512, 1))
    assert_size_stride(arg883_1, (128, ), (1, ))
    assert_size_stride(arg884_1, (128, 128), (128, 1))
    assert_size_stride(arg885_1, (128, ), (1, ))
    assert_size_stride(arg886_1, (512, 128), (128, 1))
    assert_size_stride(arg887_1, (512, ), (1, ))
    assert_size_stride(arg888_1, (128, 512), (512, 1))
    assert_size_stride(arg889_1, (128, ), (1, ))
    assert_size_stride(arg890_1, (512, 128), (128, 1))
    assert_size_stride(arg891_1, (512, ), (1, ))
    assert_size_stride(arg892_1, (128, 512), (512, 1))
    assert_size_stride(arg893_1, (128, ), (1, ))
    assert_size_stride(arg894_1, (512, 128), (128, 1))
    assert_size_stride(arg895_1, (512, ), (1, ))
    assert_size_stride(arg896_1, (128, 512), (512, 1))
    assert_size_stride(arg897_1, (128, ), (1, ))
    assert_size_stride(arg898_1, (512, 128), (128, 1))
    assert_size_stride(arg899_1, (512, ), (1, ))
    assert_size_stride(arg900_1, (128, 512), (512, 1))
    assert_size_stride(arg901_1, (128, ), (1, ))
    assert_size_stride(arg902_1, (512, 128), (128, 1))
    assert_size_stride(arg903_1, (512, ), (1, ))
    assert_size_stride(arg904_1, (128, 512), (512, 1))
    assert_size_stride(arg905_1, (128, ), (1, ))
    assert_size_stride(arg906_1, (128, 512), (512, 1))
    assert_size_stride(arg907_1, (128, ), (1, ))
    assert_size_stride(arg908_1, (128, 128), (128, 1))
    assert_size_stride(arg909_1, (128, ), (1, ))
    assert_size_stride(arg910_1, (128, 128), (128, 1))
    assert_size_stride(arg911_1, (128, ), (1, ))
    assert_size_stride(arg912_1, (128, 512), (512, 1))
    assert_size_stride(arg913_1, (128, ), (1, ))
    assert_size_stride(arg914_1, (128, 128), (128, 1))
    assert_size_stride(arg915_1, (128, ), (1, ))
    assert_size_stride(arg916_1, (512, 128), (128, 1))
    assert_size_stride(arg917_1, (512, ), (1, ))
    assert_size_stride(arg918_1, (128, 512), (512, 1))
    assert_size_stride(arg919_1, (128, ), (1, ))
    assert_size_stride(arg920_1, (512, 128), (128, 1))
    assert_size_stride(arg921_1, (512, ), (1, ))
    assert_size_stride(arg922_1, (128, 512), (512, 1))
    assert_size_stride(arg923_1, (128, ), (1, ))
    assert_size_stride(arg924_1, (512, 128), (128, 1))
    assert_size_stride(arg925_1, (512, ), (1, ))
    assert_size_stride(arg926_1, (128, 512), (512, 1))
    assert_size_stride(arg927_1, (128, ), (1, ))
    assert_size_stride(arg928_1, (512, 128), (128, 1))
    assert_size_stride(arg929_1, (512, ), (1, ))
    assert_size_stride(arg930_1, (128, 512), (512, 1))
    assert_size_stride(arg931_1, (128, ), (1, ))
    assert_size_stride(arg932_1, (512, 128), (128, 1))
    assert_size_stride(arg933_1, (512, ), (1, ))
    assert_size_stride(arg934_1, (128, 512), (512, 1))
    assert_size_stride(arg935_1, (128, ), (1, ))
    assert_size_stride(arg936_1, (128, 512), (512, 1))
    assert_size_stride(arg937_1, (128, ), (1, ))
    assert_size_stride(arg938_1, (128, 128), (128, 1))
    assert_size_stride(arg939_1, (128, ), (1, ))
    assert_size_stride(arg940_1, (128, 128), (128, 1))
    assert_size_stride(arg941_1, (128, ), (1, ))
    assert_size_stride(arg942_1, (128, 512), (512, 1))
    assert_size_stride(arg943_1, (128, ), (1, ))
    assert_size_stride(arg944_1, (128, 128), (128, 1))
    assert_size_stride(arg945_1, (128, ), (1, ))
    assert_size_stride(arg946_1, (512, 128), (128, 1))
    assert_size_stride(arg947_1, (512, ), (1, ))
    assert_size_stride(arg948_1, (128, 512), (512, 1))
    assert_size_stride(arg949_1, (128, ), (1, ))
    assert_size_stride(arg950_1, (512, 128), (128, 1))
    assert_size_stride(arg951_1, (512, ), (1, ))
    assert_size_stride(arg952_1, (128, 512), (512, 1))
    assert_size_stride(arg953_1, (128, ), (1, ))
    assert_size_stride(arg954_1, (512, 128), (128, 1))
    assert_size_stride(arg955_1, (512, ), (1, ))
    assert_size_stride(arg956_1, (128, 512), (512, 1))
    assert_size_stride(arg957_1, (128, ), (1, ))
    assert_size_stride(arg958_1, (512, 128), (128, 1))
    assert_size_stride(arg959_1, (512, ), (1, ))
    assert_size_stride(arg960_1, (128, 512), (512, 1))
    assert_size_stride(arg961_1, (128, ), (1, ))
    assert_size_stride(arg962_1, (512, 128), (128, 1))
    assert_size_stride(arg963_1, (512, ), (1, ))
    assert_size_stride(arg964_1, (128, 512), (512, 1))
    assert_size_stride(arg965_1, (128, ), (1, ))
    assert_size_stride(arg966_1, (128, 512), (512, 1))
    assert_size_stride(arg967_1, (128, ), (1, ))
    assert_size_stride(arg968_1, (128, 128), (128, 1))
    assert_size_stride(arg969_1, (128, ), (1, ))
    assert_size_stride(arg970_1, (128, 128), (128, 1))
    assert_size_stride(arg971_1, (128, ), (1, ))
    assert_size_stride(arg972_1, (128, 512), (512, 1))
    assert_size_stride(arg973_1, (128, ), (1, ))
    assert_size_stride(arg974_1, (128, 128), (128, 1))
    assert_size_stride(arg975_1, (128, ), (1, ))
    assert_size_stride(arg976_1, (512, 128), (128, 1))
    assert_size_stride(arg977_1, (512, ), (1, ))
    assert_size_stride(arg978_1, (128, 512), (512, 1))
    assert_size_stride(arg979_1, (128, ), (1, ))
    assert_size_stride(arg980_1, (512, 128), (128, 1))
    assert_size_stride(arg981_1, (512, ), (1, ))
    assert_size_stride(arg982_1, (128, 512), (512, 1))
    assert_size_stride(arg983_1, (128, ), (1, ))
    assert_size_stride(arg984_1, (512, 128), (128, 1))
    assert_size_stride(arg985_1, (512, ), (1, ))
    assert_size_stride(arg986_1, (128, 512), (512, 1))
    assert_size_stride(arg987_1, (128, ), (1, ))
    assert_size_stride(arg988_1, (512, 128), (128, 1))
    assert_size_stride(arg989_1, (512, ), (1, ))
    assert_size_stride(arg990_1, (128, 512), (512, 1))
    assert_size_stride(arg991_1, (128, ), (1, ))
    assert_size_stride(arg992_1, (512, 128), (128, 1))
    assert_size_stride(arg993_1, (512, ), (1, ))
    assert_size_stride(arg994_1, (128, 512), (512, 1))
    assert_size_stride(arg995_1, (128, ), (1, ))
    assert_size_stride(arg996_1, (128, 512), (512, 1))
    assert_size_stride(arg997_1, (128, ), (1, ))
    assert_size_stride(arg998_1, (128, 128), (128, 1))
    assert_size_stride(arg999_1, (128, ), (1, ))
    assert_size_stride(arg1000_1, (128, 128), (128, 1))
    assert_size_stride(arg1001_1, (128, ), (1, ))
    assert_size_stride(arg1002_1, (128, 512), (512, 1))
    assert_size_stride(arg1003_1, (128, ), (1, ))
    assert_size_stride(arg1004_1, (128, 128), (128, 1))
    assert_size_stride(arg1005_1, (128, ), (1, ))
    assert_size_stride(arg1006_1, (512, 128), (128, 1))
    assert_size_stride(arg1007_1, (512, ), (1, ))
    assert_size_stride(arg1008_1, (128, 512), (512, 1))
    assert_size_stride(arg1009_1, (128, ), (1, ))
    assert_size_stride(arg1010_1, (512, 128), (128, 1))
    assert_size_stride(arg1011_1, (512, ), (1, ))
    assert_size_stride(arg1012_1, (128, 512), (512, 1))
    assert_size_stride(arg1013_1, (128, ), (1, ))
    assert_size_stride(arg1014_1, (512, 128), (128, 1))
    assert_size_stride(arg1015_1, (512, ), (1, ))
    assert_size_stride(arg1016_1, (128, 512), (512, 1))
    assert_size_stride(arg1017_1, (128, ), (1, ))
    assert_size_stride(arg1018_1, (512, 128), (128, 1))
    assert_size_stride(arg1019_1, (512, ), (1, ))
    assert_size_stride(arg1020_1, (128, 512), (512, 1))
    assert_size_stride(arg1021_1, (128, ), (1, ))
    assert_size_stride(arg1022_1, (512, 128), (128, 1))
    assert_size_stride(arg1023_1, (512, ), (1, ))
    assert_size_stride(arg1024_1, (128, 512), (512, 1))
    assert_size_stride(arg1025_1, (128, ), (1, ))
    assert_size_stride(arg1026_1, (128, 512), (512, 1))
    assert_size_stride(arg1027_1, (128, ), (1, ))
    assert_size_stride(arg1028_1, (128, 128), (128, 1))
    assert_size_stride(arg1029_1, (128, ), (1, ))
    assert_size_stride(arg1030_1, (128, 128), (128, 1))
    assert_size_stride(arg1031_1, (128, ), (1, ))
    assert_size_stride(arg1032_1, (128, 512), (512, 1))
    assert_size_stride(arg1033_1, (128, ), (1, ))
    assert_size_stride(arg1034_1, (128, 128), (128, 1))
    assert_size_stride(arg1035_1, (128, ), (1, ))
    assert_size_stride(arg1036_1, (512, 128), (128, 1))
    assert_size_stride(arg1037_1, (512, ), (1, ))
    assert_size_stride(arg1038_1, (128, 512), (512, 1))
    assert_size_stride(arg1039_1, (128, ), (1, ))
    assert_size_stride(arg1040_1, (512, 128), (128, 1))
    assert_size_stride(arg1041_1, (512, ), (1, ))
    assert_size_stride(arg1042_1, (128, 512), (512, 1))
    assert_size_stride(arg1043_1, (128, ), (1, ))
    assert_size_stride(arg1044_1, (512, 128), (128, 1))
    assert_size_stride(arg1045_1, (512, ), (1, ))
    assert_size_stride(arg1046_1, (128, 512), (512, 1))
    assert_size_stride(arg1047_1, (128, ), (1, ))
    assert_size_stride(arg1048_1, (512, 128), (128, 1))
    assert_size_stride(arg1049_1, (512, ), (1, ))
    assert_size_stride(arg1050_1, (128, 512), (512, 1))
    assert_size_stride(arg1051_1, (128, ), (1, ))
    assert_size_stride(arg1052_1, (512, 128), (128, 1))
    assert_size_stride(arg1053_1, (512, ), (1, ))
    assert_size_stride(arg1054_1, (128, 512), (512, 1))
    assert_size_stride(arg1055_1, (128, ), (1, ))
    assert_size_stride(arg1056_1, (128, 512), (512, 1))
    assert_size_stride(arg1057_1, (128, ), (1, ))
    assert_size_stride(arg1058_1, (128, 128), (128, 1))
    assert_size_stride(arg1059_1, (128, ), (1, ))
    assert_size_stride(arg1060_1, (128, 128), (128, 1))
    assert_size_stride(arg1061_1, (128, ), (1, ))
    assert_size_stride(arg1062_1, (128, 512), (512, 1))
    assert_size_stride(arg1063_1, (128, ), (1, ))
    assert_size_stride(arg1064_1, (128, 128), (128, 1))
    assert_size_stride(arg1065_1, (128, ), (1, ))
    assert_size_stride(arg1066_1, (512, 128), (128, 1))
    assert_size_stride(arg1067_1, (512, ), (1, ))
    assert_size_stride(arg1068_1, (128, 512), (512, 1))
    assert_size_stride(arg1069_1, (128, ), (1, ))
    assert_size_stride(arg1070_1, (512, 128), (128, 1))
    assert_size_stride(arg1071_1, (512, ), (1, ))
    assert_size_stride(arg1072_1, (128, 512), (512, 1))
    assert_size_stride(arg1073_1, (128, ), (1, ))
    assert_size_stride(arg1074_1, (512, 128), (128, 1))
    assert_size_stride(arg1075_1, (512, ), (1, ))
    assert_size_stride(arg1076_1, (128, 512), (512, 1))
    assert_size_stride(arg1077_1, (128, ), (1, ))
    assert_size_stride(arg1078_1, (512, 128), (128, 1))
    assert_size_stride(arg1079_1, (512, ), (1, ))
    assert_size_stride(arg1080_1, (128, 512), (512, 1))
    assert_size_stride(arg1081_1, (128, ), (1, ))
    assert_size_stride(arg1082_1, (512, 128), (128, 1))
    assert_size_stride(arg1083_1, (512, ), (1, ))
    assert_size_stride(arg1084_1, (128, 512), (512, 1))
    assert_size_stride(arg1085_1, (128, ), (1, ))
    assert_size_stride(arg1086_1, (128, 512), (512, 1))
    assert_size_stride(arg1087_1, (128, ), (1, ))
    assert_size_stride(arg1088_1, (128, 128), (128, 1))
    assert_size_stride(arg1089_1, (128, ), (1, ))
    assert_size_stride(arg1090_1, (128, 128), (128, 1))
    assert_size_stride(arg1091_1, (128, ), (1, ))
    assert_size_stride(arg1092_1, (128, 512), (512, 1))
    assert_size_stride(arg1093_1, (128, ), (1, ))
    assert_size_stride(arg1094_1, (128, 128), (128, 1))
    assert_size_stride(arg1095_1, (128, ), (1, ))
    assert_size_stride(arg1096_1, (512, 128), (128, 1))
    assert_size_stride(arg1097_1, (512, ), (1, ))
    assert_size_stride(arg1098_1, (128, 512), (512, 1))
    assert_size_stride(arg1099_1, (128, ), (1, ))
    assert_size_stride(arg1100_1, (512, 128), (128, 1))
    assert_size_stride(arg1101_1, (512, ), (1, ))
    assert_size_stride(arg1102_1, (128, 512), (512, 1))
    assert_size_stride(arg1103_1, (128, ), (1, ))
    assert_size_stride(arg1104_1, (512, 128), (128, 1))
    assert_size_stride(arg1105_1, (512, ), (1, ))
    assert_size_stride(arg1106_1, (128, 512), (512, 1))
    assert_size_stride(arg1107_1, (128, ), (1, ))
    assert_size_stride(arg1108_1, (512, 128), (128, 1))
    assert_size_stride(arg1109_1, (512, ), (1, ))
    assert_size_stride(arg1110_1, (128, 512), (512, 1))
    assert_size_stride(arg1111_1, (128, ), (1, ))
    assert_size_stride(arg1112_1, (512, 128), (128, 1))
    assert_size_stride(arg1113_1, (512, ), (1, ))
    assert_size_stride(arg1114_1, (512, 512), (512, 1))
    assert_size_stride(arg1115_1, (512, ), (1, ))
    assert_size_stride(arg1116_1, (512, ), (1, ))
    assert_size_stride(arg1117_1, (512, ), (1, ))
    assert_size_stride(arg1118_1, (1, 512), (512, 1))
    assert_size_stride(arg1119_1, (1, 128), (128, 1))
    assert_size_stride(arg1120_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_cat_0.run(arg1119_1, arg389_1, buf0, 49152, grid=grid(49152), stream=stream0)
        del arg1119_1
        del arg389_1
        buf1 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (128, 384), (384, 1), 0), reinterpret_tensor(arg390_1, (384, 512), (1, 384), 0), out=buf1)
        del arg390_1
        del buf0
        buf2 = reinterpret_tensor(buf1, (1, 128, 512), (65536, 512, 1), 0); del buf1  # reuse
        # Source Nodes: [add, embeddings, embeddings_1, mul_1, position_embeddings, token_type_embeddings, token_type_ids], Original ATen: [aten.add, aten.embedding, aten.mul, aten.zeros]
        triton_poi_fused_add_embedding_mul_zeros_1.run(buf2, arg391_1, arg1118_1, arg392_1, arg393_1, arg0_1, arg1_1, 65536, grid=grid(65536), stream=stream0)
        del arg0_1
        del arg1118_1
        del arg1_1
        del arg391_1
        del arg392_1
        del arg393_1
        buf3 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg396_1, (512, 128), (1, 512), 0), out=buf3)
        del arg396_1
        buf4 = reinterpret_tensor(buf3, (1, 128, 128), (16384, 128, 1), 0); del buf3  # reuse
        # Source Nodes: [key_tensor, mul_3], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf4, arg397_1, arg4_1, arg5_1, 16384, grid=grid(16384), stream=stream0)
        del arg397_1
        del arg4_1
        del arg5_1
        buf5 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (128, 128), (128, 1), 0), reinterpret_tensor(arg398_1, (128, 128), (1, 128), 0), out=buf5)
        del arg398_1
        buf6 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (128, 128), (128, 1), 0), reinterpret_tensor(arg400_1, (128, 128), (1, 128), 0), out=buf6)
        del arg400_1
        buf7 = reinterpret_tensor(buf4, (128, 128), (128, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg402_1, (512, 128), (1, 512), 0), out=buf7)
        del arg402_1
        buf8 = empty((1, 4, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf5, arg399_1, buf8, 16384, grid=grid(16384), stream=stream0)
        del arg399_1
        buf9 = reinterpret_tensor(buf5, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf6, arg401_1, buf9, 16384, grid=grid(16384), stream=stream0)
        del arg401_1
        buf10 = reinterpret_tensor(buf6, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf7, arg403_1, buf10, 16384, grid=grid(16384), stream=stream0)
        del arg403_1
        del buf7
        # Source Nodes: [], Original ATen: []
        buf11 = aten._scaled_dot_product_efficient_attention(buf8, buf9, buf10, None, False, scale=0.17677669529663687)
        buf12 = buf11[0]
        del buf11
        buf16 = reinterpret_tensor(buf9, (128, 128), (128, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (128, 128), (128, 1), 0), reinterpret_tensor(arg404_1, (128, 128), (1, 128), 0), out=buf16)
        del arg404_1
        buf17 = reinterpret_tensor(buf12, (128, 128), (128, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg394_1, (512, 128), (1, 512), 0), out=buf17)
        del arg394_1
        buf18 = reinterpret_tensor(buf16, (1, 128, 128), (16384, 128, 1), 0); del buf16  # reuse
        # Source Nodes: [add_6, attention_output, layer_input_4, mul_2, mul_4], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf18, arg405_1, buf17, arg395_1, arg2_1, arg3_1, arg6_1, arg7_1, 16384, grid=grid(16384), stream=stream0)
        del arg2_1
        del arg395_1
        del arg3_1
        del arg405_1
        del arg6_1
        del arg7_1
        buf19 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 128), (128, 1), 0), reinterpret_tensor(arg406_1, (128, 512), (1, 128), 0), out=buf19)
        del arg406_1
        buf20 = reinterpret_tensor(buf19, (1, 128, 512), (65536, 512, 1), 0); del buf19  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf20, arg407_1, 65536, grid=grid(65536), stream=stream0)
        del arg407_1
        buf21 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (128, 512), (512, 1), 0), reinterpret_tensor(arg408_1, (512, 128), (1, 512), 0), out=buf21)
        del arg408_1
        buf22 = buf18; del buf18  # reuse
        # Source Nodes: [add_8, attention_output_1, mul_5], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf22, buf21, arg409_1, arg8_1, arg9_1, 16384, grid=grid(16384), stream=stream0)
        del arg409_1
        del arg8_1
        del arg9_1
        buf23 = reinterpret_tensor(buf20, (128, 512), (512, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (128, 128), (128, 1), 0), reinterpret_tensor(arg410_1, (128, 512), (1, 128), 0), out=buf23)
        del arg410_1
        buf24 = reinterpret_tensor(buf23, (1, 128, 512), (65536, 512, 1), 0); del buf23  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf24, arg411_1, 65536, grid=grid(65536), stream=stream0)
        del arg411_1
        buf25 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (128, 512), (512, 1), 0), reinterpret_tensor(arg412_1, (512, 128), (1, 512), 0), out=buf25)
        del arg412_1
        buf26 = buf22; del buf22  # reuse
        # Source Nodes: [add_10, attention_output_2, mul_6], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf26, buf25, arg413_1, arg10_1, arg11_1, 16384, grid=grid(16384), stream=stream0)
        del arg10_1
        del arg11_1
        del arg413_1
        buf27 = reinterpret_tensor(buf24, (128, 512), (512, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (128, 128), (128, 1), 0), reinterpret_tensor(arg414_1, (128, 512), (1, 128), 0), out=buf27)
        del arg414_1
        buf28 = reinterpret_tensor(buf27, (1, 128, 512), (65536, 512, 1), 0); del buf27  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf28, arg415_1, 65536, grid=grid(65536), stream=stream0)
        del arg415_1
        buf29 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (128, 512), (512, 1), 0), reinterpret_tensor(arg416_1, (512, 128), (1, 512), 0), out=buf29)
        del arg416_1
        buf30 = buf26; del buf26  # reuse
        # Source Nodes: [add_12, attention_output_3, mul_7], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf30, buf29, arg417_1, arg12_1, arg13_1, 16384, grid=grid(16384), stream=stream0)
        del arg12_1
        del arg13_1
        del arg417_1
        buf31 = reinterpret_tensor(buf28, (128, 512), (512, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (128, 128), (128, 1), 0), reinterpret_tensor(arg418_1, (128, 512), (1, 128), 0), out=buf31)
        del arg418_1
        buf32 = reinterpret_tensor(buf31, (1, 128, 512), (65536, 512, 1), 0); del buf31  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf32, arg419_1, 65536, grid=grid(65536), stream=stream0)
        del arg419_1
        buf33 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (128, 512), (512, 1), 0), reinterpret_tensor(arg420_1, (512, 128), (1, 512), 0), out=buf33)
        del arg420_1
        buf34 = buf30; del buf30  # reuse
        # Source Nodes: [add_14, layer_output_1, mul_8], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf34, buf33, arg421_1, arg14_1, arg15_1, 16384, grid=grid(16384), stream=stream0)
        del arg14_1
        del arg15_1
        del arg421_1
        buf35 = reinterpret_tensor(buf32, (128, 512), (512, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (128, 128), (128, 1), 0), reinterpret_tensor(arg422_1, (128, 512), (1, 128), 0), out=buf35)
        del arg422_1
        buf36 = buf2; del buf2  # reuse
        # Source Nodes: [add_16, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf36, buf35, arg423_1, arg16_1, arg17_1, 65536, grid=grid(65536), stream=stream0)
        del arg16_1
        del arg17_1
        del arg423_1
        buf37 = reinterpret_tensor(buf34, (128, 128), (128, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (128, 512), (512, 1), 0), reinterpret_tensor(arg426_1, (512, 128), (1, 512), 0), out=buf37)
        del arg426_1
        buf38 = reinterpret_tensor(buf37, (1, 128, 128), (16384, 128, 1), 0); del buf37  # reuse
        # Source Nodes: [key_tensor_1, mul_11], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf38, arg427_1, arg20_1, arg21_1, 16384, grid=grid(16384), stream=stream0)
        del arg20_1
        del arg21_1
        del arg427_1
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (128, 128), (128, 1), 0), reinterpret_tensor(arg428_1, (128, 128), (1, 128), 0), out=buf39)
        del arg428_1
        buf40 = reinterpret_tensor(buf8, (128, 128), (128, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (128, 128), (128, 1), 0), reinterpret_tensor(arg430_1, (128, 128), (1, 128), 0), out=buf40)
        del arg430_1
        buf41 = reinterpret_tensor(buf38, (128, 128), (128, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (128, 512), (512, 1), 0), reinterpret_tensor(arg432_1, (512, 128), (1, 512), 0), out=buf41)
        del arg432_1
        buf42 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf39, arg429_1, buf42, 16384, grid=grid(16384), stream=stream0)
        del arg429_1
        buf43 = reinterpret_tensor(buf39, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf40, arg431_1, buf43, 16384, grid=grid(16384), stream=stream0)
        del arg431_1
        buf44 = reinterpret_tensor(buf40, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf41, arg433_1, buf44, 16384, grid=grid(16384), stream=stream0)
        del arg433_1
        del buf41
        # Source Nodes: [], Original ATen: []
        buf45 = aten._scaled_dot_product_efficient_attention(buf42, buf43, buf44, None, False, scale=0.17677669529663687)
        buf46 = buf45[0]
        del buf45
        buf50 = reinterpret_tensor(buf44, (128, 128), (128, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (128, 128), (128, 1), 0), reinterpret_tensor(arg434_1, (128, 128), (1, 128), 0), out=buf50)
        del arg434_1
        buf51 = reinterpret_tensor(buf46, (128, 128), (128, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (128, 512), (512, 1), 0), reinterpret_tensor(arg424_1, (512, 128), (1, 512), 0), out=buf51)
        del arg424_1
        buf52 = reinterpret_tensor(buf50, (1, 128, 128), (16384, 128, 1), 0); del buf50  # reuse
        # Source Nodes: [add_21, attention_output_5, layer_input_9, mul_10, mul_12], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf52, arg435_1, buf51, arg425_1, arg18_1, arg19_1, arg22_1, arg23_1, 16384, grid=grid(16384), stream=stream0)
        del arg18_1
        del arg19_1
        del arg22_1
        del arg23_1
        del arg425_1
        del arg435_1
        buf53 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (128, 128), (128, 1), 0), reinterpret_tensor(arg436_1, (128, 512), (1, 128), 0), out=buf53)
        del arg436_1
        buf54 = reinterpret_tensor(buf53, (1, 128, 512), (65536, 512, 1), 0); del buf53  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf54, arg437_1, 65536, grid=grid(65536), stream=stream0)
        del arg437_1
        buf55 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (128, 512), (512, 1), 0), reinterpret_tensor(arg438_1, (512, 128), (1, 512), 0), out=buf55)
        del arg438_1
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [add_23, attention_output_6, mul_13], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf56, buf55, arg439_1, arg24_1, arg25_1, 16384, grid=grid(16384), stream=stream0)
        del arg24_1
        del arg25_1
        del arg439_1
        buf57 = reinterpret_tensor(buf54, (128, 512), (512, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 128), (128, 1), 0), reinterpret_tensor(arg440_1, (128, 512), (1, 128), 0), out=buf57)
        del arg440_1
        buf58 = reinterpret_tensor(buf57, (1, 128, 512), (65536, 512, 1), 0); del buf57  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf58, arg441_1, 65536, grid=grid(65536), stream=stream0)
        del arg441_1
        buf59 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (128, 512), (512, 1), 0), reinterpret_tensor(arg442_1, (512, 128), (1, 512), 0), out=buf59)
        del arg442_1
        buf60 = buf56; del buf56  # reuse
        # Source Nodes: [add_25, attention_output_7, mul_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf60, buf59, arg443_1, arg26_1, arg27_1, 16384, grid=grid(16384), stream=stream0)
        del arg26_1
        del arg27_1
        del arg443_1
        buf61 = reinterpret_tensor(buf58, (128, 512), (512, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (128, 128), (128, 1), 0), reinterpret_tensor(arg444_1, (128, 512), (1, 128), 0), out=buf61)
        del arg444_1
        buf62 = reinterpret_tensor(buf61, (1, 128, 512), (65536, 512, 1), 0); del buf61  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf62, arg445_1, 65536, grid=grid(65536), stream=stream0)
        del arg445_1
        buf63 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (128, 512), (512, 1), 0), reinterpret_tensor(arg446_1, (512, 128), (1, 512), 0), out=buf63)
        del arg446_1
        buf64 = buf60; del buf60  # reuse
        # Source Nodes: [add_27, attention_output_8, mul_15], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf64, buf63, arg447_1, arg28_1, arg29_1, 16384, grid=grid(16384), stream=stream0)
        del arg28_1
        del arg29_1
        del arg447_1
        buf65 = reinterpret_tensor(buf62, (128, 512), (512, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (128, 128), (128, 1), 0), reinterpret_tensor(arg448_1, (128, 512), (1, 128), 0), out=buf65)
        del arg448_1
        buf66 = reinterpret_tensor(buf65, (1, 128, 512), (65536, 512, 1), 0); del buf65  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf66, arg449_1, 65536, grid=grid(65536), stream=stream0)
        del arg449_1
        buf67 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 512), (512, 1), 0), reinterpret_tensor(arg450_1, (512, 128), (1, 512), 0), out=buf67)
        del arg450_1
        buf68 = buf64; del buf64  # reuse
        # Source Nodes: [add_29, layer_output_5, mul_16], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf68, buf67, arg451_1, arg30_1, arg31_1, 16384, grid=grid(16384), stream=stream0)
        del arg30_1
        del arg31_1
        del arg451_1
        buf69 = reinterpret_tensor(buf66, (128, 512), (512, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (128, 128), (128, 1), 0), reinterpret_tensor(arg452_1, (128, 512), (1, 128), 0), out=buf69)
        del arg452_1
        buf70 = buf36; del buf36  # reuse
        # Source Nodes: [add_31, mul_17, value_tensor_2], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf70, buf69, arg453_1, arg32_1, arg33_1, 65536, grid=grid(65536), stream=stream0)
        del arg32_1
        del arg33_1
        del arg453_1
        buf71 = reinterpret_tensor(buf68, (128, 128), (128, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (128, 512), (512, 1), 0), reinterpret_tensor(arg456_1, (512, 128), (1, 512), 0), out=buf71)
        del arg456_1
        buf72 = reinterpret_tensor(buf71, (1, 128, 128), (16384, 128, 1), 0); del buf71  # reuse
        # Source Nodes: [key_tensor_2, mul_19], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf72, arg457_1, arg36_1, arg37_1, 16384, grid=grid(16384), stream=stream0)
        del arg36_1
        del arg37_1
        del arg457_1
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 128), (128, 1), 0), reinterpret_tensor(arg458_1, (128, 128), (1, 128), 0), out=buf73)
        del arg458_1
        buf74 = reinterpret_tensor(buf43, (128, 128), (128, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 128), (128, 1), 0), reinterpret_tensor(arg460_1, (128, 128), (1, 128), 0), out=buf74)
        del arg460_1
        buf75 = reinterpret_tensor(buf72, (128, 128), (128, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (128, 512), (512, 1), 0), reinterpret_tensor(arg462_1, (512, 128), (1, 512), 0), out=buf75)
        del arg462_1
        buf76 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf73, arg459_1, buf76, 16384, grid=grid(16384), stream=stream0)
        del arg459_1
        buf77 = reinterpret_tensor(buf73, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf74, arg461_1, buf77, 16384, grid=grid(16384), stream=stream0)
        del arg461_1
        buf78 = reinterpret_tensor(buf74, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf75, arg463_1, buf78, 16384, grid=grid(16384), stream=stream0)
        del arg463_1
        del buf75
        # Source Nodes: [], Original ATen: []
        buf79 = aten._scaled_dot_product_efficient_attention(buf76, buf77, buf78, None, False, scale=0.17677669529663687)
        buf80 = buf79[0]
        del buf79
        buf84 = reinterpret_tensor(buf78, (128, 128), (128, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (128, 128), (128, 1), 0), reinterpret_tensor(arg464_1, (128, 128), (1, 128), 0), out=buf84)
        del arg464_1
        buf85 = reinterpret_tensor(buf80, (128, 128), (128, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (128, 512), (512, 1), 0), reinterpret_tensor(arg454_1, (512, 128), (1, 512), 0), out=buf85)
        del arg454_1
        buf86 = reinterpret_tensor(buf84, (1, 128, 128), (16384, 128, 1), 0); del buf84  # reuse
        # Source Nodes: [add_36, attention_output_10, layer_input_14, mul_18, mul_20], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf86, arg465_1, buf85, arg455_1, arg34_1, arg35_1, arg38_1, arg39_1, 16384, grid=grid(16384), stream=stream0)
        del arg34_1
        del arg35_1
        del arg38_1
        del arg39_1
        del arg455_1
        del arg465_1
        buf87 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (128, 128), (128, 1), 0), reinterpret_tensor(arg466_1, (128, 512), (1, 128), 0), out=buf87)
        del arg466_1
        buf88 = reinterpret_tensor(buf87, (1, 128, 512), (65536, 512, 1), 0); del buf87  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf88, arg467_1, 65536, grid=grid(65536), stream=stream0)
        del arg467_1
        buf89 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (128, 512), (512, 1), 0), reinterpret_tensor(arg468_1, (512, 128), (1, 512), 0), out=buf89)
        del arg468_1
        buf90 = buf86; del buf86  # reuse
        # Source Nodes: [add_38, attention_output_11, mul_21], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf90, buf89, arg469_1, arg40_1, arg41_1, 16384, grid=grid(16384), stream=stream0)
        del arg40_1
        del arg41_1
        del arg469_1
        buf91 = reinterpret_tensor(buf88, (128, 512), (512, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (128, 128), (128, 1), 0), reinterpret_tensor(arg470_1, (128, 512), (1, 128), 0), out=buf91)
        del arg470_1
        buf92 = reinterpret_tensor(buf91, (1, 128, 512), (65536, 512, 1), 0); del buf91  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf92, arg471_1, 65536, grid=grid(65536), stream=stream0)
        del arg471_1
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (128, 512), (512, 1), 0), reinterpret_tensor(arg472_1, (512, 128), (1, 512), 0), out=buf93)
        del arg472_1
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [add_40, attention_output_12, mul_22], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf94, buf93, arg473_1, arg42_1, arg43_1, 16384, grid=grid(16384), stream=stream0)
        del arg42_1
        del arg43_1
        del arg473_1
        buf95 = reinterpret_tensor(buf92, (128, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (128, 128), (128, 1), 0), reinterpret_tensor(arg474_1, (128, 512), (1, 128), 0), out=buf95)
        del arg474_1
        buf96 = reinterpret_tensor(buf95, (1, 128, 512), (65536, 512, 1), 0); del buf95  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf96, arg475_1, 65536, grid=grid(65536), stream=stream0)
        del arg475_1
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (128, 512), (512, 1), 0), reinterpret_tensor(arg476_1, (512, 128), (1, 512), 0), out=buf97)
        del arg476_1
        buf98 = buf94; del buf94  # reuse
        # Source Nodes: [add_42, attention_output_13, mul_23], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf98, buf97, arg477_1, arg44_1, arg45_1, 16384, grid=grid(16384), stream=stream0)
        del arg44_1
        del arg45_1
        del arg477_1
        buf99 = reinterpret_tensor(buf96, (128, 512), (512, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (128, 128), (128, 1), 0), reinterpret_tensor(arg478_1, (128, 512), (1, 128), 0), out=buf99)
        del arg478_1
        buf100 = reinterpret_tensor(buf99, (1, 128, 512), (65536, 512, 1), 0); del buf99  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf100, arg479_1, 65536, grid=grid(65536), stream=stream0)
        del arg479_1
        buf101 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (128, 512), (512, 1), 0), reinterpret_tensor(arg480_1, (512, 128), (1, 512), 0), out=buf101)
        del arg480_1
        buf102 = reinterpret_tensor(buf101, (1, 128, 128), (16384, 128, 1), 0); del buf101  # reuse
        # Source Nodes: [add_44, layer_output_9, mul_24], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf102, arg481_1, buf98, arg46_1, arg47_1, 16384, grid=grid(16384), stream=stream0)
        del arg46_1
        del arg47_1
        del arg481_1
        buf103 = reinterpret_tensor(buf100, (128, 512), (512, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (128, 128), (128, 1), 0), reinterpret_tensor(arg482_1, (128, 512), (1, 128), 0), out=buf103)
        del arg482_1
        buf104 = reinterpret_tensor(buf103, (1, 128, 512), (65536, 512, 1), 0); del buf103  # reuse
        # Source Nodes: [add_46, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(buf104, arg483_1, buf70, arg48_1, arg49_1, 65536, grid=grid(65536), stream=stream0)
        del arg483_1
        del arg48_1
        del arg49_1
        buf105 = reinterpret_tensor(buf102, (128, 128), (128, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (128, 512), (512, 1), 0), reinterpret_tensor(arg486_1, (512, 128), (1, 512), 0), out=buf105)
        del arg486_1
        buf106 = reinterpret_tensor(buf105, (1, 128, 128), (16384, 128, 1), 0); del buf105  # reuse
        # Source Nodes: [key_tensor_3, mul_27], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf106, arg487_1, arg52_1, arg53_1, 16384, grid=grid(16384), stream=stream0)
        del arg487_1
        del arg52_1
        del arg53_1
        buf107 = reinterpret_tensor(buf98, (128, 128), (128, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 128), (128, 1), 0), reinterpret_tensor(arg488_1, (128, 128), (1, 128), 0), out=buf107)
        del arg488_1
        buf108 = reinterpret_tensor(buf77, (128, 128), (128, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 128), (128, 1), 0), reinterpret_tensor(arg490_1, (128, 128), (1, 128), 0), out=buf108)
        del arg490_1
        buf109 = reinterpret_tensor(buf106, (128, 128), (128, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (128, 512), (512, 1), 0), reinterpret_tensor(arg492_1, (512, 128), (1, 512), 0), out=buf109)
        del arg492_1
        buf110 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf107, arg489_1, buf110, 16384, grid=grid(16384), stream=stream0)
        del arg489_1
        buf111 = reinterpret_tensor(buf107, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf108, arg491_1, buf111, 16384, grid=grid(16384), stream=stream0)
        del arg491_1
        buf112 = reinterpret_tensor(buf108, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf109, arg493_1, buf112, 16384, grid=grid(16384), stream=stream0)
        del arg493_1
        del buf109
        # Source Nodes: [], Original ATen: []
        buf113 = aten._scaled_dot_product_efficient_attention(buf110, buf111, buf112, None, False, scale=0.17677669529663687)
        buf114 = buf113[0]
        del buf113
        buf118 = reinterpret_tensor(buf112, (128, 128), (128, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (128, 128), (128, 1), 0), reinterpret_tensor(arg494_1, (128, 128), (1, 128), 0), out=buf118)
        del arg494_1
        buf119 = reinterpret_tensor(buf114, (128, 128), (128, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (128, 512), (512, 1), 0), reinterpret_tensor(arg484_1, (512, 128), (1, 512), 0), out=buf119)
        del arg484_1
        buf120 = reinterpret_tensor(buf118, (1, 128, 128), (16384, 128, 1), 0); del buf118  # reuse
        # Source Nodes: [add_51, attention_output_15, layer_input_19, mul_26, mul_28], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf120, arg495_1, buf119, arg485_1, arg50_1, arg51_1, arg54_1, arg55_1, 16384, grid=grid(16384), stream=stream0)
        del arg485_1
        del arg495_1
        del arg50_1
        del arg51_1
        del arg54_1
        del arg55_1
        buf121 = reinterpret_tensor(buf70, (128, 512), (512, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (128, 128), (128, 1), 0), reinterpret_tensor(arg496_1, (128, 512), (1, 128), 0), out=buf121)
        del arg496_1
        buf122 = reinterpret_tensor(buf121, (1, 128, 512), (65536, 512, 1), 0); del buf121  # reuse
        # Source Nodes: [intermediate_output_12], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf122, arg497_1, 65536, grid=grid(65536), stream=stream0)
        del arg497_1
        buf123 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (128, 512), (512, 1), 0), reinterpret_tensor(arg498_1, (512, 128), (1, 512), 0), out=buf123)
        del arg498_1
        buf124 = buf120; del buf120  # reuse
        # Source Nodes: [add_53, attention_output_16, mul_29], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf124, buf123, arg499_1, arg56_1, arg57_1, 16384, grid=grid(16384), stream=stream0)
        del arg499_1
        del arg56_1
        del arg57_1
        buf125 = reinterpret_tensor(buf122, (128, 512), (512, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (128, 128), (128, 1), 0), reinterpret_tensor(arg500_1, (128, 512), (1, 128), 0), out=buf125)
        del arg500_1
        buf126 = reinterpret_tensor(buf125, (1, 128, 512), (65536, 512, 1), 0); del buf125  # reuse
        # Source Nodes: [intermediate_output_13], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf126, arg501_1, 65536, grid=grid(65536), stream=stream0)
        del arg501_1
        buf127 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (128, 512), (512, 1), 0), reinterpret_tensor(arg502_1, (512, 128), (1, 512), 0), out=buf127)
        del arg502_1
        buf128 = buf124; del buf124  # reuse
        # Source Nodes: [add_55, attention_output_17, mul_30], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf128, buf127, arg503_1, arg58_1, arg59_1, 16384, grid=grid(16384), stream=stream0)
        del arg503_1
        del arg58_1
        del arg59_1
        buf129 = reinterpret_tensor(buf126, (128, 512), (512, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 128), (128, 1), 0), reinterpret_tensor(arg504_1, (128, 512), (1, 128), 0), out=buf129)
        del arg504_1
        buf130 = reinterpret_tensor(buf129, (1, 128, 512), (65536, 512, 1), 0); del buf129  # reuse
        # Source Nodes: [intermediate_output_14], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf130, arg505_1, 65536, grid=grid(65536), stream=stream0)
        del arg505_1
        buf131 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (128, 512), (512, 1), 0), reinterpret_tensor(arg506_1, (512, 128), (1, 512), 0), out=buf131)
        del arg506_1
        buf132 = buf128; del buf128  # reuse
        # Source Nodes: [add_57, attention_output_18, mul_31], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf132, buf131, arg507_1, arg60_1, arg61_1, 16384, grid=grid(16384), stream=stream0)
        del arg507_1
        del arg60_1
        del arg61_1
        buf133 = reinterpret_tensor(buf130, (128, 512), (512, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (128, 128), (128, 1), 0), reinterpret_tensor(arg508_1, (128, 512), (1, 128), 0), out=buf133)
        del arg508_1
        buf134 = reinterpret_tensor(buf133, (1, 128, 512), (65536, 512, 1), 0); del buf133  # reuse
        # Source Nodes: [intermediate_output_15], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf134, arg509_1, 65536, grid=grid(65536), stream=stream0)
        del arg509_1
        buf135 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (128, 512), (512, 1), 0), reinterpret_tensor(arg510_1, (512, 128), (1, 512), 0), out=buf135)
        del arg510_1
        buf136 = buf132; del buf132  # reuse
        # Source Nodes: [add_59, layer_output_13, mul_32], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf136, buf135, arg511_1, arg62_1, arg63_1, 16384, grid=grid(16384), stream=stream0)
        del arg511_1
        del arg62_1
        del arg63_1
        buf137 = reinterpret_tensor(buf134, (128, 512), (512, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (128, 128), (128, 1), 0), reinterpret_tensor(arg512_1, (128, 512), (1, 128), 0), out=buf137)
        del arg512_1
        buf138 = buf104; del buf104  # reuse
        # Source Nodes: [add_61, mul_33, value_tensor_4], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf138, buf137, arg513_1, arg64_1, arg65_1, 65536, grid=grid(65536), stream=stream0)
        del arg513_1
        del arg64_1
        del arg65_1
        buf139 = reinterpret_tensor(buf136, (128, 128), (128, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg516_1, (512, 128), (1, 512), 0), out=buf139)
        del arg516_1
        buf140 = reinterpret_tensor(buf139, (1, 128, 128), (16384, 128, 1), 0); del buf139  # reuse
        # Source Nodes: [key_tensor_4, mul_35], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf140, arg517_1, arg68_1, arg69_1, 16384, grid=grid(16384), stream=stream0)
        del arg517_1
        del arg68_1
        del arg69_1
        buf141 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 128), (128, 1), 0), reinterpret_tensor(arg518_1, (128, 128), (1, 128), 0), out=buf141)
        del arg518_1
        buf142 = reinterpret_tensor(buf111, (128, 128), (128, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 128), (128, 1), 0), reinterpret_tensor(arg520_1, (128, 128), (1, 128), 0), out=buf142)
        del arg520_1
        buf143 = reinterpret_tensor(buf140, (128, 128), (128, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg522_1, (512, 128), (1, 512), 0), out=buf143)
        del arg522_1
        buf144 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf141, arg519_1, buf144, 16384, grid=grid(16384), stream=stream0)
        del arg519_1
        buf145 = reinterpret_tensor(buf141, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf142, arg521_1, buf145, 16384, grid=grid(16384), stream=stream0)
        del arg521_1
        buf146 = reinterpret_tensor(buf142, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf143, arg523_1, buf146, 16384, grid=grid(16384), stream=stream0)
        del arg523_1
        del buf143
        # Source Nodes: [], Original ATen: []
        buf147 = aten._scaled_dot_product_efficient_attention(buf144, buf145, buf146, None, False, scale=0.17677669529663687)
        buf148 = buf147[0]
        del buf147
        buf152 = reinterpret_tensor(buf146, (128, 128), (128, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (128, 128), (128, 1), 0), reinterpret_tensor(arg524_1, (128, 128), (1, 128), 0), out=buf152)
        del arg524_1
        buf153 = reinterpret_tensor(buf148, (128, 128), (128, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg514_1, (512, 128), (1, 512), 0), out=buf153)
        del arg514_1
        buf154 = reinterpret_tensor(buf152, (1, 128, 128), (16384, 128, 1), 0); del buf152  # reuse
        # Source Nodes: [add_66, attention_output_20, layer_input_24, mul_34, mul_36], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf154, arg525_1, buf153, arg515_1, arg66_1, arg67_1, arg70_1, arg71_1, 16384, grid=grid(16384), stream=stream0)
        del arg515_1
        del arg525_1
        del arg66_1
        del arg67_1
        del arg70_1
        del arg71_1
        buf155 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (128, 128), (128, 1), 0), reinterpret_tensor(arg526_1, (128, 512), (1, 128), 0), out=buf155)
        del arg526_1
        buf156 = reinterpret_tensor(buf155, (1, 128, 512), (65536, 512, 1), 0); del buf155  # reuse
        # Source Nodes: [intermediate_output_16], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf156, arg527_1, 65536, grid=grid(65536), stream=stream0)
        del arg527_1
        buf157 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 512), (512, 1), 0), reinterpret_tensor(arg528_1, (512, 128), (1, 512), 0), out=buf157)
        del arg528_1
        buf158 = buf154; del buf154  # reuse
        # Source Nodes: [add_68, attention_output_21, mul_37], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf158, buf157, arg529_1, arg72_1, arg73_1, 16384, grid=grid(16384), stream=stream0)
        del arg529_1
        del arg72_1
        del arg73_1
        buf159 = reinterpret_tensor(buf156, (128, 512), (512, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (128, 128), (128, 1), 0), reinterpret_tensor(arg530_1, (128, 512), (1, 128), 0), out=buf159)
        del arg530_1
        buf160 = reinterpret_tensor(buf159, (1, 128, 512), (65536, 512, 1), 0); del buf159  # reuse
        # Source Nodes: [intermediate_output_17], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf160, arg531_1, 65536, grid=grid(65536), stream=stream0)
        del arg531_1
        buf161 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (128, 512), (512, 1), 0), reinterpret_tensor(arg532_1, (512, 128), (1, 512), 0), out=buf161)
        del arg532_1
        buf162 = buf158; del buf158  # reuse
        # Source Nodes: [add_70, attention_output_22, mul_38], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf162, buf161, arg533_1, arg74_1, arg75_1, 16384, grid=grid(16384), stream=stream0)
        del arg533_1
        del arg74_1
        del arg75_1
        buf163 = reinterpret_tensor(buf160, (128, 512), (512, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (128, 128), (128, 1), 0), reinterpret_tensor(arg534_1, (128, 512), (1, 128), 0), out=buf163)
        del arg534_1
        buf164 = reinterpret_tensor(buf163, (1, 128, 512), (65536, 512, 1), 0); del buf163  # reuse
        # Source Nodes: [intermediate_output_18], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf164, arg535_1, 65536, grid=grid(65536), stream=stream0)
        del arg535_1
        buf165 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (128, 512), (512, 1), 0), reinterpret_tensor(arg536_1, (512, 128), (1, 512), 0), out=buf165)
        del arg536_1
        buf166 = buf162; del buf162  # reuse
        # Source Nodes: [add_72, attention_output_23, mul_39], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf166, buf165, arg537_1, arg76_1, arg77_1, 16384, grid=grid(16384), stream=stream0)
        del arg537_1
        del arg76_1
        del arg77_1
        buf167 = reinterpret_tensor(buf164, (128, 512), (512, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (128, 128), (128, 1), 0), reinterpret_tensor(arg538_1, (128, 512), (1, 128), 0), out=buf167)
        del arg538_1
        buf168 = reinterpret_tensor(buf167, (1, 128, 512), (65536, 512, 1), 0); del buf167  # reuse
        # Source Nodes: [intermediate_output_19], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf168, arg539_1, 65536, grid=grid(65536), stream=stream0)
        del arg539_1
        buf169 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (128, 512), (512, 1), 0), reinterpret_tensor(arg540_1, (512, 128), (1, 512), 0), out=buf169)
        del arg540_1
        buf170 = buf166; del buf166  # reuse
        # Source Nodes: [add_74, layer_output_17, mul_40], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf170, buf169, arg541_1, arg78_1, arg79_1, 16384, grid=grid(16384), stream=stream0)
        del arg541_1
        del arg78_1
        del arg79_1
        buf171 = reinterpret_tensor(buf168, (128, 512), (512, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (128, 128), (128, 1), 0), reinterpret_tensor(arg542_1, (128, 512), (1, 128), 0), out=buf171)
        del arg542_1
        buf172 = buf138; del buf138  # reuse
        # Source Nodes: [add_76, mul_41, value_tensor_5], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf172, buf171, arg543_1, arg80_1, arg81_1, 65536, grid=grid(65536), stream=stream0)
        del arg543_1
        del arg80_1
        del arg81_1
        buf173 = reinterpret_tensor(buf170, (128, 128), (128, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (128, 512), (512, 1), 0), reinterpret_tensor(arg546_1, (512, 128), (1, 512), 0), out=buf173)
        del arg546_1
        buf174 = reinterpret_tensor(buf173, (1, 128, 128), (16384, 128, 1), 0); del buf173  # reuse
        # Source Nodes: [key_tensor_5, mul_43], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf174, arg547_1, arg84_1, arg85_1, 16384, grid=grid(16384), stream=stream0)
        del arg547_1
        del arg84_1
        del arg85_1
        buf175 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (128, 128), (128, 1), 0), reinterpret_tensor(arg548_1, (128, 128), (1, 128), 0), out=buf175)
        del arg548_1
        buf176 = reinterpret_tensor(buf145, (128, 128), (128, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (128, 128), (128, 1), 0), reinterpret_tensor(arg550_1, (128, 128), (1, 128), 0), out=buf176)
        del arg550_1
        buf177 = reinterpret_tensor(buf174, (128, 128), (128, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (128, 512), (512, 1), 0), reinterpret_tensor(arg552_1, (512, 128), (1, 512), 0), out=buf177)
        del arg552_1
        buf178 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf175, arg549_1, buf178, 16384, grid=grid(16384), stream=stream0)
        del arg549_1
        buf179 = reinterpret_tensor(buf175, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf176, arg551_1, buf179, 16384, grid=grid(16384), stream=stream0)
        del arg551_1
        buf180 = reinterpret_tensor(buf176, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf177, arg553_1, buf180, 16384, grid=grid(16384), stream=stream0)
        del arg553_1
        del buf177
        # Source Nodes: [], Original ATen: []
        buf181 = aten._scaled_dot_product_efficient_attention(buf178, buf179, buf180, None, False, scale=0.17677669529663687)
        buf182 = buf181[0]
        del buf181
        buf186 = reinterpret_tensor(buf180, (128, 128), (128, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (128, 128), (128, 1), 0), reinterpret_tensor(arg554_1, (128, 128), (1, 128), 0), out=buf186)
        del arg554_1
        buf187 = reinterpret_tensor(buf182, (128, 128), (128, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (128, 512), (512, 1), 0), reinterpret_tensor(arg544_1, (512, 128), (1, 512), 0), out=buf187)
        del arg544_1
        buf188 = reinterpret_tensor(buf186, (1, 128, 128), (16384, 128, 1), 0); del buf186  # reuse
        # Source Nodes: [add_81, attention_output_25, layer_input_29, mul_42, mul_44], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf188, arg555_1, buf187, arg545_1, arg82_1, arg83_1, arg86_1, arg87_1, 16384, grid=grid(16384), stream=stream0)
        del arg545_1
        del arg555_1
        del arg82_1
        del arg83_1
        del arg86_1
        del arg87_1
        buf189 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (128, 128), (128, 1), 0), reinterpret_tensor(arg556_1, (128, 512), (1, 128), 0), out=buf189)
        del arg556_1
        buf190 = reinterpret_tensor(buf189, (1, 128, 512), (65536, 512, 1), 0); del buf189  # reuse
        # Source Nodes: [intermediate_output_20], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf190, arg557_1, 65536, grid=grid(65536), stream=stream0)
        del arg557_1
        buf191 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (128, 512), (512, 1), 0), reinterpret_tensor(arg558_1, (512, 128), (1, 512), 0), out=buf191)
        del arg558_1
        buf192 = buf188; del buf188  # reuse
        # Source Nodes: [add_83, attention_output_26, mul_45], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf192, buf191, arg559_1, arg88_1, arg89_1, 16384, grid=grid(16384), stream=stream0)
        del arg559_1
        del arg88_1
        del arg89_1
        buf193 = reinterpret_tensor(buf190, (128, 512), (512, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (128, 128), (128, 1), 0), reinterpret_tensor(arg560_1, (128, 512), (1, 128), 0), out=buf193)
        del arg560_1
        buf194 = reinterpret_tensor(buf193, (1, 128, 512), (65536, 512, 1), 0); del buf193  # reuse
        # Source Nodes: [intermediate_output_21], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf194, arg561_1, 65536, grid=grid(65536), stream=stream0)
        del arg561_1
        buf195 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (128, 512), (512, 1), 0), reinterpret_tensor(arg562_1, (512, 128), (1, 512), 0), out=buf195)
        del arg562_1
        buf196 = buf192; del buf192  # reuse
        # Source Nodes: [add_85, attention_output_27, mul_46], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf196, buf195, arg563_1, arg90_1, arg91_1, 16384, grid=grid(16384), stream=stream0)
        del arg563_1
        del arg90_1
        del arg91_1
        buf197 = reinterpret_tensor(buf194, (128, 512), (512, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 128), (128, 1), 0), reinterpret_tensor(arg564_1, (128, 512), (1, 128), 0), out=buf197)
        del arg564_1
        buf198 = reinterpret_tensor(buf197, (1, 128, 512), (65536, 512, 1), 0); del buf197  # reuse
        # Source Nodes: [intermediate_output_22], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf198, arg565_1, 65536, grid=grid(65536), stream=stream0)
        del arg565_1
        buf199 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 512), (512, 1), 0), reinterpret_tensor(arg566_1, (512, 128), (1, 512), 0), out=buf199)
        del arg566_1
        buf200 = buf196; del buf196  # reuse
        # Source Nodes: [add_87, attention_output_28, mul_47], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf200, buf199, arg567_1, arg92_1, arg93_1, 16384, grid=grid(16384), stream=stream0)
        del arg567_1
        del arg92_1
        del arg93_1
        buf201 = reinterpret_tensor(buf198, (128, 512), (512, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (128, 128), (128, 1), 0), reinterpret_tensor(arg568_1, (128, 512), (1, 128), 0), out=buf201)
        del arg568_1
        buf202 = reinterpret_tensor(buf201, (1, 128, 512), (65536, 512, 1), 0); del buf201  # reuse
        # Source Nodes: [intermediate_output_23], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf202, arg569_1, 65536, grid=grid(65536), stream=stream0)
        del arg569_1
        buf203 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (128, 512), (512, 1), 0), reinterpret_tensor(arg570_1, (512, 128), (1, 512), 0), out=buf203)
        del arg570_1
        buf204 = buf200; del buf200  # reuse
        # Source Nodes: [add_89, layer_output_21, mul_48], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf204, buf203, arg571_1, arg94_1, arg95_1, 16384, grid=grid(16384), stream=stream0)
        del arg571_1
        del arg94_1
        del arg95_1
        buf205 = reinterpret_tensor(buf202, (128, 512), (512, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (128, 128), (128, 1), 0), reinterpret_tensor(arg572_1, (128, 512), (1, 128), 0), out=buf205)
        del arg572_1
        buf206 = buf172; del buf172  # reuse
        # Source Nodes: [add_91, mul_49, value_tensor_6], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf206, buf205, arg573_1, arg96_1, arg97_1, 65536, grid=grid(65536), stream=stream0)
        del arg573_1
        del arg96_1
        del arg97_1
        buf207 = reinterpret_tensor(buf204, (128, 128), (128, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 512), (512, 1), 0), reinterpret_tensor(arg576_1, (512, 128), (1, 512), 0), out=buf207)
        del arg576_1
        buf208 = reinterpret_tensor(buf207, (1, 128, 128), (16384, 128, 1), 0); del buf207  # reuse
        # Source Nodes: [key_tensor_6, mul_51], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf208, arg577_1, arg100_1, arg101_1, 16384, grid=grid(16384), stream=stream0)
        del arg100_1
        del arg101_1
        del arg577_1
        buf209 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 128), (128, 1), 0), reinterpret_tensor(arg578_1, (128, 128), (1, 128), 0), out=buf209)
        del arg578_1
        buf210 = reinterpret_tensor(buf179, (128, 128), (128, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 128), (128, 1), 0), reinterpret_tensor(arg580_1, (128, 128), (1, 128), 0), out=buf210)
        del arg580_1
        buf211 = reinterpret_tensor(buf208, (128, 128), (128, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 512), (512, 1), 0), reinterpret_tensor(arg582_1, (512, 128), (1, 512), 0), out=buf211)
        del arg582_1
        buf212 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf209, arg579_1, buf212, 16384, grid=grid(16384), stream=stream0)
        del arg579_1
        buf213 = reinterpret_tensor(buf209, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf210, arg581_1, buf213, 16384, grid=grid(16384), stream=stream0)
        del arg581_1
        buf214 = reinterpret_tensor(buf210, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf211, arg583_1, buf214, 16384, grid=grid(16384), stream=stream0)
        del arg583_1
        del buf211
        # Source Nodes: [], Original ATen: []
        buf215 = aten._scaled_dot_product_efficient_attention(buf212, buf213, buf214, None, False, scale=0.17677669529663687)
        buf216 = buf215[0]
        del buf215
        buf220 = reinterpret_tensor(buf214, (128, 128), (128, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (128, 128), (128, 1), 0), reinterpret_tensor(arg584_1, (128, 128), (1, 128), 0), out=buf220)
        del arg584_1
        buf221 = reinterpret_tensor(buf216, (128, 128), (128, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 512), (512, 1), 0), reinterpret_tensor(arg574_1, (512, 128), (1, 512), 0), out=buf221)
        del arg574_1
        buf222 = reinterpret_tensor(buf220, (1, 128, 128), (16384, 128, 1), 0); del buf220  # reuse
        # Source Nodes: [add_96, attention_output_30, layer_input_34, mul_50, mul_52], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf222, arg585_1, buf221, arg575_1, arg98_1, arg99_1, arg102_1, arg103_1, 16384, grid=grid(16384), stream=stream0)
        del arg102_1
        del arg103_1
        del arg575_1
        del arg585_1
        del arg98_1
        del arg99_1
        buf223 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (128, 128), (128, 1), 0), reinterpret_tensor(arg586_1, (128, 512), (1, 128), 0), out=buf223)
        del arg586_1
        buf224 = reinterpret_tensor(buf223, (1, 128, 512), (65536, 512, 1), 0); del buf223  # reuse
        # Source Nodes: [intermediate_output_24], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf224, arg587_1, 65536, grid=grid(65536), stream=stream0)
        del arg587_1
        buf225 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (128, 512), (512, 1), 0), reinterpret_tensor(arg588_1, (512, 128), (1, 512), 0), out=buf225)
        del arg588_1
        buf226 = buf222; del buf222  # reuse
        # Source Nodes: [add_98, attention_output_31, mul_53], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf226, buf225, arg589_1, arg104_1, arg105_1, 16384, grid=grid(16384), stream=stream0)
        del arg104_1
        del arg105_1
        del arg589_1
        buf227 = reinterpret_tensor(buf224, (128, 512), (512, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (128, 128), (128, 1), 0), reinterpret_tensor(arg590_1, (128, 512), (1, 128), 0), out=buf227)
        del arg590_1
        buf228 = reinterpret_tensor(buf227, (1, 128, 512), (65536, 512, 1), 0); del buf227  # reuse
        # Source Nodes: [intermediate_output_25], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf228, arg591_1, 65536, grid=grid(65536), stream=stream0)
        del arg591_1
        buf229 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 512), (512, 1), 0), reinterpret_tensor(arg592_1, (512, 128), (1, 512), 0), out=buf229)
        del arg592_1
        buf230 = buf226; del buf226  # reuse
        # Source Nodes: [add_100, attention_output_32, mul_54], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf230, buf229, arg593_1, arg106_1, arg107_1, 16384, grid=grid(16384), stream=stream0)
        del arg106_1
        del arg107_1
        del arg593_1
        buf231 = reinterpret_tensor(buf228, (128, 512), (512, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (128, 128), (128, 1), 0), reinterpret_tensor(arg594_1, (128, 512), (1, 128), 0), out=buf231)
        del arg594_1
        buf232 = reinterpret_tensor(buf231, (1, 128, 512), (65536, 512, 1), 0); del buf231  # reuse
        # Source Nodes: [intermediate_output_26], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf232, arg595_1, 65536, grid=grid(65536), stream=stream0)
        del arg595_1
        buf233 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 512), (512, 1), 0), reinterpret_tensor(arg596_1, (512, 128), (1, 512), 0), out=buf233)
        del arg596_1
        buf234 = buf230; del buf230  # reuse
        # Source Nodes: [add_102, attention_output_33, mul_55], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf234, buf233, arg597_1, arg108_1, arg109_1, 16384, grid=grid(16384), stream=stream0)
        del arg108_1
        del arg109_1
        del arg597_1
        buf235 = reinterpret_tensor(buf232, (128, 512), (512, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (128, 128), (128, 1), 0), reinterpret_tensor(arg598_1, (128, 512), (1, 128), 0), out=buf235)
        del arg598_1
        buf236 = reinterpret_tensor(buf235, (1, 128, 512), (65536, 512, 1), 0); del buf235  # reuse
        # Source Nodes: [intermediate_output_27], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf236, arg599_1, 65536, grid=grid(65536), stream=stream0)
        del arg599_1
        buf237 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (128, 512), (512, 1), 0), reinterpret_tensor(arg600_1, (512, 128), (1, 512), 0), out=buf237)
        del arg600_1
        buf238 = buf234; del buf234  # reuse
        # Source Nodes: [add_104, layer_output_25, mul_56], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf238, buf237, arg601_1, arg110_1, arg111_1, 16384, grid=grid(16384), stream=stream0)
        del arg110_1
        del arg111_1
        del arg601_1
        buf239 = reinterpret_tensor(buf236, (128, 512), (512, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (128, 128), (128, 1), 0), reinterpret_tensor(arg602_1, (128, 512), (1, 128), 0), out=buf239)
        del arg602_1
        buf240 = buf206; del buf206  # reuse
        # Source Nodes: [add_106, mul_57, value_tensor_7], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf240, buf239, arg603_1, arg112_1, arg113_1, 65536, grid=grid(65536), stream=stream0)
        del arg112_1
        del arg113_1
        del arg603_1
        buf241 = reinterpret_tensor(buf238, (128, 128), (128, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg606_1, (512, 128), (1, 512), 0), out=buf241)
        del arg606_1
        buf242 = reinterpret_tensor(buf241, (1, 128, 128), (16384, 128, 1), 0); del buf241  # reuse
        # Source Nodes: [key_tensor_7, mul_59], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf242, arg607_1, arg116_1, arg117_1, 16384, grid=grid(16384), stream=stream0)
        del arg116_1
        del arg117_1
        del arg607_1
        buf243 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (128, 128), (128, 1), 0), reinterpret_tensor(arg608_1, (128, 128), (1, 128), 0), out=buf243)
        del arg608_1
        buf244 = reinterpret_tensor(buf213, (128, 128), (128, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (128, 128), (128, 1), 0), reinterpret_tensor(arg610_1, (128, 128), (1, 128), 0), out=buf244)
        del arg610_1
        buf245 = reinterpret_tensor(buf242, (128, 128), (128, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg612_1, (512, 128), (1, 512), 0), out=buf245)
        del arg612_1
        buf246 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf243, arg609_1, buf246, 16384, grid=grid(16384), stream=stream0)
        del arg609_1
        buf247 = reinterpret_tensor(buf243, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf244, arg611_1, buf247, 16384, grid=grid(16384), stream=stream0)
        del arg611_1
        buf248 = reinterpret_tensor(buf244, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf245, arg613_1, buf248, 16384, grid=grid(16384), stream=stream0)
        del arg613_1
        del buf245
        # Source Nodes: [], Original ATen: []
        buf249 = aten._scaled_dot_product_efficient_attention(buf246, buf247, buf248, None, False, scale=0.17677669529663687)
        buf250 = buf249[0]
        del buf249
        buf254 = reinterpret_tensor(buf248, (128, 128), (128, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (128, 128), (128, 1), 0), reinterpret_tensor(arg614_1, (128, 128), (1, 128), 0), out=buf254)
        del arg614_1
        buf255 = reinterpret_tensor(buf250, (128, 128), (128, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg604_1, (512, 128), (1, 512), 0), out=buf255)
        del arg604_1
        buf256 = reinterpret_tensor(buf254, (1, 128, 128), (16384, 128, 1), 0); del buf254  # reuse
        # Source Nodes: [add_111, attention_output_35, layer_input_39, mul_58, mul_60], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf256, arg615_1, buf255, arg605_1, arg114_1, arg115_1, arg118_1, arg119_1, 16384, grid=grid(16384), stream=stream0)
        del arg114_1
        del arg115_1
        del arg118_1
        del arg119_1
        del arg605_1
        del arg615_1
        buf257 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (128, 128), (128, 1), 0), reinterpret_tensor(arg616_1, (128, 512), (1, 128), 0), out=buf257)
        del arg616_1
        buf258 = reinterpret_tensor(buf257, (1, 128, 512), (65536, 512, 1), 0); del buf257  # reuse
        # Source Nodes: [intermediate_output_28], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf258, arg617_1, 65536, grid=grid(65536), stream=stream0)
        del arg617_1
        buf259 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (128, 512), (512, 1), 0), reinterpret_tensor(arg618_1, (512, 128), (1, 512), 0), out=buf259)
        del arg618_1
        buf260 = buf256; del buf256  # reuse
        # Source Nodes: [add_113, attention_output_36, mul_61], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf260, buf259, arg619_1, arg120_1, arg121_1, 16384, grid=grid(16384), stream=stream0)
        del arg120_1
        del arg121_1
        del arg619_1
        buf261 = reinterpret_tensor(buf258, (128, 512), (512, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (128, 128), (128, 1), 0), reinterpret_tensor(arg620_1, (128, 512), (1, 128), 0), out=buf261)
        del arg620_1
        buf262 = reinterpret_tensor(buf261, (1, 128, 512), (65536, 512, 1), 0); del buf261  # reuse
        # Source Nodes: [intermediate_output_29], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf262, arg621_1, 65536, grid=grid(65536), stream=stream0)
        del arg621_1
        buf263 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (128, 512), (512, 1), 0), reinterpret_tensor(arg622_1, (512, 128), (1, 512), 0), out=buf263)
        del arg622_1
        buf264 = buf260; del buf260  # reuse
        # Source Nodes: [add_115, attention_output_37, mul_62], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf264, buf263, arg623_1, arg122_1, arg123_1, 16384, grid=grid(16384), stream=stream0)
        del arg122_1
        del arg123_1
        del arg623_1
        buf265 = reinterpret_tensor(buf262, (128, 512), (512, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (128, 128), (128, 1), 0), reinterpret_tensor(arg624_1, (128, 512), (1, 128), 0), out=buf265)
        del arg624_1
        buf266 = reinterpret_tensor(buf265, (1, 128, 512), (65536, 512, 1), 0); del buf265  # reuse
        # Source Nodes: [intermediate_output_30], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf266, arg625_1, 65536, grid=grid(65536), stream=stream0)
        del arg625_1
        buf267 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (128, 512), (512, 1), 0), reinterpret_tensor(arg626_1, (512, 128), (1, 512), 0), out=buf267)
        del arg626_1
        buf268 = buf264; del buf264  # reuse
        # Source Nodes: [add_117, attention_output_38, mul_63], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf268, buf267, arg627_1, arg124_1, arg125_1, 16384, grid=grid(16384), stream=stream0)
        del arg124_1
        del arg125_1
        del arg627_1
        buf269 = reinterpret_tensor(buf266, (128, 512), (512, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (128, 128), (128, 1), 0), reinterpret_tensor(arg628_1, (128, 512), (1, 128), 0), out=buf269)
        del arg628_1
        buf270 = reinterpret_tensor(buf269, (1, 128, 512), (65536, 512, 1), 0); del buf269  # reuse
        # Source Nodes: [intermediate_output_31], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf270, arg629_1, 65536, grid=grid(65536), stream=stream0)
        del arg629_1
        buf271 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (128, 512), (512, 1), 0), reinterpret_tensor(arg630_1, (512, 128), (1, 512), 0), out=buf271)
        del arg630_1
        buf272 = buf268; del buf268  # reuse
        # Source Nodes: [add_119, layer_output_29, mul_64], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf272, buf271, arg631_1, arg126_1, arg127_1, 16384, grid=grid(16384), stream=stream0)
        del arg126_1
        del arg127_1
        del arg631_1
        buf273 = reinterpret_tensor(buf270, (128, 512), (512, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (128, 128), (128, 1), 0), reinterpret_tensor(arg632_1, (128, 512), (1, 128), 0), out=buf273)
        del arg632_1
        buf274 = buf240; del buf240  # reuse
        # Source Nodes: [add_121, mul_65, value_tensor_8], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf274, buf273, arg633_1, arg128_1, arg129_1, 65536, grid=grid(65536), stream=stream0)
        del arg128_1
        del arg129_1
        del arg633_1
        buf275 = reinterpret_tensor(buf272, (128, 128), (128, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), reinterpret_tensor(arg636_1, (512, 128), (1, 512), 0), out=buf275)
        del arg636_1
        buf276 = reinterpret_tensor(buf275, (1, 128, 128), (16384, 128, 1), 0); del buf275  # reuse
        # Source Nodes: [key_tensor_8, mul_67], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf276, arg637_1, arg132_1, arg133_1, 16384, grid=grid(16384), stream=stream0)
        del arg132_1
        del arg133_1
        del arg637_1
        buf277 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (128, 128), (128, 1), 0), reinterpret_tensor(arg638_1, (128, 128), (1, 128), 0), out=buf277)
        del arg638_1
        buf278 = reinterpret_tensor(buf247, (128, 128), (128, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (128, 128), (128, 1), 0), reinterpret_tensor(arg640_1, (128, 128), (1, 128), 0), out=buf278)
        del arg640_1
        buf279 = reinterpret_tensor(buf276, (128, 128), (128, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), reinterpret_tensor(arg642_1, (512, 128), (1, 512), 0), out=buf279)
        del arg642_1
        buf280 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf277, arg639_1, buf280, 16384, grid=grid(16384), stream=stream0)
        del arg639_1
        buf281 = reinterpret_tensor(buf277, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf278, arg641_1, buf281, 16384, grid=grid(16384), stream=stream0)
        del arg641_1
        buf282 = reinterpret_tensor(buf278, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf279, arg643_1, buf282, 16384, grid=grid(16384), stream=stream0)
        del arg643_1
        del buf279
        # Source Nodes: [], Original ATen: []
        buf283 = aten._scaled_dot_product_efficient_attention(buf280, buf281, buf282, None, False, scale=0.17677669529663687)
        buf284 = buf283[0]
        del buf283
        buf288 = reinterpret_tensor(buf282, (128, 128), (128, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (128, 128), (128, 1), 0), reinterpret_tensor(arg644_1, (128, 128), (1, 128), 0), out=buf288)
        del arg644_1
        buf289 = reinterpret_tensor(buf284, (128, 128), (128, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), reinterpret_tensor(arg634_1, (512, 128), (1, 512), 0), out=buf289)
        del arg634_1
        buf290 = reinterpret_tensor(buf288, (1, 128, 128), (16384, 128, 1), 0); del buf288  # reuse
        # Source Nodes: [add_126, attention_output_40, layer_input_44, mul_66, mul_68], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf290, arg645_1, buf289, arg635_1, arg130_1, arg131_1, arg134_1, arg135_1, 16384, grid=grid(16384), stream=stream0)
        del arg130_1
        del arg131_1
        del arg134_1
        del arg135_1
        del arg635_1
        del arg645_1
        buf291 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (128, 128), (128, 1), 0), reinterpret_tensor(arg646_1, (128, 512), (1, 128), 0), out=buf291)
        del arg646_1
        buf292 = reinterpret_tensor(buf291, (1, 128, 512), (65536, 512, 1), 0); del buf291  # reuse
        # Source Nodes: [intermediate_output_32], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf292, arg647_1, 65536, grid=grid(65536), stream=stream0)
        del arg647_1
        buf293 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 512), (512, 1), 0), reinterpret_tensor(arg648_1, (512, 128), (1, 512), 0), out=buf293)
        del arg648_1
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [add_128, attention_output_41, mul_69], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf294, buf293, arg649_1, arg136_1, arg137_1, 16384, grid=grid(16384), stream=stream0)
        del arg136_1
        del arg137_1
        del arg649_1
        buf295 = reinterpret_tensor(buf292, (128, 512), (512, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (128, 128), (128, 1), 0), reinterpret_tensor(arg650_1, (128, 512), (1, 128), 0), out=buf295)
        del arg650_1
        buf296 = reinterpret_tensor(buf295, (1, 128, 512), (65536, 512, 1), 0); del buf295  # reuse
        # Source Nodes: [intermediate_output_33], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf296, arg651_1, 65536, grid=grid(65536), stream=stream0)
        del arg651_1
        buf297 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (128, 512), (512, 1), 0), reinterpret_tensor(arg652_1, (512, 128), (1, 512), 0), out=buf297)
        del arg652_1
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [add_130, attention_output_42, mul_70], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf298, buf297, arg653_1, arg138_1, arg139_1, 16384, grid=grid(16384), stream=stream0)
        del arg138_1
        del arg139_1
        del arg653_1
        buf299 = reinterpret_tensor(buf296, (128, 512), (512, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 128), (128, 1), 0), reinterpret_tensor(arg654_1, (128, 512), (1, 128), 0), out=buf299)
        del arg654_1
        buf300 = reinterpret_tensor(buf299, (1, 128, 512), (65536, 512, 1), 0); del buf299  # reuse
        # Source Nodes: [intermediate_output_34], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf300, arg655_1, 65536, grid=grid(65536), stream=stream0)
        del arg655_1
        buf301 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (128, 512), (512, 1), 0), reinterpret_tensor(arg656_1, (512, 128), (1, 512), 0), out=buf301)
        del arg656_1
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [add_132, attention_output_43, mul_71], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf302, buf301, arg657_1, arg140_1, arg141_1, 16384, grid=grid(16384), stream=stream0)
        del arg140_1
        del arg141_1
        del arg657_1
        buf303 = reinterpret_tensor(buf300, (128, 512), (512, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (128, 128), (128, 1), 0), reinterpret_tensor(arg658_1, (128, 512), (1, 128), 0), out=buf303)
        del arg658_1
        buf304 = reinterpret_tensor(buf303, (1, 128, 512), (65536, 512, 1), 0); del buf303  # reuse
        # Source Nodes: [intermediate_output_35], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf304, arg659_1, 65536, grid=grid(65536), stream=stream0)
        del arg659_1
        buf305 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (128, 512), (512, 1), 0), reinterpret_tensor(arg660_1, (512, 128), (1, 512), 0), out=buf305)
        del arg660_1
        buf306 = buf302; del buf302  # reuse
        # Source Nodes: [add_134, layer_output_33, mul_72], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf306, buf305, arg661_1, arg142_1, arg143_1, 16384, grid=grid(16384), stream=stream0)
        del arg142_1
        del arg143_1
        del arg661_1
        buf307 = reinterpret_tensor(buf304, (128, 512), (512, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (128, 128), (128, 1), 0), reinterpret_tensor(arg662_1, (128, 512), (1, 128), 0), out=buf307)
        del arg662_1
        buf308 = buf274; del buf274  # reuse
        # Source Nodes: [add_136, mul_73, value_tensor_9], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf308, buf307, arg663_1, arg144_1, arg145_1, 65536, grid=grid(65536), stream=stream0)
        del arg144_1
        del arg145_1
        del arg663_1
        buf309 = reinterpret_tensor(buf306, (128, 128), (128, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (128, 512), (512, 1), 0), reinterpret_tensor(arg666_1, (512, 128), (1, 512), 0), out=buf309)
        del arg666_1
        buf310 = reinterpret_tensor(buf309, (1, 128, 128), (16384, 128, 1), 0); del buf309  # reuse
        # Source Nodes: [key_tensor_9, mul_75], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf310, arg667_1, arg148_1, arg149_1, 16384, grid=grid(16384), stream=stream0)
        del arg148_1
        del arg149_1
        del arg667_1
        buf311 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (128, 1), 0), reinterpret_tensor(arg668_1, (128, 128), (1, 128), 0), out=buf311)
        del arg668_1
        buf312 = reinterpret_tensor(buf281, (128, 128), (128, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (128, 1), 0), reinterpret_tensor(arg670_1, (128, 128), (1, 128), 0), out=buf312)
        del arg670_1
        buf313 = reinterpret_tensor(buf310, (128, 128), (128, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (128, 512), (512, 1), 0), reinterpret_tensor(arg672_1, (512, 128), (1, 512), 0), out=buf313)
        del arg672_1
        buf314 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf311, arg669_1, buf314, 16384, grid=grid(16384), stream=stream0)
        del arg669_1
        buf315 = reinterpret_tensor(buf311, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf312, arg671_1, buf315, 16384, grid=grid(16384), stream=stream0)
        del arg671_1
        buf316 = reinterpret_tensor(buf312, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf313, arg673_1, buf316, 16384, grid=grid(16384), stream=stream0)
        del arg673_1
        del buf313
        # Source Nodes: [], Original ATen: []
        buf317 = aten._scaled_dot_product_efficient_attention(buf314, buf315, buf316, None, False, scale=0.17677669529663687)
        buf318 = buf317[0]
        del buf317
        buf322 = reinterpret_tensor(buf316, (128, 128), (128, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf318, (128, 128), (128, 1), 0), reinterpret_tensor(arg674_1, (128, 128), (1, 128), 0), out=buf322)
        del arg674_1
        buf323 = reinterpret_tensor(buf318, (128, 128), (128, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (128, 512), (512, 1), 0), reinterpret_tensor(arg664_1, (512, 128), (1, 512), 0), out=buf323)
        del arg664_1
        buf324 = reinterpret_tensor(buf322, (1, 128, 128), (16384, 128, 1), 0); del buf322  # reuse
        # Source Nodes: [add_141, attention_output_45, layer_input_49, mul_74, mul_76], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf324, arg675_1, buf323, arg665_1, arg146_1, arg147_1, arg150_1, arg151_1, 16384, grid=grid(16384), stream=stream0)
        del arg146_1
        del arg147_1
        del arg150_1
        del arg151_1
        del arg665_1
        del arg675_1
        buf325 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (128, 128), (128, 1), 0), reinterpret_tensor(arg676_1, (128, 512), (1, 128), 0), out=buf325)
        del arg676_1
        buf326 = reinterpret_tensor(buf325, (1, 128, 512), (65536, 512, 1), 0); del buf325  # reuse
        # Source Nodes: [intermediate_output_36], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf326, arg677_1, 65536, grid=grid(65536), stream=stream0)
        del arg677_1
        buf327 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (128, 512), (512, 1), 0), reinterpret_tensor(arg678_1, (512, 128), (1, 512), 0), out=buf327)
        del arg678_1
        buf328 = buf324; del buf324  # reuse
        # Source Nodes: [add_143, attention_output_46, mul_77], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf328, buf327, arg679_1, arg152_1, arg153_1, 16384, grid=grid(16384), stream=stream0)
        del arg152_1
        del arg153_1
        del arg679_1
        buf329 = reinterpret_tensor(buf326, (128, 512), (512, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (128, 128), (128, 1), 0), reinterpret_tensor(arg680_1, (128, 512), (1, 128), 0), out=buf329)
        del arg680_1
        buf330 = reinterpret_tensor(buf329, (1, 128, 512), (65536, 512, 1), 0); del buf329  # reuse
        # Source Nodes: [intermediate_output_37], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf330, arg681_1, 65536, grid=grid(65536), stream=stream0)
        del arg681_1
        buf331 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 512), (512, 1), 0), reinterpret_tensor(arg682_1, (512, 128), (1, 512), 0), out=buf331)
        del arg682_1
        buf332 = buf328; del buf328  # reuse
        # Source Nodes: [add_145, attention_output_47, mul_78], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf332, buf331, arg683_1, arg154_1, arg155_1, 16384, grid=grid(16384), stream=stream0)
        del arg154_1
        del arg155_1
        del arg683_1
        buf333 = reinterpret_tensor(buf330, (128, 512), (512, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 128), (128, 1), 0), reinterpret_tensor(arg684_1, (128, 512), (1, 128), 0), out=buf333)
        del arg684_1
        buf334 = reinterpret_tensor(buf333, (1, 128, 512), (65536, 512, 1), 0); del buf333  # reuse
        # Source Nodes: [intermediate_output_38], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf334, arg685_1, 65536, grid=grid(65536), stream=stream0)
        del arg685_1
        buf335 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (128, 512), (512, 1), 0), reinterpret_tensor(arg686_1, (512, 128), (1, 512), 0), out=buf335)
        del arg686_1
        buf336 = buf332; del buf332  # reuse
        # Source Nodes: [add_147, attention_output_48, mul_79], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf336, buf335, arg687_1, arg156_1, arg157_1, 16384, grid=grid(16384), stream=stream0)
        del arg156_1
        del arg157_1
        del arg687_1
        buf337 = reinterpret_tensor(buf334, (128, 512), (512, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (128, 128), (128, 1), 0), reinterpret_tensor(arg688_1, (128, 512), (1, 128), 0), out=buf337)
        del arg688_1
        buf338 = reinterpret_tensor(buf337, (1, 128, 512), (65536, 512, 1), 0); del buf337  # reuse
        # Source Nodes: [intermediate_output_39], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf338, arg689_1, 65536, grid=grid(65536), stream=stream0)
        del arg689_1
        buf339 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf338, (128, 512), (512, 1), 0), reinterpret_tensor(arg690_1, (512, 128), (1, 512), 0), out=buf339)
        del arg690_1
        buf340 = buf336; del buf336  # reuse
        # Source Nodes: [add_149, layer_output_37, mul_80], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf340, buf339, arg691_1, arg158_1, arg159_1, 16384, grid=grid(16384), stream=stream0)
        del arg158_1
        del arg159_1
        del arg691_1
        buf341 = reinterpret_tensor(buf338, (128, 512), (512, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (128, 128), (128, 1), 0), reinterpret_tensor(arg692_1, (128, 512), (1, 128), 0), out=buf341)
        del arg692_1
        buf342 = buf308; del buf308  # reuse
        # Source Nodes: [add_151, mul_81, value_tensor_10], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf342, buf341, arg693_1, arg160_1, arg161_1, 65536, grid=grid(65536), stream=stream0)
        del arg160_1
        del arg161_1
        del arg693_1
        buf343 = reinterpret_tensor(buf340, (128, 128), (128, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 512), (512, 1), 0), reinterpret_tensor(arg696_1, (512, 128), (1, 512), 0), out=buf343)
        del arg696_1
        buf344 = reinterpret_tensor(buf343, (1, 128, 128), (16384, 128, 1), 0); del buf343  # reuse
        # Source Nodes: [key_tensor_10, mul_83], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf344, arg697_1, arg164_1, arg165_1, 16384, grid=grid(16384), stream=stream0)
        del arg164_1
        del arg165_1
        del arg697_1
        buf345 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (128, 128), (128, 1), 0), reinterpret_tensor(arg698_1, (128, 128), (1, 128), 0), out=buf345)
        del arg698_1
        buf346 = reinterpret_tensor(buf315, (128, 128), (128, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (128, 128), (128, 1), 0), reinterpret_tensor(arg700_1, (128, 128), (1, 128), 0), out=buf346)
        del arg700_1
        buf347 = reinterpret_tensor(buf344, (128, 128), (128, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 512), (512, 1), 0), reinterpret_tensor(arg702_1, (512, 128), (1, 512), 0), out=buf347)
        del arg702_1
        buf348 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf345, arg699_1, buf348, 16384, grid=grid(16384), stream=stream0)
        del arg699_1
        buf349 = reinterpret_tensor(buf345, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf346, arg701_1, buf349, 16384, grid=grid(16384), stream=stream0)
        del arg701_1
        buf350 = reinterpret_tensor(buf346, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf347, arg703_1, buf350, 16384, grid=grid(16384), stream=stream0)
        del arg703_1
        del buf347
        # Source Nodes: [], Original ATen: []
        buf351 = aten._scaled_dot_product_efficient_attention(buf348, buf349, buf350, None, False, scale=0.17677669529663687)
        buf352 = buf351[0]
        del buf351
        buf356 = reinterpret_tensor(buf350, (128, 128), (128, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (128, 128), (128, 1), 0), reinterpret_tensor(arg704_1, (128, 128), (1, 128), 0), out=buf356)
        del arg704_1
        buf357 = reinterpret_tensor(buf352, (128, 128), (128, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 512), (512, 1), 0), reinterpret_tensor(arg694_1, (512, 128), (1, 512), 0), out=buf357)
        del arg694_1
        buf358 = reinterpret_tensor(buf356, (1, 128, 128), (16384, 128, 1), 0); del buf356  # reuse
        # Source Nodes: [add_156, attention_output_50, layer_input_54, mul_82, mul_84], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf358, arg705_1, buf357, arg695_1, arg162_1, arg163_1, arg166_1, arg167_1, 16384, grid=grid(16384), stream=stream0)
        del arg162_1
        del arg163_1
        del arg166_1
        del arg167_1
        del arg695_1
        del arg705_1
        buf359 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (128, 128), (128, 1), 0), reinterpret_tensor(arg706_1, (128, 512), (1, 128), 0), out=buf359)
        del arg706_1
        buf360 = reinterpret_tensor(buf359, (1, 128, 512), (65536, 512, 1), 0); del buf359  # reuse
        # Source Nodes: [intermediate_output_40], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf360, arg707_1, 65536, grid=grid(65536), stream=stream0)
        del arg707_1
        buf361 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf360, (128, 512), (512, 1), 0), reinterpret_tensor(arg708_1, (512, 128), (1, 512), 0), out=buf361)
        del arg708_1
        buf362 = buf358; del buf358  # reuse
        # Source Nodes: [add_158, attention_output_51, mul_85], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf362, buf361, arg709_1, arg168_1, arg169_1, 16384, grid=grid(16384), stream=stream0)
        del arg168_1
        del arg169_1
        del arg709_1
        buf363 = reinterpret_tensor(buf360, (128, 512), (512, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (128, 128), (128, 1), 0), reinterpret_tensor(arg710_1, (128, 512), (1, 128), 0), out=buf363)
        del arg710_1
        buf364 = reinterpret_tensor(buf363, (1, 128, 512), (65536, 512, 1), 0); del buf363  # reuse
        # Source Nodes: [intermediate_output_41], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf364, arg711_1, 65536, grid=grid(65536), stream=stream0)
        del arg711_1
        buf365 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (128, 512), (512, 1), 0), reinterpret_tensor(arg712_1, (512, 128), (1, 512), 0), out=buf365)
        del arg712_1
        buf366 = buf362; del buf362  # reuse
        # Source Nodes: [add_160, attention_output_52, mul_86], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf366, buf365, arg713_1, arg170_1, arg171_1, 16384, grid=grid(16384), stream=stream0)
        del arg170_1
        del arg171_1
        del arg713_1
        buf367 = reinterpret_tensor(buf364, (128, 512), (512, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (128, 128), (128, 1), 0), reinterpret_tensor(arg714_1, (128, 512), (1, 128), 0), out=buf367)
        del arg714_1
        buf368 = reinterpret_tensor(buf367, (1, 128, 512), (65536, 512, 1), 0); del buf367  # reuse
        # Source Nodes: [intermediate_output_42], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf368, arg715_1, 65536, grid=grid(65536), stream=stream0)
        del arg715_1
        buf369 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf368, (128, 512), (512, 1), 0), reinterpret_tensor(arg716_1, (512, 128), (1, 512), 0), out=buf369)
        del arg716_1
        buf370 = buf366; del buf366  # reuse
        # Source Nodes: [add_162, attention_output_53, mul_87], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf370, buf369, arg717_1, arg172_1, arg173_1, 16384, grid=grid(16384), stream=stream0)
        del arg172_1
        del arg173_1
        del arg717_1
        buf371 = reinterpret_tensor(buf368, (128, 512), (512, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (128, 128), (128, 1), 0), reinterpret_tensor(arg718_1, (128, 512), (1, 128), 0), out=buf371)
        del arg718_1
        buf372 = reinterpret_tensor(buf371, (1, 128, 512), (65536, 512, 1), 0); del buf371  # reuse
        # Source Nodes: [intermediate_output_43], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf372, arg719_1, 65536, grid=grid(65536), stream=stream0)
        del arg719_1
        buf373 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (128, 512), (512, 1), 0), reinterpret_tensor(arg720_1, (512, 128), (1, 512), 0), out=buf373)
        del arg720_1
        buf374 = buf370; del buf370  # reuse
        # Source Nodes: [add_164, layer_output_41, mul_88], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf374, buf373, arg721_1, arg174_1, arg175_1, 16384, grid=grid(16384), stream=stream0)
        del arg174_1
        del arg175_1
        del arg721_1
        buf375 = reinterpret_tensor(buf372, (128, 512), (512, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (128, 128), (128, 1), 0), reinterpret_tensor(arg722_1, (128, 512), (1, 128), 0), out=buf375)
        del arg722_1
        buf376 = buf342; del buf342  # reuse
        # Source Nodes: [add_166, mul_89, value_tensor_11], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf376, buf375, arg723_1, arg176_1, arg177_1, 65536, grid=grid(65536), stream=stream0)
        del arg176_1
        del arg177_1
        del arg723_1
        buf377 = reinterpret_tensor(buf374, (128, 128), (128, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (128, 512), (512, 1), 0), reinterpret_tensor(arg726_1, (512, 128), (1, 512), 0), out=buf377)
        del arg726_1
        buf378 = reinterpret_tensor(buf377, (1, 128, 128), (16384, 128, 1), 0); del buf377  # reuse
        # Source Nodes: [key_tensor_11, mul_91], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf378, arg727_1, arg180_1, arg181_1, 16384, grid=grid(16384), stream=stream0)
        del arg180_1
        del arg181_1
        del arg727_1
        buf379 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 128), (128, 1), 0), reinterpret_tensor(arg728_1, (128, 128), (1, 128), 0), out=buf379)
        del arg728_1
        buf380 = reinterpret_tensor(buf349, (128, 128), (128, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 128), (128, 1), 0), reinterpret_tensor(arg730_1, (128, 128), (1, 128), 0), out=buf380)
        del arg730_1
        buf381 = reinterpret_tensor(buf378, (128, 128), (128, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (128, 512), (512, 1), 0), reinterpret_tensor(arg732_1, (512, 128), (1, 512), 0), out=buf381)
        del arg732_1
        buf382 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf379, arg729_1, buf382, 16384, grid=grid(16384), stream=stream0)
        del arg729_1
        buf383 = reinterpret_tensor(buf379, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf380, arg731_1, buf383, 16384, grid=grid(16384), stream=stream0)
        del arg731_1
        buf384 = reinterpret_tensor(buf380, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf381, arg733_1, buf384, 16384, grid=grid(16384), stream=stream0)
        del arg733_1
        del buf381
        # Source Nodes: [], Original ATen: []
        buf385 = aten._scaled_dot_product_efficient_attention(buf382, buf383, buf384, None, False, scale=0.17677669529663687)
        buf386 = buf385[0]
        del buf385
        buf390 = reinterpret_tensor(buf384, (128, 128), (128, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf386, (128, 128), (128, 1), 0), reinterpret_tensor(arg734_1, (128, 128), (1, 128), 0), out=buf390)
        del arg734_1
        buf391 = reinterpret_tensor(buf386, (128, 128), (128, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (128, 512), (512, 1), 0), reinterpret_tensor(arg724_1, (512, 128), (1, 512), 0), out=buf391)
        del arg724_1
        buf392 = reinterpret_tensor(buf390, (1, 128, 128), (16384, 128, 1), 0); del buf390  # reuse
        # Source Nodes: [add_171, attention_output_55, layer_input_59, mul_90, mul_92], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf392, arg735_1, buf391, arg725_1, arg178_1, arg179_1, arg182_1, arg183_1, 16384, grid=grid(16384), stream=stream0)
        del arg178_1
        del arg179_1
        del arg182_1
        del arg183_1
        del arg725_1
        del arg735_1
        buf393 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (128, 128), (128, 1), 0), reinterpret_tensor(arg736_1, (128, 512), (1, 128), 0), out=buf393)
        del arg736_1
        buf394 = reinterpret_tensor(buf393, (1, 128, 512), (65536, 512, 1), 0); del buf393  # reuse
        # Source Nodes: [intermediate_output_44], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf394, arg737_1, 65536, grid=grid(65536), stream=stream0)
        del arg737_1
        buf395 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (128, 512), (512, 1), 0), reinterpret_tensor(arg738_1, (512, 128), (1, 512), 0), out=buf395)
        del arg738_1
        buf396 = buf392; del buf392  # reuse
        # Source Nodes: [add_173, attention_output_56, mul_93], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf396, buf395, arg739_1, arg184_1, arg185_1, 16384, grid=grid(16384), stream=stream0)
        del arg184_1
        del arg185_1
        del arg739_1
        buf397 = reinterpret_tensor(buf394, (128, 512), (512, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (128, 128), (128, 1), 0), reinterpret_tensor(arg740_1, (128, 512), (1, 128), 0), out=buf397)
        del arg740_1
        buf398 = reinterpret_tensor(buf397, (1, 128, 512), (65536, 512, 1), 0); del buf397  # reuse
        # Source Nodes: [intermediate_output_45], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf398, arg741_1, 65536, grid=grid(65536), stream=stream0)
        del arg741_1
        buf399 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 512), (512, 1), 0), reinterpret_tensor(arg742_1, (512, 128), (1, 512), 0), out=buf399)
        del arg742_1
        buf400 = buf396; del buf396  # reuse
        # Source Nodes: [add_175, attention_output_57, mul_94], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf400, buf399, arg743_1, arg186_1, arg187_1, 16384, grid=grid(16384), stream=stream0)
        del arg186_1
        del arg187_1
        del arg743_1
        buf401 = reinterpret_tensor(buf398, (128, 512), (512, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf400, (128, 128), (128, 1), 0), reinterpret_tensor(arg744_1, (128, 512), (1, 128), 0), out=buf401)
        del arg744_1
        buf402 = reinterpret_tensor(buf401, (1, 128, 512), (65536, 512, 1), 0); del buf401  # reuse
        # Source Nodes: [intermediate_output_46], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf402, arg745_1, 65536, grid=grid(65536), stream=stream0)
        del arg745_1
        buf403 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (128, 512), (512, 1), 0), reinterpret_tensor(arg746_1, (512, 128), (1, 512), 0), out=buf403)
        del arg746_1
        buf404 = buf400; del buf400  # reuse
        # Source Nodes: [add_177, attention_output_58, mul_95], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf404, buf403, arg747_1, arg188_1, arg189_1, 16384, grid=grid(16384), stream=stream0)
        del arg188_1
        del arg189_1
        del arg747_1
        buf405 = reinterpret_tensor(buf402, (128, 512), (512, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (128, 128), (128, 1), 0), reinterpret_tensor(arg748_1, (128, 512), (1, 128), 0), out=buf405)
        del arg748_1
        buf406 = reinterpret_tensor(buf405, (1, 128, 512), (65536, 512, 1), 0); del buf405  # reuse
        # Source Nodes: [intermediate_output_47], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf406, arg749_1, 65536, grid=grid(65536), stream=stream0)
        del arg749_1
        buf407 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (128, 512), (512, 1), 0), reinterpret_tensor(arg750_1, (512, 128), (1, 512), 0), out=buf407)
        del arg750_1
        buf408 = buf404; del buf404  # reuse
        # Source Nodes: [add_179, layer_output_45, mul_96], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf408, buf407, arg751_1, arg190_1, arg191_1, 16384, grid=grid(16384), stream=stream0)
        del arg190_1
        del arg191_1
        del arg751_1
        buf409 = reinterpret_tensor(buf406, (128, 512), (512, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (128, 128), (128, 1), 0), reinterpret_tensor(arg752_1, (128, 512), (1, 128), 0), out=buf409)
        del arg752_1
        buf410 = buf376; del buf376  # reuse
        # Source Nodes: [add_181, mul_97, value_tensor_12], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf410, buf409, arg753_1, arg192_1, arg193_1, 65536, grid=grid(65536), stream=stream0)
        del arg192_1
        del arg193_1
        del arg753_1
        buf411 = reinterpret_tensor(buf408, (128, 128), (128, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (128, 512), (512, 1), 0), reinterpret_tensor(arg756_1, (512, 128), (1, 512), 0), out=buf411)
        del arg756_1
        buf412 = reinterpret_tensor(buf411, (1, 128, 128), (16384, 128, 1), 0); del buf411  # reuse
        # Source Nodes: [key_tensor_12, mul_99], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf412, arg757_1, arg196_1, arg197_1, 16384, grid=grid(16384), stream=stream0)
        del arg196_1
        del arg197_1
        del arg757_1
        buf413 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (128, 128), (128, 1), 0), reinterpret_tensor(arg758_1, (128, 128), (1, 128), 0), out=buf413)
        del arg758_1
        buf414 = reinterpret_tensor(buf383, (128, 128), (128, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (128, 128), (128, 1), 0), reinterpret_tensor(arg760_1, (128, 128), (1, 128), 0), out=buf414)
        del arg760_1
        buf415 = reinterpret_tensor(buf412, (128, 128), (128, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (128, 512), (512, 1), 0), reinterpret_tensor(arg762_1, (512, 128), (1, 512), 0), out=buf415)
        del arg762_1
        buf416 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf413, arg759_1, buf416, 16384, grid=grid(16384), stream=stream0)
        del arg759_1
        buf417 = reinterpret_tensor(buf413, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf414, arg761_1, buf417, 16384, grid=grid(16384), stream=stream0)
        del arg761_1
        buf418 = reinterpret_tensor(buf414, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf415, arg763_1, buf418, 16384, grid=grid(16384), stream=stream0)
        del arg763_1
        del buf415
        # Source Nodes: [], Original ATen: []
        buf419 = aten._scaled_dot_product_efficient_attention(buf416, buf417, buf418, None, False, scale=0.17677669529663687)
        buf420 = buf419[0]
        del buf419
        buf424 = reinterpret_tensor(buf418, (128, 128), (128, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf420, (128, 128), (128, 1), 0), reinterpret_tensor(arg764_1, (128, 128), (1, 128), 0), out=buf424)
        del arg764_1
        buf425 = reinterpret_tensor(buf420, (128, 128), (128, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (128, 512), (512, 1), 0), reinterpret_tensor(arg754_1, (512, 128), (1, 512), 0), out=buf425)
        del arg754_1
        buf426 = reinterpret_tensor(buf424, (1, 128, 128), (16384, 128, 1), 0); del buf424  # reuse
        # Source Nodes: [add_186, attention_output_60, layer_input_64, mul_100, mul_98], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf426, arg765_1, buf425, arg755_1, arg194_1, arg195_1, arg198_1, arg199_1, 16384, grid=grid(16384), stream=stream0)
        del arg194_1
        del arg195_1
        del arg198_1
        del arg199_1
        del arg755_1
        del arg765_1
        buf427 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf426, (128, 128), (128, 1), 0), reinterpret_tensor(arg766_1, (128, 512), (1, 128), 0), out=buf427)
        del arg766_1
        buf428 = reinterpret_tensor(buf427, (1, 128, 512), (65536, 512, 1), 0); del buf427  # reuse
        # Source Nodes: [intermediate_output_48], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf428, arg767_1, 65536, grid=grid(65536), stream=stream0)
        del arg767_1
        buf429 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (128, 512), (512, 1), 0), reinterpret_tensor(arg768_1, (512, 128), (1, 512), 0), out=buf429)
        del arg768_1
        buf430 = buf426; del buf426  # reuse
        # Source Nodes: [add_188, attention_output_61, mul_101], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf430, buf429, arg769_1, arg200_1, arg201_1, 16384, grid=grid(16384), stream=stream0)
        del arg200_1
        del arg201_1
        del arg769_1
        buf431 = reinterpret_tensor(buf428, (128, 512), (512, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (128, 128), (128, 1), 0), reinterpret_tensor(arg770_1, (128, 512), (1, 128), 0), out=buf431)
        del arg770_1
        buf432 = reinterpret_tensor(buf431, (1, 128, 512), (65536, 512, 1), 0); del buf431  # reuse
        # Source Nodes: [intermediate_output_49], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf432, arg771_1, 65536, grid=grid(65536), stream=stream0)
        del arg771_1
        buf433 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf432, (128, 512), (512, 1), 0), reinterpret_tensor(arg772_1, (512, 128), (1, 512), 0), out=buf433)
        del arg772_1
        buf434 = buf430; del buf430  # reuse
        # Source Nodes: [add_190, attention_output_62, mul_102], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf434, buf433, arg773_1, arg202_1, arg203_1, 16384, grid=grid(16384), stream=stream0)
        del arg202_1
        del arg203_1
        del arg773_1
        buf435 = reinterpret_tensor(buf432, (128, 512), (512, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (128, 128), (128, 1), 0), reinterpret_tensor(arg774_1, (128, 512), (1, 128), 0), out=buf435)
        del arg774_1
        buf436 = reinterpret_tensor(buf435, (1, 128, 512), (65536, 512, 1), 0); del buf435  # reuse
        # Source Nodes: [intermediate_output_50], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf436, arg775_1, 65536, grid=grid(65536), stream=stream0)
        del arg775_1
        buf437 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (128, 512), (512, 1), 0), reinterpret_tensor(arg776_1, (512, 128), (1, 512), 0), out=buf437)
        del arg776_1
        buf438 = buf434; del buf434  # reuse
        # Source Nodes: [add_192, attention_output_63, mul_103], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf438, buf437, arg777_1, arg204_1, arg205_1, 16384, grid=grid(16384), stream=stream0)
        del arg204_1
        del arg205_1
        del arg777_1
        buf439 = reinterpret_tensor(buf436, (128, 512), (512, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (128, 128), (128, 1), 0), reinterpret_tensor(arg778_1, (128, 512), (1, 128), 0), out=buf439)
        del arg778_1
        buf440 = reinterpret_tensor(buf439, (1, 128, 512), (65536, 512, 1), 0); del buf439  # reuse
        # Source Nodes: [intermediate_output_51], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf440, arg779_1, 65536, grid=grid(65536), stream=stream0)
        del arg779_1
        buf441 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (128, 512), (512, 1), 0), reinterpret_tensor(arg780_1, (512, 128), (1, 512), 0), out=buf441)
        del arg780_1
        buf442 = buf438; del buf438  # reuse
        # Source Nodes: [add_194, layer_output_49, mul_104], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf442, buf441, arg781_1, arg206_1, arg207_1, 16384, grid=grid(16384), stream=stream0)
        del arg206_1
        del arg207_1
        del arg781_1
        buf443 = reinterpret_tensor(buf440, (128, 512), (512, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf442, (128, 128), (128, 1), 0), reinterpret_tensor(arg782_1, (128, 512), (1, 128), 0), out=buf443)
        del arg782_1
        buf444 = buf410; del buf410  # reuse
        # Source Nodes: [add_196, mul_105, value_tensor_13], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf444, buf443, arg783_1, arg208_1, arg209_1, 65536, grid=grid(65536), stream=stream0)
        del arg208_1
        del arg209_1
        del arg783_1
        buf445 = reinterpret_tensor(buf442, (128, 128), (128, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (128, 512), (512, 1), 0), reinterpret_tensor(arg786_1, (512, 128), (1, 512), 0), out=buf445)
        del arg786_1
        buf446 = reinterpret_tensor(buf445, (1, 128, 128), (16384, 128, 1), 0); del buf445  # reuse
        # Source Nodes: [key_tensor_13, mul_107], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf446, arg787_1, arg212_1, arg213_1, 16384, grid=grid(16384), stream=stream0)
        del arg212_1
        del arg213_1
        del arg787_1
        buf447 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (128, 1), 0), reinterpret_tensor(arg788_1, (128, 128), (1, 128), 0), out=buf447)
        del arg788_1
        buf448 = reinterpret_tensor(buf417, (128, 128), (128, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (128, 1), 0), reinterpret_tensor(arg790_1, (128, 128), (1, 128), 0), out=buf448)
        del arg790_1
        buf449 = reinterpret_tensor(buf446, (128, 128), (128, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (128, 512), (512, 1), 0), reinterpret_tensor(arg792_1, (512, 128), (1, 512), 0), out=buf449)
        del arg792_1
        buf450 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf447, arg789_1, buf450, 16384, grid=grid(16384), stream=stream0)
        del arg789_1
        buf451 = reinterpret_tensor(buf447, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf448, arg791_1, buf451, 16384, grid=grid(16384), stream=stream0)
        del arg791_1
        buf452 = reinterpret_tensor(buf448, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf449, arg793_1, buf452, 16384, grid=grid(16384), stream=stream0)
        del arg793_1
        del buf449
        # Source Nodes: [], Original ATen: []
        buf453 = aten._scaled_dot_product_efficient_attention(buf450, buf451, buf452, None, False, scale=0.17677669529663687)
        buf454 = buf453[0]
        del buf453
        buf458 = reinterpret_tensor(buf452, (128, 128), (128, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf454, (128, 128), (128, 1), 0), reinterpret_tensor(arg794_1, (128, 128), (1, 128), 0), out=buf458)
        del arg794_1
        buf459 = reinterpret_tensor(buf454, (128, 128), (128, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (128, 512), (512, 1), 0), reinterpret_tensor(arg784_1, (512, 128), (1, 512), 0), out=buf459)
        del arg784_1
        buf460 = reinterpret_tensor(buf458, (1, 128, 128), (16384, 128, 1), 0); del buf458  # reuse
        # Source Nodes: [add_201, attention_output_65, layer_input_69, mul_106, mul_108], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf460, arg795_1, buf459, arg785_1, arg210_1, arg211_1, arg214_1, arg215_1, 16384, grid=grid(16384), stream=stream0)
        del arg210_1
        del arg211_1
        del arg214_1
        del arg215_1
        del arg785_1
        del arg795_1
        buf461 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (128, 128), (128, 1), 0), reinterpret_tensor(arg796_1, (128, 512), (1, 128), 0), out=buf461)
        del arg796_1
        buf462 = reinterpret_tensor(buf461, (1, 128, 512), (65536, 512, 1), 0); del buf461  # reuse
        # Source Nodes: [intermediate_output_52], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf462, arg797_1, 65536, grid=grid(65536), stream=stream0)
        del arg797_1
        buf463 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (128, 512), (512, 1), 0), reinterpret_tensor(arg798_1, (512, 128), (1, 512), 0), out=buf463)
        del arg798_1
        buf464 = buf460; del buf460  # reuse
        # Source Nodes: [add_203, attention_output_66, mul_109], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf464, buf463, arg799_1, arg216_1, arg217_1, 16384, grid=grid(16384), stream=stream0)
        del arg216_1
        del arg217_1
        del arg799_1
        buf465 = reinterpret_tensor(buf462, (128, 512), (512, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf464, (128, 128), (128, 1), 0), reinterpret_tensor(arg800_1, (128, 512), (1, 128), 0), out=buf465)
        del arg800_1
        buf466 = reinterpret_tensor(buf465, (1, 128, 512), (65536, 512, 1), 0); del buf465  # reuse
        # Source Nodes: [intermediate_output_53], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf466, arg801_1, 65536, grid=grid(65536), stream=stream0)
        del arg801_1
        buf467 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf466, (128, 512), (512, 1), 0), reinterpret_tensor(arg802_1, (512, 128), (1, 512), 0), out=buf467)
        del arg802_1
        buf468 = buf464; del buf464  # reuse
        # Source Nodes: [add_205, attention_output_67, mul_110], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf468, buf467, arg803_1, arg218_1, arg219_1, 16384, grid=grid(16384), stream=stream0)
        del arg218_1
        del arg219_1
        del arg803_1
        buf469 = reinterpret_tensor(buf466, (128, 512), (512, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf468, (128, 128), (128, 1), 0), reinterpret_tensor(arg804_1, (128, 512), (1, 128), 0), out=buf469)
        del arg804_1
        buf470 = reinterpret_tensor(buf469, (1, 128, 512), (65536, 512, 1), 0); del buf469  # reuse
        # Source Nodes: [intermediate_output_54], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf470, arg805_1, 65536, grid=grid(65536), stream=stream0)
        del arg805_1
        buf471 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf470, (128, 512), (512, 1), 0), reinterpret_tensor(arg806_1, (512, 128), (1, 512), 0), out=buf471)
        del arg806_1
        buf472 = buf468; del buf468  # reuse
        # Source Nodes: [add_207, attention_output_68, mul_111], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf472, buf471, arg807_1, arg220_1, arg221_1, 16384, grid=grid(16384), stream=stream0)
        del arg220_1
        del arg221_1
        del arg807_1
        buf473 = reinterpret_tensor(buf470, (128, 512), (512, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf472, (128, 128), (128, 1), 0), reinterpret_tensor(arg808_1, (128, 512), (1, 128), 0), out=buf473)
        del arg808_1
        buf474 = reinterpret_tensor(buf473, (1, 128, 512), (65536, 512, 1), 0); del buf473  # reuse
        # Source Nodes: [intermediate_output_55], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf474, arg809_1, 65536, grid=grid(65536), stream=stream0)
        del arg809_1
        buf475 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 512), (512, 1), 0), reinterpret_tensor(arg810_1, (512, 128), (1, 512), 0), out=buf475)
        del arg810_1
        buf476 = buf472; del buf472  # reuse
        # Source Nodes: [add_209, layer_output_53, mul_112], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf476, buf475, arg811_1, arg222_1, arg223_1, 16384, grid=grid(16384), stream=stream0)
        del arg222_1
        del arg223_1
        del arg811_1
        buf477 = reinterpret_tensor(buf474, (128, 512), (512, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (128, 128), (128, 1), 0), reinterpret_tensor(arg812_1, (128, 512), (1, 128), 0), out=buf477)
        del arg812_1
        buf478 = buf444; del buf444  # reuse
        # Source Nodes: [add_211, mul_113, value_tensor_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf478, buf477, arg813_1, arg224_1, arg225_1, 65536, grid=grid(65536), stream=stream0)
        del arg224_1
        del arg225_1
        del arg813_1
        buf479 = reinterpret_tensor(buf476, (128, 128), (128, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 512), (512, 1), 0), reinterpret_tensor(arg816_1, (512, 128), (1, 512), 0), out=buf479)
        del arg816_1
        buf480 = reinterpret_tensor(buf479, (1, 128, 128), (16384, 128, 1), 0); del buf479  # reuse
        # Source Nodes: [key_tensor_14, mul_115], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf480, arg817_1, arg228_1, arg229_1, 16384, grid=grid(16384), stream=stream0)
        del arg228_1
        del arg229_1
        del arg817_1
        buf481 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (128, 128), (128, 1), 0), reinterpret_tensor(arg818_1, (128, 128), (1, 128), 0), out=buf481)
        del arg818_1
        buf482 = reinterpret_tensor(buf451, (128, 128), (128, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (128, 128), (128, 1), 0), reinterpret_tensor(arg820_1, (128, 128), (1, 128), 0), out=buf482)
        del arg820_1
        buf483 = reinterpret_tensor(buf480, (128, 128), (128, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 512), (512, 1), 0), reinterpret_tensor(arg822_1, (512, 128), (1, 512), 0), out=buf483)
        del arg822_1
        buf484 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf481, arg819_1, buf484, 16384, grid=grid(16384), stream=stream0)
        del arg819_1
        buf485 = reinterpret_tensor(buf481, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf482, arg821_1, buf485, 16384, grid=grid(16384), stream=stream0)
        del arg821_1
        buf486 = reinterpret_tensor(buf482, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf483, arg823_1, buf486, 16384, grid=grid(16384), stream=stream0)
        del arg823_1
        del buf483
        # Source Nodes: [], Original ATen: []
        buf487 = aten._scaled_dot_product_efficient_attention(buf484, buf485, buf486, None, False, scale=0.17677669529663687)
        buf488 = buf487[0]
        del buf487
        buf492 = reinterpret_tensor(buf486, (128, 128), (128, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (128, 128), (128, 1), 0), reinterpret_tensor(arg824_1, (128, 128), (1, 128), 0), out=buf492)
        del arg824_1
        buf493 = reinterpret_tensor(buf488, (128, 128), (128, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 512), (512, 1), 0), reinterpret_tensor(arg814_1, (512, 128), (1, 512), 0), out=buf493)
        del arg814_1
        buf494 = reinterpret_tensor(buf492, (1, 128, 128), (16384, 128, 1), 0); del buf492  # reuse
        # Source Nodes: [add_216, attention_output_70, layer_input_74, mul_114, mul_116], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf494, arg825_1, buf493, arg815_1, arg226_1, arg227_1, arg230_1, arg231_1, 16384, grid=grid(16384), stream=stream0)
        del arg226_1
        del arg227_1
        del arg230_1
        del arg231_1
        del arg815_1
        del arg825_1
        buf495 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (128, 128), (128, 1), 0), reinterpret_tensor(arg826_1, (128, 512), (1, 128), 0), out=buf495)
        del arg826_1
        buf496 = reinterpret_tensor(buf495, (1, 128, 512), (65536, 512, 1), 0); del buf495  # reuse
        # Source Nodes: [intermediate_output_56], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf496, arg827_1, 65536, grid=grid(65536), stream=stream0)
        del arg827_1
        buf497 = buf493; del buf493  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 512), (512, 1), 0), reinterpret_tensor(arg828_1, (512, 128), (1, 512), 0), out=buf497)
        del arg828_1
        buf498 = buf494; del buf494  # reuse
        # Source Nodes: [add_218, attention_output_71, mul_117], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf498, buf497, arg829_1, arg232_1, arg233_1, 16384, grid=grid(16384), stream=stream0)
        del arg232_1
        del arg233_1
        del arg829_1
        buf499 = reinterpret_tensor(buf496, (128, 512), (512, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (128, 128), (128, 1), 0), reinterpret_tensor(arg830_1, (128, 512), (1, 128), 0), out=buf499)
        del arg830_1
        buf500 = reinterpret_tensor(buf499, (1, 128, 512), (65536, 512, 1), 0); del buf499  # reuse
        # Source Nodes: [intermediate_output_57], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf500, arg831_1, 65536, grid=grid(65536), stream=stream0)
        del arg831_1
        buf501 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (128, 512), (512, 1), 0), reinterpret_tensor(arg832_1, (512, 128), (1, 512), 0), out=buf501)
        del arg832_1
        buf502 = buf498; del buf498  # reuse
        # Source Nodes: [add_220, attention_output_72, mul_118], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf502, buf501, arg833_1, arg234_1, arg235_1, 16384, grid=grid(16384), stream=stream0)
        del arg234_1
        del arg235_1
        del arg833_1
        buf503 = reinterpret_tensor(buf500, (128, 512), (512, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (128, 128), (128, 1), 0), reinterpret_tensor(arg834_1, (128, 512), (1, 128), 0), out=buf503)
        del arg834_1
        buf504 = reinterpret_tensor(buf503, (1, 128, 512), (65536, 512, 1), 0); del buf503  # reuse
        # Source Nodes: [intermediate_output_58], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf504, arg835_1, 65536, grid=grid(65536), stream=stream0)
        del arg835_1
        buf505 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf504, (128, 512), (512, 1), 0), reinterpret_tensor(arg836_1, (512, 128), (1, 512), 0), out=buf505)
        del arg836_1
        buf506 = buf502; del buf502  # reuse
        # Source Nodes: [add_222, attention_output_73, mul_119], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf506, buf505, arg837_1, arg236_1, arg237_1, 16384, grid=grid(16384), stream=stream0)
        del arg236_1
        del arg237_1
        del arg837_1
        buf507 = reinterpret_tensor(buf504, (128, 512), (512, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf506, (128, 128), (128, 1), 0), reinterpret_tensor(arg838_1, (128, 512), (1, 128), 0), out=buf507)
        del arg838_1
        buf508 = reinterpret_tensor(buf507, (1, 128, 512), (65536, 512, 1), 0); del buf507  # reuse
        # Source Nodes: [intermediate_output_59], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf508, arg839_1, 65536, grid=grid(65536), stream=stream0)
        del arg839_1
        buf509 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (128, 512), (512, 1), 0), reinterpret_tensor(arg840_1, (512, 128), (1, 512), 0), out=buf509)
        del arg840_1
        buf510 = buf506; del buf506  # reuse
        # Source Nodes: [add_224, layer_output_57, mul_120], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf510, buf509, arg841_1, arg238_1, arg239_1, 16384, grid=grid(16384), stream=stream0)
        del arg238_1
        del arg239_1
        del arg841_1
        buf511 = reinterpret_tensor(buf508, (128, 512), (512, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf510, (128, 128), (128, 1), 0), reinterpret_tensor(arg842_1, (128, 512), (1, 128), 0), out=buf511)
        del arg842_1
        buf512 = buf478; del buf478  # reuse
        # Source Nodes: [add_226, mul_121, value_tensor_15], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf512, buf511, arg843_1, arg240_1, arg241_1, 65536, grid=grid(65536), stream=stream0)
        del arg240_1
        del arg241_1
        del arg843_1
        buf513 = reinterpret_tensor(buf510, (128, 128), (128, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (128, 512), (512, 1), 0), reinterpret_tensor(arg846_1, (512, 128), (1, 512), 0), out=buf513)
        del arg846_1
        buf514 = reinterpret_tensor(buf513, (1, 128, 128), (16384, 128, 1), 0); del buf513  # reuse
        # Source Nodes: [key_tensor_15, mul_123], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf514, arg847_1, arg244_1, arg245_1, 16384, grid=grid(16384), stream=stream0)
        del arg244_1
        del arg245_1
        del arg847_1
        buf515 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (128, 128), (128, 1), 0), reinterpret_tensor(arg848_1, (128, 128), (1, 128), 0), out=buf515)
        del arg848_1
        buf516 = reinterpret_tensor(buf485, (128, 128), (128, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (128, 128), (128, 1), 0), reinterpret_tensor(arg850_1, (128, 128), (1, 128), 0), out=buf516)
        del arg850_1
        buf517 = reinterpret_tensor(buf514, (128, 128), (128, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (128, 512), (512, 1), 0), reinterpret_tensor(arg852_1, (512, 128), (1, 512), 0), out=buf517)
        del arg852_1
        buf518 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf515, arg849_1, buf518, 16384, grid=grid(16384), stream=stream0)
        del arg849_1
        buf519 = reinterpret_tensor(buf515, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf516, arg851_1, buf519, 16384, grid=grid(16384), stream=stream0)
        del arg851_1
        buf520 = reinterpret_tensor(buf516, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf517, arg853_1, buf520, 16384, grid=grid(16384), stream=stream0)
        del arg853_1
        del buf517
        # Source Nodes: [], Original ATen: []
        buf521 = aten._scaled_dot_product_efficient_attention(buf518, buf519, buf520, None, False, scale=0.17677669529663687)
        buf522 = buf521[0]
        del buf521
        buf526 = reinterpret_tensor(buf520, (128, 128), (128, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (128, 128), (128, 1), 0), reinterpret_tensor(arg854_1, (128, 128), (1, 128), 0), out=buf526)
        del arg854_1
        buf527 = reinterpret_tensor(buf522, (128, 128), (128, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (128, 512), (512, 1), 0), reinterpret_tensor(arg844_1, (512, 128), (1, 512), 0), out=buf527)
        del arg844_1
        buf528 = reinterpret_tensor(buf526, (1, 128, 128), (16384, 128, 1), 0); del buf526  # reuse
        # Source Nodes: [add_231, attention_output_75, layer_input_79, mul_122, mul_124], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf528, arg855_1, buf527, arg845_1, arg242_1, arg243_1, arg246_1, arg247_1, 16384, grid=grid(16384), stream=stream0)
        del arg242_1
        del arg243_1
        del arg246_1
        del arg247_1
        del arg845_1
        del arg855_1
        buf529 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (128, 128), (128, 1), 0), reinterpret_tensor(arg856_1, (128, 512), (1, 128), 0), out=buf529)
        del arg856_1
        buf530 = reinterpret_tensor(buf529, (1, 128, 512), (65536, 512, 1), 0); del buf529  # reuse
        # Source Nodes: [intermediate_output_60], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf530, arg857_1, 65536, grid=grid(65536), stream=stream0)
        del arg857_1
        buf531 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf530, (128, 512), (512, 1), 0), reinterpret_tensor(arg858_1, (512, 128), (1, 512), 0), out=buf531)
        del arg858_1
        buf532 = buf528; del buf528  # reuse
        # Source Nodes: [add_233, attention_output_76, mul_125], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf532, buf531, arg859_1, arg248_1, arg249_1, 16384, grid=grid(16384), stream=stream0)
        del arg248_1
        del arg249_1
        del arg859_1
        buf533 = reinterpret_tensor(buf530, (128, 512), (512, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (128, 128), (128, 1), 0), reinterpret_tensor(arg860_1, (128, 512), (1, 128), 0), out=buf533)
        del arg860_1
        buf534 = reinterpret_tensor(buf533, (1, 128, 512), (65536, 512, 1), 0); del buf533  # reuse
        # Source Nodes: [intermediate_output_61], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf534, arg861_1, 65536, grid=grid(65536), stream=stream0)
        del arg861_1
        buf535 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf534, (128, 512), (512, 1), 0), reinterpret_tensor(arg862_1, (512, 128), (1, 512), 0), out=buf535)
        del arg862_1
        buf536 = buf532; del buf532  # reuse
        # Source Nodes: [add_235, attention_output_77, mul_126], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf536, buf535, arg863_1, arg250_1, arg251_1, 16384, grid=grid(16384), stream=stream0)
        del arg250_1
        del arg251_1
        del arg863_1
        buf537 = reinterpret_tensor(buf534, (128, 512), (512, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf536, (128, 128), (128, 1), 0), reinterpret_tensor(arg864_1, (128, 512), (1, 128), 0), out=buf537)
        del arg864_1
        buf538 = reinterpret_tensor(buf537, (1, 128, 512), (65536, 512, 1), 0); del buf537  # reuse
        # Source Nodes: [intermediate_output_62], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf538, arg865_1, 65536, grid=grid(65536), stream=stream0)
        del arg865_1
        buf539 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf538, (128, 512), (512, 1), 0), reinterpret_tensor(arg866_1, (512, 128), (1, 512), 0), out=buf539)
        del arg866_1
        buf540 = buf536; del buf536  # reuse
        # Source Nodes: [add_237, attention_output_78, mul_127], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf540, buf539, arg867_1, arg252_1, arg253_1, 16384, grid=grid(16384), stream=stream0)
        del arg252_1
        del arg253_1
        del arg867_1
        buf541 = reinterpret_tensor(buf538, (128, 512), (512, 1), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (128, 128), (128, 1), 0), reinterpret_tensor(arg868_1, (128, 512), (1, 128), 0), out=buf541)
        del arg868_1
        buf542 = reinterpret_tensor(buf541, (1, 128, 512), (65536, 512, 1), 0); del buf541  # reuse
        # Source Nodes: [intermediate_output_63], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf542, arg869_1, 65536, grid=grid(65536), stream=stream0)
        del arg869_1
        buf543 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf542, (128, 512), (512, 1), 0), reinterpret_tensor(arg870_1, (512, 128), (1, 512), 0), out=buf543)
        del arg870_1
        buf544 = buf540; del buf540  # reuse
        # Source Nodes: [add_239, layer_output_61, mul_128], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf544, buf543, arg871_1, arg254_1, arg255_1, 16384, grid=grid(16384), stream=stream0)
        del arg254_1
        del arg255_1
        del arg871_1
        buf545 = reinterpret_tensor(buf542, (128, 512), (512, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf544, (128, 128), (128, 1), 0), reinterpret_tensor(arg872_1, (128, 512), (1, 128), 0), out=buf545)
        del arg872_1
        buf546 = buf512; del buf512  # reuse
        # Source Nodes: [add_241, mul_129, value_tensor_16], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf546, buf545, arg873_1, arg256_1, arg257_1, 65536, grid=grid(65536), stream=stream0)
        del arg256_1
        del arg257_1
        del arg873_1
        buf547 = reinterpret_tensor(buf544, (128, 128), (128, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 512), (512, 1), 0), reinterpret_tensor(arg876_1, (512, 128), (1, 512), 0), out=buf547)
        del arg876_1
        buf548 = reinterpret_tensor(buf547, (1, 128, 128), (16384, 128, 1), 0); del buf547  # reuse
        # Source Nodes: [key_tensor_16, mul_131], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf548, arg877_1, arg260_1, arg261_1, 16384, grid=grid(16384), stream=stream0)
        del arg260_1
        del arg261_1
        del arg877_1
        buf549 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (128, 128), (128, 1), 0), reinterpret_tensor(arg878_1, (128, 128), (1, 128), 0), out=buf549)
        del arg878_1
        buf550 = reinterpret_tensor(buf519, (128, 128), (128, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (128, 128), (128, 1), 0), reinterpret_tensor(arg880_1, (128, 128), (1, 128), 0), out=buf550)
        del arg880_1
        buf551 = reinterpret_tensor(buf548, (128, 128), (128, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 512), (512, 1), 0), reinterpret_tensor(arg882_1, (512, 128), (1, 512), 0), out=buf551)
        del arg882_1
        buf552 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf549, arg879_1, buf552, 16384, grid=grid(16384), stream=stream0)
        del arg879_1
        buf553 = reinterpret_tensor(buf549, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf550, arg881_1, buf553, 16384, grid=grid(16384), stream=stream0)
        del arg881_1
        buf554 = reinterpret_tensor(buf550, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf551, arg883_1, buf554, 16384, grid=grid(16384), stream=stream0)
        del arg883_1
        del buf551
        # Source Nodes: [], Original ATen: []
        buf555 = aten._scaled_dot_product_efficient_attention(buf552, buf553, buf554, None, False, scale=0.17677669529663687)
        buf556 = buf555[0]
        del buf555
        buf560 = reinterpret_tensor(buf554, (128, 128), (128, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf556, (128, 128), (128, 1), 0), reinterpret_tensor(arg884_1, (128, 128), (1, 128), 0), out=buf560)
        del arg884_1
        buf561 = reinterpret_tensor(buf556, (128, 128), (128, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 512), (512, 1), 0), reinterpret_tensor(arg874_1, (512, 128), (1, 512), 0), out=buf561)
        del arg874_1
        buf562 = reinterpret_tensor(buf560, (1, 128, 128), (16384, 128, 1), 0); del buf560  # reuse
        # Source Nodes: [add_246, attention_output_80, layer_input_84, mul_130, mul_132], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf562, arg885_1, buf561, arg875_1, arg258_1, arg259_1, arg262_1, arg263_1, 16384, grid=grid(16384), stream=stream0)
        del arg258_1
        del arg259_1
        del arg262_1
        del arg263_1
        del arg875_1
        del arg885_1
        buf563 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (128, 128), (128, 1), 0), reinterpret_tensor(arg886_1, (128, 512), (1, 128), 0), out=buf563)
        del arg886_1
        buf564 = reinterpret_tensor(buf563, (1, 128, 512), (65536, 512, 1), 0); del buf563  # reuse
        # Source Nodes: [intermediate_output_64], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf564, arg887_1, 65536, grid=grid(65536), stream=stream0)
        del arg887_1
        buf565 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (128, 512), (512, 1), 0), reinterpret_tensor(arg888_1, (512, 128), (1, 512), 0), out=buf565)
        del arg888_1
        buf566 = buf562; del buf562  # reuse
        # Source Nodes: [add_248, attention_output_81, mul_133], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf566, buf565, arg889_1, arg264_1, arg265_1, 16384, grid=grid(16384), stream=stream0)
        del arg264_1
        del arg265_1
        del arg889_1
        buf567 = reinterpret_tensor(buf564, (128, 512), (512, 1), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf566, (128, 128), (128, 1), 0), reinterpret_tensor(arg890_1, (128, 512), (1, 128), 0), out=buf567)
        del arg890_1
        buf568 = reinterpret_tensor(buf567, (1, 128, 512), (65536, 512, 1), 0); del buf567  # reuse
        # Source Nodes: [intermediate_output_65], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf568, arg891_1, 65536, grid=grid(65536), stream=stream0)
        del arg891_1
        buf569 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (128, 512), (512, 1), 0), reinterpret_tensor(arg892_1, (512, 128), (1, 512), 0), out=buf569)
        del arg892_1
        buf570 = buf566; del buf566  # reuse
        # Source Nodes: [add_250, attention_output_82, mul_134], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf570, buf569, arg893_1, arg266_1, arg267_1, 16384, grid=grid(16384), stream=stream0)
        del arg266_1
        del arg267_1
        del arg893_1
        buf571 = reinterpret_tensor(buf568, (128, 512), (512, 1), 0); del buf568  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (128, 128), (128, 1), 0), reinterpret_tensor(arg894_1, (128, 512), (1, 128), 0), out=buf571)
        del arg894_1
        buf572 = reinterpret_tensor(buf571, (1, 128, 512), (65536, 512, 1), 0); del buf571  # reuse
        # Source Nodes: [intermediate_output_66], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf572, arg895_1, 65536, grid=grid(65536), stream=stream0)
        del arg895_1
        buf573 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (128, 512), (512, 1), 0), reinterpret_tensor(arg896_1, (512, 128), (1, 512), 0), out=buf573)
        del arg896_1
        buf574 = buf570; del buf570  # reuse
        # Source Nodes: [add_252, attention_output_83, mul_135], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf574, buf573, arg897_1, arg268_1, arg269_1, 16384, grid=grid(16384), stream=stream0)
        del arg268_1
        del arg269_1
        del arg897_1
        buf575 = reinterpret_tensor(buf572, (128, 512), (512, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf574, (128, 128), (128, 1), 0), reinterpret_tensor(arg898_1, (128, 512), (1, 128), 0), out=buf575)
        del arg898_1
        buf576 = reinterpret_tensor(buf575, (1, 128, 512), (65536, 512, 1), 0); del buf575  # reuse
        # Source Nodes: [intermediate_output_67], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf576, arg899_1, 65536, grid=grid(65536), stream=stream0)
        del arg899_1
        buf577 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf576, (128, 512), (512, 1), 0), reinterpret_tensor(arg900_1, (512, 128), (1, 512), 0), out=buf577)
        del arg900_1
        buf578 = buf574; del buf574  # reuse
        # Source Nodes: [add_254, layer_output_65, mul_136], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf578, buf577, arg901_1, arg270_1, arg271_1, 16384, grid=grid(16384), stream=stream0)
        del arg270_1
        del arg271_1
        del arg901_1
        buf579 = reinterpret_tensor(buf576, (128, 512), (512, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (128, 128), (128, 1), 0), reinterpret_tensor(arg902_1, (128, 512), (1, 128), 0), out=buf579)
        del arg902_1
        buf580 = buf546; del buf546  # reuse
        # Source Nodes: [add_256, mul_137, value_tensor_17], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf580, buf579, arg903_1, arg272_1, arg273_1, 65536, grid=grid(65536), stream=stream0)
        del arg272_1
        del arg273_1
        del arg903_1
        buf581 = reinterpret_tensor(buf578, (128, 128), (128, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (128, 512), (512, 1), 0), reinterpret_tensor(arg906_1, (512, 128), (1, 512), 0), out=buf581)
        del arg906_1
        buf582 = reinterpret_tensor(buf581, (1, 128, 128), (16384, 128, 1), 0); del buf581  # reuse
        # Source Nodes: [key_tensor_17, mul_139], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf582, arg907_1, arg276_1, arg277_1, 16384, grid=grid(16384), stream=stream0)
        del arg276_1
        del arg277_1
        del arg907_1
        buf583 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (128, 128), (128, 1), 0), reinterpret_tensor(arg908_1, (128, 128), (1, 128), 0), out=buf583)
        del arg908_1
        buf584 = reinterpret_tensor(buf553, (128, 128), (128, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (128, 128), (128, 1), 0), reinterpret_tensor(arg910_1, (128, 128), (1, 128), 0), out=buf584)
        del arg910_1
        buf585 = reinterpret_tensor(buf582, (128, 128), (128, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (128, 512), (512, 1), 0), reinterpret_tensor(arg912_1, (512, 128), (1, 512), 0), out=buf585)
        del arg912_1
        buf586 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf583, arg909_1, buf586, 16384, grid=grid(16384), stream=stream0)
        del arg909_1
        buf587 = reinterpret_tensor(buf583, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf584, arg911_1, buf587, 16384, grid=grid(16384), stream=stream0)
        del arg911_1
        buf588 = reinterpret_tensor(buf584, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf585, arg913_1, buf588, 16384, grid=grid(16384), stream=stream0)
        del arg913_1
        del buf585
        # Source Nodes: [], Original ATen: []
        buf589 = aten._scaled_dot_product_efficient_attention(buf586, buf587, buf588, None, False, scale=0.17677669529663687)
        buf590 = buf589[0]
        del buf589
        buf594 = reinterpret_tensor(buf588, (128, 128), (128, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf590, (128, 128), (128, 1), 0), reinterpret_tensor(arg914_1, (128, 128), (1, 128), 0), out=buf594)
        del arg914_1
        buf595 = reinterpret_tensor(buf590, (128, 128), (128, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (128, 512), (512, 1), 0), reinterpret_tensor(arg904_1, (512, 128), (1, 512), 0), out=buf595)
        del arg904_1
        buf596 = reinterpret_tensor(buf594, (1, 128, 128), (16384, 128, 1), 0); del buf594  # reuse
        # Source Nodes: [add_261, attention_output_85, layer_input_89, mul_138, mul_140], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf596, arg915_1, buf595, arg905_1, arg274_1, arg275_1, arg278_1, arg279_1, 16384, grid=grid(16384), stream=stream0)
        del arg274_1
        del arg275_1
        del arg278_1
        del arg279_1
        del arg905_1
        del arg915_1
        buf597 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (128, 128), (128, 1), 0), reinterpret_tensor(arg916_1, (128, 512), (1, 128), 0), out=buf597)
        del arg916_1
        buf598 = reinterpret_tensor(buf597, (1, 128, 512), (65536, 512, 1), 0); del buf597  # reuse
        # Source Nodes: [intermediate_output_68], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf598, arg917_1, 65536, grid=grid(65536), stream=stream0)
        del arg917_1
        buf599 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (128, 512), (512, 1), 0), reinterpret_tensor(arg918_1, (512, 128), (1, 512), 0), out=buf599)
        del arg918_1
        buf600 = buf596; del buf596  # reuse
        # Source Nodes: [add_263, attention_output_86, mul_141], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf600, buf599, arg919_1, arg280_1, arg281_1, 16384, grid=grid(16384), stream=stream0)
        del arg280_1
        del arg281_1
        del arg919_1
        buf601 = reinterpret_tensor(buf598, (128, 512), (512, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf600, (128, 128), (128, 1), 0), reinterpret_tensor(arg920_1, (128, 512), (1, 128), 0), out=buf601)
        del arg920_1
        buf602 = reinterpret_tensor(buf601, (1, 128, 512), (65536, 512, 1), 0); del buf601  # reuse
        # Source Nodes: [intermediate_output_69], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf602, arg921_1, 65536, grid=grid(65536), stream=stream0)
        del arg921_1
        buf603 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf602, (128, 512), (512, 1), 0), reinterpret_tensor(arg922_1, (512, 128), (1, 512), 0), out=buf603)
        del arg922_1
        buf604 = buf600; del buf600  # reuse
        # Source Nodes: [add_265, attention_output_87, mul_142], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf604, buf603, arg923_1, arg282_1, arg283_1, 16384, grid=grid(16384), stream=stream0)
        del arg282_1
        del arg283_1
        del arg923_1
        buf605 = reinterpret_tensor(buf602, (128, 512), (512, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (128, 128), (128, 1), 0), reinterpret_tensor(arg924_1, (128, 512), (1, 128), 0), out=buf605)
        del arg924_1
        buf606 = reinterpret_tensor(buf605, (1, 128, 512), (65536, 512, 1), 0); del buf605  # reuse
        # Source Nodes: [intermediate_output_70], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf606, arg925_1, 65536, grid=grid(65536), stream=stream0)
        del arg925_1
        buf607 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf606, (128, 512), (512, 1), 0), reinterpret_tensor(arg926_1, (512, 128), (1, 512), 0), out=buf607)
        del arg926_1
        buf608 = buf604; del buf604  # reuse
        # Source Nodes: [add_267, attention_output_88, mul_143], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf608, buf607, arg927_1, arg284_1, arg285_1, 16384, grid=grid(16384), stream=stream0)
        del arg284_1
        del arg285_1
        del arg927_1
        buf609 = reinterpret_tensor(buf606, (128, 512), (512, 1), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf608, (128, 128), (128, 1), 0), reinterpret_tensor(arg928_1, (128, 512), (1, 128), 0), out=buf609)
        del arg928_1
        buf610 = reinterpret_tensor(buf609, (1, 128, 512), (65536, 512, 1), 0); del buf609  # reuse
        # Source Nodes: [intermediate_output_71], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf610, arg929_1, 65536, grid=grid(65536), stream=stream0)
        del arg929_1
        buf611 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf610, (128, 512), (512, 1), 0), reinterpret_tensor(arg930_1, (512, 128), (1, 512), 0), out=buf611)
        del arg930_1
        buf612 = buf608; del buf608  # reuse
        # Source Nodes: [add_269, layer_output_69, mul_144], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf612, buf611, arg931_1, arg286_1, arg287_1, 16384, grid=grid(16384), stream=stream0)
        del arg286_1
        del arg287_1
        del arg931_1
        buf613 = reinterpret_tensor(buf610, (128, 512), (512, 1), 0); del buf610  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf612, (128, 128), (128, 1), 0), reinterpret_tensor(arg932_1, (128, 512), (1, 128), 0), out=buf613)
        del arg932_1
        buf614 = buf580; del buf580  # reuse
        # Source Nodes: [add_271, mul_145, value_tensor_18], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf614, buf613, arg933_1, arg288_1, arg289_1, 65536, grid=grid(65536), stream=stream0)
        del arg288_1
        del arg289_1
        del arg933_1
        buf615 = reinterpret_tensor(buf612, (128, 128), (128, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (128, 512), (512, 1), 0), reinterpret_tensor(arg936_1, (512, 128), (1, 512), 0), out=buf615)
        del arg936_1
        buf616 = reinterpret_tensor(buf615, (1, 128, 128), (16384, 128, 1), 0); del buf615  # reuse
        # Source Nodes: [key_tensor_18, mul_147], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf616, arg937_1, arg292_1, arg293_1, 16384, grid=grid(16384), stream=stream0)
        del arg292_1
        del arg293_1
        del arg937_1
        buf617 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf616, (128, 128), (128, 1), 0), reinterpret_tensor(arg938_1, (128, 128), (1, 128), 0), out=buf617)
        del arg938_1
        buf618 = reinterpret_tensor(buf587, (128, 128), (128, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf616, (128, 128), (128, 1), 0), reinterpret_tensor(arg940_1, (128, 128), (1, 128), 0), out=buf618)
        del arg940_1
        buf619 = reinterpret_tensor(buf616, (128, 128), (128, 1), 0); del buf616  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (128, 512), (512, 1), 0), reinterpret_tensor(arg942_1, (512, 128), (1, 512), 0), out=buf619)
        del arg942_1
        buf620 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf617, arg939_1, buf620, 16384, grid=grid(16384), stream=stream0)
        del arg939_1
        buf621 = reinterpret_tensor(buf617, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf618, arg941_1, buf621, 16384, grid=grid(16384), stream=stream0)
        del arg941_1
        buf622 = reinterpret_tensor(buf618, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf619, arg943_1, buf622, 16384, grid=grid(16384), stream=stream0)
        del arg943_1
        del buf619
        # Source Nodes: [], Original ATen: []
        buf623 = aten._scaled_dot_product_efficient_attention(buf620, buf621, buf622, None, False, scale=0.17677669529663687)
        buf624 = buf623[0]
        del buf623
        buf628 = reinterpret_tensor(buf622, (128, 128), (128, 1), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf624, (128, 128), (128, 1), 0), reinterpret_tensor(arg944_1, (128, 128), (1, 128), 0), out=buf628)
        del arg944_1
        buf629 = reinterpret_tensor(buf624, (128, 128), (128, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (128, 512), (512, 1), 0), reinterpret_tensor(arg934_1, (512, 128), (1, 512), 0), out=buf629)
        del arg934_1
        buf630 = reinterpret_tensor(buf628, (1, 128, 128), (16384, 128, 1), 0); del buf628  # reuse
        # Source Nodes: [add_276, attention_output_90, layer_input_94, mul_146, mul_148], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf630, arg945_1, buf629, arg935_1, arg290_1, arg291_1, arg294_1, arg295_1, 16384, grid=grid(16384), stream=stream0)
        del arg290_1
        del arg291_1
        del arg294_1
        del arg295_1
        del arg935_1
        del arg945_1
        buf631 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf630, (128, 128), (128, 1), 0), reinterpret_tensor(arg946_1, (128, 512), (1, 128), 0), out=buf631)
        del arg946_1
        buf632 = reinterpret_tensor(buf631, (1, 128, 512), (65536, 512, 1), 0); del buf631  # reuse
        # Source Nodes: [intermediate_output_72], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf632, arg947_1, 65536, grid=grid(65536), stream=stream0)
        del arg947_1
        buf633 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf632, (128, 512), (512, 1), 0), reinterpret_tensor(arg948_1, (512, 128), (1, 512), 0), out=buf633)
        del arg948_1
        buf634 = buf630; del buf630  # reuse
        # Source Nodes: [add_278, attention_output_91, mul_149], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf634, buf633, arg949_1, arg296_1, arg297_1, 16384, grid=grid(16384), stream=stream0)
        del arg296_1
        del arg297_1
        del arg949_1
        buf635 = reinterpret_tensor(buf632, (128, 512), (512, 1), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf634, (128, 128), (128, 1), 0), reinterpret_tensor(arg950_1, (128, 512), (1, 128), 0), out=buf635)
        del arg950_1
        buf636 = reinterpret_tensor(buf635, (1, 128, 512), (65536, 512, 1), 0); del buf635  # reuse
        # Source Nodes: [intermediate_output_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf636, arg951_1, 65536, grid=grid(65536), stream=stream0)
        del arg951_1
        buf637 = buf633; del buf633  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf636, (128, 512), (512, 1), 0), reinterpret_tensor(arg952_1, (512, 128), (1, 512), 0), out=buf637)
        del arg952_1
        buf638 = buf634; del buf634  # reuse
        # Source Nodes: [add_280, attention_output_92, mul_150], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf638, buf637, arg953_1, arg298_1, arg299_1, 16384, grid=grid(16384), stream=stream0)
        del arg298_1
        del arg299_1
        del arg953_1
        buf639 = reinterpret_tensor(buf636, (128, 512), (512, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf638, (128, 128), (128, 1), 0), reinterpret_tensor(arg954_1, (128, 512), (1, 128), 0), out=buf639)
        del arg954_1
        buf640 = reinterpret_tensor(buf639, (1, 128, 512), (65536, 512, 1), 0); del buf639  # reuse
        # Source Nodes: [intermediate_output_74], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf640, arg955_1, 65536, grid=grid(65536), stream=stream0)
        del arg955_1
        buf641 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (128, 512), (512, 1), 0), reinterpret_tensor(arg956_1, (512, 128), (1, 512), 0), out=buf641)
        del arg956_1
        buf642 = buf638; del buf638  # reuse
        # Source Nodes: [add_282, attention_output_93, mul_151], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf642, buf641, arg957_1, arg300_1, arg301_1, 16384, grid=grid(16384), stream=stream0)
        del arg300_1
        del arg301_1
        del arg957_1
        buf643 = reinterpret_tensor(buf640, (128, 512), (512, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf642, (128, 128), (128, 1), 0), reinterpret_tensor(arg958_1, (128, 512), (1, 128), 0), out=buf643)
        del arg958_1
        buf644 = reinterpret_tensor(buf643, (1, 128, 512), (65536, 512, 1), 0); del buf643  # reuse
        # Source Nodes: [intermediate_output_75], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf644, arg959_1, 65536, grid=grid(65536), stream=stream0)
        del arg959_1
        buf645 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (128, 512), (512, 1), 0), reinterpret_tensor(arg960_1, (512, 128), (1, 512), 0), out=buf645)
        del arg960_1
        buf646 = buf642; del buf642  # reuse
        # Source Nodes: [add_284, layer_output_73, mul_152], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf646, buf645, arg961_1, arg302_1, arg303_1, 16384, grid=grid(16384), stream=stream0)
        del arg302_1
        del arg303_1
        del arg961_1
        buf647 = reinterpret_tensor(buf644, (128, 512), (512, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf646, (128, 128), (128, 1), 0), reinterpret_tensor(arg962_1, (128, 512), (1, 128), 0), out=buf647)
        del arg962_1
        buf648 = buf614; del buf614  # reuse
        # Source Nodes: [add_286, mul_153, value_tensor_19], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf648, buf647, arg963_1, arg304_1, arg305_1, 65536, grid=grid(65536), stream=stream0)
        del arg304_1
        del arg305_1
        del arg963_1
        buf649 = reinterpret_tensor(buf646, (128, 128), (128, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 512), (512, 1), 0), reinterpret_tensor(arg966_1, (512, 128), (1, 512), 0), out=buf649)
        del arg966_1
        buf650 = reinterpret_tensor(buf649, (1, 128, 128), (16384, 128, 1), 0); del buf649  # reuse
        # Source Nodes: [key_tensor_19, mul_155], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf650, arg967_1, arg308_1, arg309_1, 16384, grid=grid(16384), stream=stream0)
        del arg308_1
        del arg309_1
        del arg967_1
        buf651 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf650, (128, 128), (128, 1), 0), reinterpret_tensor(arg968_1, (128, 128), (1, 128), 0), out=buf651)
        del arg968_1
        buf652 = reinterpret_tensor(buf621, (128, 128), (128, 1), 0); del buf621  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf650, (128, 128), (128, 1), 0), reinterpret_tensor(arg970_1, (128, 128), (1, 128), 0), out=buf652)
        del arg970_1
        buf653 = reinterpret_tensor(buf650, (128, 128), (128, 1), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 512), (512, 1), 0), reinterpret_tensor(arg972_1, (512, 128), (1, 512), 0), out=buf653)
        del arg972_1
        buf654 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf651, arg969_1, buf654, 16384, grid=grid(16384), stream=stream0)
        del arg969_1
        buf655 = reinterpret_tensor(buf651, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf652, arg971_1, buf655, 16384, grid=grid(16384), stream=stream0)
        del arg971_1
        buf656 = reinterpret_tensor(buf652, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf653, arg973_1, buf656, 16384, grid=grid(16384), stream=stream0)
        del arg973_1
        del buf653
        # Source Nodes: [], Original ATen: []
        buf657 = aten._scaled_dot_product_efficient_attention(buf654, buf655, buf656, None, False, scale=0.17677669529663687)
        buf658 = buf657[0]
        del buf657
        buf662 = reinterpret_tensor(buf656, (128, 128), (128, 1), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf658, (128, 128), (128, 1), 0), reinterpret_tensor(arg974_1, (128, 128), (1, 128), 0), out=buf662)
        del arg974_1
        buf663 = reinterpret_tensor(buf658, (128, 128), (128, 1), 0); del buf658  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 512), (512, 1), 0), reinterpret_tensor(arg964_1, (512, 128), (1, 512), 0), out=buf663)
        del arg964_1
        buf664 = reinterpret_tensor(buf662, (1, 128, 128), (16384, 128, 1), 0); del buf662  # reuse
        # Source Nodes: [add_291, attention_output_95, layer_input_99, mul_154, mul_156], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf664, arg975_1, buf663, arg965_1, arg306_1, arg307_1, arg310_1, arg311_1, 16384, grid=grid(16384), stream=stream0)
        del arg306_1
        del arg307_1
        del arg310_1
        del arg311_1
        del arg965_1
        del arg975_1
        buf665 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (128, 128), (128, 1), 0), reinterpret_tensor(arg976_1, (128, 512), (1, 128), 0), out=buf665)
        del arg976_1
        buf666 = reinterpret_tensor(buf665, (1, 128, 512), (65536, 512, 1), 0); del buf665  # reuse
        # Source Nodes: [intermediate_output_76], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf666, arg977_1, 65536, grid=grid(65536), stream=stream0)
        del arg977_1
        buf667 = buf663; del buf663  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf666, (128, 512), (512, 1), 0), reinterpret_tensor(arg978_1, (512, 128), (1, 512), 0), out=buf667)
        del arg978_1
        buf668 = buf664; del buf664  # reuse
        # Source Nodes: [add_293, attention_output_96, mul_157], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf668, buf667, arg979_1, arg312_1, arg313_1, 16384, grid=grid(16384), stream=stream0)
        del arg312_1
        del arg313_1
        del arg979_1
        buf669 = reinterpret_tensor(buf666, (128, 512), (512, 1), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf668, (128, 128), (128, 1), 0), reinterpret_tensor(arg980_1, (128, 512), (1, 128), 0), out=buf669)
        del arg980_1
        buf670 = reinterpret_tensor(buf669, (1, 128, 512), (65536, 512, 1), 0); del buf669  # reuse
        # Source Nodes: [intermediate_output_77], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf670, arg981_1, 65536, grid=grid(65536), stream=stream0)
        del arg981_1
        buf671 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf670, (128, 512), (512, 1), 0), reinterpret_tensor(arg982_1, (512, 128), (1, 512), 0), out=buf671)
        del arg982_1
        buf672 = buf668; del buf668  # reuse
        # Source Nodes: [add_295, attention_output_97, mul_158], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf672, buf671, arg983_1, arg314_1, arg315_1, 16384, grid=grid(16384), stream=stream0)
        del arg314_1
        del arg315_1
        del arg983_1
        buf673 = reinterpret_tensor(buf670, (128, 512), (512, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (128, 128), (128, 1), 0), reinterpret_tensor(arg984_1, (128, 512), (1, 128), 0), out=buf673)
        del arg984_1
        buf674 = reinterpret_tensor(buf673, (1, 128, 512), (65536, 512, 1), 0); del buf673  # reuse
        # Source Nodes: [intermediate_output_78], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf674, arg985_1, 65536, grid=grid(65536), stream=stream0)
        del arg985_1
        buf675 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf674, (128, 512), (512, 1), 0), reinterpret_tensor(arg986_1, (512, 128), (1, 512), 0), out=buf675)
        del arg986_1
        buf676 = buf672; del buf672  # reuse
        # Source Nodes: [add_297, attention_output_98, mul_159], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf676, buf675, arg987_1, arg316_1, arg317_1, 16384, grid=grid(16384), stream=stream0)
        del arg316_1
        del arg317_1
        del arg987_1
        buf677 = reinterpret_tensor(buf674, (128, 512), (512, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf676, (128, 128), (128, 1), 0), reinterpret_tensor(arg988_1, (128, 512), (1, 128), 0), out=buf677)
        del arg988_1
        buf678 = reinterpret_tensor(buf677, (1, 128, 512), (65536, 512, 1), 0); del buf677  # reuse
        # Source Nodes: [intermediate_output_79], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf678, arg989_1, 65536, grid=grid(65536), stream=stream0)
        del arg989_1
        buf679 = buf675; del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf678, (128, 512), (512, 1), 0), reinterpret_tensor(arg990_1, (512, 128), (1, 512), 0), out=buf679)
        del arg990_1
        buf680 = buf676; del buf676  # reuse
        # Source Nodes: [add_299, layer_output_77, mul_160], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf680, buf679, arg991_1, arg318_1, arg319_1, 16384, grid=grid(16384), stream=stream0)
        del arg318_1
        del arg319_1
        del arg991_1
        buf681 = reinterpret_tensor(buf678, (128, 512), (512, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf680, (128, 128), (128, 1), 0), reinterpret_tensor(arg992_1, (128, 512), (1, 128), 0), out=buf681)
        del arg992_1
        buf682 = buf648; del buf648  # reuse
        # Source Nodes: [add_301, mul_161, value_tensor_20], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf682, buf681, arg993_1, arg320_1, arg321_1, 65536, grid=grid(65536), stream=stream0)
        del arg320_1
        del arg321_1
        del arg993_1
        buf683 = reinterpret_tensor(buf680, (128, 128), (128, 1), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (128, 512), (512, 1), 0), reinterpret_tensor(arg996_1, (512, 128), (1, 512), 0), out=buf683)
        del arg996_1
        buf684 = reinterpret_tensor(buf683, (1, 128, 128), (16384, 128, 1), 0); del buf683  # reuse
        # Source Nodes: [key_tensor_20, mul_163], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf684, arg997_1, arg324_1, arg325_1, 16384, grid=grid(16384), stream=stream0)
        del arg324_1
        del arg325_1
        del arg997_1
        buf685 = buf679; del buf679  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (128, 128), (128, 1), 0), reinterpret_tensor(arg998_1, (128, 128), (1, 128), 0), out=buf685)
        del arg998_1
        buf686 = reinterpret_tensor(buf655, (128, 128), (128, 1), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (128, 128), (128, 1), 0), reinterpret_tensor(arg1000_1, (128, 128), (1, 128), 0), out=buf686)
        del arg1000_1
        buf687 = reinterpret_tensor(buf684, (128, 128), (128, 1), 0); del buf684  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (128, 512), (512, 1), 0), reinterpret_tensor(arg1002_1, (512, 128), (1, 512), 0), out=buf687)
        del arg1002_1
        buf688 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf685, arg999_1, buf688, 16384, grid=grid(16384), stream=stream0)
        del arg999_1
        buf689 = reinterpret_tensor(buf685, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf686, arg1001_1, buf689, 16384, grid=grid(16384), stream=stream0)
        del arg1001_1
        buf690 = reinterpret_tensor(buf686, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf686  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf687, arg1003_1, buf690, 16384, grid=grid(16384), stream=stream0)
        del arg1003_1
        del buf687
        # Source Nodes: [], Original ATen: []
        buf691 = aten._scaled_dot_product_efficient_attention(buf688, buf689, buf690, None, False, scale=0.17677669529663687)
        buf692 = buf691[0]
        del buf691
        buf696 = reinterpret_tensor(buf690, (128, 128), (128, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf692, (128, 128), (128, 1), 0), reinterpret_tensor(arg1004_1, (128, 128), (1, 128), 0), out=buf696)
        del arg1004_1
        buf697 = reinterpret_tensor(buf692, (128, 128), (128, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (128, 512), (512, 1), 0), reinterpret_tensor(arg994_1, (512, 128), (1, 512), 0), out=buf697)
        del arg994_1
        buf698 = reinterpret_tensor(buf696, (1, 128, 128), (16384, 128, 1), 0); del buf696  # reuse
        # Source Nodes: [add_306, attention_output_100, layer_input_104, mul_162, mul_164], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf698, arg1005_1, buf697, arg995_1, arg322_1, arg323_1, arg326_1, arg327_1, 16384, grid=grid(16384), stream=stream0)
        del arg1005_1
        del arg322_1
        del arg323_1
        del arg326_1
        del arg327_1
        del arg995_1
        buf699 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf698, (128, 128), (128, 1), 0), reinterpret_tensor(arg1006_1, (128, 512), (1, 128), 0), out=buf699)
        del arg1006_1
        buf700 = reinterpret_tensor(buf699, (1, 128, 512), (65536, 512, 1), 0); del buf699  # reuse
        # Source Nodes: [intermediate_output_80], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf700, arg1007_1, 65536, grid=grid(65536), stream=stream0)
        del arg1007_1
        buf701 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf700, (128, 512), (512, 1), 0), reinterpret_tensor(arg1008_1, (512, 128), (1, 512), 0), out=buf701)
        del arg1008_1
        buf702 = buf698; del buf698  # reuse
        # Source Nodes: [add_308, attention_output_101, mul_165], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf702, buf701, arg1009_1, arg328_1, arg329_1, 16384, grid=grid(16384), stream=stream0)
        del arg1009_1
        del arg328_1
        del arg329_1
        buf703 = reinterpret_tensor(buf700, (128, 512), (512, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf702, (128, 128), (128, 1), 0), reinterpret_tensor(arg1010_1, (128, 512), (1, 128), 0), out=buf703)
        del arg1010_1
        buf704 = reinterpret_tensor(buf703, (1, 128, 512), (65536, 512, 1), 0); del buf703  # reuse
        # Source Nodes: [intermediate_output_81], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf704, arg1011_1, 65536, grid=grid(65536), stream=stream0)
        del arg1011_1
        buf705 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf704, (128, 512), (512, 1), 0), reinterpret_tensor(arg1012_1, (512, 128), (1, 512), 0), out=buf705)
        del arg1012_1
        buf706 = buf702; del buf702  # reuse
        # Source Nodes: [add_310, attention_output_102, mul_166], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf706, buf705, arg1013_1, arg330_1, arg331_1, 16384, grid=grid(16384), stream=stream0)
        del arg1013_1
        del arg330_1
        del arg331_1
        buf707 = reinterpret_tensor(buf704, (128, 512), (512, 1), 0); del buf704  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf706, (128, 128), (128, 1), 0), reinterpret_tensor(arg1014_1, (128, 512), (1, 128), 0), out=buf707)
        del arg1014_1
        buf708 = reinterpret_tensor(buf707, (1, 128, 512), (65536, 512, 1), 0); del buf707  # reuse
        # Source Nodes: [intermediate_output_82], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf708, arg1015_1, 65536, grid=grid(65536), stream=stream0)
        del arg1015_1
        buf709 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf708, (128, 512), (512, 1), 0), reinterpret_tensor(arg1016_1, (512, 128), (1, 512), 0), out=buf709)
        del arg1016_1
        buf710 = buf706; del buf706  # reuse
        # Source Nodes: [add_312, attention_output_103, mul_167], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf710, buf709, arg1017_1, arg332_1, arg333_1, 16384, grid=grid(16384), stream=stream0)
        del arg1017_1
        del arg332_1
        del arg333_1
        buf711 = reinterpret_tensor(buf708, (128, 512), (512, 1), 0); del buf708  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf710, (128, 128), (128, 1), 0), reinterpret_tensor(arg1018_1, (128, 512), (1, 128), 0), out=buf711)
        del arg1018_1
        buf712 = reinterpret_tensor(buf711, (1, 128, 512), (65536, 512, 1), 0); del buf711  # reuse
        # Source Nodes: [intermediate_output_83], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf712, arg1019_1, 65536, grid=grid(65536), stream=stream0)
        del arg1019_1
        buf713 = buf709; del buf709  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf712, (128, 512), (512, 1), 0), reinterpret_tensor(arg1020_1, (512, 128), (1, 512), 0), out=buf713)
        del arg1020_1
        buf714 = buf710; del buf710  # reuse
        # Source Nodes: [add_314, layer_output_81, mul_168], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf714, buf713, arg1021_1, arg334_1, arg335_1, 16384, grid=grid(16384), stream=stream0)
        del arg1021_1
        del arg334_1
        del arg335_1
        buf715 = reinterpret_tensor(buf712, (128, 512), (512, 1), 0); del buf712  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf714, (128, 128), (128, 1), 0), reinterpret_tensor(arg1022_1, (128, 512), (1, 128), 0), out=buf715)
        del arg1022_1
        buf716 = buf682; del buf682  # reuse
        # Source Nodes: [add_316, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf716, buf715, arg1023_1, arg336_1, arg337_1, 65536, grid=grid(65536), stream=stream0)
        del arg1023_1
        del arg336_1
        del arg337_1
        buf717 = reinterpret_tensor(buf714, (128, 128), (128, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf716, (128, 512), (512, 1), 0), reinterpret_tensor(arg1026_1, (512, 128), (1, 512), 0), out=buf717)
        del arg1026_1
        buf718 = reinterpret_tensor(buf717, (1, 128, 128), (16384, 128, 1), 0); del buf717  # reuse
        # Source Nodes: [key_tensor_21, mul_171], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf718, arg1027_1, arg340_1, arg341_1, 16384, grid=grid(16384), stream=stream0)
        del arg1027_1
        del arg340_1
        del arg341_1
        buf719 = buf713; del buf713  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf718, (128, 128), (128, 1), 0), reinterpret_tensor(arg1028_1, (128, 128), (1, 128), 0), out=buf719)
        del arg1028_1
        buf720 = reinterpret_tensor(buf689, (128, 128), (128, 1), 0); del buf689  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf718, (128, 128), (128, 1), 0), reinterpret_tensor(arg1030_1, (128, 128), (1, 128), 0), out=buf720)
        del arg1030_1
        buf721 = reinterpret_tensor(buf718, (128, 128), (128, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf716, (128, 512), (512, 1), 0), reinterpret_tensor(arg1032_1, (512, 128), (1, 512), 0), out=buf721)
        del arg1032_1
        buf722 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf719, arg1029_1, buf722, 16384, grid=grid(16384), stream=stream0)
        del arg1029_1
        buf723 = reinterpret_tensor(buf719, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf720, arg1031_1, buf723, 16384, grid=grid(16384), stream=stream0)
        del arg1031_1
        buf724 = reinterpret_tensor(buf720, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf720  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf721, arg1033_1, buf724, 16384, grid=grid(16384), stream=stream0)
        del arg1033_1
        del buf721
        # Source Nodes: [], Original ATen: []
        buf725 = aten._scaled_dot_product_efficient_attention(buf722, buf723, buf724, None, False, scale=0.17677669529663687)
        buf726 = buf725[0]
        del buf725
        buf730 = reinterpret_tensor(buf724, (128, 128), (128, 1), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf726, (128, 128), (128, 1), 0), reinterpret_tensor(arg1034_1, (128, 128), (1, 128), 0), out=buf730)
        del arg1034_1
        buf731 = reinterpret_tensor(buf726, (128, 128), (128, 1), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf716, (128, 512), (512, 1), 0), reinterpret_tensor(arg1024_1, (512, 128), (1, 512), 0), out=buf731)
        del arg1024_1
        buf732 = reinterpret_tensor(buf730, (1, 128, 128), (16384, 128, 1), 0); del buf730  # reuse
        # Source Nodes: [add_321, attention_output_105, layer_input_109, mul_170, mul_172], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf732, arg1035_1, buf731, arg1025_1, arg338_1, arg339_1, arg342_1, arg343_1, 16384, grid=grid(16384), stream=stream0)
        del arg1025_1
        del arg1035_1
        del arg338_1
        del arg339_1
        del arg342_1
        del arg343_1
        buf733 = buf715; del buf715  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (128, 128), (128, 1), 0), reinterpret_tensor(arg1036_1, (128, 512), (1, 128), 0), out=buf733)
        del arg1036_1
        buf734 = reinterpret_tensor(buf733, (1, 128, 512), (65536, 512, 1), 0); del buf733  # reuse
        # Source Nodes: [intermediate_output_84], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf734, arg1037_1, 65536, grid=grid(65536), stream=stream0)
        del arg1037_1
        buf735 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf734, (128, 512), (512, 1), 0), reinterpret_tensor(arg1038_1, (512, 128), (1, 512), 0), out=buf735)
        del arg1038_1
        buf736 = buf732; del buf732  # reuse
        # Source Nodes: [add_323, attention_output_106, mul_173], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf736, buf735, arg1039_1, arg344_1, arg345_1, 16384, grid=grid(16384), stream=stream0)
        del arg1039_1
        del arg344_1
        del arg345_1
        buf737 = reinterpret_tensor(buf734, (128, 512), (512, 1), 0); del buf734  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf736, (128, 128), (128, 1), 0), reinterpret_tensor(arg1040_1, (128, 512), (1, 128), 0), out=buf737)
        del arg1040_1
        buf738 = reinterpret_tensor(buf737, (1, 128, 512), (65536, 512, 1), 0); del buf737  # reuse
        # Source Nodes: [intermediate_output_85], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf738, arg1041_1, 65536, grid=grid(65536), stream=stream0)
        del arg1041_1
        buf739 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (128, 512), (512, 1), 0), reinterpret_tensor(arg1042_1, (512, 128), (1, 512), 0), out=buf739)
        del arg1042_1
        buf740 = buf736; del buf736  # reuse
        # Source Nodes: [add_325, attention_output_107, mul_174], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf740, buf739, arg1043_1, arg346_1, arg347_1, 16384, grid=grid(16384), stream=stream0)
        del arg1043_1
        del arg346_1
        del arg347_1
        buf741 = reinterpret_tensor(buf738, (128, 512), (512, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf740, (128, 128), (128, 1), 0), reinterpret_tensor(arg1044_1, (128, 512), (1, 128), 0), out=buf741)
        del arg1044_1
        buf742 = reinterpret_tensor(buf741, (1, 128, 512), (65536, 512, 1), 0); del buf741  # reuse
        # Source Nodes: [intermediate_output_86], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf742, arg1045_1, 65536, grid=grid(65536), stream=stream0)
        del arg1045_1
        buf743 = buf739; del buf739  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (128, 512), (512, 1), 0), reinterpret_tensor(arg1046_1, (512, 128), (1, 512), 0), out=buf743)
        del arg1046_1
        buf744 = buf740; del buf740  # reuse
        # Source Nodes: [add_327, attention_output_108, mul_175], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf744, buf743, arg1047_1, arg348_1, arg349_1, 16384, grid=grid(16384), stream=stream0)
        del arg1047_1
        del arg348_1
        del arg349_1
        buf745 = reinterpret_tensor(buf742, (128, 512), (512, 1), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf744, (128, 128), (128, 1), 0), reinterpret_tensor(arg1048_1, (128, 512), (1, 128), 0), out=buf745)
        del arg1048_1
        buf746 = reinterpret_tensor(buf745, (1, 128, 512), (65536, 512, 1), 0); del buf745  # reuse
        # Source Nodes: [intermediate_output_87], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf746, arg1049_1, 65536, grid=grid(65536), stream=stream0)
        del arg1049_1
        buf747 = buf743; del buf743  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf746, (128, 512), (512, 1), 0), reinterpret_tensor(arg1050_1, (512, 128), (1, 512), 0), out=buf747)
        del arg1050_1
        buf748 = buf744; del buf744  # reuse
        # Source Nodes: [add_329, layer_output_85, mul_176], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf748, buf747, arg1051_1, arg350_1, arg351_1, 16384, grid=grid(16384), stream=stream0)
        del arg1051_1
        del arg350_1
        del arg351_1
        buf749 = reinterpret_tensor(buf746, (128, 512), (512, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf748, (128, 128), (128, 1), 0), reinterpret_tensor(arg1052_1, (128, 512), (1, 128), 0), out=buf749)
        del arg1052_1
        buf750 = buf716; del buf716  # reuse
        # Source Nodes: [add_331, mul_177, value_tensor_22], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf750, buf749, arg1053_1, arg352_1, arg353_1, 65536, grid=grid(65536), stream=stream0)
        del arg1053_1
        del arg352_1
        del arg353_1
        buf751 = reinterpret_tensor(buf748, (128, 128), (128, 1), 0); del buf748  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (128, 512), (512, 1), 0), reinterpret_tensor(arg1056_1, (512, 128), (1, 512), 0), out=buf751)
        del arg1056_1
        buf752 = reinterpret_tensor(buf751, (1, 128, 128), (16384, 128, 1), 0); del buf751  # reuse
        # Source Nodes: [key_tensor_22, mul_179], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf752, arg1057_1, arg356_1, arg357_1, 16384, grid=grid(16384), stream=stream0)
        del arg1057_1
        del arg356_1
        del arg357_1
        buf753 = buf747; del buf747  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf752, (128, 128), (128, 1), 0), reinterpret_tensor(arg1058_1, (128, 128), (1, 128), 0), out=buf753)
        del arg1058_1
        buf754 = reinterpret_tensor(buf723, (128, 128), (128, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf752, (128, 128), (128, 1), 0), reinterpret_tensor(arg1060_1, (128, 128), (1, 128), 0), out=buf754)
        del arg1060_1
        buf755 = reinterpret_tensor(buf752, (128, 128), (128, 1), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (128, 512), (512, 1), 0), reinterpret_tensor(arg1062_1, (512, 128), (1, 512), 0), out=buf755)
        del arg1062_1
        buf756 = buf722; del buf722  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf753, arg1059_1, buf756, 16384, grid=grid(16384), stream=stream0)
        del arg1059_1
        buf757 = reinterpret_tensor(buf753, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf753  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf754, arg1061_1, buf757, 16384, grid=grid(16384), stream=stream0)
        del arg1061_1
        buf758 = reinterpret_tensor(buf754, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf755, arg1063_1, buf758, 16384, grid=grid(16384), stream=stream0)
        del arg1063_1
        del buf755
        # Source Nodes: [], Original ATen: []
        buf759 = aten._scaled_dot_product_efficient_attention(buf756, buf757, buf758, None, False, scale=0.17677669529663687)
        buf760 = buf759[0]
        del buf759
        buf764 = reinterpret_tensor(buf758, (128, 128), (128, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf760, (128, 128), (128, 1), 0), reinterpret_tensor(arg1064_1, (128, 128), (1, 128), 0), out=buf764)
        del arg1064_1
        buf765 = reinterpret_tensor(buf760, (128, 128), (128, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (128, 512), (512, 1), 0), reinterpret_tensor(arg1054_1, (512, 128), (1, 512), 0), out=buf765)
        del arg1054_1
        buf766 = reinterpret_tensor(buf764, (1, 128, 128), (16384, 128, 1), 0); del buf764  # reuse
        # Source Nodes: [add_336, attention_output_110, layer_input_114, mul_178, mul_180], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf766, arg1065_1, buf765, arg1055_1, arg354_1, arg355_1, arg358_1, arg359_1, 16384, grid=grid(16384), stream=stream0)
        del arg1055_1
        del arg1065_1
        del arg354_1
        del arg355_1
        del arg358_1
        del arg359_1
        buf767 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf766, (128, 128), (128, 1), 0), reinterpret_tensor(arg1066_1, (128, 512), (1, 128), 0), out=buf767)
        del arg1066_1
        buf768 = reinterpret_tensor(buf767, (1, 128, 512), (65536, 512, 1), 0); del buf767  # reuse
        # Source Nodes: [intermediate_output_88], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf768, arg1067_1, 65536, grid=grid(65536), stream=stream0)
        del arg1067_1
        buf769 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf768, (128, 512), (512, 1), 0), reinterpret_tensor(arg1068_1, (512, 128), (1, 512), 0), out=buf769)
        del arg1068_1
        buf770 = buf766; del buf766  # reuse
        # Source Nodes: [add_338, attention_output_111, mul_181], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf770, buf769, arg1069_1, arg360_1, arg361_1, 16384, grid=grid(16384), stream=stream0)
        del arg1069_1
        del arg360_1
        del arg361_1
        buf771 = reinterpret_tensor(buf768, (128, 512), (512, 1), 0); del buf768  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf770, (128, 128), (128, 1), 0), reinterpret_tensor(arg1070_1, (128, 512), (1, 128), 0), out=buf771)
        del arg1070_1
        buf772 = reinterpret_tensor(buf771, (1, 128, 512), (65536, 512, 1), 0); del buf771  # reuse
        # Source Nodes: [intermediate_output_89], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf772, arg1071_1, 65536, grid=grid(65536), stream=stream0)
        del arg1071_1
        buf773 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 512), (512, 1), 0), reinterpret_tensor(arg1072_1, (512, 128), (1, 512), 0), out=buf773)
        del arg1072_1
        buf774 = buf770; del buf770  # reuse
        # Source Nodes: [add_340, attention_output_112, mul_182], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf774, buf773, arg1073_1, arg362_1, arg363_1, 16384, grid=grid(16384), stream=stream0)
        del arg1073_1
        del arg362_1
        del arg363_1
        buf775 = reinterpret_tensor(buf772, (128, 512), (512, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf774, (128, 128), (128, 1), 0), reinterpret_tensor(arg1074_1, (128, 512), (1, 128), 0), out=buf775)
        del arg1074_1
        buf776 = reinterpret_tensor(buf775, (1, 128, 512), (65536, 512, 1), 0); del buf775  # reuse
        # Source Nodes: [intermediate_output_90], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf776, arg1075_1, 65536, grid=grid(65536), stream=stream0)
        del arg1075_1
        buf777 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf776, (128, 512), (512, 1), 0), reinterpret_tensor(arg1076_1, (512, 128), (1, 512), 0), out=buf777)
        del arg1076_1
        buf778 = buf774; del buf774  # reuse
        # Source Nodes: [add_342, attention_output_113, mul_183], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf778, buf777, arg1077_1, arg364_1, arg365_1, 16384, grid=grid(16384), stream=stream0)
        del arg1077_1
        del arg364_1
        del arg365_1
        buf779 = reinterpret_tensor(buf776, (128, 512), (512, 1), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf778, (128, 128), (128, 1), 0), reinterpret_tensor(arg1078_1, (128, 512), (1, 128), 0), out=buf779)
        del arg1078_1
        buf780 = reinterpret_tensor(buf779, (1, 128, 512), (65536, 512, 1), 0); del buf779  # reuse
        # Source Nodes: [intermediate_output_91], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf780, arg1079_1, 65536, grid=grid(65536), stream=stream0)
        del arg1079_1
        buf781 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf780, (128, 512), (512, 1), 0), reinterpret_tensor(arg1080_1, (512, 128), (1, 512), 0), out=buf781)
        del arg1080_1
        buf782 = buf778; del buf778  # reuse
        # Source Nodes: [add_344, layer_output_89, mul_184], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf782, buf781, arg1081_1, arg366_1, arg367_1, 16384, grid=grid(16384), stream=stream0)
        del arg1081_1
        del arg366_1
        del arg367_1
        buf783 = reinterpret_tensor(buf780, (128, 512), (512, 1), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf782, (128, 128), (128, 1), 0), reinterpret_tensor(arg1082_1, (128, 512), (1, 128), 0), out=buf783)
        del arg1082_1
        buf784 = buf750; del buf750  # reuse
        # Source Nodes: [add_346, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf784, buf783, arg1083_1, arg368_1, arg369_1, 65536, grid=grid(65536), stream=stream0)
        del arg1083_1
        del arg368_1
        del arg369_1
        buf785 = reinterpret_tensor(buf782, (128, 128), (128, 1), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (128, 512), (512, 1), 0), reinterpret_tensor(arg1086_1, (512, 128), (1, 512), 0), out=buf785)
        del arg1086_1
        buf786 = reinterpret_tensor(buf785, (1, 128, 128), (16384, 128, 1), 0); del buf785  # reuse
        # Source Nodes: [key_tensor_23, mul_187], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(buf786, arg1087_1, arg372_1, arg373_1, 16384, grid=grid(16384), stream=stream0)
        del arg1087_1
        del arg372_1
        del arg373_1
        buf787 = buf781; del buf781  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf786, (128, 128), (128, 1), 0), reinterpret_tensor(arg1088_1, (128, 128), (1, 128), 0), out=buf787)
        del arg1088_1
        buf788 = reinterpret_tensor(buf757, (128, 128), (128, 1), 0); del buf757  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf786, (128, 128), (128, 1), 0), reinterpret_tensor(arg1090_1, (128, 128), (1, 128), 0), out=buf788)
        del arg1090_1
        buf789 = reinterpret_tensor(buf786, (128, 128), (128, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (128, 512), (512, 1), 0), reinterpret_tensor(arg1092_1, (512, 128), (1, 512), 0), out=buf789)
        del arg1092_1
        buf790 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf787, arg1089_1, buf790, 16384, grid=grid(16384), stream=stream0)
        del arg1089_1
        buf791 = reinterpret_tensor(buf787, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf787  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf788, arg1091_1, buf791, 16384, grid=grid(16384), stream=stream0)
        del arg1091_1
        buf792 = reinterpret_tensor(buf788, (1, 4, 128, 32), (16384, 4096, 32, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf789, arg1093_1, buf792, 16384, grid=grid(16384), stream=stream0)
        del arg1093_1
        del buf789
        # Source Nodes: [], Original ATen: []
        buf793 = aten._scaled_dot_product_efficient_attention(buf790, buf791, buf792, None, False, scale=0.17677669529663687)
        del buf790
        del buf791
        buf794 = buf793[0]
        del buf793
        buf798 = reinterpret_tensor(buf792, (128, 128), (128, 1), 0); del buf792  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf794, (128, 128), (128, 1), 0), reinterpret_tensor(arg1094_1, (128, 128), (1, 128), 0), out=buf798)
        del arg1094_1
        buf799 = reinterpret_tensor(buf794, (128, 128), (128, 1), 0); del buf794  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf784, (128, 512), (512, 1), 0), reinterpret_tensor(arg1084_1, (512, 128), (1, 512), 0), out=buf799)
        del arg1084_1
        buf800 = reinterpret_tensor(buf798, (1, 128, 128), (16384, 128, 1), 0); del buf798  # reuse
        # Source Nodes: [add_351, attention_output_115, layer_input_119, mul_186, mul_188], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf800, arg1095_1, buf799, arg1085_1, arg370_1, arg371_1, arg374_1, arg375_1, 16384, grid=grid(16384), stream=stream0)
        del arg1085_1
        del arg1095_1
        del arg370_1
        del arg371_1
        del arg374_1
        del arg375_1
        buf801 = buf783; del buf783  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf800, (128, 128), (128, 1), 0), reinterpret_tensor(arg1096_1, (128, 512), (1, 128), 0), out=buf801)
        del arg1096_1
        buf802 = reinterpret_tensor(buf801, (1, 128, 512), (65536, 512, 1), 0); del buf801  # reuse
        # Source Nodes: [intermediate_output_92], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf802, arg1097_1, 65536, grid=grid(65536), stream=stream0)
        del arg1097_1
        buf803 = buf799; del buf799  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf802, (128, 512), (512, 1), 0), reinterpret_tensor(arg1098_1, (512, 128), (1, 512), 0), out=buf803)
        del arg1098_1
        buf804 = buf800; del buf800  # reuse
        # Source Nodes: [add_353, attention_output_116, mul_189], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf804, buf803, arg1099_1, arg376_1, arg377_1, 16384, grid=grid(16384), stream=stream0)
        del arg1099_1
        del arg376_1
        del arg377_1
        buf805 = reinterpret_tensor(buf802, (128, 512), (512, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf804, (128, 128), (128, 1), 0), reinterpret_tensor(arg1100_1, (128, 512), (1, 128), 0), out=buf805)
        del arg1100_1
        buf806 = reinterpret_tensor(buf805, (1, 128, 512), (65536, 512, 1), 0); del buf805  # reuse
        # Source Nodes: [intermediate_output_93], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf806, arg1101_1, 65536, grid=grid(65536), stream=stream0)
        del arg1101_1
        buf807 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf806, (128, 512), (512, 1), 0), reinterpret_tensor(arg1102_1, (512, 128), (1, 512), 0), out=buf807)
        del arg1102_1
        buf808 = buf804; del buf804  # reuse
        # Source Nodes: [add_355, attention_output_117, mul_190], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf808, buf807, arg1103_1, arg378_1, arg379_1, 16384, grid=grid(16384), stream=stream0)
        del arg1103_1
        del arg378_1
        del arg379_1
        buf809 = reinterpret_tensor(buf806, (128, 512), (512, 1), 0); del buf806  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf808, (128, 128), (128, 1), 0), reinterpret_tensor(arg1104_1, (128, 512), (1, 128), 0), out=buf809)
        del arg1104_1
        buf810 = reinterpret_tensor(buf809, (1, 128, 512), (65536, 512, 1), 0); del buf809  # reuse
        # Source Nodes: [intermediate_output_94], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf810, arg1105_1, 65536, grid=grid(65536), stream=stream0)
        del arg1105_1
        buf811 = buf807; del buf807  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf810, (128, 512), (512, 1), 0), reinterpret_tensor(arg1106_1, (512, 128), (1, 512), 0), out=buf811)
        del arg1106_1
        buf812 = buf808; del buf808  # reuse
        # Source Nodes: [add_357, attention_output_118, mul_191], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf812, buf811, arg1107_1, arg380_1, arg381_1, 16384, grid=grid(16384), stream=stream0)
        del arg1107_1
        del arg380_1
        del arg381_1
        buf813 = reinterpret_tensor(buf810, (128, 512), (512, 1), 0); del buf810  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf812, (128, 128), (128, 1), 0), reinterpret_tensor(arg1108_1, (128, 512), (1, 128), 0), out=buf813)
        del arg1108_1
        buf814 = reinterpret_tensor(buf813, (1, 128, 512), (65536, 512, 1), 0); del buf813  # reuse
        # Source Nodes: [intermediate_output_95], Original ATen: [aten.relu]
        triton_poi_fused_relu_5.run(buf814, arg1109_1, 65536, grid=grid(65536), stream=stream0)
        del arg1109_1
        buf815 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf814, (128, 512), (512, 1), 0), reinterpret_tensor(arg1110_1, (512, 128), (1, 512), 0), out=buf815)
        del arg1110_1
        buf816 = buf812; del buf812  # reuse
        # Source Nodes: [add_359, layer_output_93, mul_192], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf816, buf815, arg1111_1, arg382_1, arg383_1, 16384, grid=grid(16384), stream=stream0)
        del arg1111_1
        del arg382_1
        del arg383_1
        del buf815
        buf817 = reinterpret_tensor(buf814, (128, 512), (512, 1), 0); del buf814  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf816, (128, 128), (128, 1), 0), reinterpret_tensor(arg1112_1, (128, 512), (1, 128), 0), out=buf817)
        del arg1112_1
        del buf816
        buf818 = buf784; del buf784  # reuse
        # Source Nodes: [add_361, mul_193, sequence_output], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf818, buf817, arg1113_1, arg384_1, arg385_1, 65536, grid=grid(65536), stream=stream0)
        del arg1113_1
        del arg384_1
        del arg385_1
        buf819 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf818, (128, 512), (512, 1), 0), reinterpret_tensor(arg1114_1, (512, 512), (1, 512), 0), out=buf819)
        del arg1114_1
        buf823 = buf818; del buf818  # reuse
        # Source Nodes: [hidden_states_217, hidden_states_219], Original ATen: [aten.native_layer_norm, aten.relu]
        triton_per_fused_native_layer_norm_relu_10.run(buf819, arg1115_1, arg1116_1, arg1117_1, buf823, 128, 512, grid=grid(128), stream=stream0)
        del arg1115_1
        del arg1116_1
        del arg1117_1
        del buf819
        buf824 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(arg386_1, arg387_1, buf824, 512, 30522, grid=grid(512, 30522), stream=stream0)
        del arg386_1
        del arg387_1
        buf825 = empty((128, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_2, hidden_states_220], Original ATen: [aten.cat, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (128, 512), (512, 1), 0), buf824, out=buf825)
        del buf823
        del buf824
        buf826 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_12.run(buf825, arg388_1, buf826, 512, 7631, grid=grid(512), stream=stream0)
        buf827 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_13.run(buf826, buf827, 128, 4, grid=grid(128), stream=stream0)
        buf828 = buf826; del buf826  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_14.run(buf825, arg388_1, buf827, buf828, 512, 7631, grid=grid(512), stream=stream0)
        buf829 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_15.run(buf828, buf829, 128, 4, grid=grid(128), stream=stream0)
        del buf828
        buf830 = empty((), device='cuda', dtype=torch.float32)
        buf833 = buf830; del buf830  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_16.run(buf833, arg1120_1, buf825, arg388_1, buf827, buf829, 1, 128, grid=grid(1), stream=stream0)
        del arg1120_1
        del buf827
        del buf829
        buf832 = reinterpret_tensor(buf825, (1, 128, 30522), (3906816, 30522, 1), 0); del buf825  # reuse
        # Source Nodes: [prediction_scores], Original ATen: [aten.view]
        triton_poi_fused_view_17.run(buf832, arg388_1, 3906816, grid=grid(3906816), stream=stream0)
        del arg388_1
        return (buf833, buf832, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((384, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1119_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1120_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForMaskedLM', benchmark_compiled_module)
