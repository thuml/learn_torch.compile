
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


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2c3iblli2m6samph3lvi3fmotjgfywwva5j5p6x64zurf43hfh.py
# Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
# diagonal_chunked_attention_scores => view_17
triton_poi_fused_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 8.0
    tmp4 = tmp2 / tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/wt/cwtssjtnejaat5tnoufhj5bhndkognx52335cjaeey5wkmfp57md.py
# Source Nodes: [hidden_states_2], Original ATen: [aten.view]
# hidden_states_2 => view_13
triton_poi_fused_view_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
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


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dr675wcx7qamzbqjwrb7477mrbfxnjijduxvrhc6rtxtqfdl3x.py
# Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
# diagonal_chunked_attention_scores => clone
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 3
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x3) + (768*x1) + (196608*x2)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswrs3seb5mxnysi5xhgnxxvee6btxt3glrdbsl55pr6hhfn3mep.py
# Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
# diagonal_chunked_attention_scores => clone_1
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64) % 3
    y2 = (yindex // 192)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*y2) + (768*x3) + (196608*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (512*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6m5r5ec7v5aj5tm4xkbvezuvpizvf6ygnixqzbx4ii2fzvkwcim.py
# Source Nodes: [diagonal_attention_scores, setitem, setitem_1], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
# diagonal_attention_scores => full
# setitem => copy, slice_scatter, slice_scatter_2
# setitem_1 => copy_1, select_scatter, slice_scatter_4
triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131328) % 4
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x3 = (xindex // 525312)
    x5 = xindex % 131328
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + x0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((512*(((656384 + x5) // 512) % 513)) + (262144*((656384 + x5) // 262656)) + (786432*x3) + (786432*((656384 + x5) // 787968)) + (x5 % 512)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.full([1], 3, tl.int64)
    tmp16 = tmp15 < tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = ((787712 + x0 + (513*x1)) // 512) % 513
    tmp19 = tmp18 < tmp7
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr0 + ((262144*(((787712 + x0 + (513*x1)) // 262656) % 3)) + (786432*(((787712 + x0 + (513*x1) + (787968*x3)) // 787968) % 12)) + ((787712 + x0 + (513*x1)) % 262656)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp17, tmp23, tmp24)
    tmp26 = 0.0
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp16, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp29, tmp26)
    tmp31 = tl.where(tmp5, tmp14, tmp30)
    tmp32 = tmp0 < tmp15
    tmp33 = tmp5 & tmp32
    tmp34 = (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 512) % 513
    tmp35 = tmp34 < tmp7
    tmp36 = tmp35 & tmp33
    tmp37 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 262656) % 36)) + (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) % 262656)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp33, tmp39, tmp40)
    tmp42 = tl.where(tmp5, tmp41, tmp26)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp32, tmp42, tmp43)
    tmp45 = tl.where(tmp32, tmp44, tmp26)
    tmp46 = tl.where(tmp2, tmp31, tmp45)
    tl.store(out_ptr0 + (x6), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjchatppxdmgayejc6se4kha3iecgnd75kx3npznvzrwala3u2lf.py
# Source Nodes: [setitem_3], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_3 => copy_3, slice_scatter_11, slice_scatter_12
triton_poi_fused_copy_slice_scatter_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1575936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 256
    x0 = xindex % 513
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    tmp50 = tl.load(in_ptr1 + (x3 + (525312*x2)), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (787968*x2)) // 262656) % 36)) + (((-256) + x0 + (513*x1) + (787968*x2)) % 262656)), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp19 & tmp2
    tmp21 = tmp6 & tmp20
    tmp22 = (((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp23 = tmp22 < tmp10
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp21, tmp27, tmp28)
    tmp30 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp20 & xmask, other=0.0)
    tmp31 = tl.where(tmp6, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp20, tmp31, tmp32)
    tmp34 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp2 & xmask, other=0.0)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp7, tmp17, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp2, tmp36, tmp37)
    tmp39 = tmp6 & tmp19
    tmp40 = tmp23 & tmp39
    tmp41 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp40 & xmask, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp39, tmp43, tmp44)
    tmp46 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp19 & xmask, other=0.0)
    tmp47 = tl.where(tmp6, tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp19, tmp47, tmp48)
    tmp51 = tl.where(tmp19, tmp49, tmp50)
    tmp52 = tl.where(tmp2, tmp38, tmp51)
    tl.store(out_ptr0 + (x4), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74pmmeoylls4iewv22wvynkii5s2f2jkhzrl6cyjnmoxp26rhrg.py
# Source Nodes: [bool_1, full_like, where], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# bool_1 => convert_element_type
# full_like => full_default_2
# where => where_1
triton_poi_fused__to_copy_full_like_where_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 789504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257) % 256
    x3 = (xindex // 257)
    x2 = (xindex // 65792)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x0 + (513*x3)), xmask)
    tmp29 = tl.load(in_ptr2 + (x0 + (513*x1) + (525312*x2)), xmask)
    tmp0 = (-255) + x0 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = x0
    tmp13 = tl.full([1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp17 = tl.full([1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((512*((((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*x1) + (787968*x2)) // 262656) % 36)) + ((x0 + (513*x1)) % 512)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.load(in_ptr2 + (x0 + (513*x1) + (525312*x2)), tmp11 & xmask, other=0.0)
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp11, tmp26, tmp27)
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp31 = tl.where(tmp8, tmp9, tmp30)
    tmp32 = float("-inf")
    tmp33 = tl.where(tmp6, tmp32, tmp31)
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4ukcftskekilnq5amk5ym5umiv2xya2mhwesbnmu72ahk2rwlz3.py
# Source Nodes: [setitem_4], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_4 => copy_4, slice_scatter_14
triton_poi_fused_copy_slice_scatter_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1575936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x3 = (xindex // 513)
    x4 = xindex
    x1 = (xindex // 513) % 256
    x2 = (xindex // 131328)
    x5 = xindex % 131328
    tmp8 = tl.load(in_ptr1 + (x4), xmask)
    tmp28 = tl.load(in_ptr3 + (x5 + (525312*x2)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 257, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (257*x3)), tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp6 == tmp6
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp13 & tmp11
    tmp15 = (((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp16 = tl.full([1], 512, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = tl.load(in_ptr2 + ((512*((((-131584) + x5 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x5 + (787968*x2)) // 262656) % 36)) + (x5 % 512)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.load(in_ptr3 + (x5 + (525312*x2)), tmp11 & xmask, other=0.0)
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tl.where(tmp7, tmp8, tmp29)
    tmp31 = tl.where(tmp2, tmp5, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5ifynnfvbvapunbrkaopn4m53fffxlc4wookyy3gnqonlfajtil.py
# Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
# setitem_4 => slice_scatter_16
triton_poi_fused_slice_scatter_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6156)
    x0 = xindex % 513
    x1 = (xindex // 513) % 12
    x4 = xindex
    tmp9 = tl.load(in_ptr1 + (x0 + (513*(x2 % 256)) + (131328*x1)), None)
    tmp28 = tl.load(in_ptr3 + (x0 + (513*x2) + (525312*x1)), None)
    tmp0 = x2
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (513*x2) + (131328*x1)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (x2 // 256)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp6 >= tmp10
    tmp12 = x0
    tmp13 = tmp12 < tmp1
    tmp14 = tmp13 & tmp11
    tmp15 = (((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 512) % 513
    tmp16 = tl.full([1], 512, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = tl.load(in_ptr2 + ((512*((((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 262656) % 36)) + ((x0 + (513*(x2 % 256))) % 512)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.load(in_ptr3 + (x0 + (513*x2) + (525312*x1)), tmp11, other=0.0)
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp9, tmp29)
    tmp31 = tl.where(tmp2, tmp5, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xqhnjmf7nymwxjy5wleepvc7gjjqpksvlzmgcx47qo2ynxouyn.py
# Source Nodes: [hidden_states_3], Original ATen: [aten.view]
# hidden_states_3 => view_37
triton_poi_fused_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjhpzjzg2txdvrbjc3p2usykvndcz5dcatqxkvqoh62gd26ftfd.py
# Source Nodes: [hidden_states_18, hidden_states_4], Original ATen: [aten.view]
# hidden_states_18 => view_113
# hidden_states_4 => view_38
triton_poi_fused_view_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyocdeccnmshsnbgcnmyuatce53vrhg5qjraxdvwo75wbiosaqhb.py
# Source Nodes: [diagonal_attention_scores_2, setitem_6, setitem_7], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
# diagonal_attention_scores_2 => full_5
# setitem_6 => copy_6, slice_scatter_22, slice_scatter_24
# setitem_7 => copy_7, select_scatter_2, slice_scatter_26
triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 513
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 256)
    y0 = yindex
    x1 = xindex % 256
    x3 = xindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = y0
    tmp4 = tl.full([1, 1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + y0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1, 1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((256*((656384 + y0 + (513*x1)) // 262656)) + (((656384 + y0 + (513*x1)) // 512) % 513)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + ((256*((656384 + y0 + (513*x1)) // 262656)) + ((y0 + (513*x1)) % 512)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.full([1, 1], 3, tl.int64)
    tmp18 = tmp17 < tmp17
    tmp19 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp4
    tmp21 = tmp20 & tmp18
    tmp22 = ((787712 + y0 + (513*x1)) // 512) % 513
    tmp23 = tmp22 < tmp7
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr0 + ((256*(((787712 + y0 + (513*x1)) // 262656) % 3)) + (((787712 + y0 + (513*x1)) // 512) % 513)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr1 + ((256*(((787712 + y0 + (513*x1)) // 262656) % 3)) + ((787712 + y0 + (513*x1)) % 512)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp24, tmp27, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp21, tmp29, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp20, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp18, tmp33, tmp34)
    tmp36 = tl.where(tmp18, tmp35, tmp32)
    tmp37 = tl.where(tmp5, tmp16, tmp36)
    tmp38 = tmp0 < tmp17
    tmp39 = tmp20 & tmp38
    tmp40 = (((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513
    tmp41 = tmp40 < tmp7
    tmp42 = tmp41 & tmp39
    tmp43 = tl.load(in_ptr0 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 3)) + ((((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513)), tmp42 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr1 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 3)) + (((-256) + y0 + (513*x1) + (262656*x2)) % 512)), tmp42 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp42, tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp39, tmp47, tmp48)
    tmp50 = tl.where(tmp20, tmp49, tmp32)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp38, tmp50, tmp51)
    tmp53 = tl.where(tmp38, tmp52, tmp32)
    tmp54 = tl.where(tmp2, tmp37, tmp53)
    tl.store(out_ptr0 + (x3 + (1024*y0)), tmp54, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/caht35sw4whfz5ps5bwf44lj7ijerbozhffe27f42opgm2lonjkw.py
# Source Nodes: [setitem_9], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_9 => copy_9, slice_scatter_33, slice_scatter_34
triton_poi_fused_copy_slice_scatter_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp56 = tl.load(in_ptr2 + (y0 + (1024*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1, 1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x1 + (513*y0)) // 512) % 513
    tmp10 = tl.full([1, 1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 3)) + ((((-256) + x1 + (513*y0)) // 512) % 513)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 3)) + (((-256) + x1 + (513*y0)) % 512)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp21 & tmp2
    tmp23 = tmp6 & tmp22
    tmp24 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp25 = tmp24 < tmp10
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp26, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp6, tmp33, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp22, tmp35, tmp36)
    tmp38 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp7, tmp19, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp2, tmp40, tmp41)
    tmp43 = tmp6 & tmp21
    tmp44 = tmp25 & tmp43
    tmp45 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 * tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp44, tmp47, tmp48)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp43, tmp49, tmp50)
    tmp52 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.where(tmp6, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp21, tmp53, tmp54)
    tmp57 = tl.where(tmp21, tmp55, tmp56)
    tmp58 = tl.where(tmp2, tmp42, tmp57)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp58, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4kp7nag37mgabcghj73uzpuw7zh4t3l7xdxzlx6hyv26e7mutv.py
# Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# bool_3 => convert_element_type_3
# full_like_2 => full_default_7
# where_2 => where_5
triton_poi_fused__to_copy_full_like_where_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp9 = tl.load(in_ptr0 + (x1 + (513*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (y0 + (1024*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-255) + x1 + y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp17 = tl.full([1, 1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.load(in_ptr3 + (y0 + (1024*x1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp14, tmp26, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp32 = tl.where(tmp11, tmp30, tmp31)
    tmp33 = tl.where(tmp8, tmp9, tmp32)
    tmp34 = float("-inf")
    tmp35 = tl.where(tmp6, tmp34, tmp33)
    tl.store(out_ptr0 + (x1 + (257*y0)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvqzzht3hitoc2phsidxgbx4k5edxv4qjemxjlffxqjhcm4sujb.py
# Source Nodes: [setitem_10], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_10 => copy_10, slice_scatter_36, slice_scatter_38
triton_poi_fused_copy_slice_scatter_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp42 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr4 + (y0 + (1024*x1)), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tl.full([1, 1], 257, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x1 + (257*y0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.broadcast_to((y0 // 256), [XBLOCK, YBLOCK])
    tmp11 = tl.full([1, 1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full([1, 1], 1, tl.int64)
    tmp15 = tmp10 >= tmp14
    tmp16 = tmp15 & tmp2
    tmp17 = tmp3 < tmp1
    tmp18 = tmp17 & tmp16
    tmp19 = (((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513
    tmp20 = tl.full([1, 1], 512, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21 & tmp18
    tmp23 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((x1 + (513*(y0 % 256))) % 512)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp16, tmp31, tmp32)
    tmp34 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp12, tmp13, tmp35)
    tmp37 = tl.where(tmp5, tmp9, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp2, tmp37, tmp38)
    tmp40 = (y0 // 256)
    tmp41 = tmp40 == tmp11
    tmp43 = tmp40 >= tmp14
    tmp44 = tmp17 & tmp43
    tmp45 = tmp21 & tmp44
    tmp46 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((x1 + (513*(y0 % 256))) % 512)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 * tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp44, tmp50, tmp51)
    tmp53 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.where(tmp17, tmp52, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp43, tmp54, tmp55)
    tmp58 = tl.where(tmp43, tmp56, tmp57)
    tmp59 = tl.where(tmp41, tmp42, tmp58)
    tmp60 = tl.where(tmp2, tmp39, tmp59)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp60, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3pdlwil2mpr5bn6ztseh6sihaakrz3hxjieyni4z257nsttq4u.py
# Source Nodes: [attn_probs, attn_scores_1], Original ATen: [aten._softmax, aten.add]
# attn_probs => amax, clone_2, exp, sub_4, sum_1
# attn_scores_1 => add_2
triton_per_fused__softmax_add_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 513
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x1 = (xindex // 12)
    r2 = rindex
    x0 = xindex % 12
    x3 = xindex
    tmp23 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r2, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = 1280 + ((-1)*r2) + ((-1)*x1)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 <= tmp8
    tmp10 = 1.0
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp6, other=0.0)
    tmp15 = float("-inf")
    tmp16 = tl.where(tmp13, tmp15, tmp14)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp2, other=0.0)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp13, tmp15, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp2, tmp30, tmp31)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp35 = tmp24 + tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, float("-inf"))
    tmp39 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp38, 0))
    tmp40 = tmp35 - tmp39
    tmp41 = tl.exp(tmp40)
    tmp42 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp44 = tl.where(rmask, tmp42, 0)
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp44, 0))
    tl.store(out_ptr0 + (r2 + (513*x3)), tmp35, rmask)
    tl.store(out_ptr1 + (x3), tmp39, None)
    tl.store(out_ptr2 + (x3), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32usrnrw2xykk6x3oao3isbjyjxv2g2tbmnz35f5jlkkcya55ie.py
# Source Nodes: [padded_value], Original ATen: [aten.constant_pad_nd]
# padded_value => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 1536
    x0 = xindex % 64
    x2 = (xindex // 98304)
    x3 = xindex
    tmp0 = (-256) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-196608) + x0 + (64*x2) + (768*x1)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, -1.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z6/cz6qq6njs7oewf2ahc6ognwamzfrblwsxn2bojzl4h5qqns36y4r.py
# Source Nodes: [chunked_hidden_states], Original ATen: [aten.constant_pad_nd]
# chunked_hidden_states => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9461760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x2 = (xindex // 9240)
    x3 = (xindex // 770)
    x1 = (xindex // 770) % 12
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr1 + (x0 + (513*x3)), tmp2, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp4 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.load(in_ptr3 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp3, tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tl.store(out_ptr0 + (x0 + (770*x2) + (788480*x1)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lhtv2ge7mw437pukbsms7og2zkx5kl2qm6y4ra6l6vwcdupwgr.py
# Source Nodes: [context], Original ATen: [aten.clone]
# context => clone_4
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 49152
    x1 = (xindex // 49152) % 4
    x2 = (xindex // 196608)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16384*x1) + (98304*x2)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cntwoxehdypqqkiarhqlvawg4wigcnyx2ewhpf2i3ubxlljihj7g.py
# Source Nodes: [reshape_6], Original ATen: [aten.clone]
# reshape_6 => clone_5
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtc3ztvufgtglzdt2xpmmbfgtxtdxz4fhlnskaj2hctveixzzcz.py
# Source Nodes: [add_3, attn_output_3], Original ATen: [aten.add, aten.native_layer_norm]
# add_3 => add_4
# attn_output_3 => add_5, add_6, mul_1, mul_2, rsqrt, sub_6, var_mean
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7nf2ngqokcqdwuyl7am5istw3nltal7tgkynouo22lucbxpj74.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_7, erf, mul_3, mul_4, mul_5
triton_poi_fused_gelu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, 768), (768, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (3072, 768), (768, 1))
    assert_size_stride(arg11_1, (3072, ), (1, ))
    assert_size_stride(arg12_1, (768, 3072), (3072, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, 768), (768, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, 768), (768, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (3072, 768), (768, 1))
    assert_size_stride(arg27_1, (3072, ), (1, ))
    assert_size_stride(arg28_1, (768, 3072), (3072, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, 768), (768, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 768), (768, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (3072, 768), (768, 1))
    assert_size_stride(arg43_1, (3072, ), (1, ))
    assert_size_stride(arg44_1, (768, 3072), (3072, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, 768), (768, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, 768), (768, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (3072, 768), (768, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 768), (768, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, 768), (768, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (3072, 768), (768, 1))
    assert_size_stride(arg75_1, (3072, ), (1, ))
    assert_size_stride(arg76_1, (768, 3072), (3072, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, 768), (768, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (3072, 768), (768, 1))
    assert_size_stride(arg91_1, (3072, ), (1, ))
    assert_size_stride(arg92_1, (768, 3072), (3072, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, 768), (768, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (3072, 768), (768, 1))
    assert_size_stride(arg123_1, (3072, ), (1, ))
    assert_size_stride(arg124_1, (768, 3072), (3072, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, 768), (768, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (3072, 768), (768, 1))
    assert_size_stride(arg139_1, (3072, ), (1, ))
    assert_size_stride(arg140_1, (768, 3072), (3072, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (3072, 768), (768, 1))
    assert_size_stride(arg155_1, (3072, ), (1, ))
    assert_size_stride(arg156_1, (768, 3072), (3072, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 768), (768, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (3072, 768), (768, 1))
    assert_size_stride(arg171_1, (3072, ), (1, ))
    assert_size_stride(arg172_1, (768, 3072), (3072, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, 768), (768, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, 768), (768, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (3072, 768), (768, 1))
    assert_size_stride(arg187_1, (3072, ), (1, ))
    assert_size_stride(arg188_1, (768, 3072), (3072, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(arg193_1, (1, 1024), (1024, 1))
    assert_size_stride(arg194_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(arg192_1, (1024, 768), (768, 1), 0), reinterpret_tensor(arg0_1, (768, 768), (1, 768), 0), out=buf0)
        del arg0_1
        buf1 = reinterpret_tensor(buf0, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf0  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_view_0.run(buf1, arg1_1, 786432, grid=grid(786432), stream=stream0)
        del arg1_1
        buf2 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(arg192_1, (1024, 768), (768, 1), 0), reinterpret_tensor(arg2_1, (768, 768), (1, 768), 0), out=buf2)
        del arg2_1
        buf3 = reinterpret_tensor(buf2, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf2  # reuse
        # Source Nodes: [hidden_states_2], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf3, arg3_1, 786432, grid=grid(786432), stream=stream0)
        del arg3_1
        buf4 = empty((12, 3, 512, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf1, buf4, 1179648, grid=grid(1179648), stream=stream0)
        buf5 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf3, buf5, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf6 = empty((36, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf5, (36, 64, 512), (32768, 512, 1), 0), out=buf6)
        buf7 = empty((12, 4, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_attention_scores, setitem, setitem_1], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf6, buf7, 6303744, grid=grid(6303744), stream=stream0)
        buf8 = empty((12, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_3], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf6, buf7, buf8, 1575936, grid=grid(1575936), stream=stream0)
        buf9 = empty_strided((1, 256, 12, 257), (789504, 257, 65792, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bool_1, full_like, where], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf8, buf6, buf7, buf9, 789504, grid=grid(789504), stream=stream0)
        buf10 = empty_strided((1, 256, 12, 513), (1575936, 513, 131328, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_4], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf9, buf8, buf6, buf7, buf10, 1575936, grid=grid(1575936), stream=stream0)
        buf11 = empty((1, 1024, 12, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf10, buf8, buf6, buf7, buf11, 6303744, grid=grid(6303744), stream=stream0)
        buf12 = empty((1, 2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf12, 1024, grid=grid(1024), stream=stream0)
        buf13 = empty((1, 2, 512, 1), device='cuda', dtype=torch.float32)
        buf52 = empty((1, 2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, hidden_states_4], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf13, buf52, 1024, grid=grid(1024), stream=stream0)
        buf14 = empty_strided((1, 4, 256, 513), (525312, 256, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_attention_scores_2, setitem_6, setitem_7], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf12, buf13, buf14, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf15 = empty((1, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_9], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf12, buf13, buf14, buf15, 256, 513, grid=grid(256, 513), stream=stream0)
        buf16 = empty_strided((1, 256, 1, 257), (65792, 257, 65792, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf15, buf12, buf13, buf14, buf16, 256, 257, grid=grid(256, 257), stream=stream0)
        buf17 = empty_strided((1, 1024, 1, 513), (525312, 513, 525312, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_10], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf16, buf15, buf12, buf13, buf14, buf17, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf18 = reinterpret_tensor(buf7, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf7  # reuse
        buf19 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs, attn_scores_1], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf11, buf17, buf18, buf19, buf20, 12288, 513, grid=grid(12288), stream=stream0)
        buf21 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [value_vectors], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(arg192_1, (1024, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf21)
        del arg4_1
        del arg5_1
        buf22 = reinterpret_tensor(buf5, (12, 1536, 64), (98304, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [padded_value], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf21, buf22, 1179648, grid=grid(1179648), stream=stream0)
        buf23 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf18, buf19, buf20, buf23, 9461760, grid=grid(9461760), stream=stream0)
        buf24 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf22, buf24, 2359296, grid=grid(2359296), stream=stream0)
        buf25 = reinterpret_tensor(buf21, (48, 256, 64), (16384, 64, 1), 0); del buf21  # reuse
        # Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf24, (48, 768, 64), (49152, 64, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf1, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf1  # reuse
        # Source Nodes: [reshape_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf25, buf26, 786432, grid=grid(786432), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (1024, 768), (768, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), out=buf27)
        del arg6_1
        buf31 = reinterpret_tensor(buf26, (1, 1024, 768), (786432, 768, 1), 0); del buf26  # reuse
        # Source Nodes: [add_3, attn_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf27, arg7_1, arg192_1, arg8_1, arg9_1, buf31, 1024, 768, grid=grid(1024), stream=stream0)
        del arg192_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf32 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 3072), (1, 768), 0), out=buf32)
        del arg10_1
        buf33 = reinterpret_tensor(buf32, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf32  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf33, arg11_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg11_1
        buf34 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg12_1, (3072, 768), (1, 3072), 0), out=buf34)
        del arg12_1
        buf38 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, hidden_states_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf34, arg13_1, buf31, arg14_1, arg15_1, buf38, 1024, 768, grid=grid(1024), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        buf39 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1024, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 768), (1, 768), 0), out=buf39)
        del arg16_1
        buf40 = reinterpret_tensor(buf39, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf39  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf40, arg17_1, 786432, grid=grid(786432), stream=stream0)
        del arg17_1
        buf41 = reinterpret_tensor(buf31, (1024, 768), (768, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1024, 768), (768, 1), 0), reinterpret_tensor(arg18_1, (768, 768), (1, 768), 0), out=buf41)
        del arg18_1
        buf42 = reinterpret_tensor(buf41, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf41  # reuse
        # Source Nodes: [hidden_states_16], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf42, arg19_1, 786432, grid=grid(786432), stream=stream0)
        del arg19_1
        buf43 = reinterpret_tensor(buf22, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf22  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf40, buf43, 1179648, grid=grid(1179648), stream=stream0)
        buf44 = reinterpret_tensor(buf4, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf4  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf42, buf44, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf45 = buf6; del buf6  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf44, (36, 64, 512), (32768, 512, 1), 0), out=buf45)
        buf46 = reinterpret_tensor(buf18, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf18  # reuse
        # Source Nodes: [diagonal_attention_scores_4, setitem_12, setitem_13], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf45, buf46, 6303744, grid=grid(6303744), stream=stream0)
        buf47 = buf8; del buf8  # reuse
        # Source Nodes: [setitem_15], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf45, buf46, buf47, 1575936, grid=grid(1575936), stream=stream0)
        buf48 = buf9; del buf9  # reuse
        # Source Nodes: [bool_5, full_like_4, where_4], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf47, buf45, buf46, buf48, 789504, grid=grid(789504), stream=stream0)
        buf49 = buf10; del buf10  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf48, buf47, buf45, buf46, buf49, 1575936, grid=grid(1575936), stream=stream0)
        buf50 = buf11; del buf11  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf49, buf47, buf45, buf46, buf50, 6303744, grid=grid(6303744), stream=stream0)
        buf51 = buf13; del buf13  # reuse
        # Source Nodes: [hidden_states_17], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf51, 1024, grid=grid(1024), stream=stream0)
        buf53 = reinterpret_tensor(buf17, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf17  # reuse
        # Source Nodes: [diagonal_attention_scores_6, setitem_18, setitem_19], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf51, buf52, buf53, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf54 = buf15; del buf15  # reuse
        # Source Nodes: [setitem_21], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf51, buf52, buf53, buf54, 256, 513, grid=grid(256, 513), stream=stream0)
        buf55 = buf16; del buf16  # reuse
        # Source Nodes: [bool_7, full_like_6, where_6], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf54, buf51, buf52, buf53, buf55, 256, 257, grid=grid(256, 257), stream=stream0)
        buf56 = reinterpret_tensor(buf14, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf14  # reuse
        # Source Nodes: [setitem_22], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf55, buf54, buf51, buf52, buf53, buf56, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf57 = reinterpret_tensor(buf46, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf46  # reuse
        buf58 = buf20; del buf20  # reuse
        buf59 = buf19; del buf19  # reuse
        # Source Nodes: [attn_probs_4, attn_scores_3], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf50, buf56, buf57, buf58, buf59, 12288, 513, grid=grid(12288), stream=stream0)
        buf60 = reinterpret_tensor(buf42, (1024, 768), (768, 1), 0); del buf42  # reuse
        # Source Nodes: [value_vectors_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg21_1, reinterpret_tensor(buf38, (1024, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf60)
        del arg20_1
        del arg21_1
        buf61 = reinterpret_tensor(buf44, (12, 1536, 64), (98304, 64, 1), 0); del buf44  # reuse
        # Source Nodes: [padded_value_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf60, buf61, 1179648, grid=grid(1179648), stream=stream0)
        buf62 = buf23; del buf23  # reuse
        # Source Nodes: [chunked_hidden_states_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf57, buf58, buf59, buf62, 9461760, grid=grid(9461760), stream=stream0)
        buf63 = buf24; del buf24  # reuse
        # Source Nodes: [context_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf61, buf63, 2359296, grid=grid(2359296), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (48, 256, 64), (16384, 64, 1), 0); del buf60  # reuse
        # Source Nodes: [context_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf63, (48, 768, 64), (49152, 64, 1), 0), out=buf64)
        buf65 = reinterpret_tensor(buf40, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [reshape_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf64, buf65, 786432, grid=grid(786432), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (1024, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (1024, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf66)
        del arg22_1
        buf70 = reinterpret_tensor(buf65, (1, 1024, 768), (786432, 768, 1), 0); del buf65  # reuse
        # Source Nodes: [add_8, attn_output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf66, arg23_1, buf38, arg24_1, arg25_1, buf70, 1024, 768, grid=grid(1024), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf71 = reinterpret_tensor(buf33, (1024, 3072), (3072, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (1024, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 3072), (1, 768), 0), out=buf71)
        del arg26_1
        buf72 = reinterpret_tensor(buf71, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf71  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf72, arg27_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg27_1
        buf73 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg28_1, (3072, 768), (1, 3072), 0), out=buf73)
        del arg28_1
        buf77 = buf38; del buf38  # reuse
        # Source Nodes: [add_9, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf73, arg29_1, buf70, arg30_1, arg31_1, buf77, 1024, 768, grid=grid(1024), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf78 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (1024, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 768), (1, 768), 0), out=buf78)
        del arg32_1
        buf79 = reinterpret_tensor(buf78, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf78  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf79, arg33_1, 786432, grid=grid(786432), stream=stream0)
        del arg33_1
        buf80 = reinterpret_tensor(buf70, (1024, 768), (768, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (1024, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 768), (1, 768), 0), out=buf80)
        del arg34_1
        buf81 = reinterpret_tensor(buf80, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf80  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf81, arg35_1, 786432, grid=grid(786432), stream=stream0)
        del arg35_1
        buf82 = reinterpret_tensor(buf61, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf61  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf79, buf82, 1179648, grid=grid(1179648), stream=stream0)
        buf83 = reinterpret_tensor(buf43, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf43  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf81, buf83, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf84 = buf45; del buf45  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf83, (36, 64, 512), (32768, 512, 1), 0), out=buf84)
        buf85 = reinterpret_tensor(buf57, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf57  # reuse
        # Source Nodes: [diagonal_attention_scores_8, setitem_24, setitem_25], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf84, buf85, 6303744, grid=grid(6303744), stream=stream0)
        buf86 = reinterpret_tensor(buf49, (12, 256, 513), (131328, 513, 1), 0); del buf49  # reuse
        # Source Nodes: [setitem_27], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf84, buf85, buf86, 1575936, grid=grid(1575936), stream=stream0)
        buf87 = buf48; del buf48  # reuse
        # Source Nodes: [bool_9, full_like_8, where_8], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf86, buf84, buf85, buf87, 789504, grid=grid(789504), stream=stream0)
        buf88 = reinterpret_tensor(buf47, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf47  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf87, buf86, buf84, buf85, buf88, 1575936, grid=grid(1575936), stream=stream0)
        buf89 = buf50; del buf50  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf88, buf86, buf84, buf85, buf89, 6303744, grid=grid(6303744), stream=stream0)
        buf90 = buf52; del buf52  # reuse
        # Source Nodes: [hidden_states_31], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf90, 1024, grid=grid(1024), stream=stream0)
        buf91 = buf51; del buf51  # reuse
        buf130 = buf12; del buf12  # reuse
        # Source Nodes: [hidden_states_32, hidden_states_46], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf91, buf130, 1024, grid=grid(1024), stream=stream0)
        buf92 = reinterpret_tensor(buf56, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf56  # reuse
        # Source Nodes: [diagonal_attention_scores_10, setitem_30, setitem_31], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf90, buf91, buf92, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf93 = buf54; del buf54  # reuse
        # Source Nodes: [setitem_33], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf90, buf91, buf92, buf93, 256, 513, grid=grid(256, 513), stream=stream0)
        buf94 = buf55; del buf55  # reuse
        # Source Nodes: [bool_11, full_like_10, where_10], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf93, buf90, buf91, buf92, buf94, 256, 257, grid=grid(256, 257), stream=stream0)
        buf95 = reinterpret_tensor(buf53, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf53  # reuse
        # Source Nodes: [setitem_34], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf94, buf93, buf90, buf91, buf92, buf95, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf96 = reinterpret_tensor(buf85, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf85  # reuse
        buf97 = buf59; del buf59  # reuse
        buf98 = buf58; del buf58  # reuse
        # Source Nodes: [attn_probs_8, attn_scores_5], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf89, buf95, buf96, buf97, buf98, 12288, 513, grid=grid(12288), stream=stream0)
        buf99 = reinterpret_tensor(buf81, (1024, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [value_vectors_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg37_1, reinterpret_tensor(buf77, (1024, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf99)
        del arg36_1
        del arg37_1
        buf100 = reinterpret_tensor(buf83, (12, 1536, 64), (98304, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [padded_value_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf99, buf100, 1179648, grid=grid(1179648), stream=stream0)
        buf101 = buf62; del buf62  # reuse
        # Source Nodes: [chunked_hidden_states_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf96, buf97, buf98, buf101, 9461760, grid=grid(9461760), stream=stream0)
        buf102 = buf63; del buf63  # reuse
        # Source Nodes: [context_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf100, buf102, 2359296, grid=grid(2359296), stream=stream0)
        buf103 = reinterpret_tensor(buf99, (48, 256, 64), (16384, 64, 1), 0); del buf99  # reuse
        # Source Nodes: [context_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf102, (48, 768, 64), (49152, 64, 1), 0), out=buf103)
        buf104 = reinterpret_tensor(buf79, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf79  # reuse
        # Source Nodes: [reshape_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf103, buf104, 786432, grid=grid(786432), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (1024, 768), (768, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (1024, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), out=buf105)
        del arg38_1
        buf109 = reinterpret_tensor(buf104, (1, 1024, 768), (786432, 768, 1), 0); del buf104  # reuse
        # Source Nodes: [add_13, attn_output_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf105, arg39_1, buf77, arg40_1, arg41_1, buf109, 1024, 768, grid=grid(1024), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        buf110 = reinterpret_tensor(buf72, (1024, 3072), (3072, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 3072), (1, 768), 0), out=buf110)
        del arg42_1
        buf111 = reinterpret_tensor(buf110, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf110  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf111, arg43_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg43_1
        buf112 = reinterpret_tensor(buf77, (1024, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg44_1, (3072, 768), (1, 3072), 0), out=buf112)
        del arg44_1
        buf116 = reinterpret_tensor(buf105, (1, 1024, 768), (786432, 768, 1), 0); del buf105  # reuse
        # Source Nodes: [add_14, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf112, arg45_1, buf109, arg46_1, arg47_1, buf116, 1024, 768, grid=grid(1024), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 768), (1, 768), 0), out=buf117)
        del arg48_1
        buf118 = reinterpret_tensor(buf117, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf117  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf118, arg49_1, 786432, grid=grid(786432), stream=stream0)
        del arg49_1
        buf119 = reinterpret_tensor(buf109, (1024, 768), (768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 768), (1, 768), 0), out=buf119)
        del arg50_1
        buf120 = reinterpret_tensor(buf119, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_44], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf120, arg51_1, 786432, grid=grid(786432), stream=stream0)
        del arg51_1
        buf121 = reinterpret_tensor(buf100, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf100  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf118, buf121, 1179648, grid=grid(1179648), stream=stream0)
        buf122 = reinterpret_tensor(buf82, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf82  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf120, buf122, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf123 = buf84; del buf84  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf122, (36, 64, 512), (32768, 512, 1), 0), out=buf123)
        buf124 = reinterpret_tensor(buf96, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf96  # reuse
        # Source Nodes: [diagonal_attention_scores_12, setitem_36, setitem_37], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf123, buf124, 6303744, grid=grid(6303744), stream=stream0)
        buf125 = reinterpret_tensor(buf88, (12, 256, 513), (131328, 513, 1), 0); del buf88  # reuse
        # Source Nodes: [setitem_39], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf123, buf124, buf125, 1575936, grid=grid(1575936), stream=stream0)
        buf126 = buf87; del buf87  # reuse
        # Source Nodes: [bool_13, full_like_12, where_12], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf125, buf123, buf124, buf126, 789504, grid=grid(789504), stream=stream0)
        buf127 = reinterpret_tensor(buf86, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf86  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf126, buf125, buf123, buf124, buf127, 1575936, grid=grid(1575936), stream=stream0)
        buf128 = buf89; del buf89  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf127, buf125, buf123, buf124, buf128, 6303744, grid=grid(6303744), stream=stream0)
        buf129 = buf91; del buf91  # reuse
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf129, 1024, grid=grid(1024), stream=stream0)
        buf131 = reinterpret_tensor(buf95, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf95  # reuse
        # Source Nodes: [diagonal_attention_scores_14, setitem_42, setitem_43], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf129, buf130, buf131, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf132 = buf93; del buf93  # reuse
        # Source Nodes: [setitem_45], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf129, buf130, buf131, buf132, 256, 513, grid=grid(256, 513), stream=stream0)
        buf133 = buf94; del buf94  # reuse
        # Source Nodes: [bool_15, full_like_14, where_14], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf132, buf129, buf130, buf131, buf133, 256, 257, grid=grid(256, 257), stream=stream0)
        buf134 = reinterpret_tensor(buf92, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf92  # reuse
        # Source Nodes: [setitem_46], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf133, buf132, buf129, buf130, buf131, buf134, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf135 = reinterpret_tensor(buf124, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf124  # reuse
        buf136 = buf98; del buf98  # reuse
        buf137 = buf97; del buf97  # reuse
        # Source Nodes: [attn_probs_12, attn_scores_7], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf128, buf134, buf135, buf136, buf137, 12288, 513, grid=grid(12288), stream=stream0)
        buf138 = reinterpret_tensor(buf120, (1024, 768), (768, 1), 0); del buf120  # reuse
        # Source Nodes: [value_vectors_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg53_1, reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf138)
        del arg52_1
        del arg53_1
        buf139 = reinterpret_tensor(buf122, (12, 1536, 64), (98304, 64, 1), 0); del buf122  # reuse
        # Source Nodes: [padded_value_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf138, buf139, 1179648, grid=grid(1179648), stream=stream0)
        buf140 = buf101; del buf101  # reuse
        # Source Nodes: [chunked_hidden_states_15], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf135, buf136, buf137, buf140, 9461760, grid=grid(9461760), stream=stream0)
        buf141 = buf102; del buf102  # reuse
        # Source Nodes: [context_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf139, buf141, 2359296, grid=grid(2359296), stream=stream0)
        buf142 = reinterpret_tensor(buf138, (48, 256, 64), (16384, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [context_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf140, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf141, (48, 768, 64), (49152, 64, 1), 0), out=buf142)
        buf143 = reinterpret_tensor(buf118, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [reshape_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf142, buf143, 786432, grid=grid(786432), stream=stream0)
        buf144 = reinterpret_tensor(buf142, (1024, 768), (768, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1024, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), out=buf144)
        del arg54_1
        buf148 = reinterpret_tensor(buf143, (1, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
        # Source Nodes: [add_18, attn_output_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf144, arg55_1, buf116, arg56_1, arg57_1, buf148, 1024, 768, grid=grid(1024), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf149 = reinterpret_tensor(buf111, (1024, 3072), (3072, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (1024, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 3072), (1, 768), 0), out=buf149)
        del arg58_1
        buf150 = reinterpret_tensor(buf149, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf149  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf150, arg59_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg59_1
        buf151 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg60_1, (3072, 768), (1, 3072), 0), out=buf151)
        del arg60_1
        buf155 = buf116; del buf116  # reuse
        # Source Nodes: [add_19, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf151, arg61_1, buf148, arg62_1, arg63_1, buf155, 1024, 768, grid=grid(1024), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf156 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1024, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 768), (1, 768), 0), out=buf156)
        del arg64_1
        buf157 = reinterpret_tensor(buf156, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf156  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf157, arg65_1, 786432, grid=grid(786432), stream=stream0)
        del arg65_1
        buf158 = reinterpret_tensor(buf148, (1024, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1024, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 768), (1, 768), 0), out=buf158)
        del arg66_1
        buf159 = reinterpret_tensor(buf158, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf158  # reuse
        # Source Nodes: [hidden_states_58], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf159, arg67_1, 786432, grid=grid(786432), stream=stream0)
        del arg67_1
        buf160 = reinterpret_tensor(buf139, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf139  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf157, buf160, 1179648, grid=grid(1179648), stream=stream0)
        buf161 = reinterpret_tensor(buf121, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf121  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf159, buf161, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf162 = buf123; del buf123  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf161, (36, 64, 512), (32768, 512, 1), 0), out=buf162)
        buf163 = reinterpret_tensor(buf135, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf135  # reuse
        # Source Nodes: [diagonal_attention_scores_16, setitem_48, setitem_49], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf162, buf163, 6303744, grid=grid(6303744), stream=stream0)
        buf164 = reinterpret_tensor(buf127, (12, 256, 513), (131328, 513, 1), 0); del buf127  # reuse
        # Source Nodes: [setitem_51], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf162, buf163, buf164, 1575936, grid=grid(1575936), stream=stream0)
        buf165 = buf126; del buf126  # reuse
        # Source Nodes: [bool_17, full_like_16, where_16], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf164, buf162, buf163, buf165, 789504, grid=grid(789504), stream=stream0)
        buf166 = reinterpret_tensor(buf125, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf125  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf165, buf164, buf162, buf163, buf166, 1575936, grid=grid(1575936), stream=stream0)
        buf167 = buf128; del buf128  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf166, buf164, buf162, buf163, buf167, 6303744, grid=grid(6303744), stream=stream0)
        buf168 = buf130; del buf130  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf168, 1024, grid=grid(1024), stream=stream0)
        buf169 = buf129; del buf129  # reuse
        buf208 = buf90; del buf90  # reuse
        # Source Nodes: [hidden_states_60, hidden_states_74], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf169, buf208, 1024, grid=grid(1024), stream=stream0)
        buf170 = reinterpret_tensor(buf134, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf134  # reuse
        # Source Nodes: [diagonal_attention_scores_18, setitem_54, setitem_55], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf168, buf169, buf170, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf171 = buf132; del buf132  # reuse
        # Source Nodes: [setitem_57], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf168, buf169, buf170, buf171, 256, 513, grid=grid(256, 513), stream=stream0)
        buf172 = buf133; del buf133  # reuse
        # Source Nodes: [bool_19, full_like_18, where_18], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf171, buf168, buf169, buf170, buf172, 256, 257, grid=grid(256, 257), stream=stream0)
        buf173 = reinterpret_tensor(buf131, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf131  # reuse
        # Source Nodes: [setitem_58], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf172, buf171, buf168, buf169, buf170, buf173, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf174 = reinterpret_tensor(buf163, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf163  # reuse
        buf175 = buf137; del buf137  # reuse
        buf176 = buf136; del buf136  # reuse
        # Source Nodes: [attn_probs_16, attn_scores_9], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf167, buf173, buf174, buf175, buf176, 12288, 513, grid=grid(12288), stream=stream0)
        buf177 = reinterpret_tensor(buf159, (1024, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [value_vectors_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg69_1, reinterpret_tensor(buf155, (1024, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg68_1
        del arg69_1
        buf178 = reinterpret_tensor(buf161, (12, 1536, 64), (98304, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [padded_value_4], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf177, buf178, 1179648, grid=grid(1179648), stream=stream0)
        buf179 = buf140; del buf140  # reuse
        # Source Nodes: [chunked_hidden_states_20], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf174, buf175, buf176, buf179, 9461760, grid=grid(9461760), stream=stream0)
        buf180 = buf141; del buf141  # reuse
        # Source Nodes: [context_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf178, buf180, 2359296, grid=grid(2359296), stream=stream0)
        buf181 = reinterpret_tensor(buf177, (48, 256, 64), (16384, 64, 1), 0); del buf177  # reuse
        # Source Nodes: [context_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf180, (48, 768, 64), (49152, 64, 1), 0), out=buf181)
        buf182 = reinterpret_tensor(buf157, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf157  # reuse
        # Source Nodes: [reshape_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf181, buf182, 786432, grid=grid(786432), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (1024, 768), (768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), out=buf183)
        del arg70_1
        buf187 = reinterpret_tensor(buf182, (1, 1024, 768), (786432, 768, 1), 0); del buf182  # reuse
        # Source Nodes: [add_23, attn_output_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf183, arg71_1, buf155, arg72_1, arg73_1, buf187, 1024, 768, grid=grid(1024), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        buf188 = reinterpret_tensor(buf150, (1024, 3072), (3072, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1024, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg74_1
        buf189 = reinterpret_tensor(buf188, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf188  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf189, arg75_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg75_1
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg76_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg76_1
        buf194 = buf155; del buf155  # reuse
        # Source Nodes: [add_24, hidden_states_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf190, arg77_1, buf187, arg78_1, arg79_1, buf194, 1024, 768, grid=grid(1024), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        buf195 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0), out=buf195)
        del arg80_1
        buf196 = reinterpret_tensor(buf195, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf195  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf196, arg81_1, 786432, grid=grid(786432), stream=stream0)
        del arg81_1
        buf197 = reinterpret_tensor(buf187, (1024, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 768), (1, 768), 0), out=buf197)
        del arg82_1
        buf198 = reinterpret_tensor(buf197, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf197  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf198, arg83_1, 786432, grid=grid(786432), stream=stream0)
        del arg83_1
        buf199 = reinterpret_tensor(buf178, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf178  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf196, buf199, 1179648, grid=grid(1179648), stream=stream0)
        buf200 = reinterpret_tensor(buf160, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf160  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf198, buf200, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf201 = buf162; del buf162  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf200, (36, 64, 512), (32768, 512, 1), 0), out=buf201)
        buf202 = reinterpret_tensor(buf174, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf174  # reuse
        # Source Nodes: [diagonal_attention_scores_20, setitem_60, setitem_61], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf201, buf202, 6303744, grid=grid(6303744), stream=stream0)
        buf203 = reinterpret_tensor(buf166, (12, 256, 513), (131328, 513, 1), 0); del buf166  # reuse
        # Source Nodes: [setitem_63], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf201, buf202, buf203, 1575936, grid=grid(1575936), stream=stream0)
        buf204 = buf165; del buf165  # reuse
        # Source Nodes: [bool_21, full_like_20, where_20], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf203, buf201, buf202, buf204, 789504, grid=grid(789504), stream=stream0)
        buf205 = reinterpret_tensor(buf164, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf164  # reuse
        # Source Nodes: [setitem_64], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf204, buf203, buf201, buf202, buf205, 1575936, grid=grid(1575936), stream=stream0)
        buf206 = buf167; del buf167  # reuse
        # Source Nodes: [setitem_64], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf205, buf203, buf201, buf202, buf206, 6303744, grid=grid(6303744), stream=stream0)
        buf207 = buf169; del buf169  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf207, 1024, grid=grid(1024), stream=stream0)
        buf209 = reinterpret_tensor(buf173, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf173  # reuse
        # Source Nodes: [diagonal_attention_scores_22, setitem_66, setitem_67], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf207, buf208, buf209, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf210 = buf171; del buf171  # reuse
        # Source Nodes: [setitem_69], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf207, buf208, buf209, buf210, 256, 513, grid=grid(256, 513), stream=stream0)
        buf211 = buf172; del buf172  # reuse
        # Source Nodes: [bool_23, full_like_22, where_22], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf210, buf207, buf208, buf209, buf211, 256, 257, grid=grid(256, 257), stream=stream0)
        buf212 = reinterpret_tensor(buf170, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf170  # reuse
        # Source Nodes: [setitem_70], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf211, buf210, buf207, buf208, buf209, buf212, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf213 = reinterpret_tensor(buf202, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf202  # reuse
        buf214 = buf176; del buf176  # reuse
        buf215 = buf175; del buf175  # reuse
        # Source Nodes: [attn_probs_20, attn_scores_11], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf206, buf212, buf213, buf214, buf215, 12288, 513, grid=grid(12288), stream=stream0)
        buf216 = reinterpret_tensor(buf198, (1024, 768), (768, 1), 0); del buf198  # reuse
        # Source Nodes: [value_vectors_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
        del arg84_1
        del arg85_1
        buf217 = reinterpret_tensor(buf200, (12, 1536, 64), (98304, 64, 1), 0); del buf200  # reuse
        # Source Nodes: [padded_value_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf216, buf217, 1179648, grid=grid(1179648), stream=stream0)
        buf218 = buf179; del buf179  # reuse
        # Source Nodes: [chunked_hidden_states_25], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf213, buf214, buf215, buf218, 9461760, grid=grid(9461760), stream=stream0)
        buf219 = buf180; del buf180  # reuse
        # Source Nodes: [context_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf217, buf219, 2359296, grid=grid(2359296), stream=stream0)
        buf220 = reinterpret_tensor(buf216, (48, 256, 64), (16384, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [context_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf218, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf219, (48, 768, 64), (49152, 64, 1), 0), out=buf220)
        buf221 = reinterpret_tensor(buf196, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [reshape_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf220, buf221, 786432, grid=grid(786432), stream=stream0)
        buf222 = reinterpret_tensor(buf220, (1024, 768), (768, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (1024, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf222)
        del arg86_1
        buf226 = reinterpret_tensor(buf221, (1, 1024, 768), (786432, 768, 1), 0); del buf221  # reuse
        # Source Nodes: [add_28, attn_output_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf222, arg87_1, buf194, arg88_1, arg89_1, buf226, 1024, 768, grid=grid(1024), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        buf227 = reinterpret_tensor(buf189, (1024, 3072), (3072, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1024, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 3072), (1, 768), 0), out=buf227)
        del arg90_1
        buf228 = reinterpret_tensor(buf227, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf227  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf228, arg91_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg91_1
        buf229 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg92_1, (3072, 768), (1, 3072), 0), out=buf229)
        del arg92_1
        buf233 = buf194; del buf194  # reuse
        # Source Nodes: [add_29, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf229, arg93_1, buf226, arg94_1, arg95_1, buf233, 1024, 768, grid=grid(1024), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        buf234 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (1024, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 768), (1, 768), 0), out=buf234)
        del arg96_1
        buf235 = reinterpret_tensor(buf234, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf234  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf235, arg97_1, 786432, grid=grid(786432), stream=stream0)
        del arg97_1
        buf236 = reinterpret_tensor(buf226, (1024, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (1024, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), out=buf236)
        del arg98_1
        buf237 = reinterpret_tensor(buf236, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf236  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf237, arg99_1, 786432, grid=grid(786432), stream=stream0)
        del arg99_1
        buf238 = reinterpret_tensor(buf217, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf217  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf235, buf238, 1179648, grid=grid(1179648), stream=stream0)
        buf239 = reinterpret_tensor(buf199, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf199  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf237, buf239, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf240 = buf201; del buf201  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf239, (36, 64, 512), (32768, 512, 1), 0), out=buf240)
        buf241 = reinterpret_tensor(buf213, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf213  # reuse
        # Source Nodes: [diagonal_attention_scores_24, setitem_72, setitem_73], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf240, buf241, 6303744, grid=grid(6303744), stream=stream0)
        buf242 = reinterpret_tensor(buf205, (12, 256, 513), (131328, 513, 1), 0); del buf205  # reuse
        # Source Nodes: [setitem_75], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf240, buf241, buf242, 1575936, grid=grid(1575936), stream=stream0)
        buf243 = buf204; del buf204  # reuse
        # Source Nodes: [bool_25, full_like_24, where_24], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf242, buf240, buf241, buf243, 789504, grid=grid(789504), stream=stream0)
        buf244 = reinterpret_tensor(buf203, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf203  # reuse
        # Source Nodes: [setitem_76], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf243, buf242, buf240, buf241, buf244, 1575936, grid=grid(1575936), stream=stream0)
        buf245 = buf206; del buf206  # reuse
        # Source Nodes: [setitem_76], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf244, buf242, buf240, buf241, buf245, 6303744, grid=grid(6303744), stream=stream0)
        buf246 = buf208; del buf208  # reuse
        # Source Nodes: [hidden_states_87], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf246, 1024, grid=grid(1024), stream=stream0)
        buf247 = buf207; del buf207  # reuse
        buf286 = buf168; del buf168  # reuse
        # Source Nodes: [hidden_states_102, hidden_states_88], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf247, buf286, 1024, grid=grid(1024), stream=stream0)
        buf248 = reinterpret_tensor(buf212, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf212  # reuse
        # Source Nodes: [diagonal_attention_scores_26, setitem_78, setitem_79], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf246, buf247, buf248, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf249 = buf210; del buf210  # reuse
        # Source Nodes: [setitem_81], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf246, buf247, buf248, buf249, 256, 513, grid=grid(256, 513), stream=stream0)
        buf250 = buf211; del buf211  # reuse
        # Source Nodes: [bool_27, full_like_26, where_26], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf249, buf246, buf247, buf248, buf250, 256, 257, grid=grid(256, 257), stream=stream0)
        buf251 = reinterpret_tensor(buf209, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf209  # reuse
        # Source Nodes: [setitem_82], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf250, buf249, buf246, buf247, buf248, buf251, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf252 = reinterpret_tensor(buf241, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf241  # reuse
        buf253 = buf215; del buf215  # reuse
        buf254 = buf214; del buf214  # reuse
        # Source Nodes: [attn_probs_24, attn_scores_13], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf245, buf251, buf252, buf253, buf254, 12288, 513, grid=grid(12288), stream=stream0)
        buf255 = reinterpret_tensor(buf237, (1024, 768), (768, 1), 0); del buf237  # reuse
        # Source Nodes: [value_vectors_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf233, (1024, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf255)
        del arg100_1
        del arg101_1
        buf256 = reinterpret_tensor(buf239, (12, 1536, 64), (98304, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [padded_value_6], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf255, buf256, 1179648, grid=grid(1179648), stream=stream0)
        buf257 = buf218; del buf218  # reuse
        # Source Nodes: [chunked_hidden_states_30], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf252, buf253, buf254, buf257, 9461760, grid=grid(9461760), stream=stream0)
        buf258 = buf219; del buf219  # reuse
        # Source Nodes: [context_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf256, buf258, 2359296, grid=grid(2359296), stream=stream0)
        buf259 = reinterpret_tensor(buf255, (48, 256, 64), (16384, 64, 1), 0); del buf255  # reuse
        # Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf258, (48, 768, 64), (49152, 64, 1), 0), out=buf259)
        buf260 = reinterpret_tensor(buf235, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [reshape_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf259, buf260, 786432, grid=grid(786432), stream=stream0)
        buf261 = reinterpret_tensor(buf259, (1024, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (1024, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf261)
        del arg102_1
        buf265 = reinterpret_tensor(buf260, (1, 1024, 768), (786432, 768, 1), 0); del buf260  # reuse
        # Source Nodes: [add_33, attn_output_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf261, arg103_1, buf233, arg104_1, arg105_1, buf265, 1024, 768, grid=grid(1024), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        buf266 = reinterpret_tensor(buf228, (1024, 3072), (3072, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (1024, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), out=buf266)
        del arg106_1
        buf267 = reinterpret_tensor(buf266, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf266  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf267, arg107_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg107_1
        buf268 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), out=buf268)
        del arg108_1
        buf272 = buf233; del buf233  # reuse
        # Source Nodes: [add_34, hidden_states_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf268, arg109_1, buf265, arg110_1, arg111_1, buf272, 1024, 768, grid=grid(1024), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        buf273 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (1024, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), out=buf273)
        del arg112_1
        buf274 = reinterpret_tensor(buf273, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf273  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf274, arg113_1, 786432, grid=grid(786432), stream=stream0)
        del arg113_1
        buf275 = reinterpret_tensor(buf265, (1024, 768), (768, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (1024, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), out=buf275)
        del arg114_1
        buf276 = reinterpret_tensor(buf275, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_100], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf276, arg115_1, 786432, grid=grid(786432), stream=stream0)
        del arg115_1
        buf277 = reinterpret_tensor(buf256, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf256  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf274, buf277, 1179648, grid=grid(1179648), stream=stream0)
        buf278 = reinterpret_tensor(buf238, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf238  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf276, buf278, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf279 = buf240; del buf240  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf277, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf278, (36, 64, 512), (32768, 512, 1), 0), out=buf279)
        buf280 = reinterpret_tensor(buf252, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf252  # reuse
        # Source Nodes: [diagonal_attention_scores_28, setitem_84, setitem_85], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf279, buf280, 6303744, grid=grid(6303744), stream=stream0)
        buf281 = reinterpret_tensor(buf244, (12, 256, 513), (131328, 513, 1), 0); del buf244  # reuse
        # Source Nodes: [setitem_87], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf279, buf280, buf281, 1575936, grid=grid(1575936), stream=stream0)
        buf282 = buf243; del buf243  # reuse
        # Source Nodes: [bool_29, full_like_28, where_28], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf281, buf279, buf280, buf282, 789504, grid=grid(789504), stream=stream0)
        buf283 = reinterpret_tensor(buf242, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf242  # reuse
        # Source Nodes: [setitem_88], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf282, buf281, buf279, buf280, buf283, 1575936, grid=grid(1575936), stream=stream0)
        buf284 = buf245; del buf245  # reuse
        # Source Nodes: [setitem_88], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf283, buf281, buf279, buf280, buf284, 6303744, grid=grid(6303744), stream=stream0)
        buf285 = buf247; del buf247  # reuse
        # Source Nodes: [hidden_states_101], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf285, 1024, grid=grid(1024), stream=stream0)
        buf287 = reinterpret_tensor(buf251, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf251  # reuse
        # Source Nodes: [diagonal_attention_scores_30, setitem_90, setitem_91], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf285, buf286, buf287, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf288 = buf249; del buf249  # reuse
        # Source Nodes: [setitem_93], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf285, buf286, buf287, buf288, 256, 513, grid=grid(256, 513), stream=stream0)
        buf289 = buf250; del buf250  # reuse
        # Source Nodes: [bool_31, full_like_30, where_30], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf288, buf285, buf286, buf287, buf289, 256, 257, grid=grid(256, 257), stream=stream0)
        buf290 = reinterpret_tensor(buf248, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf248  # reuse
        # Source Nodes: [setitem_94], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf289, buf288, buf285, buf286, buf287, buf290, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf291 = reinterpret_tensor(buf280, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf280  # reuse
        buf292 = buf254; del buf254  # reuse
        buf293 = buf253; del buf253  # reuse
        # Source Nodes: [attn_probs_28, attn_scores_15], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf284, buf290, buf291, buf292, buf293, 12288, 513, grid=grid(12288), stream=stream0)
        buf294 = reinterpret_tensor(buf276, (1024, 768), (768, 1), 0); del buf276  # reuse
        # Source Nodes: [value_vectors_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg117_1, reinterpret_tensor(buf272, (1024, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf294)
        del arg116_1
        del arg117_1
        buf295 = reinterpret_tensor(buf278, (12, 1536, 64), (98304, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [padded_value_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf294, buf295, 1179648, grid=grid(1179648), stream=stream0)
        buf296 = buf257; del buf257  # reuse
        # Source Nodes: [chunked_hidden_states_35], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf291, buf292, buf293, buf296, 9461760, grid=grid(9461760), stream=stream0)
        buf297 = buf258; del buf258  # reuse
        # Source Nodes: [context_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf295, buf297, 2359296, grid=grid(2359296), stream=stream0)
        buf298 = reinterpret_tensor(buf294, (48, 256, 64), (16384, 64, 1), 0); del buf294  # reuse
        # Source Nodes: [context_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf297, (48, 768, 64), (49152, 64, 1), 0), out=buf298)
        buf299 = reinterpret_tensor(buf274, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [reshape_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf298, buf299, 786432, grid=grid(786432), stream=stream0)
        buf300 = reinterpret_tensor(buf298, (1024, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (1024, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf300)
        del arg118_1
        buf304 = reinterpret_tensor(buf299, (1, 1024, 768), (786432, 768, 1), 0); del buf299  # reuse
        # Source Nodes: [add_38, attn_output_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf300, arg119_1, buf272, arg120_1, arg121_1, buf304, 1024, 768, grid=grid(1024), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        buf305 = reinterpret_tensor(buf267, (1024, 3072), (3072, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (1024, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 3072), (1, 768), 0), out=buf305)
        del arg122_1
        buf306 = reinterpret_tensor(buf305, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf305  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf306, arg123_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg123_1
        buf307 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg124_1, (3072, 768), (1, 3072), 0), out=buf307)
        del arg124_1
        buf311 = buf272; del buf272  # reuse
        # Source Nodes: [add_39, hidden_states_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf307, arg125_1, buf304, arg126_1, arg127_1, buf311, 1024, 768, grid=grid(1024), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        buf312 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf312)
        del arg128_1
        buf313 = reinterpret_tensor(buf312, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf312  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf313, arg129_1, 786432, grid=grid(786432), stream=stream0)
        del arg129_1
        buf314 = reinterpret_tensor(buf304, (1024, 768), (768, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 768), (1, 768), 0), out=buf314)
        del arg130_1
        buf315 = reinterpret_tensor(buf314, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf314  # reuse
        # Source Nodes: [hidden_states_114], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf315, arg131_1, 786432, grid=grid(786432), stream=stream0)
        del arg131_1
        buf316 = reinterpret_tensor(buf295, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf295  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf313, buf316, 1179648, grid=grid(1179648), stream=stream0)
        buf317 = reinterpret_tensor(buf277, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf277  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf315, buf317, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf318 = buf279; del buf279  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf317, (36, 64, 512), (32768, 512, 1), 0), out=buf318)
        buf319 = reinterpret_tensor(buf291, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf291  # reuse
        # Source Nodes: [diagonal_attention_scores_32, setitem_96, setitem_97], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf318, buf319, 6303744, grid=grid(6303744), stream=stream0)
        buf320 = reinterpret_tensor(buf283, (12, 256, 513), (131328, 513, 1), 0); del buf283  # reuse
        # Source Nodes: [setitem_99], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf318, buf319, buf320, 1575936, grid=grid(1575936), stream=stream0)
        buf321 = buf282; del buf282  # reuse
        # Source Nodes: [bool_33, full_like_32, where_32], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf320, buf318, buf319, buf321, 789504, grid=grid(789504), stream=stream0)
        buf322 = reinterpret_tensor(buf281, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf281  # reuse
        # Source Nodes: [setitem_100], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf321, buf320, buf318, buf319, buf322, 1575936, grid=grid(1575936), stream=stream0)
        buf323 = buf284; del buf284  # reuse
        # Source Nodes: [setitem_100], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf322, buf320, buf318, buf319, buf323, 6303744, grid=grid(6303744), stream=stream0)
        buf324 = buf286; del buf286  # reuse
        # Source Nodes: [hidden_states_115], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf324, 1024, grid=grid(1024), stream=stream0)
        buf325 = buf285; del buf285  # reuse
        buf364 = buf246; del buf246  # reuse
        # Source Nodes: [hidden_states_116, hidden_states_130], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf325, buf364, 1024, grid=grid(1024), stream=stream0)
        buf326 = reinterpret_tensor(buf290, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf290  # reuse
        # Source Nodes: [diagonal_attention_scores_34, setitem_102, setitem_103], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf324, buf325, buf326, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf327 = buf288; del buf288  # reuse
        # Source Nodes: [setitem_105], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf324, buf325, buf326, buf327, 256, 513, grid=grid(256, 513), stream=stream0)
        buf328 = buf289; del buf289  # reuse
        # Source Nodes: [bool_35, full_like_34, where_34], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf327, buf324, buf325, buf326, buf328, 256, 257, grid=grid(256, 257), stream=stream0)
        buf329 = reinterpret_tensor(buf287, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf287  # reuse
        # Source Nodes: [setitem_106], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf328, buf327, buf324, buf325, buf326, buf329, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf330 = reinterpret_tensor(buf319, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf319  # reuse
        buf331 = buf293; del buf293  # reuse
        buf332 = buf292; del buf292  # reuse
        # Source Nodes: [attn_probs_32, attn_scores_17], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf323, buf329, buf330, buf331, buf332, 12288, 513, grid=grid(12288), stream=stream0)
        buf333 = reinterpret_tensor(buf315, (1024, 768), (768, 1), 0); del buf315  # reuse
        # Source Nodes: [value_vectors_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg133_1, reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf333)
        del arg132_1
        del arg133_1
        buf334 = reinterpret_tensor(buf317, (12, 1536, 64), (98304, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [padded_value_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf333, buf334, 1179648, grid=grid(1179648), stream=stream0)
        buf335 = buf296; del buf296  # reuse
        # Source Nodes: [chunked_hidden_states_40], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf330, buf331, buf332, buf335, 9461760, grid=grid(9461760), stream=stream0)
        buf336 = buf297; del buf297  # reuse
        # Source Nodes: [context_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf334, buf336, 2359296, grid=grid(2359296), stream=stream0)
        buf337 = reinterpret_tensor(buf333, (48, 256, 64), (16384, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [context_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf336, (48, 768, 64), (49152, 64, 1), 0), out=buf337)
        buf338 = reinterpret_tensor(buf313, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [reshape_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf337, buf338, 786432, grid=grid(786432), stream=stream0)
        buf339 = reinterpret_tensor(buf337, (1024, 768), (768, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf338, (1024, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), out=buf339)
        del arg134_1
        buf343 = reinterpret_tensor(buf338, (1, 1024, 768), (786432, 768, 1), 0); del buf338  # reuse
        # Source Nodes: [add_43, attn_output_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf339, arg135_1, buf311, arg136_1, arg137_1, buf343, 1024, 768, grid=grid(1024), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        buf344 = reinterpret_tensor(buf306, (1024, 3072), (3072, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (1024, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 3072), (1, 768), 0), out=buf344)
        del arg138_1
        buf345 = reinterpret_tensor(buf344, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf344  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf345, arg139_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg139_1
        buf346 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg140_1, (3072, 768), (1, 3072), 0), out=buf346)
        del arg140_1
        buf350 = buf311; del buf311  # reuse
        # Source Nodes: [add_44, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf346, arg141_1, buf343, arg142_1, arg143_1, buf350, 1024, 768, grid=grid(1024), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        buf351 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (1024, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 768), (1, 768), 0), out=buf351)
        del arg144_1
        buf352 = reinterpret_tensor(buf351, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf351  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf352, arg145_1, 786432, grid=grid(786432), stream=stream0)
        del arg145_1
        buf353 = reinterpret_tensor(buf343, (1024, 768), (768, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (1024, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), out=buf353)
        del arg146_1
        buf354 = reinterpret_tensor(buf353, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf353  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf354, arg147_1, 786432, grid=grid(786432), stream=stream0)
        del arg147_1
        buf355 = reinterpret_tensor(buf334, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf334  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf352, buf355, 1179648, grid=grid(1179648), stream=stream0)
        buf356 = reinterpret_tensor(buf316, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf316  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf354, buf356, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf357 = buf318; del buf318  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf356, (36, 64, 512), (32768, 512, 1), 0), out=buf357)
        buf358 = reinterpret_tensor(buf330, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf330  # reuse
        # Source Nodes: [diagonal_attention_scores_36, setitem_108, setitem_109], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf357, buf358, 6303744, grid=grid(6303744), stream=stream0)
        buf359 = reinterpret_tensor(buf322, (12, 256, 513), (131328, 513, 1), 0); del buf322  # reuse
        # Source Nodes: [setitem_111], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf357, buf358, buf359, 1575936, grid=grid(1575936), stream=stream0)
        buf360 = buf321; del buf321  # reuse
        # Source Nodes: [bool_37, full_like_36, where_36], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf359, buf357, buf358, buf360, 789504, grid=grid(789504), stream=stream0)
        buf361 = reinterpret_tensor(buf320, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf320  # reuse
        # Source Nodes: [setitem_112], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf360, buf359, buf357, buf358, buf361, 1575936, grid=grid(1575936), stream=stream0)
        buf362 = buf323; del buf323  # reuse
        # Source Nodes: [setitem_112], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf361, buf359, buf357, buf358, buf362, 6303744, grid=grid(6303744), stream=stream0)
        buf363 = buf325; del buf325  # reuse
        # Source Nodes: [hidden_states_129], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf363, 1024, grid=grid(1024), stream=stream0)
        buf365 = reinterpret_tensor(buf329, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf329  # reuse
        # Source Nodes: [diagonal_attention_scores_38, setitem_114, setitem_115], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf363, buf364, buf365, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf366 = buf327; del buf327  # reuse
        # Source Nodes: [setitem_117], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf363, buf364, buf365, buf366, 256, 513, grid=grid(256, 513), stream=stream0)
        buf367 = buf328; del buf328  # reuse
        # Source Nodes: [bool_39, full_like_38, where_38], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf366, buf363, buf364, buf365, buf367, 256, 257, grid=grid(256, 257), stream=stream0)
        buf368 = reinterpret_tensor(buf326, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf326  # reuse
        # Source Nodes: [setitem_118], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf367, buf366, buf363, buf364, buf365, buf368, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf369 = reinterpret_tensor(buf358, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf358  # reuse
        buf370 = buf332; del buf332  # reuse
        buf371 = buf331; del buf331  # reuse
        # Source Nodes: [attn_probs_36, attn_scores_19], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf362, buf368, buf369, buf370, buf371, 12288, 513, grid=grid(12288), stream=stream0)
        buf372 = reinterpret_tensor(buf354, (1024, 768), (768, 1), 0); del buf354  # reuse
        # Source Nodes: [value_vectors_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg149_1, reinterpret_tensor(buf350, (1024, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf372)
        del arg148_1
        del arg149_1
        buf373 = reinterpret_tensor(buf356, (12, 1536, 64), (98304, 64, 1), 0); del buf356  # reuse
        # Source Nodes: [padded_value_9], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf372, buf373, 1179648, grid=grid(1179648), stream=stream0)
        buf374 = buf335; del buf335  # reuse
        # Source Nodes: [chunked_hidden_states_45], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf369, buf370, buf371, buf374, 9461760, grid=grid(9461760), stream=stream0)
        buf375 = buf336; del buf336  # reuse
        # Source Nodes: [context_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf373, buf375, 2359296, grid=grid(2359296), stream=stream0)
        buf376 = reinterpret_tensor(buf372, (48, 256, 64), (16384, 64, 1), 0); del buf372  # reuse
        # Source Nodes: [context_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf375, (48, 768, 64), (49152, 64, 1), 0), out=buf376)
        buf377 = reinterpret_tensor(buf352, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf352  # reuse
        # Source Nodes: [reshape_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf376, buf377, 786432, grid=grid(786432), stream=stream0)
        buf378 = reinterpret_tensor(buf376, (1024, 768), (768, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1024, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), out=buf378)
        del arg150_1
        buf382 = reinterpret_tensor(buf377, (1, 1024, 768), (786432, 768, 1), 0); del buf377  # reuse
        # Source Nodes: [add_48, attn_output_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf378, arg151_1, buf350, arg152_1, arg153_1, buf382, 1024, 768, grid=grid(1024), stream=stream0)
        del arg151_1
        del arg152_1
        del arg153_1
        buf383 = reinterpret_tensor(buf345, (1024, 3072), (3072, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (1024, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 3072), (1, 768), 0), out=buf383)
        del arg154_1
        buf384 = reinterpret_tensor(buf383, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf383  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf384, arg155_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg155_1
        buf385 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf384, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg156_1, (3072, 768), (1, 3072), 0), out=buf385)
        del arg156_1
        buf389 = buf350; del buf350  # reuse
        # Source Nodes: [add_49, hidden_states_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf385, arg157_1, buf382, arg158_1, arg159_1, buf389, 1024, 768, grid=grid(1024), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        buf390 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1024, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), out=buf390)
        del arg160_1
        buf391 = reinterpret_tensor(buf390, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf390  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf391, arg161_1, 786432, grid=grid(786432), stream=stream0)
        del arg161_1
        buf392 = reinterpret_tensor(buf382, (1024, 768), (768, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1024, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), out=buf392)
        del arg162_1
        buf393 = reinterpret_tensor(buf392, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf392  # reuse
        # Source Nodes: [hidden_states_142], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf393, arg163_1, 786432, grid=grid(786432), stream=stream0)
        del arg163_1
        buf394 = reinterpret_tensor(buf373, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf373  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf391, buf394, 1179648, grid=grid(1179648), stream=stream0)
        buf395 = reinterpret_tensor(buf355, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf355  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf393, buf395, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf396 = buf357; del buf357  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf394, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf395, (36, 64, 512), (32768, 512, 1), 0), out=buf396)
        buf397 = reinterpret_tensor(buf369, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf369  # reuse
        # Source Nodes: [diagonal_attention_scores_40, setitem_120, setitem_121], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf396, buf397, 6303744, grid=grid(6303744), stream=stream0)
        buf398 = reinterpret_tensor(buf361, (12, 256, 513), (131328, 513, 1), 0); del buf361  # reuse
        # Source Nodes: [setitem_123], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf396, buf397, buf398, 1575936, grid=grid(1575936), stream=stream0)
        buf399 = buf360; del buf360  # reuse
        # Source Nodes: [bool_41, full_like_40, where_40], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf398, buf396, buf397, buf399, 789504, grid=grid(789504), stream=stream0)
        buf400 = reinterpret_tensor(buf359, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf359  # reuse
        # Source Nodes: [setitem_124], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf399, buf398, buf396, buf397, buf400, 1575936, grid=grid(1575936), stream=stream0)
        buf401 = buf362; del buf362  # reuse
        # Source Nodes: [setitem_124], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf400, buf398, buf396, buf397, buf401, 6303744, grid=grid(6303744), stream=stream0)
        buf402 = buf364; del buf364  # reuse
        # Source Nodes: [hidden_states_143], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf402, 1024, grid=grid(1024), stream=stream0)
        buf403 = buf363; del buf363  # reuse
        buf442 = buf324; del buf324  # reuse
        # Source Nodes: [hidden_states_144, hidden_states_158], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(arg193_1, buf403, buf442, 1024, grid=grid(1024), stream=stream0)
        del arg193_1
        buf404 = reinterpret_tensor(buf368, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf368  # reuse
        # Source Nodes: [diagonal_attention_scores_42, setitem_126, setitem_127], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf402, buf403, buf404, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf405 = buf366; del buf366  # reuse
        # Source Nodes: [setitem_129], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf402, buf403, buf404, buf405, 256, 513, grid=grid(256, 513), stream=stream0)
        buf406 = buf367; del buf367  # reuse
        # Source Nodes: [bool_43, full_like_42, where_42], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf405, buf402, buf403, buf404, buf406, 256, 257, grid=grid(256, 257), stream=stream0)
        buf407 = reinterpret_tensor(buf365, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf365  # reuse
        # Source Nodes: [setitem_130], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf406, buf405, buf402, buf403, buf404, buf407, 1024, 513, grid=grid(1024, 513), stream=stream0)
        del buf402
        buf408 = reinterpret_tensor(buf397, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf397  # reuse
        buf409 = buf371; del buf371  # reuse
        buf410 = buf370; del buf370  # reuse
        # Source Nodes: [attn_probs_40, attn_scores_21], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf401, buf407, buf408, buf409, buf410, 12288, 513, grid=grid(12288), stream=stream0)
        buf411 = reinterpret_tensor(buf393, (1024, 768), (768, 1), 0); del buf393  # reuse
        # Source Nodes: [value_vectors_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg165_1, reinterpret_tensor(buf389, (1024, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf411)
        del arg164_1
        del arg165_1
        buf412 = reinterpret_tensor(buf395, (12, 1536, 64), (98304, 64, 1), 0); del buf395  # reuse
        # Source Nodes: [padded_value_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf411, buf412, 1179648, grid=grid(1179648), stream=stream0)
        buf413 = buf374; del buf374  # reuse
        # Source Nodes: [chunked_hidden_states_50], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf408, buf409, buf410, buf413, 9461760, grid=grid(9461760), stream=stream0)
        buf414 = buf375; del buf375  # reuse
        # Source Nodes: [context_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf412, buf414, 2359296, grid=grid(2359296), stream=stream0)
        buf415 = reinterpret_tensor(buf411, (48, 256, 64), (16384, 64, 1), 0); del buf411  # reuse
        # Source Nodes: [context_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf413, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf414, (48, 768, 64), (49152, 64, 1), 0), out=buf415)
        buf416 = reinterpret_tensor(buf391, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf391  # reuse
        # Source Nodes: [reshape_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf415, buf416, 786432, grid=grid(786432), stream=stream0)
        buf417 = reinterpret_tensor(buf415, (1024, 768), (768, 1), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (1024, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), out=buf417)
        del arg166_1
        buf421 = reinterpret_tensor(buf416, (1, 1024, 768), (786432, 768, 1), 0); del buf416  # reuse
        # Source Nodes: [add_53, attn_output_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf417, arg167_1, buf389, arg168_1, arg169_1, buf421, 1024, 768, grid=grid(1024), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        buf422 = reinterpret_tensor(buf384, (1024, 3072), (3072, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (1024, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 3072), (1, 768), 0), out=buf422)
        del arg170_1
        buf423 = reinterpret_tensor(buf422, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf422  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf423, arg171_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg171_1
        buf424 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg172_1, (3072, 768), (1, 3072), 0), out=buf424)
        del arg172_1
        buf428 = buf389; del buf389  # reuse
        # Source Nodes: [add_54, hidden_states_153], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf424, arg173_1, buf421, arg174_1, arg175_1, buf428, 1024, 768, grid=grid(1024), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        buf429 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (1024, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 768), (1, 768), 0), out=buf429)
        del arg176_1
        buf430 = reinterpret_tensor(buf429, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf429  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf430, arg177_1, 786432, grid=grid(786432), stream=stream0)
        del arg177_1
        buf431 = reinterpret_tensor(buf421, (1024, 768), (768, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (1024, 768), (768, 1), 0), reinterpret_tensor(arg178_1, (768, 768), (1, 768), 0), out=buf431)
        del arg178_1
        buf432 = reinterpret_tensor(buf431, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf431  # reuse
        # Source Nodes: [hidden_states_156], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf432, arg179_1, 786432, grid=grid(786432), stream=stream0)
        del arg179_1
        buf433 = reinterpret_tensor(buf412, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf412  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf430, buf433, 1179648, grid=grid(1179648), stream=stream0)
        buf434 = reinterpret_tensor(buf394, (12, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf394  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf432, buf434, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf435 = buf396; del buf396  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf433, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf434, (36, 64, 512), (32768, 512, 1), 0), out=buf435)
        del buf433
        buf436 = reinterpret_tensor(buf408, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf408  # reuse
        # Source Nodes: [diagonal_attention_scores_44, setitem_132, setitem_133], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf435, buf436, 6303744, grid=grid(6303744), stream=stream0)
        buf437 = reinterpret_tensor(buf400, (12, 256, 513), (131328, 513, 1), 0); del buf400  # reuse
        # Source Nodes: [setitem_135], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_5.run(buf435, buf436, buf437, 1575936, grid=grid(1575936), stream=stream0)
        buf438 = buf399; del buf399  # reuse
        # Source Nodes: [bool_45, full_like_44, where_44], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_6.run(buf437, buf435, buf436, buf438, 789504, grid=grid(789504), stream=stream0)
        buf439 = reinterpret_tensor(buf398, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf398  # reuse
        # Source Nodes: [setitem_136], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_7.run(buf438, buf437, buf435, buf436, buf439, 1575936, grid=grid(1575936), stream=stream0)
        del buf438
        buf440 = buf401; del buf401  # reuse
        # Source Nodes: [setitem_136], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf439, buf437, buf435, buf436, buf440, 6303744, grid=grid(6303744), stream=stream0)
        del buf435
        del buf437
        del buf439
        buf441 = buf403; del buf403  # reuse
        # Source Nodes: [hidden_states_157], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf441, 1024, grid=grid(1024), stream=stream0)
        buf443 = reinterpret_tensor(buf407, (1, 4, 256, 513), (525312, 256, 1, 1024), 0); del buf407  # reuse
        # Source Nodes: [diagonal_attention_scores_46, setitem_138, setitem_139], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf441, buf442, buf443, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf444 = buf405; del buf405  # reuse
        # Source Nodes: [setitem_141], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf441, buf442, buf443, buf444, 256, 513, grid=grid(256, 513), stream=stream0)
        buf445 = buf406; del buf406  # reuse
        # Source Nodes: [bool_47, full_like_46, where_46], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf444, buf441, buf442, buf443, buf445, 256, 257, grid=grid(256, 257), stream=stream0)
        buf446 = reinterpret_tensor(buf404, (1, 1024, 1, 513), (525312, 513, 525312, 1), 0); del buf404  # reuse
        # Source Nodes: [setitem_142], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf445, buf444, buf441, buf442, buf443, buf446, 1024, 513, grid=grid(1024, 513), stream=stream0)
        del buf441
        del buf442
        del buf443
        del buf444
        del buf445
        buf447 = reinterpret_tensor(buf436, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf436  # reuse
        buf448 = buf410; del buf410  # reuse
        buf449 = buf409; del buf409  # reuse
        # Source Nodes: [attn_probs_44, attn_scores_23], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_15.run(buf440, buf446, buf447, buf448, buf449, 12288, 513, grid=grid(12288), stream=stream0)
        del buf440
        del buf446
        buf450 = reinterpret_tensor(buf432, (1024, 768), (768, 1), 0); del buf432  # reuse
        # Source Nodes: [value_vectors_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg181_1, reinterpret_tensor(buf428, (1024, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf450)
        del arg180_1
        del arg181_1
        buf451 = reinterpret_tensor(buf434, (12, 1536, 64), (98304, 64, 1), 0); del buf434  # reuse
        # Source Nodes: [padded_value_11], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf450, buf451, 1179648, grid=grid(1179648), stream=stream0)
        buf452 = buf413; del buf413  # reuse
        # Source Nodes: [chunked_hidden_states_55], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg194_1, buf447, buf448, buf449, buf452, 9461760, grid=grid(9461760), stream=stream0)
        del arg194_1
        del buf447
        del buf448
        del buf449
        buf453 = buf414; del buf414  # reuse
        # Source Nodes: [context_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf451, buf453, 2359296, grid=grid(2359296), stream=stream0)
        del buf451
        buf454 = reinterpret_tensor(buf450, (48, 256, 64), (16384, 64, 1), 0); del buf450  # reuse
        # Source Nodes: [context_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf453, (48, 768, 64), (49152, 64, 1), 0), out=buf454)
        del buf452
        del buf453
        buf455 = reinterpret_tensor(buf430, (1024, 1, 12, 64), (768, 768, 64, 1), 0); del buf430  # reuse
        # Source Nodes: [reshape_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf454, buf455, 786432, grid=grid(786432), stream=stream0)
        buf456 = reinterpret_tensor(buf454, (1024, 768), (768, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf455, (1024, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), out=buf456)
        del arg182_1
        buf460 = reinterpret_tensor(buf455, (1, 1024, 768), (786432, 768, 1), 0); del buf455  # reuse
        # Source Nodes: [add_58, attn_output_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf456, arg183_1, buf428, arg184_1, arg185_1, buf460, 1024, 768, grid=grid(1024), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        buf461 = reinterpret_tensor(buf423, (1024, 3072), (3072, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (1024, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 3072), (1, 768), 0), out=buf461)
        del arg186_1
        buf462 = reinterpret_tensor(buf461, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf461  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf462, arg187_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg187_1
        buf463 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg188_1, (3072, 768), (1, 3072), 0), out=buf463)
        del arg188_1
        del buf462
        buf467 = buf428; del buf428  # reuse
        # Source Nodes: [add_59, hidden_states_167, hidden_states_168], Original ATen: [aten.add, aten.native_layer_norm, aten.slice]
        triton_per_fused_add_native_layer_norm_20.run(buf463, arg189_1, buf460, arg190_1, arg191_1, buf467, 1024, 768, grid=grid(1024), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        return (buf467, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.bool)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
