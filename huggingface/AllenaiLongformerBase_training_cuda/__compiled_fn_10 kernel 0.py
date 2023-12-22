
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


# kernel path: /tmp/torchinductor_youkaichao/yd/cydxkcfowkbinpweg2fv63qkt2sgv3vfs7voxn5ffxi2xfm2qzep.py
# Source Nodes: [hidden_states_2], Original ATen: [aten.view]
# hidden_states_2 => view_13
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/hq/chqcea2ngeamcg54t7nphbfa2pdnvqe57ywct7gpgvjafkgph5ct.py
# Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
# diagonal_chunked_attention_scores => view_17
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
    tmp3 = 8.0
    tmp4 = tmp2 / tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgq6dvxgexn4o3h6ql5a2nakub4yabzmjdrydqyzpp7dqxvcso7h.py
# Source Nodes: [beginning_mask], Original ATen: [aten.slice]
# beginning_mask => slice_64
triton_poi_fused_slice_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257)
    x2 = xindex
    tmp0 = (-255) + x0 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/cls7zj7jvvozoiqr7uq3e5fjr77tvfx2v52ki46rwztpvihkwlsl.py
# Source Nodes: [bool_1, full_like, setitem_3, setitem_4, where], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
# bool_1 => convert_element_type
# full_like => full_default_2
# setitem_3 => copy_3, slice_scatter_11, slice_scatter_12
# setitem_4 => copy_4, slice_scatter_14
# where => where_1
triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp53 = tl.full([1], 257, tl.int64)
    tmp54 = tmp3 < tmp53
    tmp55 = tl.load(in_ptr2 + (x0 + (257*x1)), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = (tmp55 != 0)
    tmp57 = tl.full([1], 0, tl.int32)
    tmp58 = tmp57 == tmp57
    tmp59 = tmp19 & tmp54
    tmp60 = tmp6 & tmp59
    tmp61 = tmp23 & tmp60
    tmp62 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp61 & xmask, other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp60, tmp64, tmp65)
    tmp67 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp59 & xmask, other=0.0)
    tmp68 = tl.where(tmp6, tmp66, tmp67)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp59, tmp68, tmp69)
    tmp71 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp54 & xmask, other=0.0)
    tmp72 = tl.where(tmp19, tmp70, tmp71)
    tmp73 = tl.where(tmp58, tmp52, tmp72)
    tmp74 = float("-inf")
    tmp75 = tl.where(tmp56, tmp74, tmp73)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp54, tmp75, tmp76)
    tmp78 = tl.where(tmp58, tmp52, tmp51)
    tmp79 = tl.where(tmp54, tmp77, tmp78)
    tl.store(out_ptr0 + (x4), tmp52, xmask)
    tl.store(out_ptr1 + (x4), tmp79, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/cluvs3wuxyxokfc4jcoltnwyhhehpfwvr5jsx5ll77xcqowunyyb.py
# Source Nodes: [ending_mask], Original ATen: [aten.flip]
# ending_mask => rev_1
triton_poi_fused_flip_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_flip_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257)
    x2 = xindex
    tmp0 = 256 + ((-1)*x0) + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlcu5xfmazh4z5slpm6rr4aliuae57lvfebqr6srfue4uwczocf.py
# Source Nodes: [hidden_states_4], Original ATen: [aten.view]
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/tf/ctf24ahfharag27ngfcaxuvtzblykihbfxkidusq6jfivjapyoyq.py
# Source Nodes: [attn_probs, attn_probs_1, attn_scores_1, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
# attn_probs => amax, clone_2, div_7, exp, sub_4, sum_1
# attn_probs_1 => where_7
# attn_scores_1 => add_2
# tril => full_default_1
triton_per_fused__softmax_add_detach_masked_fill_tril_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_detach_masked_fill_tril_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp18 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask, other=0.0)
    tmp35 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r2, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((-197632) + r2 + (257*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp6, other=0.0)
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp2, other=0.0)
    tmp15 = tl.where(tmp5, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp19 = tl.where(tmp2, tmp17, tmp18)
    tmp20 = 1280 + ((-1)*r2) + ((-1)*x1)
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp20 <= tmp21
    tmp23 = 1.0
    tmp24 = 0.0
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = (tmp25 != 0)
    tmp27 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp26, tmp10, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp6, tmp28, tmp29)
    tmp31 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp5, tmp30, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tmp19 + tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask, tmp38, float("-inf"))
    tmp41 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp40, 0))
    tmp42 = tmp37 - tmp41
    tmp43 = tl.exp(tmp42)
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp49 = tmp43 / tmp47
    tmp50 = tl.where(tmp48, tmp24, tmp49)
    tl.store(out_ptr3 + (r2 + (513*x3)), tmp50, rmask)
    tl.store(out_ptr4 + (r2 + (513*x3)), tmp49, rmask)
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmmbmcjc3rfsymo2ga3sapinnofmnyjlquiqc4a2hs2ds7edktf.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9461760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x1 = (xindex // 770) % 1024
    x2 = (xindex // 788480)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (513*x2) + (6156*x1)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lhtv2ge7mw437pukbsms7og2zkx5kl2qm6y4ra6l6vwcdupwgr.py
# Source Nodes: [context], Original ATen: [aten.clone]
# context => clone_3
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


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4nz5lakfltb26bilcelyemwvndcuga4iwx6tjgaaph6reb4usp.py
# Source Nodes: [hidden_states_5], Original ATen: [aten.view]
# hidden_states_5 => view_69
triton_poi_fused_view_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybtpgupanwksya7ubiczp2sn42bs6ija7t47khxklxicwkg4fdb.py
# Source Nodes: [add_3, attn_output_3, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_4
# attn_output_3 => add_5, add_6, mul_1, mul_2, rsqrt, sub_6, var_mean
# hidden_states_8 => view_71
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp22 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ep/cep22wujy6btzwy3jz3vrldtpc45hzhifewcvven4i7pklaavp5z.py
# Source Nodes: [hidden_states_10, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_10 => view_73
# intermediate_output => add_7, erf, mul_3, mul_4, mul_5
triton_poi_fused_gelu_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
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


# kernel path: /tmp/torchinductor_youkaichao/uy/cuy4sligd5yjtej4l7vovghol4rzbxb2bbjn4qonv66m7wjoa5m2.py
# Source Nodes: [add_4, attn_output_3, hidden_states_13, query_vectors_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_4 => add_8
# attn_output_3 => add_6, mul_2
# hidden_states_13 => add_9, mul_6, rsqrt_1, sub_7, var_mean_1
# query_vectors_3 => view_75
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp26 = 1e-05
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195 = args
    args.clear()
    assert_size_stride(primals_1, (768, 768), (768, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (3072, 768), (768, 1))
    assert_size_stride(primals_12, (3072, ), (1, ))
    assert_size_stride(primals_13, (768, 3072), (3072, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768), (768, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, 768), (768, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, 768), (768, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (3072, 768), (768, 1))
    assert_size_stride(primals_28, (3072, ), (1, ))
    assert_size_stride(primals_29, (768, 3072), (3072, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768), (768, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, 768), (768, 1))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 768), (768, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, 768), (768, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (3072, 768), (768, 1))
    assert_size_stride(primals_44, (3072, ), (1, ))
    assert_size_stride(primals_45, (768, 3072), (3072, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768), (768, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, 768), (768, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768), (768, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, 768), (768, 1))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, 768), (768, 1))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (3072, 768), (768, 1))
    assert_size_stride(primals_76, (3072, ), (1, ))
    assert_size_stride(primals_77, (768, 3072), (3072, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, 768), (768, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (3072, 768), (768, 1))
    assert_size_stride(primals_92, (3072, ), (1, ))
    assert_size_stride(primals_93, (768, 3072), (3072, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768), (768, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 768), (768, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, 768), (768, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, 768), (768, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (3072, 768), (768, 1))
    assert_size_stride(primals_124, (3072, ), (1, ))
    assert_size_stride(primals_125, (768, 3072), (3072, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, 768), (768, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 768), (768, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, 768), (768, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (3072, 768), (768, 1))
    assert_size_stride(primals_140, (3072, ), (1, ))
    assert_size_stride(primals_141, (768, 3072), (3072, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768), (768, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, 768), (768, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, 768), (768, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (3072, 768), (768, 1))
    assert_size_stride(primals_156, (3072, ), (1, ))
    assert_size_stride(primals_157, (768, 3072), (3072, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, 768), (768, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 768), (768, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, 768), (768, 1))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (3072, 768), (768, 1))
    assert_size_stride(primals_172, (3072, ), (1, ))
    assert_size_stride(primals_173, (768, 3072), (3072, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768), (768, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, 768), (768, 1))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 768), (768, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768), (768, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (3072, 768), (768, 1))
    assert_size_stride(primals_188, (3072, ), (1, ))
    assert_size_stride(primals_189, (768, 3072), (3072, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(primals_194, (1, 1024), (1024, 1))
    assert_size_stride(primals_195, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_1, (768, 768), (1, 768), 0), out=buf0)
        buf1 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_3, (768, 768), (1, 768), 0), out=buf1)
        buf2 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf2)
        del primals_6
        buf3 = reinterpret_tensor(buf1, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf1  # reuse
        # Source Nodes: [hidden_states_2], Original ATen: [aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_view_0.run(buf3, primals_4, 786432, grid=grid(786432), stream=stream0)
        del primals_4
        buf4 = reinterpret_tensor(buf0, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf0  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf4, primals_2, 786432, grid=grid(786432), stream=stream0)
        del primals_2
        buf5 = empty((12, 3, 512, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, buf5, 1179648, grid=grid(1179648), stream=stream0)
        buf6 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf3, buf6, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf7 = empty((36, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf6, (36, 64, 512), (32768, 512, 1), 0), out=buf7)
        buf8 = empty((12, 4, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_attention_scores, setitem, setitem_1], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf7, buf8, 6303744, grid=grid(6303744), stream=stream0)
        buf10 = empty((1, 256, 1, 257), device='cuda', dtype=torch.float32)
        # Source Nodes: [beginning_mask], Original ATen: [aten.slice]
        triton_poi_fused_slice_5.run(buf10, 65792, grid=grid(65792), stream=stream0)
        buf9 = empty((12, 256, 513), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 256, 12, 513), (1575936, 513, 131328, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bool_1, full_like, setitem_3, setitem_4, where], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf7, buf8, buf10, buf9, buf12, 1575936, grid=grid(1575936), stream=stream0)
        buf11 = empty((1, 256, 1, 257), device='cuda', dtype=torch.float32)
        # Source Nodes: [ending_mask], Original ATen: [aten.flip]
        triton_poi_fused_flip_7.run(buf11, 65792, grid=grid(65792), stream=stream0)
        buf13 = empty((1, 1024, 12, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf12, buf9, buf7, buf8, buf13, 6303744, grid=grid(6303744), stream=stream0)
        buf14 = empty((1, 2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf14, 1024, grid=grid(1024), stream=stream0)
        buf15 = empty((1, 2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(primals_194, buf15, 1024, grid=grid(1024), stream=stream0)
        del primals_194
        buf16 = empty_strided((1, 4, 256, 513), (525312, 256, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_attention_scores_2, setitem_6, setitem_7], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11.run(buf14, buf15, buf16, 513, 1024, grid=grid(513, 1024), stream=stream0)
        buf17 = empty((1, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_9], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_12.run(buf14, buf15, buf16, buf17, 256, 513, grid=grid(256, 513), stream=stream0)
        buf18 = empty_strided((1, 256, 1, 257), (65792, 257, 65792, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_13.run(buf17, buf14, buf15, buf16, buf18, 256, 257, grid=grid(256, 257), stream=stream0)
        buf19 = empty_strided((1, 1024, 1, 513), (525312, 513, 525312, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [setitem_10], Original ATen: [aten.copy, aten.slice_scatter]
        triton_poi_fused_copy_slice_scatter_14.run(buf18, buf17, buf14, buf15, buf16, buf19, 1024, 513, grid=grid(1024, 513), stream=stream0)
        del buf16
        del buf17
        del buf18
        buf23 = reinterpret_tensor(buf8, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf8  # reuse
        buf571 = empty((1, 1024, 12, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs, attn_probs_1, attn_scores_1, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf13, buf19, primals_195, buf23, buf571, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs, attn_probs_1, attn_probs_3, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf24 = aten.native_dropout(buf23, 0.1, True)
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf27 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf2, buf27, 1179648, grid=grid(1179648), stream=stream0)
        buf28 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf25, buf28, 9461760, grid=grid(9461760), stream=stream0)
        buf29 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf27, buf29, 2359296, grid=grid(2359296), stream=stream0)
        buf30 = reinterpret_tensor(buf2, (48, 256, 64), (16384, 64, 1), 0); del buf2  # reuse
        # Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf29, (48, 768, 64), (49152, 64, 1), 0), out=buf30)
        buf31 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf30, buf31, 786432, grid=grid(786432), stream=stream0)
        buf32 = reinterpret_tensor(buf30, (1024, 768), (768, 1), 0); del buf30  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf31, reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del primals_8
        # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
        buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf39 = reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0); del buf32  # reuse
        buf40 = reinterpret_tensor(buf4, (1024, 768), (768, 1), 0); del buf4  # reuse
        buf570 = reinterpret_tensor(buf15, (1, 1024, 1), (1024, 1, 1), 0); del buf15  # reuse
        # Source Nodes: [add_3, attn_output_3, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf34, primals_193, primals_9, primals_10, buf39, buf40, buf570, 1024, 768, grid=grid(1024), stream=stream0)
        buf41 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf40, reinterpret_tensor(primals_11, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf41)
        del primals_12
        buf42 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf41, buf42, 3145728, grid=grid(3145728), stream=stream0)
        buf43 = reinterpret_tensor(buf34, (1024, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf42, reinterpret_tensor(primals_13, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf43)
        del primals_14
        # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
        buf44 = aten.native_dropout(reinterpret_tensor(buf43, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf50 = reinterpret_tensor(buf43, (1, 1024, 768), (786432, 768, 1), 0); del buf43  # reuse
        buf51 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf569 = reinterpret_tensor(buf14, (1, 1024, 1), (1024, 1, 1), 0); del buf14  # reuse
        # Source Nodes: [add_4, attn_output_3, hidden_states_13, query_vectors_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf45, buf39, primals_9, primals_10, primals_15, primals_16, buf50, buf51, buf569, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_10
        buf52 = reinterpret_tensor(buf45, (1024, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf51, reinterpret_tensor(primals_17, (768, 768), (1, 768), 0), out=buf52)
        buf53 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf51, reinterpret_tensor(primals_19, (768, 768), (1, 768), 0), out=buf53)
        buf54 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_22, buf51, reinterpret_tensor(primals_21, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
        del primals_22
        buf55 = reinterpret_tensor(buf53, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf53  # reuse
        # Source Nodes: [hidden_states_16], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf55, primals_20, 786432, grid=grid(786432), stream=stream0)
        del primals_20
        buf56 = reinterpret_tensor(buf52, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf52  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf56, primals_18, 786432, grid=grid(786432), stream=stream0)
        del primals_18
        buf57 = reinterpret_tensor(buf27, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf27  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf56, buf57, 1179648, grid=grid(1179648), stream=stream0)
        buf58 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf55, buf58, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf59 = buf7; del buf7  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf58, (36, 64, 512), (32768, 512, 1), 0), out=buf59)
        buf60 = reinterpret_tensor(buf25, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf25  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_12, setitem_13], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf59, buf60, 6303744, grid=grid(6303744), stream=stream0)
        buf61 = buf9; del buf9  # reuse
        buf62 = buf12; del buf12  # reuse
        # Source Nodes: [bool_1, full_like, setitem_15, setitem_16, where_4], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf59, buf60, buf10, buf61, buf62, 1575936, grid=grid(1575936), stream=stream0)
        buf63 = buf23; del buf23  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf62, buf61, buf59, buf60, buf63, 6303744, grid=grid(6303744), stream=stream0)
        buf67 = reinterpret_tensor(buf60, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf60  # reuse
        buf568 = buf13; del buf13  # reuse
        # Source Nodes: [attn_probs_4, attn_probs_5, attn_scores_3, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf63, buf19, primals_195, buf67, buf568, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_4, attn_probs_5, attn_probs_7, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf68 = aten.native_dropout(buf67, 0.1, True)
        buf69 = buf68[0]
        buf70 = buf68[1]
        del buf68
        buf71 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf54, buf71, 1179648, grid=grid(1179648), stream=stream0)
        buf72 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf69, buf72, 9461760, grid=grid(9461760), stream=stream0)
        buf73 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf71, buf73, 2359296, grid=grid(2359296), stream=stream0)
        buf74 = reinterpret_tensor(buf54, (48, 256, 64), (16384, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [context_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf73, (48, 768, 64), (49152, 64, 1), 0), out=buf74)
        buf75 = reinterpret_tensor(buf55, (1024, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [hidden_states_19], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf74, buf75, 786432, grid=grid(786432), stream=stream0)
        buf76 = reinterpret_tensor(buf74, (1024, 768), (768, 1), 0); del buf74  # reuse
        # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf75, reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf76)
        del primals_24
        # Source Nodes: [hidden_states_20], Original ATen: [aten.native_dropout]
        buf77 = aten.native_dropout(reinterpret_tensor(buf76, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf83 = reinterpret_tensor(buf76, (1, 1024, 768), (786432, 768, 1), 0); del buf76  # reuse
        buf84 = reinterpret_tensor(buf56, (1024, 768), (768, 1), 0); del buf56  # reuse
        buf567 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attn_output_7, hidden_states_13, hidden_states_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf78, buf50, primals_15, primals_16, primals_25, primals_26, buf83, buf84, buf567, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_16
        buf85 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_28, buf84, reinterpret_tensor(primals_27, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del primals_28
        buf86 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf85, buf86, 3145728, grid=grid(3145728), stream=stream0)
        buf87 = reinterpret_tensor(buf78, (1024, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_30, buf86, reinterpret_tensor(primals_29, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf87)
        del primals_30
        # Source Nodes: [hidden_states_25], Original ATen: [aten.native_dropout]
        buf88 = aten.native_dropout(reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        buf94 = reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
        buf95 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf566 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attn_output_7, hidden_states_27, query_vectors_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf89, buf83, primals_25, primals_26, primals_31, primals_32, buf94, buf95, buf566, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_26
        buf96 = reinterpret_tensor(buf89, (1024, 768), (768, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf95, reinterpret_tensor(primals_33, (768, 768), (1, 768), 0), out=buf96)
        buf97 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf95, reinterpret_tensor(primals_35, (768, 768), (1, 768), 0), out=buf97)
        buf98 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_38, buf95, reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf98)
        del primals_38
        buf99 = reinterpret_tensor(buf97, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf97  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf99, primals_36, 786432, grid=grid(786432), stream=stream0)
        del primals_36
        buf100 = reinterpret_tensor(buf96, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf96  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf100, primals_34, 786432, grid=grid(786432), stream=stream0)
        del primals_34
        buf101 = reinterpret_tensor(buf71, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf71  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf100, buf101, 1179648, grid=grid(1179648), stream=stream0)
        buf102 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf99, buf102, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf103 = buf59; del buf59  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf102, (36, 64, 512), (32768, 512, 1), 0), out=buf103)
        buf104 = reinterpret_tensor(buf69, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf69  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_24, setitem_25], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf103, buf104, 6303744, grid=grid(6303744), stream=stream0)
        buf105 = reinterpret_tensor(buf62, (12, 256, 513), (131328, 513, 1), 0); del buf62  # reuse
        buf106 = reinterpret_tensor(buf61, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf61  # reuse
        # Source Nodes: [bool_1, full_like, setitem_27, setitem_28, where_8], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf103, buf104, buf10, buf105, buf106, 1575936, grid=grid(1575936), stream=stream0)
        buf107 = buf67; del buf67  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf106, buf105, buf103, buf104, buf107, 6303744, grid=grid(6303744), stream=stream0)
        buf111 = reinterpret_tensor(buf104, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf104  # reuse
        buf565 = buf63; del buf63  # reuse
        # Source Nodes: [attn_probs_8, attn_probs_9, attn_scores_5, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf107, buf19, primals_195, buf111, buf565, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_11, attn_probs_8, attn_probs_9, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf112 = aten.native_dropout(buf111, 0.1, True)
        buf113 = buf112[0]
        buf114 = buf112[1]
        del buf112
        buf115 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf98, buf115, 1179648, grid=grid(1179648), stream=stream0)
        buf116 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf113, buf116, 9461760, grid=grid(9461760), stream=stream0)
        buf117 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf115, buf117, 2359296, grid=grid(2359296), stream=stream0)
        buf118 = reinterpret_tensor(buf98, (48, 256, 64), (16384, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [context_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf117, (48, 768, 64), (49152, 64, 1), 0), out=buf118)
        buf119 = reinterpret_tensor(buf99, (1024, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf118, buf119, 786432, grid=grid(786432), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (1024, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_40, buf119, reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
        del primals_40
        # Source Nodes: [hidden_states_34], Original ATen: [aten.native_dropout]
        buf121 = aten.native_dropout(reinterpret_tensor(buf120, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf122 = buf121[0]
        buf123 = buf121[1]
        del buf121
        buf127 = reinterpret_tensor(buf120, (1, 1024, 768), (786432, 768, 1), 0); del buf120  # reuse
        buf128 = reinterpret_tensor(buf100, (1024, 768), (768, 1), 0); del buf100  # reuse
        buf564 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, attn_output_11, hidden_states_27, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf122, buf94, primals_31, primals_32, primals_41, primals_42, buf127, buf128, buf564, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_32
        buf129 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_44, buf128, reinterpret_tensor(primals_43, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf129)
        del primals_44
        buf130 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_38, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf129, buf130, 3145728, grid=grid(3145728), stream=stream0)
        buf131 = reinterpret_tensor(buf122, (1024, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_46, buf130, reinterpret_tensor(primals_45, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf131)
        del primals_46
        # Source Nodes: [hidden_states_39], Original ATen: [aten.native_dropout]
        buf132 = aten.native_dropout(reinterpret_tensor(buf131, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf133 = buf132[0]
        buf134 = buf132[1]
        del buf132
        buf138 = reinterpret_tensor(buf131, (1, 1024, 768), (786432, 768, 1), 0); del buf131  # reuse
        buf139 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf563 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attn_output_11, hidden_states_41, query_vectors_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf133, buf127, primals_41, primals_42, primals_47, primals_48, buf138, buf139, buf563, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_42
        buf140 = reinterpret_tensor(buf133, (1024, 768), (768, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_49, (768, 768), (1, 768), 0), out=buf140)
        buf141 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_51, (768, 768), (1, 768), 0), out=buf141)
        buf142 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_54, buf139, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf142)
        del primals_54
        buf143 = reinterpret_tensor(buf141, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_44], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf143, primals_52, 786432, grid=grid(786432), stream=stream0)
        del primals_52
        buf144 = reinterpret_tensor(buf140, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf140  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf144, primals_50, 786432, grid=grid(786432), stream=stream0)
        del primals_50
        buf145 = reinterpret_tensor(buf115, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf115  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf144, buf145, 1179648, grid=grid(1179648), stream=stream0)
        buf146 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf143, buf146, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf147 = buf103; del buf103  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf146, (36, 64, 512), (32768, 512, 1), 0), out=buf147)
        buf148 = reinterpret_tensor(buf113, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf113  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_36, setitem_37], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf147, buf148, 6303744, grid=grid(6303744), stream=stream0)
        buf149 = reinterpret_tensor(buf106, (12, 256, 513), (131328, 513, 1), 0); del buf106  # reuse
        buf150 = reinterpret_tensor(buf105, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf105  # reuse
        # Source Nodes: [bool_1, full_like, setitem_39, setitem_40, where_12], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf147, buf148, buf10, buf149, buf150, 1575936, grid=grid(1575936), stream=stream0)
        buf151 = buf111; del buf111  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf150, buf149, buf147, buf148, buf151, 6303744, grid=grid(6303744), stream=stream0)
        buf155 = reinterpret_tensor(buf148, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf148  # reuse
        buf562 = buf107; del buf107  # reuse
        # Source Nodes: [attn_probs_12, attn_probs_13, attn_scores_7, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf151, buf19, primals_195, buf155, buf562, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_12, attn_probs_13, attn_probs_15, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf156 = aten.native_dropout(buf155, 0.1, True)
        buf157 = buf156[0]
        buf158 = buf156[1]
        del buf156
        buf159 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf142, buf159, 1179648, grid=grid(1179648), stream=stream0)
        buf160 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_15], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf157, buf160, 9461760, grid=grid(9461760), stream=stream0)
        buf161 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf159, buf161, 2359296, grid=grid(2359296), stream=stream0)
        buf162 = reinterpret_tensor(buf142, (48, 256, 64), (16384, 64, 1), 0); del buf142  # reuse
        # Source Nodes: [context_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf161, (48, 768, 64), (49152, 64, 1), 0), out=buf162)
        buf163 = reinterpret_tensor(buf143, (1024, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [hidden_states_47], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf162, buf163, 786432, grid=grid(786432), stream=stream0)
        buf164 = reinterpret_tensor(buf162, (1024, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_56, buf163, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf164)
        del primals_56
        # Source Nodes: [hidden_states_48], Original ATen: [aten.native_dropout]
        buf165 = aten.native_dropout(reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf171 = reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0); del buf164  # reuse
        buf172 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
        buf561 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attn_output_15, hidden_states_41, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf166, buf138, primals_47, primals_48, primals_57, primals_58, buf171, buf172, buf561, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_48
        buf173 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf172, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf173)
        del primals_60
        buf174 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_52, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf173, buf174, 3145728, grid=grid(3145728), stream=stream0)
        buf175 = reinterpret_tensor(buf166, (1024, 768), (768, 1), 0); del buf166  # reuse
        # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_62, buf174, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf175)
        del primals_62
        # Source Nodes: [hidden_states_53], Original ATen: [aten.native_dropout]
        buf176 = aten.native_dropout(reinterpret_tensor(buf175, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf177 = buf176[0]
        buf178 = buf176[1]
        del buf176
        buf182 = reinterpret_tensor(buf175, (1, 1024, 768), (786432, 768, 1), 0); del buf175  # reuse
        buf183 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf560 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_19, attn_output_15, hidden_states_55, query_vectors_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf177, buf171, primals_57, primals_58, primals_63, primals_64, buf182, buf183, buf560, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_58
        buf184 = reinterpret_tensor(buf177, (1024, 768), (768, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_65, (768, 768), (1, 768), 0), out=buf184)
        buf185 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_67, (768, 768), (1, 768), 0), out=buf185)
        buf186 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_70, buf183, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
        del primals_70
        buf187 = reinterpret_tensor(buf185, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf185  # reuse
        # Source Nodes: [hidden_states_58], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf187, primals_68, 786432, grid=grid(786432), stream=stream0)
        del primals_68
        buf188 = reinterpret_tensor(buf184, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf184  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf188, primals_66, 786432, grid=grid(786432), stream=stream0)
        del primals_66
        buf189 = reinterpret_tensor(buf159, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf159  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf188, buf189, 1179648, grid=grid(1179648), stream=stream0)
        buf190 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf187, buf190, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf191 = buf147; del buf147  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf190, (36, 64, 512), (32768, 512, 1), 0), out=buf191)
        buf192 = reinterpret_tensor(buf157, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf157  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_48, setitem_49], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf191, buf192, 6303744, grid=grid(6303744), stream=stream0)
        buf193 = reinterpret_tensor(buf150, (12, 256, 513), (131328, 513, 1), 0); del buf150  # reuse
        buf194 = reinterpret_tensor(buf149, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf149  # reuse
        # Source Nodes: [bool_1, full_like, setitem_51, setitem_52, where_16], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf191, buf192, buf10, buf193, buf194, 1575936, grid=grid(1575936), stream=stream0)
        buf195 = buf155; del buf155  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf194, buf193, buf191, buf192, buf195, 6303744, grid=grid(6303744), stream=stream0)
        buf199 = reinterpret_tensor(buf192, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf192  # reuse
        buf559 = buf151; del buf151  # reuse
        # Source Nodes: [attn_probs_16, attn_probs_17, attn_scores_9, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf195, buf19, primals_195, buf199, buf559, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_16, attn_probs_17, attn_probs_19, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf200 = aten.native_dropout(buf199, 0.1, True)
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_4], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf186, buf203, 1179648, grid=grid(1179648), stream=stream0)
        buf204 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_20], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf201, buf204, 9461760, grid=grid(9461760), stream=stream0)
        buf205 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf203, buf205, 2359296, grid=grid(2359296), stream=stream0)
        buf206 = reinterpret_tensor(buf186, (48, 256, 64), (16384, 64, 1), 0); del buf186  # reuse
        # Source Nodes: [context_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf205, (48, 768, 64), (49152, 64, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf187, (1024, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [hidden_states_61], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf206, buf207, 786432, grid=grid(786432), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (1024, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf207, reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf208)
        del primals_72
        # Source Nodes: [hidden_states_62], Original ATen: [aten.native_dropout]
        buf209 = aten.native_dropout(reinterpret_tensor(buf208, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf215 = reinterpret_tensor(buf208, (1, 1024, 768), (786432, 768, 1), 0); del buf208  # reuse
        buf216 = reinterpret_tensor(buf188, (1024, 768), (768, 1), 0); del buf188  # reuse
        buf558 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attn_output_19, hidden_states_55, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf210, buf182, primals_63, primals_64, primals_73, primals_74, buf215, buf216, buf558, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_64
        buf217 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_76, buf216, reinterpret_tensor(primals_75, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del primals_76
        buf218 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf217, buf218, 3145728, grid=grid(3145728), stream=stream0)
        buf219 = reinterpret_tensor(buf210, (1024, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_78, buf218, reinterpret_tensor(primals_77, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf219)
        del primals_78
        # Source Nodes: [hidden_states_67], Original ATen: [aten.native_dropout]
        buf220 = aten.native_dropout(reinterpret_tensor(buf219, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf221 = buf220[0]
        buf222 = buf220[1]
        del buf220
        buf226 = reinterpret_tensor(buf219, (1, 1024, 768), (786432, 768, 1), 0); del buf219  # reuse
        buf227 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf557 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attn_output_19, hidden_states_69, query_vectors_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf221, buf215, primals_73, primals_74, primals_79, primals_80, buf226, buf227, buf557, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_74
        buf228 = reinterpret_tensor(buf221, (1024, 768), (768, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf227, reinterpret_tensor(primals_81, (768, 768), (1, 768), 0), out=buf228)
        buf229 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf227, reinterpret_tensor(primals_83, (768, 768), (1, 768), 0), out=buf229)
        buf230 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_86, buf227, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf230)
        del primals_86
        buf231 = reinterpret_tensor(buf229, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf229  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf231, primals_84, 786432, grid=grid(786432), stream=stream0)
        del primals_84
        buf232 = reinterpret_tensor(buf228, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf228  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf232, primals_82, 786432, grid=grid(786432), stream=stream0)
        del primals_82
        buf233 = reinterpret_tensor(buf203, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf203  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf232, buf233, 1179648, grid=grid(1179648), stream=stream0)
        buf234 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf231, buf234, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf235 = buf191; del buf191  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf234, (36, 64, 512), (32768, 512, 1), 0), out=buf235)
        buf236 = reinterpret_tensor(buf201, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf201  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_60, setitem_61], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf235, buf236, 6303744, grid=grid(6303744), stream=stream0)
        buf237 = reinterpret_tensor(buf194, (12, 256, 513), (131328, 513, 1), 0); del buf194  # reuse
        buf238 = reinterpret_tensor(buf193, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf193  # reuse
        # Source Nodes: [bool_1, full_like, setitem_63, setitem_64, where_20], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf235, buf236, buf10, buf237, buf238, 1575936, grid=grid(1575936), stream=stream0)
        buf239 = buf199; del buf199  # reuse
        # Source Nodes: [setitem_64], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf238, buf237, buf235, buf236, buf239, 6303744, grid=grid(6303744), stream=stream0)
        buf243 = reinterpret_tensor(buf236, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf236  # reuse
        buf556 = buf195; del buf195  # reuse
        # Source Nodes: [attn_probs_20, attn_probs_21, attn_scores_11, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf239, buf19, primals_195, buf243, buf556, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_20, attn_probs_21, attn_probs_23, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf244 = aten.native_dropout(buf243, 0.1, True)
        buf245 = buf244[0]
        buf246 = buf244[1]
        del buf244
        buf247 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf230, buf247, 1179648, grid=grid(1179648), stream=stream0)
        buf248 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_25], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf245, buf248, 9461760, grid=grid(9461760), stream=stream0)
        buf249 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf247, buf249, 2359296, grid=grid(2359296), stream=stream0)
        buf250 = reinterpret_tensor(buf230, (48, 256, 64), (16384, 64, 1), 0); del buf230  # reuse
        # Source Nodes: [context_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf249, (48, 768, 64), (49152, 64, 1), 0), out=buf250)
        buf251 = reinterpret_tensor(buf231, (1024, 768), (768, 1), 0); del buf231  # reuse
        # Source Nodes: [hidden_states_75], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf250, buf251, 786432, grid=grid(786432), stream=stream0)
        buf252 = reinterpret_tensor(buf250, (1024, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf251, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf252)
        del primals_88
        # Source Nodes: [hidden_states_76], Original ATen: [aten.native_dropout]
        buf253 = aten.native_dropout(reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        buf259 = reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0); del buf252  # reuse
        buf260 = reinterpret_tensor(buf232, (1024, 768), (768, 1), 0); del buf232  # reuse
        buf555 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, attn_output_23, hidden_states_69, hidden_states_78], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf254, buf226, primals_79, primals_80, primals_89, primals_90, buf259, buf260, buf555, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_80
        buf261 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf260, reinterpret_tensor(primals_91, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del primals_92
        buf262 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf261, buf262, 3145728, grid=grid(3145728), stream=stream0)
        buf263 = reinterpret_tensor(buf254, (1024, 768), (768, 1), 0); del buf254  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_94, buf262, reinterpret_tensor(primals_93, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf263)
        del primals_94
        # Source Nodes: [hidden_states_81], Original ATen: [aten.native_dropout]
        buf264 = aten.native_dropout(reinterpret_tensor(buf263, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf265 = buf264[0]
        buf266 = buf264[1]
        del buf264
        buf270 = reinterpret_tensor(buf263, (1, 1024, 768), (786432, 768, 1), 0); del buf263  # reuse
        buf271 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf554 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attn_output_23, hidden_states_83, query_vectors_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf265, buf259, primals_89, primals_90, primals_95, primals_96, buf270, buf271, buf554, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_90
        buf272 = reinterpret_tensor(buf265, (1024, 768), (768, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_97, (768, 768), (1, 768), 0), out=buf272)
        buf273 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), out=buf273)
        buf274 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf271, reinterpret_tensor(primals_101, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf274)
        del primals_102
        buf275 = reinterpret_tensor(buf273, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf273  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf275, primals_100, 786432, grid=grid(786432), stream=stream0)
        del primals_100
        buf276 = reinterpret_tensor(buf272, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf272  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf276, primals_98, 786432, grid=grid(786432), stream=stream0)
        del primals_98
        buf277 = reinterpret_tensor(buf247, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf247  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf276, buf277, 1179648, grid=grid(1179648), stream=stream0)
        buf278 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf275, buf278, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf279 = buf235; del buf235  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf277, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf278, (36, 64, 512), (32768, 512, 1), 0), out=buf279)
        buf280 = reinterpret_tensor(buf245, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf245  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_72, setitem_73], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf279, buf280, 6303744, grid=grid(6303744), stream=stream0)
        buf281 = reinterpret_tensor(buf238, (12, 256, 513), (131328, 513, 1), 0); del buf238  # reuse
        buf282 = reinterpret_tensor(buf237, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf237  # reuse
        # Source Nodes: [bool_1, full_like, setitem_75, setitem_76, where_24], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf279, buf280, buf10, buf281, buf282, 1575936, grid=grid(1575936), stream=stream0)
        buf283 = buf243; del buf243  # reuse
        # Source Nodes: [setitem_76], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf282, buf281, buf279, buf280, buf283, 6303744, grid=grid(6303744), stream=stream0)
        buf287 = reinterpret_tensor(buf280, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf280  # reuse
        buf553 = buf239; del buf239  # reuse
        # Source Nodes: [attn_probs_24, attn_probs_25, attn_scores_13, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf283, buf19, primals_195, buf287, buf553, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_24, attn_probs_25, attn_probs_27, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf288 = aten.native_dropout(buf287, 0.1, True)
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf291 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_6], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf274, buf291, 1179648, grid=grid(1179648), stream=stream0)
        buf292 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_30], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf289, buf292, 9461760, grid=grid(9461760), stream=stream0)
        buf293 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf291, buf293, 2359296, grid=grid(2359296), stream=stream0)
        buf294 = reinterpret_tensor(buf274, (48, 256, 64), (16384, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf293, (48, 768, 64), (49152, 64, 1), 0), out=buf294)
        buf295 = reinterpret_tensor(buf275, (1024, 768), (768, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_89], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf294, buf295, 786432, grid=grid(786432), stream=stream0)
        buf296 = reinterpret_tensor(buf294, (1024, 768), (768, 1), 0); del buf294  # reuse
        # Source Nodes: [hidden_states_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_104, buf295, reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf296)
        del primals_104
        # Source Nodes: [hidden_states_90], Original ATen: [aten.native_dropout]
        buf297 = aten.native_dropout(reinterpret_tensor(buf296, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        buf303 = reinterpret_tensor(buf296, (1, 1024, 768), (786432, 768, 1), 0); del buf296  # reuse
        buf304 = reinterpret_tensor(buf276, (1024, 768), (768, 1), 0); del buf276  # reuse
        buf552 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attn_output_27, hidden_states_83, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf298, buf270, primals_95, primals_96, primals_105, primals_106, buf303, buf304, buf552, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_96
        buf305 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf304, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf305)
        del primals_108
        buf306 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_94, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf305, buf306, 3145728, grid=grid(3145728), stream=stream0)
        buf307 = reinterpret_tensor(buf298, (1024, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_110, buf306, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf307)
        del primals_110
        # Source Nodes: [hidden_states_95], Original ATen: [aten.native_dropout]
        buf308 = aten.native_dropout(reinterpret_tensor(buf307, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf314 = reinterpret_tensor(buf307, (1, 1024, 768), (786432, 768, 1), 0); del buf307  # reuse
        buf315 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf551 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, attn_output_27, hidden_states_97, query_vectors_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf309, buf303, primals_105, primals_106, primals_111, primals_112, buf314, buf315, buf551, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_106
        buf316 = reinterpret_tensor(buf309, (1024, 768), (768, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf315, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), out=buf316)
        buf317 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf315, reinterpret_tensor(primals_115, (768, 768), (1, 768), 0), out=buf317)
        buf318 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_118, buf315, reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf318)
        del primals_118
        buf319 = reinterpret_tensor(buf317, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf317  # reuse
        # Source Nodes: [hidden_states_100], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf319, primals_116, 786432, grid=grid(786432), stream=stream0)
        del primals_116
        buf320 = reinterpret_tensor(buf316, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf316  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf320, primals_114, 786432, grid=grid(786432), stream=stream0)
        del primals_114
        buf321 = reinterpret_tensor(buf291, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf291  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf320, buf321, 1179648, grid=grid(1179648), stream=stream0)
        buf322 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf319, buf322, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf323 = buf279; del buf279  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf322, (36, 64, 512), (32768, 512, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf289, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf289  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_84, setitem_85], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf323, buf324, 6303744, grid=grid(6303744), stream=stream0)
        buf325 = reinterpret_tensor(buf282, (12, 256, 513), (131328, 513, 1), 0); del buf282  # reuse
        buf326 = reinterpret_tensor(buf281, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf281  # reuse
        # Source Nodes: [bool_1, full_like, setitem_87, setitem_88, where_28], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf323, buf324, buf10, buf325, buf326, 1575936, grid=grid(1575936), stream=stream0)
        buf327 = buf287; del buf287  # reuse
        # Source Nodes: [setitem_88], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf326, buf325, buf323, buf324, buf327, 6303744, grid=grid(6303744), stream=stream0)
        buf331 = reinterpret_tensor(buf324, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf324  # reuse
        buf550 = buf283; del buf283  # reuse
        # Source Nodes: [attn_probs_28, attn_probs_29, attn_scores_15, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf327, buf19, primals_195, buf331, buf550, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_28, attn_probs_29, attn_probs_31, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf332 = aten.native_dropout(buf331, 0.1, True)
        buf333 = buf332[0]
        buf334 = buf332[1]
        del buf332
        buf335 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf318, buf335, 1179648, grid=grid(1179648), stream=stream0)
        buf336 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_35], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf333, buf336, 9461760, grid=grid(9461760), stream=stream0)
        buf337 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf335, buf337, 2359296, grid=grid(2359296), stream=stream0)
        buf338 = reinterpret_tensor(buf318, (48, 256, 64), (16384, 64, 1), 0); del buf318  # reuse
        # Source Nodes: [context_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf336, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf337, (48, 768, 64), (49152, 64, 1), 0), out=buf338)
        buf339 = reinterpret_tensor(buf319, (1024, 768), (768, 1), 0); del buf319  # reuse
        # Source Nodes: [hidden_states_103], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf338, buf339, 786432, grid=grid(786432), stream=stream0)
        buf340 = reinterpret_tensor(buf338, (1024, 768), (768, 1), 0); del buf338  # reuse
        # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf339, reinterpret_tensor(primals_119, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf340)
        del primals_120
        # Source Nodes: [hidden_states_104], Original ATen: [aten.native_dropout]
        buf341 = aten.native_dropout(reinterpret_tensor(buf340, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf342 = buf341[0]
        buf343 = buf341[1]
        del buf341
        buf347 = reinterpret_tensor(buf340, (1, 1024, 768), (786432, 768, 1), 0); del buf340  # reuse
        buf348 = reinterpret_tensor(buf320, (1024, 768), (768, 1), 0); del buf320  # reuse
        buf549 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, attn_output_31, hidden_states_106, hidden_states_97], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf342, buf314, primals_111, primals_112, primals_121, primals_122, buf347, buf348, buf549, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_112
        buf349 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_124, buf348, reinterpret_tensor(primals_123, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf349)
        del primals_124
        buf350 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_108, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf349, buf350, 3145728, grid=grid(3145728), stream=stream0)
        buf351 = reinterpret_tensor(buf342, (1024, 768), (768, 1), 0); del buf342  # reuse
        # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf350, reinterpret_tensor(primals_125, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf351)
        del primals_126
        # Source Nodes: [hidden_states_109], Original ATen: [aten.native_dropout]
        buf352 = aten.native_dropout(reinterpret_tensor(buf351, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf353 = buf352[0]
        buf354 = buf352[1]
        del buf352
        buf358 = reinterpret_tensor(buf351, (1, 1024, 768), (786432, 768, 1), 0); del buf351  # reuse
        buf359 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf548 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_39, attn_output_31, hidden_states_111, query_vectors_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf353, buf347, primals_121, primals_122, primals_127, primals_128, buf358, buf359, buf548, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_122
        buf360 = reinterpret_tensor(buf353, (1024, 768), (768, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf359, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), out=buf360)
        buf361 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf359, reinterpret_tensor(primals_131, (768, 768), (1, 768), 0), out=buf361)
        buf362 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_134, buf359, reinterpret_tensor(primals_133, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf362)
        del primals_134
        buf363 = reinterpret_tensor(buf361, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf361  # reuse
        # Source Nodes: [hidden_states_114], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf363, primals_132, 786432, grid=grid(786432), stream=stream0)
        del primals_132
        buf364 = reinterpret_tensor(buf360, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf360  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf364, primals_130, 786432, grid=grid(786432), stream=stream0)
        del primals_130
        buf365 = reinterpret_tensor(buf335, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf335  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf364, buf365, 1179648, grid=grid(1179648), stream=stream0)
        buf366 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf363, buf366, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf367 = buf323; del buf323  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf365, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf366, (36, 64, 512), (32768, 512, 1), 0), out=buf367)
        buf368 = reinterpret_tensor(buf333, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf333  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_96, setitem_97], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf367, buf368, 6303744, grid=grid(6303744), stream=stream0)
        buf369 = reinterpret_tensor(buf326, (12, 256, 513), (131328, 513, 1), 0); del buf326  # reuse
        buf370 = reinterpret_tensor(buf325, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf325  # reuse
        # Source Nodes: [bool_1, full_like, setitem_100, setitem_99, where_32], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf367, buf368, buf10, buf369, buf370, 1575936, grid=grid(1575936), stream=stream0)
        buf371 = buf331; del buf331  # reuse
        # Source Nodes: [setitem_100], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf370, buf369, buf367, buf368, buf371, 6303744, grid=grid(6303744), stream=stream0)
        buf375 = reinterpret_tensor(buf368, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf368  # reuse
        buf547 = buf327; del buf327  # reuse
        # Source Nodes: [attn_probs_32, attn_probs_33, attn_scores_17, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf371, buf19, primals_195, buf375, buf547, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_32, attn_probs_33, attn_probs_35, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf376 = aten.native_dropout(buf375, 0.1, True)
        buf377 = buf376[0]
        buf378 = buf376[1]
        del buf376
        buf379 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf362, buf379, 1179648, grid=grid(1179648), stream=stream0)
        buf380 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_40], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf377, buf380, 9461760, grid=grid(9461760), stream=stream0)
        buf381 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf379, buf381, 2359296, grid=grid(2359296), stream=stream0)
        buf382 = reinterpret_tensor(buf362, (48, 256, 64), (16384, 64, 1), 0); del buf362  # reuse
        # Source Nodes: [context_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf381, (48, 768, 64), (49152, 64, 1), 0), out=buf382)
        buf383 = reinterpret_tensor(buf363, (1024, 768), (768, 1), 0); del buf363  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf382, buf383, 786432, grid=grid(786432), stream=stream0)
        buf384 = reinterpret_tensor(buf382, (1024, 768), (768, 1), 0); del buf382  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf383, reinterpret_tensor(primals_135, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf384)
        del primals_136
        # Source Nodes: [hidden_states_118], Original ATen: [aten.native_dropout]
        buf385 = aten.native_dropout(reinterpret_tensor(buf384, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf386 = buf385[0]
        buf387 = buf385[1]
        del buf385
        buf391 = reinterpret_tensor(buf384, (1, 1024, 768), (786432, 768, 1), 0); del buf384  # reuse
        buf392 = reinterpret_tensor(buf364, (1024, 768), (768, 1), 0); del buf364  # reuse
        buf546 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_43, attn_output_35, hidden_states_111, hidden_states_120], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf386, buf358, primals_127, primals_128, primals_137, primals_138, buf391, buf392, buf546, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_128
        buf393 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_120], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_140, buf392, reinterpret_tensor(primals_139, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf393)
        del primals_140
        buf394 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_122, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf393, buf394, 3145728, grid=grid(3145728), stream=stream0)
        buf395 = reinterpret_tensor(buf386, (1024, 768), (768, 1), 0); del buf386  # reuse
        # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_142, buf394, reinterpret_tensor(primals_141, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf395)
        del primals_142
        # Source Nodes: [hidden_states_123], Original ATen: [aten.native_dropout]
        buf396 = aten.native_dropout(reinterpret_tensor(buf395, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf402 = reinterpret_tensor(buf395, (1, 1024, 768), (786432, 768, 1), 0); del buf395  # reuse
        buf403 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf545 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, attn_output_35, hidden_states_125, query_vectors_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf397, buf391, primals_137, primals_138, primals_143, primals_144, buf402, buf403, buf545, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_138
        buf404 = reinterpret_tensor(buf397, (1024, 768), (768, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf403, reinterpret_tensor(primals_145, (768, 768), (1, 768), 0), out=buf404)
        buf405 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf403, reinterpret_tensor(primals_147, (768, 768), (1, 768), 0), out=buf405)
        buf406 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf403, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf406)
        del primals_150
        buf407 = reinterpret_tensor(buf405, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf405  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf407, primals_148, 786432, grid=grid(786432), stream=stream0)
        del primals_148
        buf408 = reinterpret_tensor(buf404, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf404  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf408, primals_146, 786432, grid=grid(786432), stream=stream0)
        del primals_146
        buf409 = reinterpret_tensor(buf379, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf379  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf408, buf409, 1179648, grid=grid(1179648), stream=stream0)
        buf410 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf407, buf410, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf411 = buf367; del buf367  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf410, (36, 64, 512), (32768, 512, 1), 0), out=buf411)
        buf412 = reinterpret_tensor(buf377, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf377  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_108, setitem_109], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf411, buf412, 6303744, grid=grid(6303744), stream=stream0)
        buf413 = reinterpret_tensor(buf370, (12, 256, 513), (131328, 513, 1), 0); del buf370  # reuse
        buf414 = reinterpret_tensor(buf369, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf369  # reuse
        # Source Nodes: [bool_1, full_like, setitem_111, setitem_112, where_36], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf411, buf412, buf10, buf413, buf414, 1575936, grid=grid(1575936), stream=stream0)
        buf415 = buf375; del buf375  # reuse
        # Source Nodes: [setitem_112], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf414, buf413, buf411, buf412, buf415, 6303744, grid=grid(6303744), stream=stream0)
        buf419 = reinterpret_tensor(buf412, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf412  # reuse
        buf544 = buf371; del buf371  # reuse
        # Source Nodes: [attn_probs_36, attn_probs_37, attn_scores_19, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf415, buf19, primals_195, buf419, buf544, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_36, attn_probs_37, attn_probs_39, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf420 = aten.native_dropout(buf419, 0.1, True)
        buf421 = buf420[0]
        buf422 = buf420[1]
        del buf420
        buf423 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_9], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf406, buf423, 1179648, grid=grid(1179648), stream=stream0)
        buf424 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_45], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf421, buf424, 9461760, grid=grid(9461760), stream=stream0)
        buf425 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf423, buf425, 2359296, grid=grid(2359296), stream=stream0)
        buf426 = reinterpret_tensor(buf406, (48, 256, 64), (16384, 64, 1), 0); del buf406  # reuse
        # Source Nodes: [context_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf425, (48, 768, 64), (49152, 64, 1), 0), out=buf426)
        buf427 = reinterpret_tensor(buf407, (1024, 768), (768, 1), 0); del buf407  # reuse
        # Source Nodes: [hidden_states_131], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf426, buf427, 786432, grid=grid(786432), stream=stream0)
        buf428 = reinterpret_tensor(buf426, (1024, 768), (768, 1), 0); del buf426  # reuse
        # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf427, reinterpret_tensor(primals_151, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf428)
        del primals_152
        # Source Nodes: [hidden_states_132], Original ATen: [aten.native_dropout]
        buf429 = aten.native_dropout(reinterpret_tensor(buf428, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf435 = reinterpret_tensor(buf428, (1, 1024, 768), (786432, 768, 1), 0); del buf428  # reuse
        buf436 = reinterpret_tensor(buf408, (1024, 768), (768, 1), 0); del buf408  # reuse
        buf543 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, attn_output_39, hidden_states_125, hidden_states_134], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf430, buf402, primals_143, primals_144, primals_153, primals_154, buf435, buf436, buf543, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_144
        buf437 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf436, reinterpret_tensor(primals_155, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf437)
        del primals_156
        buf438 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_136, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf437, buf438, 3145728, grid=grid(3145728), stream=stream0)
        buf439 = reinterpret_tensor(buf430, (1024, 768), (768, 1), 0); del buf430  # reuse
        # Source Nodes: [hidden_states_136], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf438, reinterpret_tensor(primals_157, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf439)
        del primals_158
        # Source Nodes: [hidden_states_137], Original ATen: [aten.native_dropout]
        buf440 = aten.native_dropout(reinterpret_tensor(buf439, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf446 = reinterpret_tensor(buf439, (1, 1024, 768), (786432, 768, 1), 0); del buf439  # reuse
        buf447 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf542 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_49, attn_output_39, hidden_states_139, query_vectors_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf441, buf435, primals_153, primals_154, primals_159, primals_160, buf446, buf447, buf542, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_154
        buf448 = reinterpret_tensor(buf441, (1024, 768), (768, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf447, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), out=buf448)
        buf449 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf447, reinterpret_tensor(primals_163, (768, 768), (1, 768), 0), out=buf449)
        buf450 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_166, buf447, reinterpret_tensor(primals_165, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf450)
        del primals_166
        buf451 = reinterpret_tensor(buf449, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf449  # reuse
        # Source Nodes: [hidden_states_142], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf451, primals_164, 786432, grid=grid(786432), stream=stream0)
        del primals_164
        buf452 = reinterpret_tensor(buf448, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf448  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf452, primals_162, 786432, grid=grid(786432), stream=stream0)
        del primals_162
        buf453 = reinterpret_tensor(buf423, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf423  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf452, buf453, 1179648, grid=grid(1179648), stream=stream0)
        buf454 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf451, buf454, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf455 = buf411; del buf411  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf453, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf454, (36, 64, 512), (32768, 512, 1), 0), out=buf455)
        buf456 = reinterpret_tensor(buf421, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf421  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_120, setitem_121], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf455, buf456, 6303744, grid=grid(6303744), stream=stream0)
        buf457 = reinterpret_tensor(buf414, (12, 256, 513), (131328, 513, 1), 0); del buf414  # reuse
        buf458 = reinterpret_tensor(buf413, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf413  # reuse
        # Source Nodes: [bool_1, full_like, setitem_123, setitem_124, where_40], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf455, buf456, buf10, buf457, buf458, 1575936, grid=grid(1575936), stream=stream0)
        buf459 = buf419; del buf419  # reuse
        # Source Nodes: [setitem_124], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf458, buf457, buf455, buf456, buf459, 6303744, grid=grid(6303744), stream=stream0)
        buf463 = reinterpret_tensor(buf456, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf456  # reuse
        buf541 = buf415; del buf415  # reuse
        # Source Nodes: [attn_probs_40, attn_probs_41, attn_scores_21, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf459, buf19, primals_195, buf463, buf541, 12288, 513, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_probs_40, attn_probs_41, attn_probs_43, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf464 = aten.native_dropout(buf463, 0.1, True)
        buf465 = buf464[0]
        buf466 = buf464[1]
        del buf464
        buf467 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf450, buf467, 1179648, grid=grid(1179648), stream=stream0)
        buf468 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_50], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf465, buf468, 9461760, grid=grid(9461760), stream=stream0)
        buf469 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf467, buf469, 2359296, grid=grid(2359296), stream=stream0)
        buf470 = reinterpret_tensor(buf450, (48, 256, 64), (16384, 64, 1), 0); del buf450  # reuse
        # Source Nodes: [context_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf468, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf469, (48, 768, 64), (49152, 64, 1), 0), out=buf470)
        buf471 = reinterpret_tensor(buf451, (1024, 768), (768, 1), 0); del buf451  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf470, buf471, 786432, grid=grid(786432), stream=stream0)
        buf472 = reinterpret_tensor(buf470, (1024, 768), (768, 1), 0); del buf470  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_168, buf471, reinterpret_tensor(primals_167, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf472)
        del primals_168
        # Source Nodes: [hidden_states_146], Original ATen: [aten.native_dropout]
        buf473 = aten.native_dropout(reinterpret_tensor(buf472, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf474 = buf473[0]
        buf475 = buf473[1]
        del buf473
        buf479 = reinterpret_tensor(buf472, (1, 1024, 768), (786432, 768, 1), 0); del buf472  # reuse
        buf480 = reinterpret_tensor(buf452, (1024, 768), (768, 1), 0); del buf452  # reuse
        buf540 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, attn_output_43, hidden_states_139, hidden_states_148], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf474, buf446, primals_159, primals_160, primals_169, primals_170, buf479, buf480, buf540, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_160
        buf481 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf480, reinterpret_tensor(primals_171, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf481)
        del primals_172
        buf482 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_150, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf481, buf482, 3145728, grid=grid(3145728), stream=stream0)
        buf483 = reinterpret_tensor(buf474, (1024, 768), (768, 1), 0); del buf474  # reuse
        # Source Nodes: [hidden_states_150], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_174, buf482, reinterpret_tensor(primals_173, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf483)
        del primals_174
        # Source Nodes: [hidden_states_151], Original ATen: [aten.native_dropout]
        buf484 = aten.native_dropout(reinterpret_tensor(buf483, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf485 = buf484[0]
        buf486 = buf484[1]
        del buf484
        buf490 = reinterpret_tensor(buf483, (1, 1024, 768), (786432, 768, 1), 0); del buf483  # reuse
        buf491 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf539 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_54, attn_output_43, hidden_states_153, query_vectors_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf485, buf479, primals_169, primals_170, primals_175, primals_176, buf490, buf491, buf539, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_170
        buf492 = reinterpret_tensor(buf485, (1024, 768), (768, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf491, reinterpret_tensor(primals_177, (768, 768), (1, 768), 0), out=buf492)
        buf493 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf491, reinterpret_tensor(primals_179, (768, 768), (1, 768), 0), out=buf493)
        buf494 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_vectors_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_182, buf491, reinterpret_tensor(primals_181, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf494)
        del primals_182
        buf495 = reinterpret_tensor(buf493, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf493  # reuse
        # Source Nodes: [hidden_states_156], Original ATen: [aten.view]
        triton_poi_fused_view_0.run(buf495, primals_180, 786432, grid=grid(786432), stream=stream0)
        del primals_180
        buf496 = reinterpret_tensor(buf492, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf492  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf496, primals_178, 786432, grid=grid(786432), stream=stream0)
        del primals_178
        buf497 = reinterpret_tensor(buf467, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf467  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf496, buf497, 1179648, grid=grid(1179648), stream=stream0)
        buf498 = empty((12, 3, 64, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf495, buf498, 2304, 512, grid=grid(2304, 512), stream=stream0)
        buf499 = buf455; del buf455  # reuse
        # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf497, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf498, (36, 64, 512), (32768, 512, 1), 0), out=buf499)
        buf500 = reinterpret_tensor(buf465, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf465  # reuse
        # Source Nodes: [diagonal_attention_scores, setitem_132, setitem_133], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice_scatter]
        triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4.run(buf499, buf500, 6303744, grid=grid(6303744), stream=stream0)
        buf501 = reinterpret_tensor(buf458, (12, 256, 513), (131328, 513, 1), 0); del buf458  # reuse
        buf502 = reinterpret_tensor(buf457, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf457  # reuse
        # Source Nodes: [bool_1, full_like, setitem_135, setitem_136, where_44], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
        triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6.run(buf499, buf500, buf10, buf501, buf502, 1575936, grid=grid(1575936), stream=stream0)
        buf503 = buf463; del buf463  # reuse
        # Source Nodes: [setitem_136], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_8.run(buf502, buf501, buf499, buf500, buf503, 6303744, grid=grid(6303744), stream=stream0)
        del buf499
        del buf501
        del buf502
        buf507 = reinterpret_tensor(buf500, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf500  # reuse
        buf538 = buf459; del buf459  # reuse
        # Source Nodes: [attn_probs_44, attn_probs_45, attn_scores_23, tril], Original ATen: [aten._softmax, aten.add, aten.detach, aten.masked_fill, aten.tril]
        triton_per_fused__softmax_add_detach_masked_fill_tril_15.run(buf11, buf503, buf19, primals_195, buf507, buf538, 12288, 513, grid=grid(12288), stream=stream0)
        del buf19
        del buf503
        # Source Nodes: [attn_probs_44, attn_probs_45, attn_probs_47, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
        buf508 = aten.native_dropout(buf507, 0.1, True)
        del buf507
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf511 = empty((12, 1536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [padded_value_11], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf494, buf511, 1179648, grid=grid(1179648), stream=stream0)
        buf512 = empty((12, 4, 256, 770), device='cuda', dtype=torch.float32)
        # Source Nodes: [chunked_hidden_states_55], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf509, buf512, 9461760, grid=grid(9461760), stream=stream0)
        del buf509
        buf513 = empty((12, 4, 768, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf511, buf513, 2359296, grid=grid(2359296), stream=stream0)
        del buf511
        buf514 = reinterpret_tensor(buf494, (48, 256, 64), (16384, 64, 1), 0); del buf494  # reuse
        # Source Nodes: [context_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf512, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf513, (48, 768, 64), (49152, 64, 1), 0), out=buf514)
        buf515 = reinterpret_tensor(buf495, (1024, 768), (768, 1), 0); del buf495  # reuse
        # Source Nodes: [hidden_states_159], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf514, buf515, 786432, grid=grid(786432), stream=stream0)
        buf516 = reinterpret_tensor(buf514, (1024, 768), (768, 1), 0); del buf514  # reuse
        # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_184, buf515, reinterpret_tensor(primals_183, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf516)
        del primals_184
        # Source Nodes: [hidden_states_160], Original ATen: [aten.native_dropout]
        buf517 = aten.native_dropout(reinterpret_tensor(buf516, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf518 = buf517[0]
        buf519 = buf517[1]
        del buf517
        buf523 = reinterpret_tensor(buf516, (1, 1024, 768), (786432, 768, 1), 0); del buf516  # reuse
        buf524 = reinterpret_tensor(buf496, (1024, 768), (768, 1), 0); del buf496  # reuse
        buf537 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_58, attn_output_47, hidden_states_153, hidden_states_162], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf518, buf490, primals_175, primals_176, primals_185, primals_186, buf523, buf524, buf537, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_176
        buf525 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_188, buf524, reinterpret_tensor(primals_187, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf525)
        del primals_188
        buf526 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_164, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf525, buf526, 3145728, grid=grid(3145728), stream=stream0)
        buf527 = reinterpret_tensor(buf518, (1024, 768), (768, 1), 0); del buf518  # reuse
        # Source Nodes: [hidden_states_164], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_190, buf526, reinterpret_tensor(primals_189, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf527)
        del primals_190
        # Source Nodes: [hidden_states_165], Original ATen: [aten.native_dropout]
        buf528 = aten.native_dropout(reinterpret_tensor(buf527, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf529 = buf528[0]
        buf530 = buf528[1]
        del buf528
        buf534 = reinterpret_tensor(buf527, (1, 1024, 768), (786432, 768, 1), 0); del buf527  # reuse
        buf535 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf536 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59, attn_output_47, hidden_states_167, hidden_states_168], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_22.run(buf529, buf523, primals_185, primals_186, primals_191, primals_192, buf534, buf535, buf536, 1024, 768, grid=grid(1024), stream=stream0)
        del buf529
        del primals_186
        del primals_192
        return (buf535, primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), buf10, buf11, reinterpret_tensor(primals_195, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf26, buf31, buf35, buf39, buf40, buf41, buf42, buf46, buf50, buf51, buf70, buf75, buf79, buf83, buf84, buf85, buf86, buf90, buf94, buf95, buf114, buf119, buf123, buf127, buf128, buf129, buf130, buf134, buf138, buf139, buf158, buf163, buf167, buf171, buf172, buf173, buf174, buf178, buf182, buf183, buf202, buf207, buf211, buf215, buf216, buf217, buf218, buf222, buf226, buf227, buf246, buf251, buf255, buf259, buf260, buf261, buf262, buf266, buf270, buf271, buf290, buf295, buf299, buf303, buf304, buf305, buf306, buf310, buf314, buf315, buf334, buf339, buf343, buf347, buf348, buf349, buf350, buf354, buf358, buf359, buf378, buf383, buf387, buf391, buf392, buf393, buf394, buf398, buf402, buf403, buf422, buf427, buf431, buf435, buf436, buf437, buf438, buf442, buf446, buf447, buf466, buf471, buf475, buf479, buf480, buf481, buf482, buf486, buf490, buf491, buf510, buf515, buf519, buf523, buf524, buf525, buf526, buf530, buf534, buf536, reinterpret_tensor(primals_189, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_187, (3072, 768), (768, 1), 0), buf537, reinterpret_tensor(primals_183, (768, 768), (768, 1), 0), reinterpret_tensor(buf512, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf513, (48, 64, 768), (49152, 1, 64), 0), buf538, reinterpret_tensor(buf497, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf498, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_181, (768, 768), (768, 1), 0), reinterpret_tensor(primals_179, (768, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 768), (768, 1), 0), buf539, reinterpret_tensor(primals_173, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_171, (3072, 768), (768, 1), 0), buf540, reinterpret_tensor(primals_167, (768, 768), (768, 1), 0), reinterpret_tensor(buf468, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf469, (48, 64, 768), (49152, 1, 64), 0), buf541, reinterpret_tensor(buf453, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf454, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_165, (768, 768), (768, 1), 0), reinterpret_tensor(primals_163, (768, 768), (768, 1), 0), reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), buf542, reinterpret_tensor(primals_157, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 768), (768, 1), 0), buf543, reinterpret_tensor(primals_151, (768, 768), (768, 1), 0), reinterpret_tensor(buf424, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf425, (48, 64, 768), (49152, 1, 64), 0), buf544, reinterpret_tensor(buf409, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf410, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 768), (768, 1), 0), buf545, reinterpret_tensor(primals_141, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_139, (3072, 768), (768, 1), 0), buf546, reinterpret_tensor(primals_135, (768, 768), (768, 1), 0), reinterpret_tensor(buf380, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf381, (48, 64, 768), (49152, 1, 64), 0), buf547, reinterpret_tensor(buf365, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf366, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_133, (768, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 768), (768, 1), 0), reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), buf548, reinterpret_tensor(primals_125, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_123, (3072, 768), (768, 1), 0), buf549, reinterpret_tensor(primals_119, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf337, (48, 64, 768), (49152, 1, 64), 0), buf550, reinterpret_tensor(buf321, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf322, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(primals_115, (768, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), buf551, reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), buf552, reinterpret_tensor(primals_103, (768, 768), (768, 1), 0), reinterpret_tensor(buf292, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf293, (48, 64, 768), (49152, 1, 64), 0), buf553, reinterpret_tensor(buf277, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf278, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_101, (768, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 768), (768, 1), 0), buf554, reinterpret_tensor(primals_93, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (768, 1), 0), buf555, reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(buf248, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf249, (48, 64, 768), (49152, 1, 64), 0), buf556, reinterpret_tensor(buf233, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf234, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), reinterpret_tensor(primals_83, (768, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 768), (768, 1), 0), buf557, reinterpret_tensor(primals_77, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_75, (3072, 768), (768, 1), 0), buf558, reinterpret_tensor(primals_71, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf205, (48, 64, 768), (49152, 1, 64), 0), buf559, reinterpret_tensor(buf189, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf190, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(primals_67, (768, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 768), (768, 1), 0), buf560, reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), buf561, reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(buf160, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf161, (48, 64, 768), (49152, 1, 64), 0), buf562, reinterpret_tensor(buf145, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf146, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), reinterpret_tensor(primals_51, (768, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 768), (768, 1), 0), buf563, reinterpret_tensor(primals_45, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 768), (768, 1), 0), buf564, reinterpret_tensor(primals_39, (768, 768), (768, 1), 0), reinterpret_tensor(buf116, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf117, (48, 64, 768), (49152, 1, 64), 0), buf565, reinterpret_tensor(buf101, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf102, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_37, (768, 768), (768, 1), 0), reinterpret_tensor(primals_35, (768, 768), (768, 1), 0), reinterpret_tensor(primals_33, (768, 768), (768, 1), 0), buf566, reinterpret_tensor(primals_29, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_27, (3072, 768), (768, 1), 0), buf567, reinterpret_tensor(primals_23, (768, 768), (768, 1), 0), reinterpret_tensor(buf72, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf73, (48, 64, 768), (49152, 1, 64), 0), buf568, reinterpret_tensor(buf57, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf58, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_21, (768, 768), (768, 1), 0), reinterpret_tensor(primals_19, (768, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 768), (768, 1), 0), buf569, reinterpret_tensor(primals_13, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 768), (768, 1), 0), buf570, reinterpret_tensor(primals_7, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf29, (48, 64, 768), (49152, 1, 64), 0), buf571, reinterpret_tensor(buf5, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf6, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_5, (768, 768), (768, 1), 0), reinterpret_tensor(primals_3, (768, 768), (768, 1), 0), reinterpret_tensor(primals_1, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.bool)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
