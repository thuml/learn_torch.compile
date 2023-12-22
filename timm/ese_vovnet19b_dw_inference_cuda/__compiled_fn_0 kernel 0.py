
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


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdh3hfruxaef6icv5pktywqty73ma3mmkmf4i25jjsxiljnhk6p.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 82944
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (82944*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (248832*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5b76sdupxyn7nezhz226yrtzhzrbav3ccwxevg2lotiu6crky7.py
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
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gbsqt73fx2ytt6ojcpbrfgcnxllbn6zrfslcny6xuhbboqnah3.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_4 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 20736
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (20736*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x2) + (1327104*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvrwjc7ort5fmpb6ly6bsau5yfxzyekibdick6j6x2jpjb5nx2s.py
# Source Nodes: [x_6], Original ATen: [aten.convolution]
# x_6 => convolution_2
triton_poi_fused_convolution_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 20736
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (20736*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1327104*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cglovbhmywjgjsq7ioh7dzwtbpxaqkj2r2u3nwjv643as365web2.py
# Source Nodes: [x_12], Original ATen: [aten.convolution]
# x_12 => convolution_4
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 5184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (5184*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (331776*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3oofk5ihfflylq7bxpyabllodg6fvi5aekgacsukybtr5qf27p.py
# Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_13 => add_5, mul_7, mul_8, sub_2
# x_17 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 5184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (5184*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x2) + (331776*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxsmirqb3vxbfjlwwkt2u3fwwx4a4usafnq4te2skggonbuipxa.py
# Source Nodes: [x_19, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_19 => add_7, mul_10, mul_11, sub_3
# x_23 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 5184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (5184*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (128*x2) + (663552*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhbmsbzvwkznah3xwcdr7euyz5bbktyalprxuvtvi3k7y7agab6.py
# Source Nodes: [x_25], Original ATen: [aten.convolution]
# x_25 => convolution_7
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 5184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (5184*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (663552*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5utlfergykvvigicqyyey3u7jsqhzetvnrghlwej5xobbsrjm6.py
# Source Nodes: [cat_7], Original ATen: [aten.cat]
# cat_7 => cat
triton_poi_fused_cat_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 41472
    xnumel = 448
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 5184
    y1 = (yindex // 5184)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 192, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-64) + x2 + (128*y3)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 320, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-192) + x2 + (128*y3)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 448, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-1658880) + y0 + (5184*x2) + (663552*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to((-320) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 - tmp26
    tmp28 = tl.load(in_ptr5 + (tl.broadcast_to((-320) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.load(in_ptr6 + (tl.broadcast_to((-320) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr7 + (tl.broadcast_to((-320) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp22, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp21, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x2 + (448*y3)), tmp45, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cengyrutr3eqmui23olzbokdgafwj5gy3es7wxse5ebfkqmoqj2q.py
# Source Nodes: [x_45, x_49, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_45 => add_15, mul_22, mul_23, sub_7
# x_49 => relu_7
# x_se => mean
triton_red_fused__native_batch_norm_legit_no_training_mean_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_relu_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 5184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (5184*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = triton_helpers.maximum(0, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tl.store(in_out_ptr0 + (r2 + (5184*x3)), tmp15, rmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = 5184.0
    tmp20 = tmp17 / tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvezmzoe5kzxrnogslnq2ypnyn6itgnbyncyrhwelcibv63452bo.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_51, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_gate => add_16, clamp_max, clamp_min, div
# x_51 => mul_24
# x_se => mean
# x_se_1 => convolution_13
triton_poi_fused_convolution_hardsigmoid_mean_mul_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 5184)
    x1 = (xindex // 5184) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7ue5yfku5ynffrn376hpvqvfg7fljz3g3ywxaltp4derqiiy2y.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_51, x_52, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_gate => add_16, clamp_max, clamp_min, div
# x_51 => mul_24
# x_52 => max_pool2d_with_indices
# x_se => mean
# x_se_1 => convolution_13
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1296
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 36)
    x2 = xindex % 36
    y4 = yindex
    x5 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = 2*x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 72, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x2) + (144*x3) + (5184*y4)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x2)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x2) + (144*x3) + (5184*y4)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x2) + (144*x3) + (5184*y4)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x3)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (72 + (2*x2) + (144*x3) + (5184*y4)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (73 + (2*x2) + (144*x3) + (5184*y4)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (74 + (2*x2) + (144*x3) + (5184*y4)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (144 + (2*x2) + (144*x3) + (5184*y4)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (145 + (2*x2) + (144*x3) + (5184*y4)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (146 + (2*x2) + (144*x3) + (5184*y4)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (256*x5) + (331776*y1)), tmp69, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7s3ip7wcwwsvg7nfr7sfnwd7ppmjy65dl7nkqbga26bx73d6zbi.py
# Source Nodes: [x_54, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_54 => add_18, mul_26, mul_27, sub_8
# x_58 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 1296
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (1296*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (160*x2) + (207360*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgnuo6sxjugus4xscwxbwgnaeah7io5hinclir7cwnazyguokdu.py
# Source Nodes: [x_60], Original ATen: [aten.convolution]
# x_60 => convolution_16
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 1296
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (1296*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (207360*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtvrt6shxe7byjkhxghcjpk6ljjhfnvbqnlf3lpeafovyztcbqf.py
# Source Nodes: [cat_6], Original ATen: [aten.cat]
# cat_6 => cat_1
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10368
    xnumel = 736
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1296
    y1 = (yindex // 1296)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (256*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 416, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-256) + x2 + (160*y3)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 576, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-416) + x2 + (160*y3)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 736, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-746496) + y0 + (1296*x2) + (207360*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to((-576) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 - tmp26
    tmp28 = tl.load(in_ptr5 + (tl.broadcast_to((-576) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.load(in_ptr6 + (tl.broadcast_to((-576) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr7 + (tl.broadcast_to((-576) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp22, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp21, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x2 + (736*y3)), tmp45, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdlqxp4h3o5k5k7enxvrdfjmhan36ca3achfeox6zdjeijav6yi.py
# Source Nodes: [x_80, x_84, x_se_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_80 => add_26, mul_38, mul_39, sub_12
# x_84 => relu_12
# x_se_2 => mean_1
triton_red_fused__native_batch_norm_legit_no_training_mean_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (1296*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = triton_helpers.maximum(0, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tl.store(in_out_ptr0 + (r2 + (1296*x3)), tmp15, rmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = 1296.0
    tmp20 = tmp17 / tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvch4n4zp4itn6ugb2hq76goldnq5xyadii23olah62of7b5bpaw.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_86, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_gate => add_27, clamp_max_1, clamp_min_1, div_1
# x_86 => mul_40
# x_se_2 => mean_1
# x_se_3 => convolution_22
triton_poi_fused_convolution_hardsigmoid_mean_mul_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 1296)
    x1 = (xindex // 1296) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxj6ry5dastg7qqy2v4trbbeozocrly7slijr5ss3yku5kkce6xw.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_86, x_87, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_gate => add_27, clamp_max_1, clamp_min_1, div_1
# x_86 => mul_40
# x_87 => max_pool2d_with_indices_1
# x_se_2 => mean_1
# x_se_3 => convolution_22
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 324
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 18)
    x2 = xindex % 18
    y4 = yindex
    x5 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = 2*x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 36, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x2) + (72*x3) + (1296*y4)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x2)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x2) + (72*x3) + (1296*y4)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x2) + (72*x3) + (1296*y4)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x3)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (36 + (2*x2) + (72*x3) + (1296*y4)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (37 + (2*x2) + (72*x3) + (1296*y4)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (38 + (2*x2) + (72*x3) + (1296*y4)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (72 + (2*x2) + (72*x3) + (1296*y4)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (73 + (2*x2) + (72*x3) + (1296*y4)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (74 + (2*x2) + (72*x3) + (1296*y4)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (512*x5) + (165888*y1)), tmp69, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmioqykgicppxwys2dl32isx7undhvbuuwbmb5rhzurxuwob6pur.py
# Source Nodes: [x_89, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_89 => add_29, mul_42, mul_43, sub_13
# x_93 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 324
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (324*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x2) + (62208*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciidzoce3ul3gx34lldksfue7tu4x4kf5b5xlxxepu6r7kwgguvl.py
# Source Nodes: [x_95], Original ATen: [aten.convolution]
# x_95 => convolution_25
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 324
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (324*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (62208*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nhkr365cihmx4b4aw6vhkaa6afwspwsuqqtq4xs6gm5tbkuwca.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat_2
triton_poi_fused_cat_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2592
    xnumel = 1088
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 324
    y1 = (yindex // 324)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (512*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 704, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-512) + x2 + (192*y3)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 896, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-704) + x2 + (192*y3)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1088, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-290304) + y0 + (324*x2) + (62208*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to((-896) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 - tmp26
    tmp28 = tl.load(in_ptr5 + (tl.broadcast_to((-896) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.load(in_ptr6 + (tl.broadcast_to((-896) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr7 + (tl.broadcast_to((-896) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp22, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp21, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x2 + (1088*y3)), tmp45, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coetwccness4ilw7ddyumde7s4om2sct5efkhy3dj34iuh2gdy5i.py
# Source Nodes: [x_115, x_119, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_115 => add_37, mul_54, mul_55, sub_17
# x_119 => relu_17
# x_se_4 => mean_2
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 324
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (r2 + (324*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 324.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (324*x3)), tmp15, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2micrxoy6yyhhx4qtxckfkm3wdjlxwzzgadwbylqpoxowhozlr.py
# Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_121, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___2___blocks___0___attn_gate => add_38, clamp_max_2, clamp_min_2, div_2
# x_121 => mul_56
# x_se_4 => mean_2
# x_se_5 => convolution_31
triton_poi_fused_convolution_hardsigmoid_mean_mul_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1990656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 324)
    x1 = (xindex // 324) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2ih7jzjvm2cxgoowhn2dniozx3bigfvn3noszyl4rtm5bvxng3u.py
# Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_121, x_122, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___2___blocks___0___attn_gate => add_38, clamp_max_2, clamp_min_2, div_2
# x_121 => mul_56
# x_122 => max_pool2d_with_indices_2
# x_se_4 => mean_2
# x_se_5 => convolution_31
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 81
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 9)
    x2 = xindex % 9
    y4 = yindex
    x5 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = 2*x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x2) + (36*x3) + (324*y4)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x2)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x2) + (36*x3) + (324*y4)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x2) + (36*x3) + (324*y4)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x3)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (18 + (2*x2) + (36*x3) + (324*y4)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (19 + (2*x2) + (36*x3) + (324*y4)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (20 + (2*x2) + (36*x3) + (324*y4)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (36 + (2*x2) + (36*x3) + (324*y4)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (37 + (2*x2) + (36*x3) + (324*y4)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (38 + (2*x2) + (36*x3) + (324*y4)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (768*x5) + (62208*y1)), tmp69, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljla45poec6qaslqoqr2j5y6liobb4xdh6epj352m22gkwyd55p.py
# Source Nodes: [x_124, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_124 => add_40, mul_58, mul_59, sub_18
# x_128 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 81
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (81*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (224*x2) + (18144*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7welsfrn37lfsqmqdwagv6qmyxsijmqmtmi3zweyyeekfzfy6fi.py
# Source Nodes: [x_130], Original ATen: [aten.convolution]
# x_130 => convolution_34
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 81
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (81*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (18144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfogo77dca3asdejxcg2kvpr7uvyi7rynp4h4mkripaswyc5epnt.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_3
triton_poi_fused_cat_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 648
    xnumel = 1440
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 81
    y1 = (yindex // 81)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (768*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 992, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-768) + x2 + (224*y3)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 1216, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-992) + x2 + (224*y3)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1440, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-98496) + y0 + (81*x2) + (18144*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 - tmp26
    tmp28 = tl.load(in_ptr5 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.load(in_ptr6 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr7 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp22, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp21, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x2 + (1440*y3)), tmp45, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chigfwvtzgpef4mvcem5iv74zg43p5fqplj47s5ez4m4odvfo6wr.py
# Source Nodes: [x_150, x_154, x_se_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_150 => add_48, mul_70, mul_71, sub_22
# x_154 => relu_22
# x_se_6 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (r2 + (81*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 81.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (81*x3)), tmp15, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3ogigwzlwtoi5fq4hao36443dmn2gopgaqhjfxgu2wc2pcsyus.py
# Source Nodes: [getattr_getattr_l__mod___stages___3___blocks___0___attn_gate, x_157, x_158, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
# getattr_getattr_l__mod___stages___3___blocks___0___attn_gate => add_49, clamp_max_3, clamp_min_3, div_3
# x_157 => mul_72
# x_158 => mean_4
# x_se_6 => mean_3
# x_se_7 => convolution_40
triton_per_fused_convolution_hardsigmoid_mean_mul_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_hardsigmoid_mean_mul_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (81*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 81.0
    tmp17 = tmp15 / tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, ), (1, ))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (160, ), (1, ))
    assert_size_stride(arg17_1, (160, ), (1, ))
    assert_size_stride(arg18_1, (160, ), (1, ))
    assert_size_stride(arg19_1, (160, ), (1, ))
    assert_size_stride(arg20_1, (160, ), (1, ))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (160, ), (1, ))
    assert_size_stride(arg23_1, (160, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (224, ), (1, ))
    assert_size_stride(arg37_1, (224, ), (1, ))
    assert_size_stride(arg38_1, (224, ), (1, ))
    assert_size_stride(arg39_1, (224, ), (1, ))
    assert_size_stride(arg40_1, (224, ), (1, ))
    assert_size_stride(arg41_1, (224, ), (1, ))
    assert_size_stride(arg42_1, (224, ), (1, ))
    assert_size_stride(arg43_1, (224, ), (1, ))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg47_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg48_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg49_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg50_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg51_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg53_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg54_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg55_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg56_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg57_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg58_1, (256, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg59_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (160, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg63_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg64_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg65_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg66_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg68_1, (512, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg69_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (192, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg72_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg74_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg75_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg76_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg77_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg78_1, (768, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(arg79_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (224, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg82_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg83_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg84_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg85_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg86_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg87_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg88_1, (1024, 1440, 1, 1), (1440, 1, 1, 1))
    assert_size_stride(arg89_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1000, ), (1, ))
    assert_size_stride(arg93_1, (64, ), (1, ))
    assert_size_stride(arg94_1, (64, ), (1, ))
    assert_size_stride(arg95_1, (64, ), (1, ))
    assert_size_stride(arg96_1, (64, ), (1, ))
    assert_size_stride(arg97_1, (64, ), (1, ))
    assert_size_stride(arg98_1, (64, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (160, ), (1, ))
    assert_size_stride(arg110_1, (160, ), (1, ))
    assert_size_stride(arg111_1, (160, ), (1, ))
    assert_size_stride(arg112_1, (160, ), (1, ))
    assert_size_stride(arg113_1, (160, ), (1, ))
    assert_size_stride(arg114_1, (160, ), (1, ))
    assert_size_stride(arg115_1, (160, ), (1, ))
    assert_size_stride(arg116_1, (160, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (192, ), (1, ))
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (192, ), (1, ))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (224, ), (1, ))
    assert_size_stride(arg130_1, (224, ), (1, ))
    assert_size_stride(arg131_1, (224, ), (1, ))
    assert_size_stride(arg132_1, (224, ), (1, ))
    assert_size_stride(arg133_1, (224, ), (1, ))
    assert_size_stride(arg134_1, (224, ), (1, ))
    assert_size_stride(arg135_1, (224, ), (1, ))
    assert_size_stride(arg136_1, (224, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 288, 288), (248832, 1, 864, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg139_1, buf0, 24, 82944, grid=grid(24, 82944), stream=stream0)
        del arg139_1
        buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg46_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg46_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 144, 144), (1327104, 20736, 144, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 64, 144, 144), (1327104, 1, 9216, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg93_1, arg94_1, arg0_1, arg1_1, buf3, 512, 20736, grid=grid(512, 20736), stream=stream0)
        del arg0_1
        del arg1_1
        del arg93_1
        del arg94_1
        del buf2
        # Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg47_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf4, (8, 64, 144, 144), (1327104, 20736, 144, 1))
        del arg47_1
        buf5 = buf3; del buf3  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(buf4, buf5, 512, 20736, grid=grid(512, 20736), stream=stream0)
        del buf4
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 144, 144), (1327104, 20736, 144, 1))
        del arg48_1
        buf7 = buf5; del buf5  # reuse
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, arg95_1, arg96_1, arg2_1, arg3_1, buf7, 512, 20736, grid=grid(512, 20736), stream=stream0)
        del arg2_1
        del arg3_1
        del arg95_1
        del arg96_1
        del buf6
        # Source Nodes: [x_10, x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf7, arg49_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf8, (8, 64, 72, 72), (331776, 5184, 72, 1))
        del arg49_1
        del buf7
        buf9 = empty_strided((8, 64, 72, 72), (331776, 1, 4608, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf8, buf9, 512, 5184, grid=grid(512, 5184), stream=stream0)
        del buf8
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 72, 72), (331776, 5184, 72, 1))
        del arg50_1
        buf11 = buf9; del buf9  # reuse
        # Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf10, arg97_1, arg98_1, arg4_1, arg5_1, buf11, 512, 5184, grid=grid(512, 5184), stream=stream0)
        del arg4_1
        del arg5_1
        del arg97_1
        del arg98_1
        del buf10
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg51_1
        buf13 = empty_strided((8, 128, 72, 72), (663552, 1, 9216, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf12, arg99_1, arg100_1, arg6_1, arg7_1, buf13, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del arg100_1
        del arg6_1
        del arg7_1
        del arg99_1
        del buf12
        # Source Nodes: [x_19, x_23, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg52_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf14, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg52_1
        buf15 = buf13; del buf13  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf14, buf15, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del buf14
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg53_1
        buf17 = buf15; del buf15  # reuse
        # Source Nodes: [x_26, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf16, arg101_1, arg102_1, arg8_1, arg9_1, buf17, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del arg101_1
        del arg102_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg54_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf18, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg54_1
        buf19 = reinterpret_tensor(buf16, (8, 128, 72, 72), (663552, 1, 9216, 128), 0); del buf16  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf18, buf19, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del buf18
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg55_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg55_1
        buf21 = buf19; del buf19  # reuse
        # Source Nodes: [x_32, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf20, arg103_1, arg104_1, arg10_1, arg11_1, buf21, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del arg103_1
        del arg104_1
        del arg10_1
        del arg11_1
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg56_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf22, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg56_1
        buf23 = reinterpret_tensor(buf20, (8, 128, 72, 72), (663552, 1, 9216, 128), 0); del buf20  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf22, buf23, 1024, 5184, grid=grid(1024, 5184), stream=stream0)
        del buf22
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg57_1
        del buf23
        buf25 = empty_strided((8, 448, 72, 72), (2322432, 1, 32256, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf11, buf17, buf21, buf24, arg105_1, arg106_1, arg12_1, arg13_1, buf25, 41472, 448, grid=grid(41472, 448), stream=stream0)
        del arg105_1
        del arg106_1
        del arg12_1
        del arg13_1
        del buf17
        del buf21
        del buf24
        # Source Nodes: [cat_7, x_44], Original ATen: [aten.cat, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 72, 72), (1327104, 5184, 72, 1))
        del arg58_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf29 = reinterpret_tensor(buf28, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf28  # reuse
        # Source Nodes: [x_45, x_49, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_relu_9.run(buf27, buf29, arg107_1, arg108_1, arg14_1, arg15_1, 2048, 5184, grid=grid(2048), stream=stream0)
        del arg107_1
        del arg108_1
        del arg14_1
        del arg15_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf30 = extern_kernels.convolution(buf29, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg59_1
        del buf29
        buf31 = buf27; del buf27  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_51, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_10.run(buf31, buf30, arg60_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg60_1
        del buf30
        buf32 = reinterpret_tensor(buf11, (8, 256, 36, 36), (331776, 1, 9216, 256), 0); del buf11  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_51, x_52, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_11.run(buf31, buf32, 2048, 1296, grid=grid(2048, 1296), stream=stream0)
        del buf31
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg61_1
        buf34 = empty_strided((8, 160, 36, 36), (207360, 1, 5760, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf33, arg109_1, arg110_1, arg16_1, arg17_1, buf34, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del arg109_1
        del arg110_1
        del arg16_1
        del arg17_1
        del buf33
        # Source Nodes: [x_54, x_58, x_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg62_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf35, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg62_1
        buf36 = buf34; del buf34  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf35, buf36, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del buf35
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg63_1
        buf38 = buf36; del buf36  # reuse
        # Source Nodes: [x_61, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf37, arg111_1, arg112_1, arg18_1, arg19_1, buf38, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del arg111_1
        del arg112_1
        del arg18_1
        del arg19_1
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg64_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf39, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg64_1
        buf40 = reinterpret_tensor(buf37, (8, 160, 36, 36), (207360, 1, 5760, 160), 0); del buf37  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf39, buf40, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del buf39
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg65_1
        buf42 = buf40; del buf40  # reuse
        # Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf41, arg113_1, arg114_1, arg20_1, arg21_1, buf42, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del arg113_1
        del arg114_1
        del arg20_1
        del arg21_1
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf43, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg66_1
        buf44 = reinterpret_tensor(buf41, (8, 160, 36, 36), (207360, 1, 5760, 160), 0); del buf41  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf43, buf44, 1280, 1296, grid=grid(1280, 1296), stream=stream0)
        del buf43
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 160, 36, 36), (207360, 1296, 36, 1))
        del arg67_1
        del buf44
        buf46 = empty_strided((8, 736, 36, 36), (953856, 1, 26496, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf32, buf38, buf42, buf45, arg115_1, arg116_1, arg22_1, arg23_1, buf46, 10368, 736, grid=grid(10368, 736), stream=stream0)
        del arg115_1
        del arg116_1
        del arg22_1
        del arg23_1
        del buf32
        del buf38
        del buf42
        del buf45
        # Source Nodes: [cat_6, x_79], Original ATen: [aten.cat, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 512, 36, 36), (663552, 1296, 36, 1))
        del arg68_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf50 = reinterpret_tensor(buf49, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf49  # reuse
        # Source Nodes: [x_80, x_84, x_se_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_relu_15.run(buf48, buf50, arg117_1, arg118_1, arg24_1, arg25_1, 4096, 1296, grid=grid(4096), stream=stream0)
        del arg117_1
        del arg118_1
        del arg24_1
        del arg25_1
        # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean]
        buf51 = extern_kernels.convolution(buf50, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg69_1
        del buf50
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_86, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_16.run(buf52, buf51, arg70_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg70_1
        del buf51
        buf53 = empty_strided((8, 512, 18, 18), (165888, 1, 9216, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_86, x_87, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17.run(buf52, buf53, 4096, 324, grid=grid(4096, 324), stream=stream0)
        del buf52
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg71_1
        buf55 = empty_strided((8, 192, 18, 18), (62208, 1, 3456, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf54, arg119_1, arg120_1, arg26_1, arg27_1, buf55, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del arg119_1
        del arg120_1
        del arg26_1
        del arg27_1
        del buf54
        # Source Nodes: [x_89, x_93, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf56, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg72_1
        buf57 = buf55; del buf55  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf56, buf57, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del buf56
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg73_1
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [x_96, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf58, arg121_1, arg122_1, arg28_1, arg29_1, buf59, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del arg121_1
        del arg122_1
        del arg28_1
        del arg29_1
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg74_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf60, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg74_1
        buf61 = reinterpret_tensor(buf58, (8, 192, 18, 18), (62208, 1, 3456, 192), 0); del buf58  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf60, buf61, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del buf60
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg75_1
        buf63 = buf61; del buf61  # reuse
        # Source Nodes: [x_102, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf62, arg123_1, arg124_1, arg30_1, arg31_1, buf63, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del arg123_1
        del arg124_1
        del arg30_1
        del arg31_1
        # Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg76_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf64, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg76_1
        buf65 = reinterpret_tensor(buf62, (8, 192, 18, 18), (62208, 1, 3456, 192), 0); del buf62  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf64, buf65, 1536, 324, grid=grid(1536, 324), stream=stream0)
        del buf64
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 192, 18, 18), (62208, 324, 18, 1))
        del arg77_1
        del buf65
        buf67 = empty_strided((8, 1088, 18, 18), (352512, 1, 19584, 1088), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf53, buf59, buf63, buf66, arg125_1, arg126_1, arg32_1, arg33_1, buf67, 2592, 1088, grid=grid(2592, 1088), stream=stream0)
        del arg125_1
        del arg126_1
        del arg32_1
        del arg33_1
        del buf53
        del buf59
        del buf63
        # Source Nodes: [cat_5, x_114], Original ATen: [aten.cat, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 768, 18, 18), (248832, 324, 18, 1))
        del arg78_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf71 = reinterpret_tensor(buf70, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf70  # reuse
        # Source Nodes: [x_115, x_119, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21.run(buf69, buf71, arg127_1, arg128_1, arg34_1, arg35_1, 6144, 324, grid=grid(6144), stream=stream0)
        del arg127_1
        del arg128_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf72 = extern_kernels.convolution(buf71, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg79_1
        del buf71
        buf73 = buf69; del buf69  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_121, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_22.run(buf73, buf72, arg80_1, 1990656, grid=grid(1990656), stream=stream0)
        del arg80_1
        del buf72
        buf74 = reinterpret_tensor(buf66, (8, 768, 9, 9), (62208, 1, 6912, 768), 0); del buf66  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_121, x_122, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_23.run(buf73, buf74, 6144, 81, grid=grid(6144, 81), stream=stream0)
        del buf73
        # Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg81_1
        buf76 = empty_strided((8, 224, 9, 9), (18144, 1, 2016, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf75, arg129_1, arg130_1, arg36_1, arg37_1, buf76, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del arg129_1
        del arg130_1
        del arg36_1
        del arg37_1
        del buf75
        # Source Nodes: [x_124, x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg82_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf77, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg82_1
        buf78 = buf76; del buf76  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf77, buf78, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del buf77
        # Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg83_1
        buf80 = buf78; del buf78  # reuse
        # Source Nodes: [x_131, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf79, arg131_1, arg132_1, arg38_1, arg39_1, buf80, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del arg131_1
        del arg132_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg84_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf81, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg84_1
        buf82 = reinterpret_tensor(buf79, (8, 224, 9, 9), (18144, 1, 2016, 224), 0); del buf79  # reuse
        # Source Nodes: [x_136], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf81, buf82, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del buf81
        # Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg85_1
        buf84 = buf82; del buf82  # reuse
        # Source Nodes: [x_137, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf83, arg133_1, arg134_1, arg40_1, arg41_1, buf84, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del arg133_1
        del arg134_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf85, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg86_1
        buf86 = reinterpret_tensor(buf83, (8, 224, 9, 9), (18144, 1, 2016, 224), 0); del buf83  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf85, buf86, 1792, 81, grid=grid(1792, 81), stream=stream0)
        del buf85
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 224, 9, 9), (18144, 81, 9, 1))
        del arg87_1
        del buf86
        buf88 = empty_strided((8, 1440, 9, 9), (116640, 1, 12960, 1440), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf74, buf80, buf84, buf87, arg135_1, arg136_1, arg42_1, arg43_1, buf88, 648, 1440, grid=grid(648, 1440), stream=stream0)
        del arg135_1
        del arg136_1
        del arg42_1
        del arg43_1
        del buf74
        del buf80
        del buf84
        del buf87
        # Source Nodes: [cat_4, x_149], Original ATen: [aten.cat, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 1024, 9, 9), (82944, 81, 9, 1))
        del arg88_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        buf91 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf92 = reinterpret_tensor(buf91, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf91  # reuse
        # Source Nodes: [x_150, x_154, x_se_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_27.run(buf90, buf92, arg137_1, arg138_1, arg44_1, arg45_1, 8192, 81, grid=grid(8192), stream=stream0)
        del arg137_1
        del arg138_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean]
        buf93 = extern_kernels.convolution(buf92, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg89_1
        del buf92
        buf94 = reinterpret_tensor(buf93, (8, 1024, 1, 1), (1024, 1, 8192, 8192), 0); del buf93  # reuse
        buf95 = reinterpret_tensor(buf94, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf94  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3___blocks___0___attn_gate, x_157, x_158, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul]
        triton_per_fused_convolution_hardsigmoid_mean_mul_28.run(buf95, buf90, arg90_1, 8192, 81, grid=grid(8192), stream=stream0)
        del arg90_1
        del buf90
        buf96 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf95, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf96)
        del arg91_1
        del arg92_1
        return (buf96, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ese_vovnet19b_dw', benchmark_compiled_module)
