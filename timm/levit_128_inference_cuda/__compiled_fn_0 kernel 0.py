
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


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlin2efbidnllolntxrzqqipguet5j7umm2bnglinmqigydpdil.py
# Source Nodes: [l__mod___stem_act1, l__mod___stem_conv1_bn, l__mod___stem_conv2_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
# l__mod___stem_act1 => add_2, clamp_max, clamp_min, div, mul_3
# l__mod___stem_conv1_bn => add_1, mul_1, mul_2, sub
# l__mod___stem_conv2_linear => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ju/cjumkzyhtdfmtx32xmdw5dcl6fro6wc5zzydpqtfsicuizedaicc.py
# Source Nodes: [l__mod___stem_act2, l__mod___stem_conv2_bn, l__mod___stem_conv3_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
# l__mod___stem_act2 => add_5, clamp_max_1, clamp_min_1, div_1, mul_7
# l__mod___stem_conv2_bn => add_4, mul_5, mul_6, sub_1
# l__mod___stem_conv3_linear => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjc6ok3jqgcm3gnhfsj2pnd7nwts2yleag27nkp4lq7nau7lqigs.py
# Source Nodes: [l__mod___stem_act3, l__mod___stem_conv3_bn, l__mod___stem_conv4_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
# l__mod___stem_act3 => add_8, clamp_max_2, clamp_min_2, div_2, mul_11
# l__mod___stem_conv3_bn => add_7, mul_10, mul_9, sub_2
# l__mod___stem_conv4_linear => convolution_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvrwmxutitzhyf5zh3dyzog3vn7twmjyk5jvmraik3kbyfu27pd.py
# Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_no_training]
# x => add_10, mul_13, mul_14, sub_3
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjbx7ozctvp63pize2xxg2zsafdlnuumhq33oz7yxpqq7bhskfr.py
# Source Nodes: [x_3], Original ATen: [aten.clone]
# x_3 => clone
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7zzrtu36lntcaijyvxj6a4iutqkgiw53isv5yjy3x4njogw2de.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 196
    x2 = (xindex // 3136) % 4
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (50176*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5yzjwrylkzszhncswzz5unpueksjgi43q7j5alfbgrbdhs2ear.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_2
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 4
    y2 = (yindex // 64)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (256*x3) + (50176*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtg4otbhv7qxahaeohyfbtqv7ry4poslzic6l7jtuhwghely7ca.py
# Source Nodes: [getitem_3], Original ATen: [aten.index]
# getitem_3 => index
triton_poi_fused_index_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 153664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 38416
    x1 = (xindex // 38416)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 196
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 196)) | ~xmask, "index out of bounds: 0 <= tmp3 < 196")
    tmp4 = tl.load(in_ptr1 + (tmp3 + (196*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c442nhvuuwzr2db4kriunruj4yjhxamyuexdwapiecvmbhdfnc2j.py
# Source Nodes: [attn, attn_1, mul], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_13
# attn_1 => amax, div_3, exp, sub_5, sum_1
# mul => mul_18
triton_per_fused__softmax_add_mul_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (196*x3)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcsfzdsbtbsryh5w2az3s7o2keou3gc25esi5kh7nlfetyzq7hb.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_3
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 4
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (256*x1) + (50176*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwo3lyjjxk6eewlwiuilcyrw4wekiy4hywmabtp33bzaf7deosu.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act => add_14, clamp_max_3, clamp_min_3, div_4, mul_19
triton_poi_fused_hardswish_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 196
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (6272*(x0 // 32)) + (25088*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bvmuxpy2oaj7u3wljlngeo3acocmhqip6fptia3wrhqyvjsp7b.py
# Source Nodes: [x_7, x_8], Original ATen: [aten.add, aten.clone]
# x_7 => add_17
# x_8 => clone_5
triton_poi_fused_add_clone_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp16, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4uycadshgv6aue5kxpbdw4cxsy4tkcoqj4yogaok3gmkwi42ii.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn => add_18, add_19, mul_23, mul_24, mul_25, reciprocal_6, sqrt_6, sub_7
# x_10 => add_20, clamp_max_4, clamp_min_4, div_5, mul_26
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqedmsptykrfy2ah42kufdpto466xaajjpcowlp7jdudse7da3n.py
# Source Nodes: [x_14, x_15], Original ATen: [aten.add, aten.clone]
# x_14 => add_23
# x_15 => clone_7
triton_poi_fused_add_clone_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cny6qazcxegma74z4lcjhriic4mmjrmenjhojgsj7q3zajxuuegx.py
# Source Nodes: [reshape_4], Original ATen: [aten.clone]
# reshape_4 => clone_29
triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 7
    x2 = (xindex // 896)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (3584*x2)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqayls2vnfmle3sheu36gdewhwqmwy6dce2bhjuw74ceabzyjdy7.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_30
triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (128*x1) + (6272*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lipxykjh6dzk2hgrvmz5qoffano6cyedpnq6jogakpa2smzxui.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_31
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (640*x3) + (125440*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clyowlrukzeb6tl3m4onzg7jwmoz5cbszqyur6gva2f4uwphlrb3.py
# Source Nodes: [getitem_19], Original ATen: [aten.index]
# getitem_19 => index_4
triton_poi_fused_index_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9604
    x1 = (xindex // 9604)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 196
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 196)) | ~xmask, "index out of bounds: 0 <= tmp3 < 196")
    tmp4 = tl.load(in_ptr1 + (tmp3 + (196*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd35d3ntjfvt5fpnpw6lt4bjtgenp4ctg426lu4c3hsek6edttbn.py
# Source Nodes: [attn_8, attn_9, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_8 => add_67
# attn_9 => amax_4, div_15, exp_4, sub_26, sum_5
# mul_4 => mul_81
triton_per_fused__softmax_add_mul_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 392
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (196*x3)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyrjctt4si4yeztdazhrj3jgwo4etn7izsaohuqcxuehl7ps47f.py
# Source Nodes: [matmul_9], Original ATen: [aten.clone]
# matmul_9 => clone_32
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 196
    x2 = (xindex // 12544) % 8
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (640*x1) + (125440*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmwmb4ezcufz6yik5wipwppkwhkk45b2xhb37lzg6pjytl752f7.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_act], Original ATen: [aten.hardswish]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_act => add_68, clamp_max_11, clamp_min_11, div_16, mul_82
triton_poi_fused_hardswish_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 49
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (3136*(x0 // 64)) + (25088*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlct7j76lrfur5ct267i6ygw5yfcqejbseztkcdxwtkhqsysso3.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn => add_69, add_70, mul_83, mul_84, mul_85, reciprocal_22, sqrt_22, sub_27
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c7/cc7yt3lnlgebcwf43spbwaoz7ywwll62qkt5b5agpnwee4zbslg3.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___stages___1___downsample_mlp_ln1_bn => add_71, add_72, mul_86, mul_87, mul_88, reciprocal_23, sqrt_23, sub_28
# x_62 => add_73, clamp_max_12, clamp_min_12, div_17, mul_89
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uo/cuorp5ffdxoiffagb6xlvsbx2lbchdlfz7sim4tk7di62d4ldgnm.py
# Source Nodes: [x_67], Original ATen: [aten.add]
# x_67 => add_76
triton_poi_fused_add_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyv2qyp7wxhsw6rl4p3ruwq6yigpnlzh2k4vpd2z7pgxdhmo4su.py
# Source Nodes: [matmul_10], Original ATen: [aten.clone]
# matmul_10 => clone_35
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (25088*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crktohe3ummvs4nlhhxpvw32mapepc6bfz3mu7xb3qhtfxbedppt.py
# Source Nodes: [matmul_10], Original ATen: [aten.clone]
# matmul_10 => clone_36
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (512*x3) + (25088*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpmbba3ei4i42may5jpeewvahlp24jt2ecqoqo3xyb3zvmgfrac.py
# Source Nodes: [getitem_23], Original ATen: [aten.index]
# getitem_23 => index_5
triton_poi_fused_index_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2401
    x1 = (xindex // 2401)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 49
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 49)) | ~xmask, "index out of bounds: 0 <= tmp3 < 49")
    tmp4 = tl.load(in_ptr1 + (tmp3 + (49*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/cky5lu3y3cjuj2it4q2l7nvqvwbohkemjbfzxyz4xfry7iix7zth.py
# Source Nodes: [attn_10, attn_11, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_10 => add_79
# attn_11 => amax_5, div_18, exp_5, sub_31, sum_6
# mul_5 => mul_96
triton_per_fused__softmax_add_mul_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 392
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (49*x3)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck374t62nyph3tjy73e6kvgn57uhqtyiaoo5bq2ajnsdb3wcvbqo.py
# Source Nodes: [matmul_11], Original ATen: [aten.clone]
# matmul_11 => clone_37
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (512*x1) + (25088*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42nuxbcewcrp3erpa664ipb4oynrxs2laepxg6cj2h77ehzh7ok.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act => add_80, clamp_max_13, clamp_min_13, div_19, mul_97
triton_poi_fused_hardswish_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 49
    x2 = (xindex // 12544)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (1568*(x0 // 32)) + (12544*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cduyhmma2uxl5ywbwfsohdwsidn67y4pst6kf23j2422nfdpydlu.py
# Source Nodes: [reshape_10], Original ATen: [aten.clone]
# reshape_10 => clone_55
triton_poi_fused_clone_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 4
    x2 = (xindex // 1024) % 4
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (3584*x2) + (12544*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfeogs5jfsz4uilsy5ktv7yp5kfyumtckx2h4jbh5ugflpaxcsx.py
# Source Nodes: [matmul_18], Original ATen: [aten.clone]
# matmul_18 => clone_56
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (256*x1) + (4096*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (16*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5z7z55aalj6n6mycodp4esinvmy5awiccgtvfq6bx5owfsgptn.py
# Source Nodes: [matmul_18], Original ATen: [aten.clone]
# matmul_18 => clone_57
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (1280*x3) + (62720*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/clujov4fui6rup4ih2634zbu5d5vyay2jlfzxgvtzbdx4y7aodni.py
# Source Nodes: [getitem_39], Original ATen: [aten.index]
# getitem_39 => index_9
triton_poi_fused_index_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 784
    x1 = (xindex // 784)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 49
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 49)) | ~xmask, "index out of bounds: 0 <= tmp3 < 49")
    tmp4 = tl.load(in_ptr1 + (tmp3 + (49*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7txqupfdeyepo5gtei7wnd6nfawx6mrjmtzf43t5cdsmfit4im.py
# Source Nodes: [attn_18, attn_19, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_18 => add_133
# attn_19 => amax_9, div_30, exp_9, sub_52, sum_10
# mul_9 => mul_159
triton_per_fused__softmax_add_mul_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (49*x3)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrqerocspsdpp5ru434c3stwvbfipd3k2dddq3c5qqrx4bdsioj.py
# Source Nodes: [matmul_19], Original ATen: [aten.clone]
# matmul_19 => clone_58
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 49
    x2 = (xindex // 3136) % 16
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (1280*x1) + (62720*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coewfmi3eohtyzr2mshbwzp7fwubbq7jdsz4tsjham3jeolc5c5b.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_act], Original ATen: [aten.hardswish]
# getattr_l__mod___stages___2___downsample_attn_downsample_proj_act => add_134, clamp_max_21, clamp_min_21, div_31, mul_160
triton_poi_fused_hardswish_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 16
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (1024*(x0 // 64)) + (16384*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxwpjegxtdjz6l5nzbjpcpok3bnbyyxj32n455yiavqwdrw652w.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn => add_135, add_136, mul_161, mul_162, mul_163, reciprocal_43, sqrt_43, sub_53
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cpln5vd5nente5jyd4xykatateqi573o7terqqv3wbvrvscch5al.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___stages___2___downsample_mlp_ln1_bn => add_137, add_138, mul_164, mul_165, mul_166, reciprocal_44, sqrt_44, sub_54
# x_127 => add_139, clamp_max_22, clamp_min_22, div_32, mul_167
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbaeyzysw4lcyiwell4meabb6fjgaubyo76d3l2vn5t6qoyt5qlp.py
# Source Nodes: [x_132], Original ATen: [aten.add]
# x_132 => add_142
triton_poi_fused_add_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyplo5bdflfhlc6ieqovuk7gae5zmdjh5a7nsn5gfnvrwrkzxv2.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_61
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 12
    x3 = (xindex // 3072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qmzdsgaqb56xauydv34tobkwpiihp54fz2ufpkhhgqmxotcl23.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_62
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 12
    y2 = (yindex // 192)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (768*x3) + (12288*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caorgwctap4mxy5jdgqne7odnda43e6ss3ghbxw4nonr347goed4.py
# Source Nodes: [getitem_43], Original ATen: [aten.index]
# getitem_43 => index_10
triton_poi_fused_index_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 16
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 16)) | ~xmask, "index out of bounds: 0 <= tmp3 < 16")
    tmp4 = tl.load(in_ptr1 + (tmp3 + (16*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpk25jhqvrme7c7niithldpuppcixivjyvxj4hjqlgbhokbjnef.py
# Source Nodes: [attn_20, attn_21, mul_10], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_20 => add_145
# attn_21 => amax_10, div_33, exp_10, sub_57, sum_11
# mul_10 => mul_174
triton_per_fused__softmax_add_mul_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (r2 + (16*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (16*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (16*x3)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacbins7hh27o2pfawqz6sjbechqfbd6iifxhqfqjtqyel7x6a6u.py
# Source Nodes: [matmul_21], Original ATen: [aten.clone]
# matmul_21 => clone_63
triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 12
    x3 = (xindex // 6144)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/czteqgb6hxwvmfgd3cbkt5yz2itbnjmk5bn3frb2tga3oivqmv4a.py
# Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
# getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act => add_146, clamp_max_23, clamp_min_23, div_34, mul_175
triton_poi_fused_hardswish_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384) % 16
    x2 = (xindex // 6144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (512*(x0 // 32)) + (6144*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrs3qjjngstuotsgdrk2nigozfpgfg722lwmrkkmr276ni4s6q4.py
# Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# l__mod___head_bn => add_195, add_196, mul_231, mul_232, mul_233, reciprocal_62, sqrt_62, sub_76
# l__mod___head_dist_bn => add_197, add_198, mul_234, mul_235, mul_236, reciprocal_63, sqrt_63, sub_77
# x_183 => add_194
# x_184 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (6144*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (6144*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 16.0
    tmp22 = tmp20 / tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 + tmp5
    tmp27 = tl.sqrt(tmp26)
    tmp28 = 1 / tmp27
    tmp29 = tmp28 * tmp9
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp22 - tmp35
    tmp38 = tmp37 + tmp5
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp9
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr1 + (x3), tmp34, xmask)
    tl.store(out_ptr2 + (x3), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywudbb3ingvbpteitm5qslpi34crdtd4kinzkfpwqjn6jx5gpao.py
# Source Nodes: [add_40, x_186], Original ATen: [aten.add, aten.div]
# add_40 => add_199
# x_186 => div_45
triton_poi_fused_add_div_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 2.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 196), (196, 1))
    assert_size_stride(arg1_1, (4, 196), (196, 1))
    assert_size_stride(arg2_1, (4, 196), (196, 1))
    assert_size_stride(arg3_1, (4, 196), (196, 1))
    assert_size_stride(arg4_1, (8, 196), (196, 1))
    assert_size_stride(arg5_1, (8, 49), (49, 1))
    assert_size_stride(arg6_1, (8, 49), (49, 1))
    assert_size_stride(arg7_1, (8, 49), (49, 1))
    assert_size_stride(arg8_1, (8, 49), (49, 1))
    assert_size_stride(arg9_1, (16, 49), (49, 1))
    assert_size_stride(arg10_1, (12, 16), (16, 1))
    assert_size_stride(arg11_1, (12, 16), (16, 1))
    assert_size_stride(arg12_1, (12, 16), (16, 1))
    assert_size_stride(arg13_1, (12, 16), (16, 1))
    assert_size_stride(arg14_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (16, ), (1, ))
    assert_size_stride(arg17_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (256, 128), (128, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (128, 128), (128, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (256, 128), (128, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (128, 256), (256, 1))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (256, 128), (128, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (128, 128), (128, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (256, 128), (128, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (128, 256), (256, 1))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (256, 128), (128, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (128, 128), (128, 1))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (256, 128), (128, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (128, 256), (256, 1))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (256, 128), (128, 1))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (128, 128), (128, 1))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (256, 128), (128, 1))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (128, 256), (256, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (640, 128), (128, 1))
    assert_size_stride(arg75_1, (640, ), (1, ))
    assert_size_stride(arg76_1, (640, ), (1, ))
    assert_size_stride(arg77_1, (128, 128), (128, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (256, 512), (512, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (512, 256), (256, 1))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (256, 512), (512, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (512, 256), (256, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (256, 256), (256, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (512, 256), (256, 1))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (256, 512), (512, 1))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (512, 256), (256, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (256, 256), (256, 1))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (512, 256), (256, 1))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (256, 512), (512, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (512, 256), (256, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (256, 256), (256, 1))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (512, 256), (256, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (256, 512), (512, 1))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (512, 256), (256, 1))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (256, 256), (256, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (512, 256), (256, 1))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (256, 512), (512, 1))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (1280, 256), (256, 1))
    assert_size_stride(arg138_1, (1280, ), (1, ))
    assert_size_stride(arg139_1, (1280, ), (1, ))
    assert_size_stride(arg140_1, (256, 256), (256, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (384, 1024), (1024, 1))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (768, 384), (384, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (384, 768), (768, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (768, 384), (384, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (384, 384), (384, 1))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (768, 384), (384, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (384, 768), (768, 1))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (768, 384), (384, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (384, 384), (384, 1))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (768, 384), (384, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (384, 768), (768, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (768, 384), (384, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (384, 384), (384, 1))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (768, 384), (384, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (384, 768), (768, 1))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (768, 384), (384, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (384, 384), (384, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (768, 384), (384, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (384, 768), (768, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (1000, 384), (384, 1))
    assert_size_stride(arg203_1, (1000, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (1000, 384), (384, 1))
    assert_size_stride(arg207_1, (1000, ), (1, ))
    assert_size_stride(arg208_1, (196, 196), (196, 1))
    assert_size_stride(arg209_1, (196, 196), (196, 1))
    assert_size_stride(arg210_1, (196, 196), (196, 1))
    assert_size_stride(arg211_1, (196, 196), (196, 1))
    assert_size_stride(arg212_1, (49, 196), (196, 1))
    assert_size_stride(arg213_1, (49, 49), (49, 1))
    assert_size_stride(arg214_1, (49, 49), (49, 1))
    assert_size_stride(arg215_1, (49, 49), (49, 1))
    assert_size_stride(arg216_1, (49, 49), (49, 1))
    assert_size_stride(arg217_1, (16, 49), (49, 1))
    assert_size_stride(arg218_1, (16, 16), (16, 1))
    assert_size_stride(arg219_1, (16, 16), (16, 1))
    assert_size_stride(arg220_1, (16, 16), (16, 1))
    assert_size_stride(arg221_1, (16, 16), (16, 1))
    assert_size_stride(arg222_1, (16, ), (1, ))
    assert_size_stride(arg223_1, (16, ), (1, ))
    assert_size_stride(arg224_1, (), ())
    assert_size_stride(arg225_1, (32, ), (1, ))
    assert_size_stride(arg226_1, (32, ), (1, ))
    assert_size_stride(arg227_1, (), ())
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (64, ), (1, ))
    assert_size_stride(arg230_1, (), ())
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (), ())
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (), ())
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (), ())
    assert_size_stride(arg240_1, (256, ), (1, ))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (), ())
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, ), (1, ))
    assert_size_stride(arg245_1, (), ())
    assert_size_stride(arg246_1, (256, ), (1, ))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (), ())
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (), ())
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (), ())
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (), ())
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (), ())
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (), ())
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (), ())
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (), ())
    assert_size_stride(arg270_1, (256, ), (1, ))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (), ())
    assert_size_stride(arg273_1, (128, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (), ())
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (256, ), (1, ))
    assert_size_stride(arg278_1, (), ())
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (), ())
    assert_size_stride(arg282_1, (640, ), (1, ))
    assert_size_stride(arg283_1, (640, ), (1, ))
    assert_size_stride(arg284_1, (), ())
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (), ())
    assert_size_stride(arg288_1, (256, ), (1, ))
    assert_size_stride(arg289_1, (256, ), (1, ))
    assert_size_stride(arg290_1, (), ())
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (), ())
    assert_size_stride(arg294_1, (256, ), (1, ))
    assert_size_stride(arg295_1, (256, ), (1, ))
    assert_size_stride(arg296_1, (), ())
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (), ())
    assert_size_stride(arg300_1, (256, ), (1, ))
    assert_size_stride(arg301_1, (256, ), (1, ))
    assert_size_stride(arg302_1, (), ())
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (), ())
    assert_size_stride(arg306_1, (256, ), (1, ))
    assert_size_stride(arg307_1, (256, ), (1, ))
    assert_size_stride(arg308_1, (), ())
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (), ())
    assert_size_stride(arg312_1, (256, ), (1, ))
    assert_size_stride(arg313_1, (256, ), (1, ))
    assert_size_stride(arg314_1, (), ())
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (), ())
    assert_size_stride(arg318_1, (256, ), (1, ))
    assert_size_stride(arg319_1, (256, ), (1, ))
    assert_size_stride(arg320_1, (), ())
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (), ())
    assert_size_stride(arg324_1, (256, ), (1, ))
    assert_size_stride(arg325_1, (256, ), (1, ))
    assert_size_stride(arg326_1, (), ())
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (), ())
    assert_size_stride(arg330_1, (256, ), (1, ))
    assert_size_stride(arg331_1, (256, ), (1, ))
    assert_size_stride(arg332_1, (), ())
    assert_size_stride(arg333_1, (512, ), (1, ))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (), ())
    assert_size_stride(arg336_1, (256, ), (1, ))
    assert_size_stride(arg337_1, (256, ), (1, ))
    assert_size_stride(arg338_1, (), ())
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (512, ), (1, ))
    assert_size_stride(arg341_1, (), ())
    assert_size_stride(arg342_1, (256, ), (1, ))
    assert_size_stride(arg343_1, (256, ), (1, ))
    assert_size_stride(arg344_1, (), ())
    assert_size_stride(arg345_1, (1280, ), (1, ))
    assert_size_stride(arg346_1, (1280, ), (1, ))
    assert_size_stride(arg347_1, (), ())
    assert_size_stride(arg348_1, (256, ), (1, ))
    assert_size_stride(arg349_1, (256, ), (1, ))
    assert_size_stride(arg350_1, (), ())
    assert_size_stride(arg351_1, (384, ), (1, ))
    assert_size_stride(arg352_1, (384, ), (1, ))
    assert_size_stride(arg353_1, (), ())
    assert_size_stride(arg354_1, (768, ), (1, ))
    assert_size_stride(arg355_1, (768, ), (1, ))
    assert_size_stride(arg356_1, (), ())
    assert_size_stride(arg357_1, (384, ), (1, ))
    assert_size_stride(arg358_1, (384, ), (1, ))
    assert_size_stride(arg359_1, (), ())
    assert_size_stride(arg360_1, (768, ), (1, ))
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (), ())
    assert_size_stride(arg363_1, (384, ), (1, ))
    assert_size_stride(arg364_1, (384, ), (1, ))
    assert_size_stride(arg365_1, (), ())
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (768, ), (1, ))
    assert_size_stride(arg368_1, (), ())
    assert_size_stride(arg369_1, (384, ), (1, ))
    assert_size_stride(arg370_1, (384, ), (1, ))
    assert_size_stride(arg371_1, (), ())
    assert_size_stride(arg372_1, (768, ), (1, ))
    assert_size_stride(arg373_1, (768, ), (1, ))
    assert_size_stride(arg374_1, (), ())
    assert_size_stride(arg375_1, (384, ), (1, ))
    assert_size_stride(arg376_1, (384, ), (1, ))
    assert_size_stride(arg377_1, (), ())
    assert_size_stride(arg378_1, (768, ), (1, ))
    assert_size_stride(arg379_1, (768, ), (1, ))
    assert_size_stride(arg380_1, (), ())
    assert_size_stride(arg381_1, (384, ), (1, ))
    assert_size_stride(arg382_1, (384, ), (1, ))
    assert_size_stride(arg383_1, (), ())
    assert_size_stride(arg384_1, (768, ), (1, ))
    assert_size_stride(arg385_1, (768, ), (1, ))
    assert_size_stride(arg386_1, (), ())
    assert_size_stride(arg387_1, (384, ), (1, ))
    assert_size_stride(arg388_1, (384, ), (1, ))
    assert_size_stride(arg389_1, (), ())
    assert_size_stride(arg390_1, (768, ), (1, ))
    assert_size_stride(arg391_1, (768, ), (1, ))
    assert_size_stride(arg392_1, (), ())
    assert_size_stride(arg393_1, (384, ), (1, ))
    assert_size_stride(arg394_1, (384, ), (1, ))
    assert_size_stride(arg395_1, (), ())
    assert_size_stride(arg396_1, (768, ), (1, ))
    assert_size_stride(arg397_1, (768, ), (1, ))
    assert_size_stride(arg398_1, (), ())
    assert_size_stride(arg399_1, (384, ), (1, ))
    assert_size_stride(arg400_1, (384, ), (1, ))
    assert_size_stride(arg401_1, (), ())
    assert_size_stride(arg402_1, (768, ), (1, ))
    assert_size_stride(arg403_1, (768, ), (1, ))
    assert_size_stride(arg404_1, (), ())
    assert_size_stride(arg405_1, (384, ), (1, ))
    assert_size_stride(arg406_1, (384, ), (1, ))
    assert_size_stride(arg407_1, (), ())
    assert_size_stride(arg408_1, (384, ), (1, ))
    assert_size_stride(arg409_1, (384, ), (1, ))
    assert_size_stride(arg410_1, (), ())
    assert_size_stride(arg411_1, (384, ), (1, ))
    assert_size_stride(arg412_1, (384, ), (1, ))
    assert_size_stride(arg413_1, (), ())
    assert_size_stride(arg414_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_conv1_linear], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg414_1, arg14_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg14_1
        del arg414_1
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Source Nodes: [l__mod___stem_act1, l__mod___stem_conv1_bn, l__mod___stem_conv2_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0.run(buf2, arg222_1, arg223_1, arg15_1, arg16_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg15_1
        del arg16_1
        del arg222_1
        del arg223_1
        # Source Nodes: [l__mod___stem_act1, l__mod___stem_conv2_linear], Original ATen: [aten.convolution, aten.hardswish]
        buf3 = extern_kernels.convolution(buf2, arg17_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg17_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [l__mod___stem_act2, l__mod___stem_conv2_bn, l__mod___stem_conv3_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1.run(buf5, arg225_1, arg226_1, arg18_1, arg19_1, 802816, grid=grid(802816), stream=stream0)
        del arg18_1
        del arg19_1
        del arg225_1
        del arg226_1
        # Source Nodes: [l__mod___stem_act2, l__mod___stem_conv3_linear], Original ATen: [aten.convolution, aten.hardswish]
        buf6 = extern_kernels.convolution(buf5, arg20_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg20_1
        buf7 = buf6; del buf6  # reuse
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [l__mod___stem_act3, l__mod___stem_conv3_bn, l__mod___stem_conv4_linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2.run(buf8, arg228_1, arg229_1, arg21_1, arg22_1, 401408, grid=grid(401408), stream=stream0)
        del arg21_1
        del arg228_1
        del arg229_1
        del arg22_1
        # Source Nodes: [l__mod___stem_act3, l__mod___stem_conv4_linear], Original ATen: [aten.convolution, aten.hardswish]
        buf9 = extern_kernels.convolution(buf8, arg23_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg23_1
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf10, arg231_1, arg232_1, arg24_1, arg25_1, 200704, grid=grid(200704), stream=stream0)
        del arg231_1
        del arg232_1
        del arg24_1
        del arg25_1
        buf11 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf10, buf11, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf12 = reinterpret_tensor(buf8, (1568, 256), (256, 1), 0); del buf8  # reuse
        # Source Nodes: [x_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (1568, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 256), (1, 128), 0), out=buf12)
        del arg26_1
        buf13 = empty((8, 4, 196, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf12, arg234_1, arg235_1, arg27_1, arg28_1, buf13, 100352, grid=grid(100352), stream=stream0)
        buf14 = empty((8, 4, 16, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf12, arg234_1, arg235_1, arg27_1, arg28_1, buf14, 512, 196, grid=grid(512, 196), stream=stream0)
        buf15 = empty((32, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf14, (32, 16, 196), (3136, 196, 1), 0), out=buf15)
        buf16 = empty((4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_3], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg208_1, arg0_1, buf16, 153664, grid=grid(153664), stream=stream0)
        del arg0_1
        del arg208_1
        buf19 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_8.run(buf15, buf16, buf19, 6272, 196, grid=grid(6272), stream=stream0)
        buf20 = reinterpret_tensor(buf11, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf11  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf12, arg234_1, arg235_1, arg27_1, arg28_1, buf20, 200704, grid=grid(200704), stream=stream0)
        del arg234_1
        del arg235_1
        del arg27_1
        del arg28_1
        buf21 = empty((32, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf19, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf20, (32, 196, 32), (6272, 32, 1), 0), out=buf21)
        buf22 = reinterpret_tensor(buf20, (8, 196, 128), (25088, 128, 1), 0); del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf21, buf22, 200704, grid=grid(200704), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (1568, 128), (128, 1), 0); del buf21  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (1568, 128), (128, 1), 0), reinterpret_tensor(arg29_1, (128, 128), (1, 128), 0), out=buf23)
        del arg29_1
        buf24 = reinterpret_tensor(buf23, (8, 196, 128), (25088, 128, 1), 0); del buf23  # reuse
        buf25 = buf22; del buf22  # reuse
        # Source Nodes: [x_7, x_8], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_11.run(buf24, buf10, arg237_1, arg238_1, arg30_1, arg31_1, buf25, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del arg237_1
        del arg238_1
        del arg30_1
        del arg31_1
        buf26 = buf12; del buf12  # reuse
        # Source Nodes: [x_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1568, 128), (128, 1), 0), reinterpret_tensor(arg32_1, (128, 256), (1, 128), 0), out=buf26)
        del arg32_1
        buf27 = buf26; del buf26  # reuse
        buf28 = reinterpret_tensor(buf27, (8, 196, 256), (50176, 256, 1), 0); del buf27  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf28, arg240_1, arg241_1, arg33_1, arg34_1, 401408, grid=grid(401408), stream=stream0)
        del arg240_1
        del arg241_1
        del arg33_1
        del arg34_1
        buf29 = reinterpret_tensor(buf25, (1568, 128), (128, 1), 0); del buf25  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (1568, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 128), (1, 256), 0), out=buf29)
        del arg35_1
        buf30 = buf24; del buf24  # reuse
        buf31 = reinterpret_tensor(buf10, (8, 196, 128), (25088, 128, 1), 0); del buf10  # reuse
        # Source Nodes: [x_14, x_15], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf30, buf29, arg243_1, arg244_1, arg36_1, arg37_1, buf31, 200704, grid=grid(200704), stream=stream0)
        del arg243_1
        del arg244_1
        del arg36_1
        del arg37_1
        buf32 = reinterpret_tensor(buf28, (1568, 256), (256, 1), 0); del buf28  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1568, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 256), (1, 128), 0), out=buf32)
        del arg38_1
        buf33 = reinterpret_tensor(buf14, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf14  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf32, arg246_1, arg247_1, arg39_1, arg40_1, buf33, 100352, grid=grid(100352), stream=stream0)
        buf34 = reinterpret_tensor(buf13, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf13  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf32, arg246_1, arg247_1, arg39_1, arg40_1, buf34, 512, 196, grid=grid(512, 196), stream=stream0)
        buf35 = reinterpret_tensor(buf19, (32, 196, 196), (38416, 196, 1), 0); del buf19  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf34, (32, 16, 196), (3136, 196, 1), 0), out=buf35)
        buf36 = empty((4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_7], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg209_1, arg1_1, buf36, 153664, grid=grid(153664), stream=stream0)
        del arg1_1
        del arg209_1
        buf39 = reinterpret_tensor(buf15, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_2, attn_3, mul_1], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_8.run(buf35, buf36, buf39, 6272, 196, grid=grid(6272), stream=stream0)
        buf40 = reinterpret_tensor(buf31, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf31  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf32, arg246_1, arg247_1, arg39_1, arg40_1, buf40, 200704, grid=grid(200704), stream=stream0)
        del arg246_1
        del arg247_1
        del arg39_1
        del arg40_1
        buf41 = reinterpret_tensor(buf29, (32, 196, 32), (6272, 32, 1), 0); del buf29  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf40, (32, 196, 32), (6272, 32, 1), 0), out=buf41)
        buf42 = reinterpret_tensor(buf40, (8, 196, 128), (25088, 128, 1), 0); del buf40  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf41, buf42, 200704, grid=grid(200704), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (1568, 128), (128, 1), 0); del buf41  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 128), (128, 1), 0), reinterpret_tensor(arg41_1, (128, 128), (1, 128), 0), out=buf43)
        del arg41_1
        buf44 = buf30; del buf30  # reuse
        buf45 = buf42; del buf42  # reuse
        # Source Nodes: [x_19, x_20], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf44, buf43, arg249_1, arg250_1, arg42_1, arg43_1, buf45, 200704, grid=grid(200704), stream=stream0)
        del arg249_1
        del arg250_1
        del arg42_1
        del arg43_1
        buf46 = buf32; del buf32  # reuse
        # Source Nodes: [x_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1568, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 256), (1, 128), 0), out=buf46)
        del arg44_1
        buf47 = buf46; del buf46  # reuse
        buf48 = reinterpret_tensor(buf47, (8, 196, 256), (50176, 256, 1), 0); del buf47  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf48, arg252_1, arg253_1, arg45_1, arg46_1, 401408, grid=grid(401408), stream=stream0)
        del arg252_1
        del arg253_1
        del arg45_1
        del arg46_1
        buf49 = reinterpret_tensor(buf45, (1568, 128), (128, 1), 0); del buf45  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (1568, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 128), (1, 256), 0), out=buf49)
        del arg47_1
        buf50 = buf44; del buf44  # reuse
        buf51 = reinterpret_tensor(buf43, (8, 196, 128), (25088, 128, 1), 0); del buf43  # reuse
        # Source Nodes: [x_26, x_27], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf50, buf49, arg255_1, arg256_1, arg48_1, arg49_1, buf51, 200704, grid=grid(200704), stream=stream0)
        del arg255_1
        del arg256_1
        del arg48_1
        del arg49_1
        buf52 = reinterpret_tensor(buf48, (1568, 256), (256, 1), 0); del buf48  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (1568, 128), (128, 1), 0), reinterpret_tensor(arg50_1, (128, 256), (1, 128), 0), out=buf52)
        del arg50_1
        buf53 = reinterpret_tensor(buf34, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf34  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf52, arg258_1, arg259_1, arg51_1, arg52_1, buf53, 100352, grid=grid(100352), stream=stream0)
        buf54 = reinterpret_tensor(buf33, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf33  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf52, arg258_1, arg259_1, arg51_1, arg52_1, buf54, 512, 196, grid=grid(512, 196), stream=stream0)
        buf55 = reinterpret_tensor(buf39, (32, 196, 196), (38416, 196, 1), 0); del buf39  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf54, (32, 16, 196), (3136, 196, 1), 0), out=buf55)
        buf56 = empty((4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_11], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg210_1, arg2_1, buf56, 153664, grid=grid(153664), stream=stream0)
        del arg210_1
        del arg2_1
        buf59 = reinterpret_tensor(buf35, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_4, attn_5, mul_2], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_8.run(buf55, buf56, buf59, 6272, 196, grid=grid(6272), stream=stream0)
        buf60 = reinterpret_tensor(buf51, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf51  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf52, arg258_1, arg259_1, arg51_1, arg52_1, buf60, 200704, grid=grid(200704), stream=stream0)
        del arg258_1
        del arg259_1
        del arg51_1
        del arg52_1
        buf61 = reinterpret_tensor(buf49, (32, 196, 32), (6272, 32, 1), 0); del buf49  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf60, (32, 196, 32), (6272, 32, 1), 0), out=buf61)
        buf62 = reinterpret_tensor(buf60, (8, 196, 128), (25088, 128, 1), 0); del buf60  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf61, buf62, 200704, grid=grid(200704), stream=stream0)
        buf63 = reinterpret_tensor(buf61, (1568, 128), (128, 1), 0); del buf61  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (1568, 128), (128, 1), 0), reinterpret_tensor(arg53_1, (128, 128), (1, 128), 0), out=buf63)
        del arg53_1
        buf64 = buf50; del buf50  # reuse
        buf65 = buf62; del buf62  # reuse
        # Source Nodes: [x_31, x_32], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf64, buf63, arg261_1, arg262_1, arg54_1, arg55_1, buf65, 200704, grid=grid(200704), stream=stream0)
        del arg261_1
        del arg262_1
        del arg54_1
        del arg55_1
        buf66 = buf52; del buf52  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (1568, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 256), (1, 128), 0), out=buf66)
        del arg56_1
        buf67 = buf66; del buf66  # reuse
        buf68 = reinterpret_tensor(buf67, (8, 196, 256), (50176, 256, 1), 0); del buf67  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf68, arg264_1, arg265_1, arg57_1, arg58_1, 401408, grid=grid(401408), stream=stream0)
        del arg264_1
        del arg265_1
        del arg57_1
        del arg58_1
        buf69 = reinterpret_tensor(buf65, (1568, 128), (128, 1), 0); del buf65  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (1568, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 128), (1, 256), 0), out=buf69)
        del arg59_1
        buf70 = buf64; del buf64  # reuse
        buf71 = reinterpret_tensor(buf63, (8, 196, 128), (25088, 128, 1), 0); del buf63  # reuse
        # Source Nodes: [x_38, x_39], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf70, buf69, arg267_1, arg268_1, arg60_1, arg61_1, buf71, 200704, grid=grid(200704), stream=stream0)
        del arg267_1
        del arg268_1
        del arg60_1
        del arg61_1
        buf72 = reinterpret_tensor(buf68, (1568, 256), (256, 1), 0); del buf68  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 128), (128, 1), 0), reinterpret_tensor(arg62_1, (128, 256), (1, 128), 0), out=buf72)
        del arg62_1
        buf73 = reinterpret_tensor(buf54, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf54  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf72, arg270_1, arg271_1, arg63_1, arg64_1, buf73, 100352, grid=grid(100352), stream=stream0)
        buf74 = reinterpret_tensor(buf53, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf53  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf72, arg270_1, arg271_1, arg63_1, arg64_1, buf74, 512, 196, grid=grid(512, 196), stream=stream0)
        buf75 = reinterpret_tensor(buf59, (32, 196, 196), (38416, 196, 1), 0); del buf59  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf74, (32, 16, 196), (3136, 196, 1), 0), out=buf75)
        buf76 = empty((4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_15], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg211_1, arg3_1, buf76, 153664, grid=grid(153664), stream=stream0)
        del arg211_1
        del arg3_1
        buf79 = reinterpret_tensor(buf55, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf55  # reuse
        # Source Nodes: [attn_6, attn_7, mul_3], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_8.run(buf75, buf76, buf79, 6272, 196, grid=grid(6272), stream=stream0)
        del buf75
        buf80 = reinterpret_tensor(buf71, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf71  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf72, arg270_1, arg271_1, arg63_1, arg64_1, buf80, 200704, grid=grid(200704), stream=stream0)
        del arg270_1
        del arg271_1
        del arg63_1
        del arg64_1
        buf81 = reinterpret_tensor(buf69, (32, 196, 32), (6272, 32, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf80, (32, 196, 32), (6272, 32, 1), 0), out=buf81)
        del buf79
        buf82 = reinterpret_tensor(buf80, (8, 196, 128), (25088, 128, 1), 0); del buf80  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf81, buf82, 200704, grid=grid(200704), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (1568, 128), (128, 1), 0); del buf81  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1568, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 128), (1, 128), 0), out=buf83)
        del arg65_1
        buf84 = buf70; del buf70  # reuse
        buf85 = buf82; del buf82  # reuse
        # Source Nodes: [x_43, x_44], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf84, buf83, arg273_1, arg274_1, arg66_1, arg67_1, buf85, 200704, grid=grid(200704), stream=stream0)
        del arg273_1
        del arg274_1
        del arg66_1
        del arg67_1
        buf86 = buf72; del buf72  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 128), (128, 1), 0), reinterpret_tensor(arg68_1, (128, 256), (1, 128), 0), out=buf86)
        del arg68_1
        buf87 = buf86; del buf86  # reuse
        buf88 = reinterpret_tensor(buf87, (8, 196, 256), (50176, 256, 1), 0); del buf87  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf88, arg276_1, arg277_1, arg69_1, arg70_1, 401408, grid=grid(401408), stream=stream0)
        del arg276_1
        del arg277_1
        del arg69_1
        del arg70_1
        buf89 = reinterpret_tensor(buf85, (1568, 128), (128, 1), 0); del buf85  # reuse
        # Source Nodes: [x_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1568, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 128), (1, 256), 0), out=buf89)
        del arg71_1
        buf90 = buf84; del buf84  # reuse
        buf91 = reinterpret_tensor(buf83, (8, 196, 128), (25088, 128, 1), 0); del buf83  # reuse
        # Source Nodes: [x_51, x_52], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf90, buf89, arg279_1, arg280_1, arg72_1, arg73_1, buf91, 200704, grid=grid(200704), stream=stream0)
        del arg279_1
        del arg280_1
        del arg72_1
        del arg73_1
        del buf89
        buf92 = empty((1568, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (1568, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 640), (1, 128), 0), out=buf92)
        del arg74_1
        buf93 = empty((8, 7, 7, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf90, buf93, 50176, grid=grid(50176), stream=stream0)
        buf94 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (392, 128), (128, 1), 0), reinterpret_tensor(arg77_1, (128, 128), (1, 128), 0), out=buf94)
        del arg77_1
        buf95 = reinterpret_tensor(buf93, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf93  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf94, arg285_1, arg286_1, arg78_1, arg79_1, buf95, 50176, grid=grid(50176), stream=stream0)
        del arg285_1
        del arg286_1
        del arg78_1
        del arg79_1
        buf96 = reinterpret_tensor(buf90, (8, 8, 16, 196), (25088, 3136, 196, 1), 0); del buf90  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf92, arg282_1, arg283_1, arg75_1, arg76_1, buf96, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf97 = empty((64, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf96, (64, 16, 196), (3136, 196, 1), 0), out=buf97)
        buf98 = empty((8, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_19], Original ATen: [aten.index]
        triton_poi_fused_index_17.run(arg212_1, arg4_1, buf98, 76832, grid=grid(76832), stream=stream0)
        del arg212_1
        del arg4_1
        buf101 = empty((8, 8, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_8, attn_9, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_18.run(buf97, buf98, buf101, 3136, 196, grid=grid(3136), stream=stream0)
        del buf97
        buf102 = reinterpret_tensor(buf5, (8, 8, 196, 64), (100352, 12544, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf92, arg282_1, arg283_1, arg75_1, arg76_1, buf102, 802816, grid=grid(802816), stream=stream0)
        del arg282_1
        del arg283_1
        del arg75_1
        del arg76_1
        del buf92
        buf103 = reinterpret_tensor(buf96, (64, 49, 64), (3136, 64, 1), 0); del buf96  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (64, 49, 196), (9604, 196, 1), 0), reinterpret_tensor(buf102, (64, 196, 64), (12544, 64, 1), 0), out=buf103)
        del buf101
        del buf102
        buf104 = reinterpret_tensor(buf91, (8, 49, 512), (25088, 512, 1), 0); del buf91  # reuse
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_20.run(buf103, buf104, 200704, grid=grid(200704), stream=stream0)
        del buf103
        buf105 = reinterpret_tensor(buf74, (392, 256), (256, 1), 0); del buf74  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (392, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 256), (1, 512), 0), out=buf105)
        del arg80_1
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf106, arg288_1, arg289_1, arg81_1, arg82_1, 100352, grid=grid(100352), stream=stream0)
        del arg288_1
        del arg289_1
        del arg81_1
        del arg82_1
        buf107 = reinterpret_tensor(buf104, (392, 512), (512, 1), 0); del buf104  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (392, 256), (256, 1), 0), reinterpret_tensor(arg83_1, (256, 512), (1, 256), 0), out=buf107)
        del arg83_1
        buf108 = buf107; del buf107  # reuse
        buf109 = reinterpret_tensor(buf108, (8, 49, 512), (25088, 512, 1), 0); del buf108  # reuse
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf109, arg291_1, arg292_1, arg84_1, arg85_1, 200704, grid=grid(200704), stream=stream0)
        del arg291_1
        del arg292_1
        del arg84_1
        del arg85_1
        buf110 = reinterpret_tensor(buf73, (392, 256), (256, 1), 0); del buf73  # reuse
        # Source Nodes: [x_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (392, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 256), (1, 512), 0), out=buf110)
        del arg86_1
        buf111 = reinterpret_tensor(buf106, (8, 49, 256), (12544, 256, 1), 0); del buf106  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf111, buf110, arg294_1, arg295_1, arg87_1, arg88_1, 100352, grid=grid(100352), stream=stream0)
        del arg294_1
        del arg295_1
        del arg87_1
        del arg88_1
        buf112 = reinterpret_tensor(buf109, (392, 512), (512, 1), 0); del buf109  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (392, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 512), (1, 256), 0), out=buf112)
        del arg89_1
        buf113 = buf95; del buf95  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf112, arg297_1, arg298_1, arg90_1, arg91_1, buf113, 50176, grid=grid(50176), stream=stream0)
        buf114 = reinterpret_tensor(buf94, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf94  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf112, arg297_1, arg298_1, arg90_1, arg91_1, buf114, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf115 = empty((64, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf113, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf114, (64, 16, 49), (784, 49, 1), 0), out=buf115)
        buf116 = empty((8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_23], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg213_1, arg5_1, buf116, 19208, grid=grid(19208), stream=stream0)
        del arg213_1
        del arg5_1
        buf119 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_11, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_27.run(buf115, buf116, buf119, 3136, 49, grid=grid(3136), stream=stream0)
        buf120 = reinterpret_tensor(buf110, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf110  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf112, arg297_1, arg298_1, arg90_1, arg91_1, buf120, 100352, grid=grid(100352), stream=stream0)
        del arg297_1
        del arg298_1
        del arg90_1
        del arg91_1
        buf121 = empty((64, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf120, (64, 49, 32), (1568, 32, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (8, 49, 256), (12544, 256, 1), 0); del buf120  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf121, buf122, 100352, grid=grid(100352), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (392, 256), (256, 1), 0); del buf121  # reuse
        # Source Nodes: [x_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (392, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf123)
        del arg92_1
        buf124 = buf111; del buf111  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf124, buf123, arg300_1, arg301_1, arg93_1, arg94_1, 100352, grid=grid(100352), stream=stream0)
        del arg300_1
        del arg301_1
        del arg93_1
        del arg94_1
        buf125 = buf112; del buf112  # reuse
        # Source Nodes: [x_73], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (392, 256), (256, 1), 0), reinterpret_tensor(arg95_1, (256, 512), (1, 256), 0), out=buf125)
        del arg95_1
        buf126 = buf125; del buf125  # reuse
        buf127 = reinterpret_tensor(buf126, (8, 49, 512), (25088, 512, 1), 0); del buf126  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf127, arg303_1, arg304_1, arg96_1, arg97_1, 200704, grid=grid(200704), stream=stream0)
        del arg303_1
        del arg304_1
        del arg96_1
        del arg97_1
        buf128 = buf123; del buf123  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (392, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 256), (1, 512), 0), out=buf128)
        del arg98_1
        buf129 = buf124; del buf124  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf129, buf128, arg306_1, arg307_1, arg99_1, arg100_1, 100352, grid=grid(100352), stream=stream0)
        del arg100_1
        del arg306_1
        del arg307_1
        del arg99_1
        buf130 = reinterpret_tensor(buf127, (392, 512), (512, 1), 0); del buf127  # reuse
        # Source Nodes: [x_80], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (392, 256), (256, 1), 0), reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf130)
        del arg101_1
        buf131 = reinterpret_tensor(buf114, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf114  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf130, arg309_1, arg310_1, arg102_1, arg103_1, buf131, 50176, grid=grid(50176), stream=stream0)
        buf132 = reinterpret_tensor(buf113, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf113  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf130, arg309_1, arg310_1, arg102_1, arg103_1, buf132, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf133 = reinterpret_tensor(buf119, (64, 49, 49), (2401, 49, 1), 0); del buf119  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf132, (64, 16, 49), (784, 49, 1), 0), out=buf133)
        buf134 = empty((8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_27], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg214_1, arg6_1, buf134, 19208, grid=grid(19208), stream=stream0)
        del arg214_1
        del arg6_1
        buf137 = reinterpret_tensor(buf115, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf115  # reuse
        # Source Nodes: [attn_12, attn_13, mul_6], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_27.run(buf133, buf134, buf137, 3136, 49, grid=grid(3136), stream=stream0)
        buf138 = reinterpret_tensor(buf128, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf128  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf130, arg309_1, arg310_1, arg102_1, arg103_1, buf138, 100352, grid=grid(100352), stream=stream0)
        del arg102_1
        del arg103_1
        del arg309_1
        del arg310_1
        buf139 = reinterpret_tensor(buf122, (64, 49, 32), (1568, 32, 1), 0); del buf122  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf138, (64, 49, 32), (1568, 32, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf138, (8, 49, 256), (12544, 256, 1), 0); del buf138  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf139, buf140, 100352, grid=grid(100352), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (392, 256), (256, 1), 0); del buf139  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (392, 256), (256, 1), 0), reinterpret_tensor(arg104_1, (256, 256), (1, 256), 0), out=buf141)
        del arg104_1
        buf142 = buf129; del buf129  # reuse
        # Source Nodes: [x_84], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf142, buf141, arg312_1, arg313_1, arg105_1, arg106_1, 100352, grid=grid(100352), stream=stream0)
        del arg105_1
        del arg106_1
        del arg312_1
        del arg313_1
        buf143 = buf130; del buf130  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (392, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 512), (1, 256), 0), out=buf143)
        del arg107_1
        buf144 = buf143; del buf143  # reuse
        buf145 = reinterpret_tensor(buf144, (8, 49, 512), (25088, 512, 1), 0); del buf144  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf145, arg315_1, arg316_1, arg108_1, arg109_1, 200704, grid=grid(200704), stream=stream0)
        del arg108_1
        del arg109_1
        del arg315_1
        del arg316_1
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (392, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 256), (1, 512), 0), out=buf146)
        del arg110_1
        buf147 = buf142; del buf142  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf147, buf146, arg318_1, arg319_1, arg111_1, arg112_1, 100352, grid=grid(100352), stream=stream0)
        del arg111_1
        del arg112_1
        del arg318_1
        del arg319_1
        buf148 = reinterpret_tensor(buf145, (392, 512), (512, 1), 0); del buf145  # reuse
        # Source Nodes: [x_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (392, 256), (256, 1), 0), reinterpret_tensor(arg113_1, (256, 512), (1, 256), 0), out=buf148)
        del arg113_1
        buf149 = reinterpret_tensor(buf132, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf132  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf148, arg321_1, arg322_1, arg114_1, arg115_1, buf149, 50176, grid=grid(50176), stream=stream0)
        buf150 = reinterpret_tensor(buf131, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf131  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf148, arg321_1, arg322_1, arg114_1, arg115_1, buf150, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf151 = reinterpret_tensor(buf137, (64, 49, 49), (2401, 49, 1), 0); del buf137  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf150, (64, 16, 49), (784, 49, 1), 0), out=buf151)
        buf152 = empty((8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_31], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg215_1, arg7_1, buf152, 19208, grid=grid(19208), stream=stream0)
        del arg215_1
        del arg7_1
        buf155 = reinterpret_tensor(buf133, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf133  # reuse
        # Source Nodes: [attn_14, attn_15, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_27.run(buf151, buf152, buf155, 3136, 49, grid=grid(3136), stream=stream0)
        buf156 = reinterpret_tensor(buf146, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf146  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf148, arg321_1, arg322_1, arg114_1, arg115_1, buf156, 100352, grid=grid(100352), stream=stream0)
        del arg114_1
        del arg115_1
        del arg321_1
        del arg322_1
        buf157 = reinterpret_tensor(buf140, (64, 49, 32), (1568, 32, 1), 0); del buf140  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf156, (64, 49, 32), (1568, 32, 1), 0), out=buf157)
        buf158 = reinterpret_tensor(buf156, (8, 49, 256), (12544, 256, 1), 0); del buf156  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf157, buf158, 100352, grid=grid(100352), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (392, 256), (256, 1), 0); del buf157  # reuse
        # Source Nodes: [x_94], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (392, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf159)
        del arg116_1
        buf160 = buf147; del buf147  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf160, buf159, arg324_1, arg325_1, arg117_1, arg118_1, 100352, grid=grid(100352), stream=stream0)
        del arg117_1
        del arg118_1
        del arg324_1
        del arg325_1
        buf161 = buf148; del buf148  # reuse
        # Source Nodes: [x_97], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (392, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 512), (1, 256), 0), out=buf161)
        del arg119_1
        buf162 = buf161; del buf161  # reuse
        buf163 = reinterpret_tensor(buf162, (8, 49, 512), (25088, 512, 1), 0); del buf162  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf163, arg327_1, arg328_1, arg120_1, arg121_1, 200704, grid=grid(200704), stream=stream0)
        del arg120_1
        del arg121_1
        del arg327_1
        del arg328_1
        buf164 = buf159; del buf159  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (392, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 256), (1, 512), 0), out=buf164)
        del arg122_1
        buf165 = buf160; del buf160  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf165, buf164, arg330_1, arg331_1, arg123_1, arg124_1, 100352, grid=grid(100352), stream=stream0)
        del arg123_1
        del arg124_1
        del arg330_1
        del arg331_1
        buf166 = reinterpret_tensor(buf163, (392, 512), (512, 1), 0); del buf163  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (392, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 512), (1, 256), 0), out=buf166)
        del arg125_1
        buf167 = reinterpret_tensor(buf150, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf150  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf166, arg333_1, arg334_1, arg126_1, arg127_1, buf167, 50176, grid=grid(50176), stream=stream0)
        buf168 = reinterpret_tensor(buf149, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf149  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf166, arg333_1, arg334_1, arg126_1, arg127_1, buf168, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf169 = reinterpret_tensor(buf155, (64, 49, 49), (2401, 49, 1), 0); del buf155  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf168, (64, 16, 49), (784, 49, 1), 0), out=buf169)
        del buf167
        del buf168
        buf170 = empty((8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_35], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg216_1, arg8_1, buf170, 19208, grid=grid(19208), stream=stream0)
        del arg216_1
        del arg8_1
        buf173 = reinterpret_tensor(buf151, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf151  # reuse
        # Source Nodes: [attn_16, attn_17, mul_8], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_27.run(buf169, buf170, buf173, 3136, 49, grid=grid(3136), stream=stream0)
        del buf169
        buf174 = reinterpret_tensor(buf164, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf164  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf166, arg333_1, arg334_1, arg126_1, arg127_1, buf174, 100352, grid=grid(100352), stream=stream0)
        del arg126_1
        del arg127_1
        del arg333_1
        del arg334_1
        buf175 = reinterpret_tensor(buf158, (64, 49, 32), (1568, 32, 1), 0); del buf158  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf174, (64, 49, 32), (1568, 32, 1), 0), out=buf175)
        del buf173
        buf176 = reinterpret_tensor(buf174, (8, 49, 256), (12544, 256, 1), 0); del buf174  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf175, buf176, 100352, grid=grid(100352), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (392, 256), (256, 1), 0); del buf175  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (392, 256), (256, 1), 0), reinterpret_tensor(arg128_1, (256, 256), (1, 256), 0), out=buf177)
        del arg128_1
        del buf176
        buf178 = buf165; del buf165  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf178, buf177, arg336_1, arg337_1, arg129_1, arg130_1, 100352, grid=grid(100352), stream=stream0)
        del arg129_1
        del arg130_1
        del arg336_1
        del arg337_1
        buf179 = buf166; del buf166  # reuse
        # Source Nodes: [x_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (392, 256), (256, 1), 0), reinterpret_tensor(arg131_1, (256, 512), (1, 256), 0), out=buf179)
        del arg131_1
        buf180 = buf179; del buf179  # reuse
        buf181 = reinterpret_tensor(buf180, (8, 49, 512), (25088, 512, 1), 0); del buf180  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf181, arg339_1, arg340_1, arg132_1, arg133_1, 200704, grid=grid(200704), stream=stream0)
        del arg132_1
        del arg133_1
        del arg339_1
        del arg340_1
        buf182 = buf177; del buf177  # reuse
        # Source Nodes: [x_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (392, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 256), (1, 512), 0), out=buf182)
        del arg134_1
        del buf181
        buf183 = buf178; del buf178  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf183, buf182, arg342_1, arg343_1, arg135_1, arg136_1, 100352, grid=grid(100352), stream=stream0)
        del arg135_1
        del arg136_1
        del arg342_1
        del arg343_1
        buf184 = empty((392, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (392, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 1280), (1, 256), 0), out=buf184)
        del arg137_1
        buf185 = empty((8, 4, 4, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf183, buf185, 32768, grid=grid(32768), stream=stream0)
        buf186 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (128, 256), (256, 1), 0), reinterpret_tensor(arg140_1, (256, 256), (1, 256), 0), out=buf186)
        del arg140_1
        buf187 = reinterpret_tensor(buf185, (8, 16, 16, 16), (4096, 256, 16, 1), 0); del buf185  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf186, arg348_1, arg349_1, arg141_1, arg142_1, buf187, 32768, grid=grid(32768), stream=stream0)
        del arg141_1
        del arg142_1
        del arg348_1
        del arg349_1
        del buf186
        buf188 = reinterpret_tensor(buf183, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf183  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf184, arg345_1, arg346_1, arg138_1, arg139_1, buf188, 2048, 49, grid=grid(2048, 49), stream=stream0)
        buf189 = reinterpret_tensor(buf182, (128, 16, 49), (784, 49, 1), 0); del buf182  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (128, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf188, (128, 16, 49), (784, 49, 1), 0), out=buf189)
        del buf187
        buf190 = empty((16, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_39], Original ATen: [aten.index]
        triton_poi_fused_index_33.run(arg217_1, arg9_1, buf190, 12544, grid=grid(12544), stream=stream0)
        del arg217_1
        del arg9_1
        buf193 = buf188; del buf188  # reuse
        # Source Nodes: [attn_18, attn_19, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_34.run(buf189, buf190, buf193, 2048, 49, grid=grid(2048), stream=stream0)
        del buf189
        buf194 = reinterpret_tensor(buf88, (8, 16, 49, 64), (50176, 3136, 64, 1), 0); del buf88  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf184, arg345_1, arg346_1, arg138_1, arg139_1, buf194, 401408, grid=grid(401408), stream=stream0)
        del arg138_1
        del arg139_1
        del arg345_1
        del arg346_1
        del buf184
        buf195 = empty((128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (128, 16, 49), (784, 49, 1), 0), reinterpret_tensor(buf194, (128, 49, 64), (3136, 64, 1), 0), out=buf195)
        del buf193
        del buf194
        buf196 = empty((8, 16, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_36.run(buf195, buf196, 131072, grid=grid(131072), stream=stream0)
        del buf195
        buf197 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 384), (1, 1024), 0), out=buf197)
        del arg143_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf198, arg351_1, arg352_1, arg144_1, arg145_1, 49152, grid=grid(49152), stream=stream0)
        del arg144_1
        del arg145_1
        del arg351_1
        del arg352_1
        buf199 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 384), (384, 1), 0), reinterpret_tensor(arg146_1, (384, 768), (1, 384), 0), out=buf199)
        del arg146_1
        buf200 = buf199; del buf199  # reuse
        buf201 = reinterpret_tensor(buf200, (8, 16, 768), (12288, 768, 1), 0); del buf200  # reuse
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf201, arg354_1, arg355_1, arg147_1, arg148_1, 98304, grid=grid(98304), stream=stream0)
        del arg147_1
        del arg148_1
        del arg354_1
        del arg355_1
        buf202 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 384), (1, 768), 0), out=buf202)
        del arg149_1
        buf203 = reinterpret_tensor(buf198, (8, 16, 384), (6144, 384, 1), 0); del buf198  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf203, buf202, arg357_1, arg358_1, arg150_1, arg151_1, 49152, grid=grid(49152), stream=stream0)
        del arg150_1
        del arg151_1
        del arg357_1
        del arg358_1
        buf204 = reinterpret_tensor(buf201, (128, 768), (768, 1), 0); del buf201  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 384), (384, 1), 0), reinterpret_tensor(arg152_1, (384, 768), (1, 384), 0), out=buf204)
        del arg152_1
        buf205 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf204, arg360_1, arg361_1, arg153_1, arg154_1, buf205, 24576, grid=grid(24576), stream=stream0)
        buf206 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf204, arg360_1, arg361_1, arg153_1, arg154_1, buf206, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf207 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf206, (96, 16, 16), (256, 16, 1), 0), out=buf207)
        buf208 = empty((12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_43], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg218_1, arg10_1, buf208, 3072, grid=grid(3072), stream=stream0)
        del arg10_1
        del arg218_1
        buf211 = buf206; del buf206  # reuse
        # Source Nodes: [attn_20, attn_21, mul_10], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_43.run(buf207, buf208, buf211, 1536, 16, grid=grid(1536), stream=stream0)
        buf212 = reinterpret_tensor(buf202, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf202  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf204, arg360_1, arg361_1, arg153_1, arg154_1, buf212, 49152, grid=grid(49152), stream=stream0)
        del arg153_1
        del arg154_1
        del arg360_1
        del arg361_1
        buf213 = empty((96, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf212, (96, 16, 32), (512, 32, 1), 0), out=buf213)
        buf214 = reinterpret_tensor(buf212, (8, 16, 384), (6144, 384, 1), 0); del buf212  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf213, buf214, 49152, grid=grid(49152), stream=stream0)
        buf215 = reinterpret_tensor(buf213, (128, 384), (384, 1), 0); del buf213  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 384), (384, 1), 0), reinterpret_tensor(arg155_1, (384, 384), (1, 384), 0), out=buf215)
        del arg155_1
        buf216 = buf203; del buf203  # reuse
        # Source Nodes: [x_137], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf216, buf215, arg363_1, arg364_1, arg156_1, arg157_1, 49152, grid=grid(49152), stream=stream0)
        del arg156_1
        del arg157_1
        del arg363_1
        del arg364_1
        buf217 = buf204; del buf204  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (128, 384), (384, 1), 0), reinterpret_tensor(arg158_1, (384, 768), (1, 384), 0), out=buf217)
        del arg158_1
        buf218 = buf217; del buf217  # reuse
        buf219 = reinterpret_tensor(buf218, (8, 16, 768), (12288, 768, 1), 0); del buf218  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf219, arg366_1, arg367_1, arg159_1, arg160_1, 98304, grid=grid(98304), stream=stream0)
        del arg159_1
        del arg160_1
        del arg366_1
        del arg367_1
        buf220 = buf215; del buf215  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (128, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 384), (1, 768), 0), out=buf220)
        del arg161_1
        buf221 = buf216; del buf216  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf221, buf220, arg369_1, arg370_1, arg162_1, arg163_1, 49152, grid=grid(49152), stream=stream0)
        del arg162_1
        del arg163_1
        del arg369_1
        del arg370_1
        buf222 = reinterpret_tensor(buf219, (128, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 384), (384, 1), 0), reinterpret_tensor(arg164_1, (384, 768), (1, 384), 0), out=buf222)
        del arg164_1
        buf223 = buf211; del buf211  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf222, arg372_1, arg373_1, arg165_1, arg166_1, buf223, 24576, grid=grid(24576), stream=stream0)
        buf224 = reinterpret_tensor(buf207, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf207  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf222, arg372_1, arg373_1, arg165_1, arg166_1, buf224, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf225 = reinterpret_tensor(buf205, (96, 16, 16), (256, 16, 1), 0); del buf205  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf224, (96, 16, 16), (256, 16, 1), 0), out=buf225)
        buf226 = empty((12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_47], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg219_1, arg11_1, buf226, 3072, grid=grid(3072), stream=stream0)
        del arg11_1
        del arg219_1
        buf229 = buf224; del buf224  # reuse
        # Source Nodes: [attn_22, attn_23, mul_11], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_43.run(buf225, buf226, buf229, 1536, 16, grid=grid(1536), stream=stream0)
        buf230 = reinterpret_tensor(buf220, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf220  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf222, arg372_1, arg373_1, arg165_1, arg166_1, buf230, 49152, grid=grid(49152), stream=stream0)
        del arg165_1
        del arg166_1
        del arg372_1
        del arg373_1
        buf231 = reinterpret_tensor(buf214, (96, 16, 32), (512, 32, 1), 0); del buf214  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf230, (96, 16, 32), (512, 32, 1), 0), out=buf231)
        buf232 = reinterpret_tensor(buf230, (8, 16, 384), (6144, 384, 1), 0); del buf230  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf231, buf232, 49152, grid=grid(49152), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (128, 384), (384, 1), 0); del buf231  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 384), (384, 1), 0), reinterpret_tensor(arg167_1, (384, 384), (1, 384), 0), out=buf233)
        del arg167_1
        buf234 = buf221; del buf221  # reuse
        # Source Nodes: [x_149], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf234, buf233, arg375_1, arg376_1, arg168_1, arg169_1, 49152, grid=grid(49152), stream=stream0)
        del arg168_1
        del arg169_1
        del arg375_1
        del arg376_1
        buf235 = buf222; del buf222  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (128, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 768), (1, 384), 0), out=buf235)
        del arg170_1
        buf236 = buf235; del buf235  # reuse
        buf237 = reinterpret_tensor(buf236, (8, 16, 768), (12288, 768, 1), 0); del buf236  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf237, arg378_1, arg379_1, arg171_1, arg172_1, 98304, grid=grid(98304), stream=stream0)
        del arg171_1
        del arg172_1
        del arg378_1
        del arg379_1
        buf238 = buf233; del buf233  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (128, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 384), (1, 768), 0), out=buf238)
        del arg173_1
        buf239 = buf234; del buf234  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf239, buf238, arg381_1, arg382_1, arg174_1, arg175_1, 49152, grid=grid(49152), stream=stream0)
        del arg174_1
        del arg175_1
        del arg381_1
        del arg382_1
        buf240 = reinterpret_tensor(buf237, (128, 768), (768, 1), 0); del buf237  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (128, 384), (384, 1), 0), reinterpret_tensor(arg176_1, (384, 768), (1, 384), 0), out=buf240)
        del arg176_1
        buf241 = buf229; del buf229  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf240, arg384_1, arg385_1, arg177_1, arg178_1, buf241, 24576, grid=grid(24576), stream=stream0)
        buf242 = reinterpret_tensor(buf225, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf225  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf240, arg384_1, arg385_1, arg177_1, arg178_1, buf242, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf243 = reinterpret_tensor(buf223, (96, 16, 16), (256, 16, 1), 0); del buf223  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf242, (96, 16, 16), (256, 16, 1), 0), out=buf243)
        buf244 = empty((12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_51], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg220_1, arg12_1, buf244, 3072, grid=grid(3072), stream=stream0)
        del arg12_1
        del arg220_1
        buf247 = buf242; del buf242  # reuse
        # Source Nodes: [attn_24, attn_25, mul_12], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_43.run(buf243, buf244, buf247, 1536, 16, grid=grid(1536), stream=stream0)
        buf248 = reinterpret_tensor(buf238, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf238  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf240, arg384_1, arg385_1, arg177_1, arg178_1, buf248, 49152, grid=grid(49152), stream=stream0)
        del arg177_1
        del arg178_1
        del arg384_1
        del arg385_1
        buf249 = reinterpret_tensor(buf232, (96, 16, 32), (512, 32, 1), 0); del buf232  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf248, (96, 16, 32), (512, 32, 1), 0), out=buf249)
        buf250 = reinterpret_tensor(buf248, (8, 16, 384), (6144, 384, 1), 0); del buf248  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf249, buf250, 49152, grid=grid(49152), stream=stream0)
        buf251 = reinterpret_tensor(buf249, (128, 384), (384, 1), 0); del buf249  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (128, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 384), (1, 384), 0), out=buf251)
        del arg179_1
        buf252 = buf239; del buf239  # reuse
        # Source Nodes: [x_161], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf252, buf251, arg387_1, arg388_1, arg180_1, arg181_1, 49152, grid=grid(49152), stream=stream0)
        del arg180_1
        del arg181_1
        del arg387_1
        del arg388_1
        buf253 = buf240; del buf240  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (128, 384), (384, 1), 0), reinterpret_tensor(arg182_1, (384, 768), (1, 384), 0), out=buf253)
        del arg182_1
        buf254 = buf253; del buf253  # reuse
        buf255 = reinterpret_tensor(buf254, (8, 16, 768), (12288, 768, 1), 0); del buf254  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf255, arg390_1, arg391_1, arg183_1, arg184_1, 98304, grid=grid(98304), stream=stream0)
        del arg183_1
        del arg184_1
        del arg390_1
        del arg391_1
        buf256 = buf251; del buf251  # reuse
        # Source Nodes: [x_166], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (128, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 384), (1, 768), 0), out=buf256)
        del arg185_1
        buf257 = buf252; del buf252  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf257, buf256, arg393_1, arg394_1, arg186_1, arg187_1, 49152, grid=grid(49152), stream=stream0)
        del arg186_1
        del arg187_1
        del arg393_1
        del arg394_1
        buf258 = reinterpret_tensor(buf255, (128, 768), (768, 1), 0); del buf255  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (128, 384), (384, 1), 0), reinterpret_tensor(arg188_1, (384, 768), (1, 384), 0), out=buf258)
        del arg188_1
        buf259 = buf247; del buf247  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf258, arg396_1, arg397_1, arg189_1, arg190_1, buf259, 24576, grid=grid(24576), stream=stream0)
        buf260 = reinterpret_tensor(buf243, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf243  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf258, arg396_1, arg397_1, arg189_1, arg190_1, buf260, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf261 = reinterpret_tensor(buf241, (96, 16, 16), (256, 16, 1), 0); del buf241  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf259, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf260, (96, 16, 16), (256, 16, 1), 0), out=buf261)
        del buf259
        buf262 = empty((12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getitem_55], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg221_1, arg13_1, buf262, 3072, grid=grid(3072), stream=stream0)
        del arg13_1
        del arg221_1
        buf265 = buf260; del buf260  # reuse
        # Source Nodes: [attn_26, attn_27, mul_13], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_43.run(buf261, buf262, buf265, 1536, 16, grid=grid(1536), stream=stream0)
        del buf261
        buf266 = reinterpret_tensor(buf256, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf256  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf258, arg396_1, arg397_1, arg189_1, arg190_1, buf266, 49152, grid=grid(49152), stream=stream0)
        del arg189_1
        del arg190_1
        del arg396_1
        del arg397_1
        buf267 = reinterpret_tensor(buf250, (96, 16, 32), (512, 32, 1), 0); del buf250  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf266, (96, 16, 32), (512, 32, 1), 0), out=buf267)
        del buf265
        buf268 = reinterpret_tensor(buf266, (8, 16, 384), (6144, 384, 1), 0); del buf266  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_act], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf267, buf268, 49152, grid=grid(49152), stream=stream0)
        buf269 = reinterpret_tensor(buf267, (128, 384), (384, 1), 0); del buf267  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (128, 384), (384, 1), 0), reinterpret_tensor(arg191_1, (384, 384), (1, 384), 0), out=buf269)
        del arg191_1
        del buf268
        buf270 = buf257; del buf257  # reuse
        # Source Nodes: [x_173], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf270, buf269, arg399_1, arg400_1, arg192_1, arg193_1, 49152, grid=grid(49152), stream=stream0)
        del arg192_1
        del arg193_1
        del arg399_1
        del arg400_1
        buf271 = buf258; del buf258  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (128, 384), (384, 1), 0), reinterpret_tensor(arg194_1, (384, 768), (1, 384), 0), out=buf271)
        del arg194_1
        buf272 = buf271; del buf271  # reuse
        buf273 = reinterpret_tensor(buf272, (8, 16, 768), (12288, 768, 1), 0); del buf272  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf273, arg402_1, arg403_1, arg195_1, arg196_1, 98304, grid=grid(98304), stream=stream0)
        del arg195_1
        del arg196_1
        del arg402_1
        del arg403_1
        buf274 = buf269; del buf269  # reuse
        # Source Nodes: [x_178], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 384), (1, 768), 0), out=buf274)
        del arg197_1
        del buf273
        buf276 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf278 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_46.run(buf270, buf274, arg405_1, arg406_1, arg198_1, arg199_1, arg408_1, arg409_1, arg200_1, arg201_1, arg411_1, arg412_1, arg204_1, arg205_1, buf276, buf278, 3072, 16, grid=grid(3072), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del arg201_1
        del arg204_1
        del arg205_1
        del arg405_1
        del arg406_1
        del arg408_1
        del arg409_1
        del arg411_1
        del arg412_1
        del buf270
        del buf274
        buf277 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_bn, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        extern_kernels.mm(buf276, reinterpret_tensor(arg202_1, (384, 1000), (1, 384), 0), out=buf277)
        del arg202_1
        del buf276
        buf279 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_dist_bn, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        extern_kernels.mm(buf278, reinterpret_tensor(arg206_1, (384, 1000), (1, 384), 0), out=buf279)
        del arg206_1
        del buf278
        buf280 = buf277; del buf277  # reuse
        # Source Nodes: [add_40, x_186], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_47.run(buf280, arg203_1, buf279, arg207_1, 8000, grid=grid(8000), stream=stream0)
        del arg203_1
        del arg207_1
        return (buf280, buf16, buf36, buf56, buf76, buf98, buf116, buf134, buf152, buf170, buf190, buf208, buf226, buf244, buf262, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((640, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1280, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg210_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg211_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg212_1 = rand_strided((49, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg213_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg214_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg216_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg217_1 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg218_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg219_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg220_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg222_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg225_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg228_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg240_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg243_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg246_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg252_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg258_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg267_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg270_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg273_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg276_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg282_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg285_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg288_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg294_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg300_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg306_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg312_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg318_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg324_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg327_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg330_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg333_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg336_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg339_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg342_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg345_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg348_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg351_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg354_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg357_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg360_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg363_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg366_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg369_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg372_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg375_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg378_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg381_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg384_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg387_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg390_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg393_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg396_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg399_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg402_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg405_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg408_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg411_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg414_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
