
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_threshold_backward_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr3 + (x3), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 / 4
    tmp7 = tl.math.max(0, (x1 // 2))
    tmp8 = tl.math.min(32, 1 + (x1 // 2))
    tmp9 = tmp7 < tmp8
    tmp10 = tl.math.max(0, (x0 // 2))
    tmp11 = tl.math.min(32, 1 + (x0 // 2))
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp1)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(tmp4, tmp1, tmp16)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.where(tmp2, tmp1, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
