
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1568)
    x3 = xindex % 1568
    x1 = (xindex // 49) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (47040 + x3 + (50176*x2)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (960 + x1 + (1024*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (47040 + x3 + (48608*x2)), xmask)
    tmp15 = tl.load(in_ptr5 + (47040 + x3 + (48608*x2)), xmask)
    tmp17 = tl.load(in_ptr6 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13 <= tmp4
    tmp16 = tl.where(tmp14, tmp4, tmp15)
    tmp18 = tmp17 + tmp7
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp12 + tmp22
    tl.store(out_ptr0 + (x4), tmp23, xmask)
