
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13 + tmp7
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp5 * tmp17
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
