
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x5 = (xindex // 196)
    x2 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(7, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(7, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp28, None)
