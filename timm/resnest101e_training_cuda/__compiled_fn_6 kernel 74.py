
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x5 = (xindex // 4096)
    x2 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2))))) >= 0, 0, 32))) + (1024*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x4), None)
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(32, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(32, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp18 = tmp16 - tmp17
    tmp20 = 3.0517578125e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x4), tmp32, None)
