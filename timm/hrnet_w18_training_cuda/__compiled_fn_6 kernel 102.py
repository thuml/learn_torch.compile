
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(24,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr12 + (x3), xmask)
    tmp35 = tl.load(in_ptr13 + (x3), xmask)
    tmp37 = tl.load(in_ptr14 + (x3), xmask)
    tmp40 = tl.load(in_ptr15 + (x3), xmask)
    tmp41 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr20 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tmp33 = 0.0
    tmp34 = tmp32 <= tmp33
    tmp36 = tmp0 + tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.where(tmp34, tmp33, tmp38)
    tmp42 = tmp40 - tmp41
    tmp44 = tmp43 * tmp5
    tmp46 = tmp45 * tmp45
    tmp47 = tmp44 * tmp46
    tmp48 = tmp42 * tmp47
    tmp49 = tmp39 - tmp48
    tmp51 = tmp50 * tmp5
    tmp52 = tmp49 - tmp51
    tmp54 = tmp45 * tmp53
    tmp55 = tmp52 * tmp54
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
