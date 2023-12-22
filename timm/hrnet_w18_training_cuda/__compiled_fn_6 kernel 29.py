
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 144
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x3), xmask)
    tmp25 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr14 + (x3), xmask)
    tmp39 = tl.load(in_ptr15 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tmp40 = tmp38 - tmp39
    tmp42 = tmp41 * tmp11
    tmp44 = tmp43 * tmp43
    tmp45 = tmp42 * tmp44
    tmp46 = tmp40 * tmp45
    tmp47 = tmp6 - tmp46
    tmp48 = tmp47 - tmp19
    tmp50 = tmp43 * tmp49
    tmp51 = tmp48 * tmp50
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tl.store(out_ptr1 + (x3), tmp37, xmask)
    tl.store(out_ptr2 + (x3), tmp51, xmask)
