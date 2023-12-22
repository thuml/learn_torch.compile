
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(23,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x3), None).to(tl.int1)
    tmp23 = tl.load(in_ptr9 + (x3), None)
    tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr15 + (x3), None)
    tmp40 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr18 + (x1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp22 = tl.where(tmp21, tmp2, tmp1)
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp8
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp22 - tmp31
    tmp34 = tmp33 * tmp8
    tmp35 = tmp32 - tmp34
    tmp37 = tmp28 * tmp36
    tmp38 = tmp35 * tmp37
    tmp41 = tmp39 - tmp40
    tmp43 = tmp42 * tmp8
    tmp45 = tmp44 * tmp44
    tmp46 = tmp43 * tmp45
    tmp47 = tmp41 * tmp46
    tmp48 = tmp22 - tmp47
    tmp49 = tmp48 - tmp34
    tmp51 = tmp44 * tmp50
    tmp52 = tmp49 * tmp51
    tl.store(out_ptr0 + (x3), tmp20, None)
    tl.store(out_ptr1 + (x3), tmp38, None)
    tl.store(out_ptr2 + (x3), tmp52, None)
