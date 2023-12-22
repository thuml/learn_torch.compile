
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*i64', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(19,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__unsafe_index_add_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 784) % 36
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x6 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x4), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr16 + (x2), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr17 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp16 + 14
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp21 = tmp20 + 14
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr7 + (tmp23 + (14*tmp19) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp26 = tmp24 - tmp25
    tmp28 = 1568.0
    tmp29 = tmp27 / tmp28
    tmp30 = tmp29 + tmp6
    tmp31 = tl.math.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp15 + tmp36
    tmp39 = tmp38 + 7
    tmp40 = tmp38 < 0
    tmp41 = tl.where(tmp40, tmp39, tmp38)
    tmp43 = tmp42 + 7
    tmp44 = tmp42 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp42)
    tmp46 = tl.load(in_ptr13 + (tmp45 + (7*tmp41) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp48 = tmp46 - tmp47
    tmp50 = 392.0
    tmp51 = tmp49 / tmp50
    tmp52 = tmp51 + tmp6
    tmp53 = tl.math.rsqrt(tmp52)
    tmp54 = tmp48 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = tmp37 + tmp58
    tmp60 = triton_helpers.maximum(0, tmp59)
    tl.store(in_out_ptr0 + (x4), tmp60, xmask)
