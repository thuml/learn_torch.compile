
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*i64', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__unsafe_index_add_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x5), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr12 + (x5), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x4), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr15 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp18 = tmp17 + 28
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tmp22 = tmp21 + 28
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr7 + (tmp24 + (28*tmp20) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 - tmp26
    tmp29 = 6272.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp30 + tmp6
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp16 + tmp37
    tmp40 = tmp39 + 14
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp44 = tmp43 + 14
    tmp45 = tmp43 < 0
    tmp46 = tl.where(tmp45, tmp44, tmp43)
    tmp47 = tl.load(in_ptr13 + (tmp46 + (14*tmp42) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp49 = tmp47 - tmp48
    tmp51 = 1568.0
    tmp52 = tmp50 / tmp51
    tmp53 = tmp52 + tmp6
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp38 + tmp59
    tmp61 = triton_helpers.maximum(0, tmp60)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(in_out_ptr0 + (x3), tmp61, xmask)
