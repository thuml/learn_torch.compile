
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp34 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = x1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp8
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp20
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.int32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp8
    tmp29 = tmp28 + tmp20
    tmp30 = tmp29 + tmp20
    tmp31 = tmp30 * tmp23
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr5 + (tmp32 + (14*tmp25) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1 / tmp38
    tmp40 = tmp39 * tmp8
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp16 + tmp45
    tmp47 = 0.25
    tmp48 = tmp22 * tmp47
    tmp49 = tmp48.to(tl.int32)
    tmp50 = tmp30 * tmp47
    tmp51 = tmp50.to(tl.int32)
    tmp52 = tl.load(in_ptr10 + (tmp51 + (7*tmp49) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp4
    tmp57 = tl.sqrt(tmp56)
    tmp58 = 1 / tmp57
    tmp59 = tmp58 * tmp8
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp46 + tmp64
    tmp66 = triton_helpers.maximum(0, tmp65)
    tl.store(in_out_ptr0 + (x4), tmp66, xmask)
