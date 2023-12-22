
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    x5 = (xindex // 14) % 14
    x4 = xindex % 14
    x6 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x3), xmask)
    tmp48 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = x5
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp8
    tmp34 = 0.0
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35 + tmp34
    tmp37 = 0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tmp38.to(tl.int32)
    tmp40 = x4
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp8
    tmp43 = tmp42 + tmp34
    tmp44 = tmp43 + tmp34
    tmp45 = tmp44 * tmp37
    tmp46 = tmp45.to(tl.int32)
    tmp47 = tl.load(in_ptr10 + (tmp46 + (7*tmp39) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp49 = tmp47 - tmp48
    tmp51 = tmp50 + tmp4
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp8
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp30 + tmp59
    tmp61 = triton_helpers.maximum(0, tmp60)
    tl.store(in_out_ptr0 + (x3), tmp61, xmask)
