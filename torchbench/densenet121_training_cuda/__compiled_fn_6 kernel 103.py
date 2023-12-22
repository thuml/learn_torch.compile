
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 25088)
    x3 = xindex % 25088
    x1 = (xindex // 784) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (301056 + x3 + (401408*x2)), None)
    tmp3 = tl.load(in_ptr1 + (301056 + x3 + (401408*x2)), None)
    tmp5 = tl.load(in_ptr2 + (384 + x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (384 + x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (301056 + x3 + (376320*x2)), None)
    tmp14 = tl.load(in_ptr5 + (301056 + x3 + (376320*x2)), None)
    tmp16 = tl.load(in_ptr6 + (384 + x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (384 + x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (301056 + x3 + (351232*x2)), None)
    tmp25 = tl.load(in_ptr9 + (301056 + x3 + (351232*x2)), None)
    tmp27 = tl.load(in_ptr10 + (384 + x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (384 + x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr12 + (301056 + x3 + (326144*x2)), None)
    tmp36 = tl.load(in_ptr13 + (301056 + x3 + (326144*x2)), None)
    tmp38 = tl.load(in_ptr14 + (384 + x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr15 + (384 + x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tmp13 = tmp12 <= tmp1
    tmp15 = tl.where(tmp13, tmp1, tmp14)
    tmp17 = tmp16 + tmp6
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp11 + tmp21
    tmp24 = tmp23 <= tmp1
    tmp26 = tl.where(tmp24, tmp1, tmp25)
    tmp28 = tmp27 + tmp6
    tmp29 = tl.math.rsqrt(tmp28)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp22 + tmp32
    tmp35 = tmp34 <= tmp1
    tmp37 = tl.where(tmp35, tmp1, tmp36)
    tmp39 = tmp38 + tmp6
    tmp40 = tl.math.rsqrt(tmp39)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp37 * tmp42
    tmp44 = tmp33 + tmp43
    tl.store(in_out_ptr0 + (x4), tmp44, None)
