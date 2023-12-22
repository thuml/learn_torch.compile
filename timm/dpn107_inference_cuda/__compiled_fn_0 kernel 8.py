
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9433088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 376
    x2 = (xindex // 1179136)
    x3 = xindex % 1179136
    x4 = xindex
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 376, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr5 + ((-802816) + x3 + (376320*x2)), tmp16, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tmp39 = tmp22 - tmp38
    tmp41 = tmp40 + tmp26
    tmp42 = tl.sqrt(tmp41)
    tmp43 = 1 / tmp42
    tmp44 = tmp43 * tmp30
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = triton_helpers.maximum(0, tmp49)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr2 + (x4), tmp50, None)
