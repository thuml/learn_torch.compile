
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp5 = tl.load(in_ptr2 + (x4), None)
    tmp7 = tl.load(in_ptr3 + (x4), None)
    tmp9 = tl.load(in_ptr4 + (x4), None)
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x4), None)
    tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 - tmp10
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp13
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp8 - tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp15 * tmp34
    tmp36 = tmp22 * tmp35
    tmp38 = tmp28 * tmp37
    tmp39 = tmp33 * tmp38
    tl.store(in_out_ptr0 + (x4), tmp36, None)
    tl.store(in_out_ptr1 + (x4), tmp39, None)
