
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 64)
    y0 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (64*x2) + (131072*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr12 + (x2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr13 + (x2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr14 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.001953125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp27 = tmp25 - tmp26
    tmp29 = tmp28 * tmp15
    tmp31 = tmp30 * tmp30
    tmp32 = tmp29 * tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp10 - tmp33
    tmp35 = tmp34 - tmp23
    tmp37 = tmp17 * tmp36
    tmp38 = tmp24 * tmp37
    tmp40 = tmp30 * tmp39
    tmp41 = tmp35 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (2048*y3)), tmp38, ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (2048*y3)), tmp41, ymask)
