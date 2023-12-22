
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (401408 + y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp36, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (128*y3)), tmp39, xmask & ymask)
