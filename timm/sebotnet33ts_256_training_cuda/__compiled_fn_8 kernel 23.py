
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (98304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (64*x2) + (98304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.001953125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp11
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp6 - tmp29
    tmp31 = tmp30 - tmp19
    tl.store(out_ptr0 + (x2 + (1536*y3)), tmp20, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1536*y3)), tmp31, xmask & ymask)
