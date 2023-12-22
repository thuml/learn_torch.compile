
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 * tmp9
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp4 - tmp30
    tmp32 = tmp31 - tmp17
    tmp34 = tmp27 * tmp33
    tmp35 = tmp32 * tmp34
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp35, xmask)
