
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_178', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 / tmp3
    tmp11 = 0.5
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp4, tmp13, tmp9)
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp15, tmp14)
    tmp19 = tmp17 - tmp18
    tmp21 = 9.964923469387754e-06
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp33, xmask & ymask)
