
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 196.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp21 = tmp19 - tmp20
    tmp23 = 0.0006377551020408163
    tmp24 = tmp22 * tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp21 * tmp27
    tmp29 = tmp18 - tmp28
    tmp31 = tmp30 * tmp23
    tmp32 = tmp29 - tmp31
    tmp34 = tmp25 * tmp33
    tmp35 = tmp32 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (480*y3)), tmp35, xmask & ymask)
