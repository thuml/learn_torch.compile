
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 760
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 95
    x2 = xindex
    y1 = (yindex // 95)
    y3 = yindex
    tmp26 = tl.load(in_ptr4 + (y0 + (95*x2) + (18620*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 84, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y0) + (20776*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x2 + (196*y0) + (20776*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x2 + (196*y3)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tmp28 = tmp26 - tmp27
    tmp30 = 0.0006377551020408163
    tmp31 = tmp29 * tmp30
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp28 * tmp34
    tmp36 = tmp25 - tmp35
    tmp38 = tmp37 * tmp30
    tmp39 = tmp36 - tmp38
    tmp41 = tmp32 * tmp40
    tmp42 = tmp39 * tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp42, xmask & ymask)
