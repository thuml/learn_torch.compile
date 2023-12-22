
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_slice_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 848
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 106
    x2 = xindex
    y1 = (yindex // 106)
    y3 = yindex
    tmp22 = tl.load(in_ptr3 + (y0 + (106*x2) + (20776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 95, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x2 + (196*y3)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tmp24 = tmp22 - tmp23
    tmp26 = 0.0006377551020408163
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tmp32 = tmp21 - tmp31
    tmp34 = tmp33 * tmp26
    tmp35 = tmp32 - tmp34
    tmp37 = tmp28 * tmp36
    tmp38 = tmp35 * tmp37
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp38, xmask & ymask)
