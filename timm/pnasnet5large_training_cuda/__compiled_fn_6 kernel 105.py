
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14112
    xnumel = 432
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y1 = (yindex // 42) % 42
    y0 = yindex % 42
    y2 = (yindex // 1764)
    tmp0 = tl.load(in_ptr0 + (x3 + (432*y4)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp32 = tl.load(in_ptr4 + (x3 + (432*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = y1
    tmp2 = tl.full([1, 1], 43, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = y0
    tmp5 = tmp4 < tmp2
    tmp6 = tmp3 & tmp5
    tmp7 = tl.load(in_ptr1 + (y0 + (43*y1) + (1849*x3) + (798768*y2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp0, tmp10, tmp9)
    tmp12 = tl.load(in_ptr2 + (x3 + (432*y0) + (18576*y1) + (798768*y2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tmp11 + tmp14
    tmp16 = 1 + y1
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tl.full([1, 1], 45, tl.int64)
    tmp20 = tmp16 < tmp19
    tmp21 = 1 + y0
    tmp22 = tmp21 >= tmp17
    tmp23 = tmp21 < tmp19
    tmp24 = tmp18 & tmp20
    tmp25 = tmp24 & tmp22
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr3 + (46 + y0 + (45*y1) + (2025*x3) + (874800*y2)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp0, tmp10, tmp29)
    tmp31 = tmp15 + tmp30
    tmp34 = tmp32 - tmp33
    tmp36 = 7.086167800453515e-05
    tmp37 = tmp35 * tmp36
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 * tmp39
    tmp41 = tmp34 * tmp40
    tmp42 = tmp31 - tmp41
    tl.store(out_ptr0 + (x3 + (432*y4)), tmp42, xmask & ymask)
