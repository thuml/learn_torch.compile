
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14112
    xnumel = 432
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 1764)
    y4 = yindex % 1764
    y1 = (yindex // 42) % 42
    y0 = yindex % 42
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + (1764*x3) + (762048*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3 + (432*y5)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp29 = tl.load(in_ptr4 + (x3 + (432*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = y1
    tmp2 = tl.full([1, 1], 43, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = y0
    tmp5 = tmp4 < tmp2
    tmp6 = tmp3 & tmp5
    tmp7 = tl.load(in_ptr1 + (x3 + (432*y0) + (18576*y1) + (798768*y2)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp0 + tmp9
    tmp12 = 2 + y1
    tmp13 = tl.full([1, 1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.full([1, 1], 47, tl.int64)
    tmp16 = tmp12 < tmp15
    tmp17 = 2 + y0
    tmp18 = tmp17 >= tmp13
    tmp19 = tmp17 < tmp15
    tmp20 = tmp14 & tmp16
    tmp21 = tmp20 & tmp18
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + (96 + y0 + (47*y1) + (2209*x3) + (954288*y2)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = 0.0
    tmp27 = tl.where(tmp11, tmp26, tmp25)
    tmp28 = tmp10 + tmp27
    tmp31 = tmp29 - tmp30
    tmp33 = 7.086167800453515e-05
    tmp34 = tmp32 * tmp33
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 * tmp36
    tmp38 = tmp31 * tmp37
    tmp39 = tmp28 - tmp38
    tmp41 = tmp40 * tmp33
    tmp42 = tmp39 - tmp41
    tl.store(out_ptr0 + (x3 + (432*y5)), tmp42, xmask & ymask)
