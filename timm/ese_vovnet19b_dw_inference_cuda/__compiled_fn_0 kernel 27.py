
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 648
    xnumel = 1440
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 81
    y1 = (yindex // 81)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (768*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 992, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-768) + x2 + (224*y3)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 1216, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-992) + x2 + (224*y3)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1440, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-98496) + y0 + (81*x2) + (18144*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 - tmp26
    tmp28 = tl.load(in_ptr5 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.load(in_ptr6 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr7 + (tl.broadcast_to((-1216) + x2, [XBLOCK, YBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp22, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp21, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x2 + (1440*y3)), tmp45, xmask & ymask)
