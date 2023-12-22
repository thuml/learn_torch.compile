
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 768
    x2 = xindex
    y1 = (yindex // 768)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (289*y0) + (55488*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1, 1], 384, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + ((-55488) + x2 + (289*y0) + (55488*y1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr6 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 + tmp9
    tmp32 = tl.sqrt(tmp31)
    tmp33 = 1 / tmp32
    tmp34 = tmp33 * tmp13
    tmp35 = tmp29 * tmp34
    tmp36 = tl.load(in_ptr8 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp26, tmp40, tmp41)
    tmp43 = tmp0 >= tmp24
    tmp44 = tl.full([1, 1], 576, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr10 + ((-110976) + x2 + (289*y0) + (55488*y1)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 - tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp9
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp13
    tmp55 = tmp49 * tmp54
    tmp56 = tl.load(in_ptr13 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.load(in_ptr14 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 + tmp58
    tmp60 = triton_helpers.maximum(0, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp46, tmp60, tmp61)
    tmp63 = tmp0 >= tmp44
    tmp64 = tl.full([1, 1], 768, tl.int64)
    tmp65 = tmp0 < tmp64
    tmp66 = tl.load(in_ptr15 + ((-166464) + x2 + (289*y0) + (55488*y1)), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr16 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = tl.load(in_ptr17 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 + tmp9
    tmp71 = tl.sqrt(tmp70)
    tmp72 = 1 / tmp71
    tmp73 = tmp72 * tmp13
    tmp74 = tmp68 * tmp73
    tmp75 = tl.load(in_ptr18 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr19 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 + tmp77
    tmp79 = triton_helpers.maximum(0, tmp78)
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp63, tmp79, tmp80)
    tmp82 = tl.where(tmp46, tmp62, tmp81)
    tmp83 = tl.where(tmp26, tmp42, tmp82)
    tmp84 = tl.where(tmp4, tmp22, tmp83)
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp84, xmask)
