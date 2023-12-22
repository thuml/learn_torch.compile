
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 602304
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 25096)
    y0 = yindex % 3137
    x3 = xindex
    y1 = (yindex // 3137) % 8
    y4 = yindex
    y5 = (yindex // 3137)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x3 + (8*y1) + (64*y0) + (200768*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x3 + (8*y1) + (64*(((-1) + y0) % 3136)) + (200704*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + (8*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-1606144) + y0 + (3137*x3) + (25096*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-1606144) + y1 + (8*x3) + (64*y0) + (200768*y2)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-512) + x3 + (8*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x3 + (8*y1)
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp25
    tmp43 = tmp42 & tmp39
    tmp44 = tl.load(in_ptr6 + ((-802816) + (3136*x3) + (25088*y1) + (50176*y2) + (((-1) + y0) % 3136)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = tmp40 >= tmp25
    tmp48 = tl.full([1, 1], 40, tl.int64)
    tmp49 = tmp40 < tmp48
    tmp50 = tmp47 & tmp49
    tmp51 = tmp50 & tmp39
    tmp52 = tl.load(in_ptr7 + ((-1254400) + (3136*x3) + (25088*y1) + (75264*y2) + (((-1) + y0) % 3136)), tmp51 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp51, tmp52, tmp53)
    tmp55 = tmp40 >= tmp48
    tmp56 = tl.full([1, 1], 64, tl.int64)
    tmp57 = tmp40 < tmp56
    tmp58 = tmp55 & tmp39
    tmp59 = tl.load(in_ptr8 + ((-1329664) + (3136*x3) + (25088*y1) + (75264*y2) + (((-1) + y0) % 3136)), tmp58 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp58, tmp59, tmp60)
    tmp62 = tl.where(tmp50, tmp54, tmp61)
    tmp63 = tl.where(tmp42, tmp46, tmp62)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp39, tmp63, tmp64)
    tmp66 = tl.where(tmp7, tmp65, tmp18)
    tmp67 = tl.load(in_ptr9 + ((-3212288) + x3 + (8*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp36, tmp68, tmp69)
    tmp71 = tl.where(tmp27, tmp35, tmp70)
    tmp72 = tl.where(tmp4, tmp23, tmp71)
    tl.store(out_ptr0 + (x3 + (8*y4)), tmp72, xmask & ymask)
