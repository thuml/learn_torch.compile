
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 768)
    x3 = xindex
    y4 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48) % 16
    y5 = yindex % 768
    y6 = (yindex // 48)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (784*y4)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y1 + (16*y0) + (768*y2), [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = 1e-12
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp5 / tmp8
    tmp10 = tmp6 >= tmp7
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to(y4, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp6 == tmp12
    tmp15 = tl.load(in_ptr3 + (y5 + (2304*x3) + (1806336*y2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 / tmp6
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tmp13 * tmp17
    tmp19 = tmp9 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1, 1], 16, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tl.load(in_ptr4 + ((-4816896) + y0 + (48*x3) + (37632*y6)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr5 + (tl.broadcast_to((-6144) + y1 + (16*y0) + (768*y2), [XBLOCK, YBLOCK])), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = triton_helpers.maximum(tmp27, tmp7)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp27 >= tmp7
    tmp31 = tl.load(in_ptr6 + (tl.broadcast_to((-6144) + y4, [XBLOCK, YBLOCK])), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp30, tmp31, tmp12)
    tmp33 = tmp27 == tmp12
    tmp34 = tl.load(in_ptr7 + ((-14450688) + y5 + (2304*x3) + (1806336*y2)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 / tmp27
    tmp36 = tl.where(tmp33, tmp12, tmp35)
    tmp37 = tmp32 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp25, tmp38, tmp39)
    tmp41 = tmp0 >= tmp23
    tmp42 = tl.full([1, 1], 24, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr8 + ((-9633792) + x3 + (784*y4)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp25, tmp40, tmp46)
    tmp48 = tl.where(tmp4, tmp21, tmp47)
    tl.store(out_ptr0 + (x3 + (784*y4)), tmp48, xmask)
