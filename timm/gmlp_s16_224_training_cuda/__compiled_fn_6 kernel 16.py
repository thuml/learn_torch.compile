
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_gelu_gelu_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1536
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
    tmp31 = tl.load(in_ptr9 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (768*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + ((-150528) + y0 + (196*x2) + (150528*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + (tl.broadcast_to((-768) + x2, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = 768.0
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr6 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 - tmp21
    tmp23 = tl.load(in_ptr7 + ((-768) + x2 + (768*y3)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr8 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tmp27 = tmp15 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp12, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp11, tmp29)
    tmp32 = 0.7071067811865476
    tmp33 = tmp31 * tmp32
    tmp34 = tl.math.erf(tmp33)
    tmp35 = 1.0
    tmp36 = tmp34 + tmp35
    tmp37 = 0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tmp31 * tmp31
    tmp40 = -0.5
    tmp41 = tmp39 * tmp40
    tmp42 = tl.exp(tmp41)
    tmp43 = 0.3989422804014327
    tmp44 = tmp42 * tmp43
    tmp45 = tmp31 * tmp44
    tmp46 = tmp38 + tmp45
    tmp47 = tmp30 * tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1536*y3)), tmp47, xmask & ymask)
