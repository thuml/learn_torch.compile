
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 24
    x2 = xindex
    y1 = (yindex // 24)
    y3 = yindex
    tmp29 = tl.load(in_ptr6 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (12*x2) + (37632*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 24, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-37632) + x2 + (3136*y0) + (37632*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-12) + y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp7, tmp27)
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp15
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp19
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp42, xmask & ymask)
