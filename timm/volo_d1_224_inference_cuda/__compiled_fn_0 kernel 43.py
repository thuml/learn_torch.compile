
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tmp17 & tmp12
    tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp15 >= tmp3
    tmp23 = tmp15 < tmp13
    tmp24 = tmp22 & tmp12
    tmp25 = tl.load(in_ptr4 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr5 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr6 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp24, tmp29, tmp30)
    tmp32 = tl.where(tmp17, tmp21, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp12, tmp32, tmp33)
    tmp35 = tl.where(tmp4, tmp11, tmp34)
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp35, xmask)