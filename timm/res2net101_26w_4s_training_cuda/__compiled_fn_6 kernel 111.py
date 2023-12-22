
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (208*x2) + (652288*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (163072*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-163072) + x2 + (3136*y0) + (163072*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 156, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-326144) + x2 + (3136*y0) + (163072*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 208, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-489216) + x2 + (3136*y0) + (163072*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp33, xmask & ymask)
