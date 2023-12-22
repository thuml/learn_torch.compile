
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_170', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (y0 + (192*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 96, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (301056*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-301056) + x2 + (3136*y0) + (301056*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp20 = tmp18 - tmp19
    tmp22 = 3.985969387755102e-05
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp34, xmask & ymask)
