
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_206', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 217800
    xnumel = 54
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 27225)
    y4 = yindex % 27225
    y1 = (yindex // 165) % 165
    y0 = yindex % 165
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + (27225*x3) + (1470150*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x3 + (54*y6)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr4 + (x3 + (54*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 1 + y1
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 167, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = 1 + y0
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp3 & tmp5
    tmp10 = tmp9 & tmp7
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr1 + (9072 + x3 + (54*y0) + (9018*y1) + (1506006*y2)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 + tmp14
    tmp17 = 3 + y1
    tmp18 = tmp17 >= tmp2
    tmp19 = tl.full([1, 1], 171, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = 3 + y0
    tmp22 = tmp21 >= tmp2
    tmp23 = tmp21 < tmp19
    tmp24 = tmp18 & tmp20
    tmp25 = tmp24 & tmp22
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr3 + (516 + y0 + (171*y1) + (29241*x3) + (1579014*y2)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = 0.0
    tmp31 = tl.where(tmp16, tmp30, tmp29)
    tmp32 = tmp15 + tmp31
    tmp35 = tmp33 - tmp34
    tmp37 = 4.591368227731864e-06
    tmp38 = tmp36 * tmp37
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 * tmp40
    tmp42 = tmp35 * tmp41
    tmp43 = tmp32 - tmp42
    tl.store(out_ptr0 + (x3 + (54*y6)), tmp43, xmask & ymask)
