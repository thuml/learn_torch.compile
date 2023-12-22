
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_threshold_backward_170', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    x3 = (xindex // 83)
    x2 = xindex % 83
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (108*x4) + (744012*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = 1 + x3
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 85, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = 1 + x2
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp3 & tmp5
    tmp10 = tmp9 & tmp7
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr1 + (86 + x2 + (85*x3) + (7225*y5)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp0, tmp15, tmp14)
    tmp17 = tl.load(in_ptr2 + (9288 + y0 + (108*x2) + (9180*x3) + (780300*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tmp16 + tmp19
    tmp21 = 2 + x3
    tmp22 = tmp21 >= tmp2
    tmp23 = tl.full([1, 1], 87, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 2 + x2
    tmp26 = tmp25 >= tmp2
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tl.load(in_ptr3 + (176 + x2 + (87*x3) + (7569*y5)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp0, tmp15, tmp33)
    tmp35 = tmp20 + tmp34
    tl.store(out_ptr0 + (x4 + (6889*y5)), tmp35, xmask & ymask)
