
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 21
    x3 = (xindex // 21)
    y0 = yindex % 432
    y1 = (yindex // 432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (432 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (864 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (18576 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (19008 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (19440 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (37152 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (37584 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (38016 + y0 + (864*x2) + (37152*x3) + (798768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x2) + (86*x3)
    tmp19 = (2*x2) + (86*x3)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x2) + (86*x3)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 43 + (2*x2) + (86*x3)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 44 + (2*x2) + (86*x3)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 45 + (2*x2) + (86*x3)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 86 + (2*x2) + (86*x3)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 87 + (2*x2) + (86*x3)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 88 + (2*x2) + (86*x3)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (y0 + (432*x4) + (190512*y1)), tmp16, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (432*x4) + (190512*y1)), tmp41, xmask & ymask)
