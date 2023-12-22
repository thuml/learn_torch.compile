
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 75264
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 3136)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 196
    y8 = (yindex // 196)
    y1 = (yindex // 196) % 16
    y2 = (yindex // 3136) % 8
    y3 = (yindex // 25088)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.42044820762685725
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-802816) + y0 + (196*x4) + (6272*y8)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp6
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp0 >= tmp11
    tmp19 = tl.full([1, 1], 24, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + ((-1605632) + x4 + (32*y7)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp13, tmp17, tmp23)
    tmp25 = tl.where(tmp4, tmp9, tmp24)
    tl.store(out_ptr0 + (x4 + (32*y1) + (512*y3) + (1536*y0) + (301056*y2)), tmp25, xmask & ymask)
