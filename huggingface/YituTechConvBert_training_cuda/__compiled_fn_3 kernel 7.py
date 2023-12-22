
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196608
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 384)
    y3 = yindex
    y0 = yindex % 384
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp2 = tmp1 + 1
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert((0 <= tmp4) & (tmp4 < 1), "index out of bounds: 0 <= tmp4 < 1")
    tmp5 = (-4) + x2 + y1
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1, 1], 512, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.load(in_ptr1 + ((-1536) + y3 + (384*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x2 + (9*y3)), tmp15, xmask)
