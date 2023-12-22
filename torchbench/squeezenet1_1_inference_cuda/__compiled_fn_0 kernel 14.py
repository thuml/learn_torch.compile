
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 13
    x3 = (xindex // 13)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (27 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (29 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (54 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (55 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (56 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (256*x5) + (43264*y1)), tmp16, xmask)