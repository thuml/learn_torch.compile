
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 768
    y1 = (yindex // 768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13056 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (13824 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14592 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26112 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26880 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (27648 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4 + (64*y0) + (81920*y1)), tmp16, xmask)
