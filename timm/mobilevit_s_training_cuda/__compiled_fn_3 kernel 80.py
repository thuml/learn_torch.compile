
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((240*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (x2 % 2)) // 4) % 16)) + (3840*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (15360*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (15360*y1) + (x2 % 2)) // 15360) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp4, xmask & ymask)
