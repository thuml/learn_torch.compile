
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y0 = yindex % 144
    y1 = (yindex // 144)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((144*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (x2 % 2)) // 4) % 256)) + (36864*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (147456*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (147456*y1) + (x2 % 2)) // 147456) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp4, xmask & ymask)
