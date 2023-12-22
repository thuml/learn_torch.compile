
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8) % 8
    y2 = (yindex // 64)
    y4 = yindex % 64
    tmp0 = tl.load(in_ptr0 + ((16*((((4*(y1 % 4)) + (y0 % 4)) // 4) % 4)) + (64*y0) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(y0 % 4)) + (64*((((4*(y1 % 4)) + (y0 % 4)) // 4) % 4)) + (256*(y0 // 4)) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(y0 % 4)) + (64*(y1 % 4)) + (256*(y0 // 4)) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (64*x3) + (8192*y2)), tmp4, xmask & ymask)
