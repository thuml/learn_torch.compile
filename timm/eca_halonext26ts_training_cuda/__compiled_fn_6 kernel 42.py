
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex % 256
    tmp0 = tl.load(in_ptr0 + ((16*((((8*(y1 % 8)) + (y0 % 8)) // 8) % 8)) + (128*y0) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(y0 % 8)) + (128*((((8*(y1 % 8)) + (y0 % 8)) // 8) % 8)) + (1024*(y0 // 8)) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(y0 % 8)) + (128*(y1 % 8)) + (1024*(y0 // 8)) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (256*x3) + (32768*y2)), tmp4, xmask)
