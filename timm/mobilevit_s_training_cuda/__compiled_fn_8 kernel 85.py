
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 32
    y2 = (yindex // 1024)
    y4 = yindex % 1024
    tmp0 = tl.load(in_ptr0 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 144.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (y4 + (1024*x3) + (147456*y2)), tmp14, xmask)
