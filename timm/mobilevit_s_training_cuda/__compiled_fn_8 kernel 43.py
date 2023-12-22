
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 192)
    y0 = yindex
    x1 = xindex % 192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 192.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp12, ymask)
