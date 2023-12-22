
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_104', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 128)
    y0 = yindex % 56
    y1 = (yindex // 56)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (y0 + (56*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (56*x5) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 128.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + (128*y0) + (7168*x3) + (401408*y1)), tmp12, xmask & ymask)
