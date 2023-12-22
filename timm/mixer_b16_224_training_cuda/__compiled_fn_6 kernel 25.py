
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 768.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)
