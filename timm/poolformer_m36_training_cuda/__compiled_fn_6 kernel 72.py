
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 6.64328231292517e-06
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tmp13 * tmp3
    tmp15 = tmp2 * tmp7
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 - tmp16
    tmp18 = tmp1 + tmp17
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp21, xmask & ymask)
