
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr9 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 6.64328231292517e-06
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
