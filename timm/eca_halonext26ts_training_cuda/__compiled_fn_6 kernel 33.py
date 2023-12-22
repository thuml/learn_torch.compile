
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y4 % 256), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y4 % 256), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y4 % 256), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y4 % 256), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (y0 + (32*x5) + (8192*y1)), tmp17, xmask)
