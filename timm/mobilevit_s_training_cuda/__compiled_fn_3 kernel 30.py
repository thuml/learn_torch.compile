
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 144
    x2 = (xindex // 144)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 144.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (36864*y0)), tmp13, ymask)
    tl.store(out_ptr1 + (x3 + (36864*y0)), tmp17, ymask)
