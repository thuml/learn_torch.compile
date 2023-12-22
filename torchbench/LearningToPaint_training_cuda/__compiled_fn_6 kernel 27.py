
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (128*x2) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19 + tmp13
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp11 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (32768*y1)), tmp18, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (128*x2) + (32768*y1)), tmp24, xmask & ymask)
