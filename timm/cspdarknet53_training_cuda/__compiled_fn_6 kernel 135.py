
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_134', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr3 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-1048576) + x2 + (16384*y0) + (1048576*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + ((-1048576) + x2 + (16384*y0) + (1048576*y1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp8, tmp16)
    tmp18 = 0.01
    tmp19 = tmp17 * tmp18
    tmp20 = tl.where(tmp0, tmp17, tmp19)
    tmp23 = tmp21 - tmp22
    tmp25 = 7.62939453125e-06
    tmp26 = tmp24 * tmp25
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp23 * tmp29
    tmp31 = tmp20 - tmp30
    tmp33 = tmp32 * tmp25
    tmp34 = tmp31 - tmp33
    tmp36 = tmp27 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (y0 + (128*x2) + (2097152*y1)), tmp37, None)
