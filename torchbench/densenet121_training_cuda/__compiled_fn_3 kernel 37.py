
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(28,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    x4 = (xindex // 196) % 256
    x5 = (xindex // 50176)
    x6 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(0, tmp22)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp23, None)
    tl.store(out_ptr2 + (x6 + (75264*x5)), tmp8, None)
    tl.store(out_ptr3 + (x6 + (81536*x5)), tmp8, None)
    tl.store(out_ptr4 + (x6 + (87808*x5)), tmp8, None)
    tl.store(out_ptr5 + (x6 + (94080*x5)), tmp8, None)
    tl.store(out_ptr6 + (x6 + (100352*x5)), tmp8, None)
    tl.store(out_ptr7 + (x6 + (106624*x5)), tmp8, None)
    tl.store(out_ptr8 + (x6 + (112896*x5)), tmp8, None)
    tl.store(out_ptr9 + (x6 + (119168*x5)), tmp8, None)
    tl.store(out_ptr10 + (x6 + (125440*x5)), tmp8, None)
    tl.store(out_ptr11 + (x6 + (131712*x5)), tmp8, None)
    tl.store(out_ptr12 + (x6 + (137984*x5)), tmp8, None)
    tl.store(out_ptr13 + (x6 + (144256*x5)), tmp8, None)
    tl.store(out_ptr14 + (x6 + (150528*x5)), tmp8, None)
    tl.store(out_ptr15 + (x6 + (156800*x5)), tmp8, None)
    tl.store(out_ptr16 + (x6 + (163072*x5)), tmp8, None)
    tl.store(out_ptr17 + (x6 + (169344*x5)), tmp8, None)
    tl.store(out_ptr18 + (x6 + (175616*x5)), tmp8, None)
    tl.store(out_ptr19 + (x6 + (181888*x5)), tmp8, None)
    tl.store(out_ptr20 + (x6 + (188160*x5)), tmp8, None)
    tl.store(out_ptr21 + (x6 + (194432*x5)), tmp8, None)
    tl.store(out_ptr22 + (x6 + (200704*x5)), tmp8, None)
