
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x4 = (xindex // 28)
    x2 = (xindex // 784) % 128
    x5 = xindex
    x3 = (xindex // 100352)
    x7 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr1 + (x7 + (200704*x3)), tmp8, None)
    tl.store(out_ptr2 + (x7 + (225792*x3)), tmp8, None)
    tl.store(out_ptr3 + (x7 + (250880*x3)), tmp8, None)
    tl.store(out_ptr4 + (x7 + (275968*x3)), tmp8, None)
    tl.store(out_ptr5 + (x7 + (301056*x3)), tmp8, None)
    tl.store(out_ptr6 + (x7 + (326144*x3)), tmp8, None)
    tl.store(out_ptr7 + (x7 + (351232*x3)), tmp8, None)
    tl.store(out_ptr8 + (x7 + (376320*x3)), tmp8, None)
    tl.store(out_ptr9 + (x7 + (401408*x3)), tmp8, None)
