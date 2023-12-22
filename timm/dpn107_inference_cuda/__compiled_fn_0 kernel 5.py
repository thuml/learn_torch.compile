
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 336
    x2 = (xindex // 1053696)
    x3 = xindex % 1053696
    x4 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 336, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-256) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 40, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-125440) + x3 + (865536*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 80, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-188160) + x3 + (865536*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
