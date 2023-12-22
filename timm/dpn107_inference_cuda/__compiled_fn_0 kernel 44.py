
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3311616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2112
    x2 = (xindex // 413952)
    x3 = xindex % 413952
    x4 = xindex
    tmp57 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (376320*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 2112, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tmp19 & tmp14
    tmp21 = tl.full([1], 960, tl.int64)
    tmp22 = tmp17 < tmp21
    tmp23 = tmp22 & tmp20
    tmp24 = tl.full([1], 896, tl.int64)
    tmp25 = tmp17 < tmp24
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr4 + ((-200704) + x3 + (376320*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp17 >= tmp24
    tmp31 = tmp30 & tmp23
    tmp32 = tl.load(in_ptr1 + ((-175616) + x3 + (213248*x2)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.where(tmp25, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp23, tmp35, tmp36)
    tmp38 = tmp17 >= tmp21
    tmp39 = tmp38 & tmp20
    tmp40 = tl.load(in_ptr2 + ((-188160) + x3 + (213248*x2)), tmp39, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp22, tmp37, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp20, tmp43, tmp44)
    tmp46 = tmp17 >= tmp3
    tmp47 = tl.full([1], 1088, tl.int64)
    tmp48 = tmp17 < tmp47
    tmp49 = tmp46 & tmp14
    tmp50 = tl.load(in_ptr3 + ((-200704) + x3 + (213248*x2)), tmp49, other=0.0)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = tl.where(tmp19, tmp45, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp14, tmp53, tmp54)
    tmp56 = tl.where(tmp4, tmp13, tmp55)
    tmp58 = tmp56 - tmp57
    tmp60 = 0.001
    tmp61 = tmp59 + tmp60
    tmp62 = tl.sqrt(tmp61)
    tmp63 = 1 / tmp62
    tmp64 = 1.0
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 * tmp65
    tmp68 = tmp66 * tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = triton_helpers.maximum(0, tmp70)
    tl.store(in_out_ptr0 + (x4), tmp71, None)
