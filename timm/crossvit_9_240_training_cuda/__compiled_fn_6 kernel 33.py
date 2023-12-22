
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 197
    x3 = xindex
    x4 = (xindex // 256)
    x0 = xindex % 256
    x2 = (xindex // 50432)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_out_ptr0 + (x3), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tl.load(in_ptr0 + (x4), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = 256.0
    tmp10 = tmp8 / tmp9
    tmp11 = tl.load(in_ptr1 + (x3), tmp2, other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 < tmp1
    tmp14 = tmp13 & tmp2
    tmp15 = tl.load(in_ptr2 + (x0 + (256*x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp13, tmp17, tmp6)
    tmp19 = tmp12 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.where(tmp2, tmp21, tmp6)
    tmp23 = tmp7 + tmp22
    tmp24 = tl.load(in_ptr3 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr4 + (x0 + (256*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr5 + (x0 + (256*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr6 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr7 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = 0.7071067811865476
    tmp32 = tmp30 * tmp31
    tmp33 = tl.math.erf(tmp32)
    tmp34 = 1.0
    tmp35 = tmp33 + tmp34
    tmp36 = 0.5
    tmp37 = tmp35 * tmp36
    tmp38 = tmp30 * tmp30
    tmp39 = -0.5
    tmp40 = tmp38 * tmp39
    tmp41 = tl.exp(tmp40)
    tmp42 = 0.3989422804014327
    tmp43 = tmp41 * tmp42
    tmp44 = tmp30 * tmp43
    tmp45 = tmp37 + tmp44
    tmp46 = tmp25 * tmp45
    tmp47 = tmp46 * tmp27
    tmp48 = tmp47 * tmp9
    tmp49 = tl.load(in_ptr8 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 - tmp49
    tmp51 = tl.load(in_ptr9 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp26 * tmp51
    tmp53 = tmp50 - tmp52
    tmp54 = tmp24 * tmp53
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp13, tmp54, tmp55)
    tmp57 = tl.where(tmp13, tmp56, tmp6)
    tmp58 = tmp23 + tmp57
    tl.store(in_out_ptr0 + (x3), tmp58, None)
