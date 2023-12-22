
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 224) % 224
    x0 = xindex % 224
    x2 = (xindex // 50176)
    x4 = xindex
    tmp25 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr18 + (x1), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr20 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 239, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp4
    tmp17 = tmp16 - tmp2
    tmp18 = tl.math.floor(tmp17)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp19 - tmp20
    tmp22 = triton_helpers.maximum(tmp21, tmp9)
    tmp23 = triton_helpers.minimum(tmp22, tmp11)
    tmp24 = tl.load(in_ptr0 + (tmp23 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp24 * tmp25
    tmp27 = triton_helpers.maximum(tmp19, tmp9)
    tmp28 = triton_helpers.minimum(tmp27, tmp11)
    tmp29 = tl.load(in_ptr0 + (tmp28 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp31 = tmp29 * tmp30
    tmp32 = tmp26 + tmp31
    tmp33 = tmp19 + tmp20
    tmp34 = triton_helpers.maximum(tmp33, tmp9)
    tmp35 = triton_helpers.minimum(tmp34, tmp11)
    tmp36 = tl.load(in_ptr0 + (tmp35 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 + tmp38
    tmp40 = tl.full([1], 2, tl.int64)
    tmp41 = tmp19 + tmp40
    tmp42 = triton_helpers.maximum(tmp41, tmp9)
    tmp43 = triton_helpers.minimum(tmp42, tmp11)
    tmp44 = tl.load(in_ptr0 + (tmp43 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp46 = tmp44 * tmp45
    tmp47 = tmp39 + tmp46
    tmp48 = tmp8 - tmp20
    tmp49 = triton_helpers.maximum(tmp48, tmp9)
    tmp50 = triton_helpers.minimum(tmp49, tmp11)
    tmp51 = tl.load(in_ptr0 + (tmp23 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp53 = tmp51 * tmp52
    tmp54 = tl.load(in_ptr0 + (tmp28 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp56 = tmp54 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = tmp8 + tmp20
    tmp59 = triton_helpers.maximum(tmp58, tmp9)
    tmp60 = triton_helpers.minimum(tmp59, tmp11)
    tmp61 = tl.load(in_ptr0 + (tmp23 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp63 = tmp61 * tmp62
    tmp64 = tl.load(in_ptr0 + (tmp28 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp66 = tmp64 * tmp65
    tmp67 = tmp63 + tmp66
    tmp68 = tmp8 + tmp40
    tmp69 = triton_helpers.maximum(tmp68, tmp9)
    tmp70 = triton_helpers.minimum(tmp69, tmp11)
    tmp71 = tl.load(in_ptr0 + (tmp23 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp73 = tmp71 * tmp72
    tmp74 = tl.load(in_ptr0 + (tmp28 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp76 = tmp74 * tmp75
    tmp77 = tmp73 + tmp76
    tmp78 = tl.load(in_ptr0 + (tmp35 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp80 = tmp78 * tmp79
    tmp81 = tmp57 + tmp80
    tmp82 = tl.load(in_ptr0 + (tmp43 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp84 = tmp82 * tmp83
    tmp85 = tmp81 + tmp84
    tmp86 = tl.load(in_ptr0 + (tmp35 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp86 * tmp87
    tmp89 = tmp67 + tmp88
    tmp90 = tl.load(in_ptr0 + (tmp43 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp92 = tmp90 * tmp91
    tmp93 = tmp89 + tmp92
    tmp94 = tl.load(in_ptr0 + (tmp35 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp96 = tmp94 * tmp95
    tmp97 = tmp77 + tmp96
    tmp98 = tl.load(in_ptr0 + (tmp43 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp100 = tmp98 * tmp99
    tmp101 = tmp97 + tmp100
    tmp103 = tmp85 * tmp102
    tmp105 = tmp47 * tmp104
    tmp106 = tmp103 + tmp105
    tmp108 = tmp93 * tmp107
    tmp109 = tmp106 + tmp108
    tmp111 = tmp101 * tmp110
    tmp112 = tmp109 + tmp111
    tl.store(in_out_ptr1 + (x4), tmp112, None)
