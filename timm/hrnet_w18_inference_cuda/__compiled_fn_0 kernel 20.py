
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = x5
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp21
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.int32)
    tmp27 = x4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp21
    tmp31 = tmp30 + tmp21
    tmp32 = tmp31 * tmp24
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.load(in_ptr5 + (tmp33 + (28*tmp26) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp8
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp17 + tmp46
    tmp48 = 0.25
    tmp49 = tmp23 * tmp48
    tmp50 = tmp49.to(tl.int32)
    tmp51 = tmp31 * tmp48
    tmp52 = tmp51.to(tl.int32)
    tmp53 = tl.load(in_ptr10 + (tmp52 + (14*tmp50) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp55 = tmp53 - tmp54
    tmp57 = tmp56 + tmp4
    tmp58 = tl.sqrt(tmp57)
    tmp59 = 1 / tmp58
    tmp60 = tmp59 * tmp8
    tmp61 = tmp55 * tmp60
    tmp63 = tmp61 * tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp47 + tmp65
    tmp67 = 0.125
    tmp68 = tmp23 * tmp67
    tmp69 = tmp68.to(tl.int32)
    tmp70 = tmp31 * tmp67
    tmp71 = tmp70.to(tl.int32)
    tmp72 = tl.load(in_ptr15 + (tmp71 + (7*tmp69) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp74 = tmp72 - tmp73
    tmp76 = tmp75 + tmp4
    tmp77 = tl.sqrt(tmp76)
    tmp78 = 1 / tmp77
    tmp79 = tmp78 * tmp8
    tmp80 = tmp74 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp66 + tmp84
    tmp86 = triton_helpers.maximum(0, tmp85)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
    tl.store(in_out_ptr1 + (x3), tmp86, xmask)
