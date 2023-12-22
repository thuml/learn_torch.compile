
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_4', 'mutated_arg_names': ['in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
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
    tmp90 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp52 = tmp51 * tmp25
    tmp53 = tl.load(in_ptr0 + (tmp28 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp54 = tmp53 * tmp30
    tmp55 = tmp52 + tmp54
    tmp56 = tmp8 + tmp20
    tmp57 = triton_helpers.maximum(tmp56, tmp9)
    tmp58 = triton_helpers.minimum(tmp57, tmp11)
    tmp59 = tl.load(in_ptr0 + (tmp23 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp60 = tmp59 * tmp25
    tmp61 = tl.load(in_ptr0 + (tmp28 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp62 = tmp61 * tmp30
    tmp63 = tmp60 + tmp62
    tmp64 = tmp8 + tmp40
    tmp65 = triton_helpers.maximum(tmp64, tmp9)
    tmp66 = triton_helpers.minimum(tmp65, tmp11)
    tmp67 = tl.load(in_ptr0 + (tmp23 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp68 = tmp67 * tmp25
    tmp69 = tl.load(in_ptr0 + (tmp28 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp70 = tmp69 * tmp30
    tmp71 = tmp68 + tmp70
    tmp72 = tl.load(in_ptr0 + (tmp35 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp73 = tmp72 * tmp37
    tmp74 = tmp55 + tmp73
    tmp75 = tl.load(in_ptr0 + (tmp43 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp76 = tmp75 * tmp45
    tmp77 = tmp74 + tmp76
    tmp78 = tl.load(in_ptr0 + (tmp35 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp79 = tmp78 * tmp37
    tmp80 = tmp63 + tmp79
    tmp81 = tl.load(in_ptr0 + (tmp43 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp45
    tmp83 = tmp80 + tmp82
    tmp84 = tl.load(in_ptr0 + (tmp35 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp85 = tmp84 * tmp37
    tmp86 = tmp71 + tmp85
    tmp87 = tl.load(in_ptr0 + (tmp43 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp87 * tmp45
    tmp89 = tmp86 + tmp88
    tmp91 = tmp77 * tmp90
    tmp93 = tmp47 * tmp92
    tmp94 = tmp91 + tmp93
    tmp96 = tmp83 * tmp95
    tmp97 = tmp94 + tmp96
    tmp99 = tmp89 * tmp98
    tmp100 = tmp97 + tmp99
    tl.store(in_out_ptr1 + (x4), tmp100, None)
