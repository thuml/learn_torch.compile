
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex
    x6 = (xindex // 1764) % 108
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp76 = tl.load(in_ptr1 + (x4), xmask)
    tmp77 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + (2*x0) + (166*x1) + (6889*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-83) + (2*x0) + (166*x1) + (6889*x2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-82) + (2*x0) + (166*x1) + (6889*x2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x0) + (166*x1) + (6889*x2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x0) + (166*x1) + (6889*x2)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x0) + (166*x1) + (6889*x2)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x1)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (82 + (2*x0) + (166*x1) + (6889*x2)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (83 + (2*x0) + (166*x1) + (6889*x2)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (84 + (2*x0) + (166*x1) + (6889*x2)), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp78 = tmp76 - tmp77
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = tl.sqrt(tmp81)
    tmp83 = 1 / tmp82
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp75
    tl.store(out_ptr1 + (x8 + (952560*x7)), tmp91, xmask)
