
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_constant_pad_nd_max_pool2d_with_indices_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21) % 21
    x0 = xindex % 21
    x3 = (xindex // 21)
    x4 = xindex
    x6 = (xindex // 441) % 432
    x7 = (xindex // 190512)
    x8 = xindex % 190512
    tmp57 = tl.load(in_ptr1 + (x4), xmask)
    tmp58 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp0 = 2*x1
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((2*x0) + (84*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, float("-inf"), tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1 + (2*x0)
    tmp10 = tmp9 < tmp1
    tmp11 = tmp2 & tmp10
    tmp12 = tl.load(in_ptr0 + (1 + (2*x0) + (84*x3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, float("-inf"), tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = 2 + (2*x0)
    tmp17 = tmp16 < tmp1
    tmp18 = tmp2 & tmp17
    tmp19 = tl.load(in_ptr0 + (2 + (2*x0) + (84*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp15)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp4
    tmp26 = tl.load(in_ptr0 + (42 + (2*x0) + (84*x3)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, float("-inf"), tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp22)
    tmp30 = tmp24 & tmp10
    tmp31 = tl.load(in_ptr0 + (43 + (2*x0) + (84*x3)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, float("-inf"), tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp33, tmp29)
    tmp35 = tmp24 & tmp17
    tmp36 = tl.load(in_ptr0 + (44 + (2*x0) + (84*x3)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full(tmp36.shape, float("-inf"), tmp36.dtype)
    tmp38 = tl.where(tmp35, tmp36, tmp37)
    tmp39 = triton_helpers.maximum(tmp38, tmp34)
    tmp40 = 2 + (2*x1)
    tmp41 = tmp40 < tmp1
    tmp42 = tmp41 & tmp4
    tmp43 = tl.load(in_ptr0 + (84 + (2*x0) + (84*x3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, float("-inf"), tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp41 & tmp10
    tmp48 = tl.load(in_ptr0 + (85 + (2*x0) + (84*x3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, float("-inf"), tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = triton_helpers.maximum(tmp50, tmp46)
    tmp52 = tmp41 & tmp17
    tmp53 = tl.load(in_ptr0 + (86 + (2*x0) + (84*x3)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, float("-inf"), tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = triton_helpers.maximum(tmp55, tmp51)
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp71 + tmp56
    tl.store(out_ptr1 + (x8 + (952560*x7)), tmp72, xmask)
