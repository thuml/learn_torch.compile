
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    x3 = xindex % 3136
    x5 = xindex
    x6 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (((-1) + x1) // 2))
    tmp10 = tl.math.min(28, 1 + (x1 // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (((-1) + x0) // 2)))
    tmp13 = tl.math.min(28, 1 + (x0 // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (((-1) + x1) // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (((-1) + x0) // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tmp38 = tmp37 <= tmp4
    tmp39 = tl.where(tmp38, tmp4, tmp36)
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp39 * tmp45
    tl.store(out_ptr0 + (x5), tmp36, None)
    tl.store(out_ptr1 + (x5), tmp46, None)
