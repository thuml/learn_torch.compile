
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_146', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 35
    x1 = (xindex // 35) % 35
    x2 = (xindex // 1225)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp25 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp42 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp59 = tl.load(in_ptr1 + (x4), xmask)
    tmp61 = tl.load(in_ptr2 + (x4), xmask)
    tmp63 = tl.load(in_ptr3 + (x4), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(35, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(35, 2 + x0)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (-1) + x0))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.math.max(0, (-1) + x0))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.math.max(0, (-1) + x1))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.math.max(0, (-1) + x1))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tmp64 = tmp62 + tmp63
    tl.store(in_out_ptr0 + (x4), tmp64, xmask)
