
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_div_mul_sigmoid_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), None)
    tmp21 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (x1 // 2))
    tmp4 = tl.math.min(14, 1 + (x1 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (x0 // 2))
    tmp7 = tl.math.min(14, 1 + (x0 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = 0.2
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp20 * tmp22
    tmp25 = 784.0
    tmp26 = tmp24 / tmp25
    tmp27 = tmp23 + tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
