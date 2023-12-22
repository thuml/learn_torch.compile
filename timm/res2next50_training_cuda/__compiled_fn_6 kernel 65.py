
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136) % 64
    x3 = (xindex // 200704)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp18 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(28, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(28, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x6), tmp29, None)
