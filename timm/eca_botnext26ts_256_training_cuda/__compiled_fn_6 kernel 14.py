
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_mul_native_batch_norm_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) >= 0, 0, 8))), None, eviction_policy='evict_last')
    tmp1 = tmp0 / 4
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(8, 1 + (x1 // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(8, 1 + (x0 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, None)
