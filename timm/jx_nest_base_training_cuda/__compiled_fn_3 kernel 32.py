
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3444736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 14848) % 29
    x1 = (xindex // 512) % 29
    x3 = (xindex // 430592)
    x4 = xindex % 14848
    x0 = xindex % 512
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (401408*x3)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, float("-inf"), tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, None)
