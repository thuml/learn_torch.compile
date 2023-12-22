
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632) % 16
    x3 = (xindex // 602112)
    x4 = (xindex // 37632)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (48*x2) + (2304*x1) + (1806336*x3)), None)
    tmp1 = tl.load(in_ptr1 + (768 + x0 + (48*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (48*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sqrt(tmp3)
    tmp5 = 1e-12
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x5), tmp7, None)
