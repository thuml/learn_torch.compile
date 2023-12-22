
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_masked_fill_rsub_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = (tmp2 != 0)
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp3, tmp4, tmp2)
    tmp6 = x0
    tmp7 = 1 + x1
    tmp8 = tmp6 < tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp9, tmp4)
    tmp11 = tmp5 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, None)
