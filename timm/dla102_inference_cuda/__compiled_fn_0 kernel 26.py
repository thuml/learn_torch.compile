
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + (1152*x3)), tmp6, None)
