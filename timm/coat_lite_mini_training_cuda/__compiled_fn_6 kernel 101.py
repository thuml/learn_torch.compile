
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 128) % 784
    x3 = (xindex // 100352)
    x5 = xindex % 100352
    x4 = xindex % 128
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    tmp6 = tl.load(in_ptr1 + (x4 + (384*x2) + (301440*x3)), None)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (128 + x5 + (100480*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0 + (16*x2) + (12544*x1) + (100352*x3)), tmp7, None)
