
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_select_scatter_zeros_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = ((-1)*(x2 % 14)) + (x1 % 14)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full([1], 1, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = ((-1)*(x2 // 14)) + (x1 // 14)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 2, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = ((x1 // 14)*(x1 // 14)) + ((x2 // 14)*(x2 // 14)) + ((x1 % 14)*(x1 % 14)) + ((x2 % 14)*(x2 % 14)) + ((-2)*(x1 // 14)*(x2 // 14)) + ((-2)*(x1 % 14)*(x2 % 14))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.0
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tl.where(tmp6, tmp8, tmp14)
    tmp16 = tl.where(tmp2, tmp4, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
