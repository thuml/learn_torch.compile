
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6156)
    x0 = xindex % 513
    x1 = (xindex // 513) % 12
    x4 = xindex
    tmp9 = tl.load(in_ptr1 + (x0 + (513*(x2 % 256)) + (131328*x1)), None)
    tmp28 = tl.load(in_ptr3 + (x0 + (513*x2) + (525312*x1)), None)
    tmp0 = x2
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (513*x2) + (131328*x1)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (x2 // 256)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp6 >= tmp10
    tmp12 = x0
    tmp13 = tmp12 < tmp1
    tmp14 = tmp13 & tmp11
    tmp15 = (((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 512) % 513
    tmp16 = tl.full([1], 512, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = tl.load(in_ptr2 + ((512*((((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*(x2 % 256)) + (262656*(x2 // 256)) + (787968*x1)) // 262656) % 36)) + ((x0 + (513*(x2 % 256))) % 512)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.load(in_ptr3 + (x0 + (513*x2) + (525312*x1)), tmp11, other=0.0)
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp9, tmp29)
    tmp31 = tl.where(tmp2, tmp5, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
