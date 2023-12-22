
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1], 16, tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp1
    tmp7 = tl.abs(tmp0)
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp7.to(tl.float32)
    tmp11 = 8.0
    tmp12 = tmp10 / tmp11
    tmp13 = tl.log(tmp12)
    tmp14 = 2.772588722239781
    tmp15 = tmp13 / tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17 + tmp8
    tmp19 = tl.full([1], 15, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp7, tmp20)
    tmp22 = tmp6 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
