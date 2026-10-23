extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* mul_c1, const float* sub_c1, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 750) {
        int a0 = _gid;
        float in2__u0 = x[a0 * 4];
        float in0__u0 = mul_c1[0];
        float in1__u0 = sub_c1[0];
        float v2__u0 = (in0__u0 * tanhf(in2__u0)) - in1__u0;
        float in2__u1 = x[a0 * 4 + 1];
        float in0__u1 = mul_c1[0];
        float in1__u1 = sub_c1[0];
        float v2__u1 = (in0__u1 * tanhf(in2__u1)) - in1__u1;
        float in2__u2 = x[a0 * 4 + 2];
        float in0__u2 = mul_c1[0];
        float in1__u2 = sub_c1[0];
        float v2__u2 = (in0__u2 * tanhf(in2__u2)) - in1__u2;
        float in2__u3 = x[a0 * 4 + 3];
        float in0__u3 = mul_c1[0];
        float in1__u3 = sub_c1[0];
        float v2__u3 = (in0__u3 * tanhf(in2__u3)) - in1__u3;
        float4 _vs_v2__u0_05a843af = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&sub[a0 * 4]) = _vs_v2__u0_05a843af;
    }
}
