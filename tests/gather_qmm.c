#include "mlx/c/mlx.h"
#include <stdint.h>
#include <stdio.h>

int main(void) {
  float x_data[128];
  uint32_t weights_data[256] = {0};
  float scale_data[32];
  float bias_data[32];
  int32_t ids_data[2] = {0, 1};
  for (int i = 0; i < 128; ++i) x_data[i] = 1.0f;
  for (int i = 0; i < 32; ++i) {
    scale_data[i] = 1.0f;
    bias_data[i] = i < 16 ? 1.0f : 3.0f;
  }
  const int xs[3] = {2, 1, 64};
  const int ws[3] = {2, 16, 8};
  const int ss[3] = {2, 16, 1};
  const int is[1] = {2};
  mlx_array x = mlx_array_new_data(x_data, xs, 3, MLX_FLOAT32);
  mlx_array w = mlx_array_new_data(weights_data, ws, 3, MLX_UINT32);
  mlx_array scales = mlx_array_new_data(scale_data, ss, 3, MLX_FLOAT32);
  mlx_array biases = mlx_array_new_data(bias_data, ss, 3, MLX_FLOAT32);
  mlx_array ids = mlx_array_new_data(ids_data, is, 1, MLX_INT32);
  mlx_array absent = {0};
  mlx_array out = mlx_array_new();
  mlx_stream stream = mlx_default_cpu_stream_new();
  const mlx_optional_int group = {64, true};
  const mlx_optional_int bits = {4, true};
  int code = 0;
  for (int sorted = 0; sorted < 2; ++sorted) {
    if (mlx_gather_qmm(&out, x, w, scales, biases, absent, ids,
                       true, group, bits, "affine", sorted != 0, stream) ||
        mlx_array_eval(out) || mlx_array_size(out) != 32) {
      code = 1;
      break;
    }
    const float *data = mlx_array_data_float32(out);
    for (int i = 0; i < 32; ++i) {
      const float expected = i < 16 ? 64.0f : 192.0f;
      if (data[i] != expected) code = 2;
    }
  }
  mlx_array_free(out);
  mlx_array_free(ids);
  mlx_array_free(biases);
  mlx_array_free(scales);
  mlx_array_free(w);
  mlx_array_free(x);
  mlx_stream_free(stream);
  printf("gather_qmm sorted/unsorted independent scalar oracle: %s\n", code ? "FAIL" : "PASS");
  return code;
}
