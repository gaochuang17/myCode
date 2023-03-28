#include "stdafx.h"
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "mt19937ar.h"

#ifndef MIN
#define MIN(A,B)	(((A) <= (B)) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A,B)	(((A) >= (B)) ? (A) : (B))
#endif

#define uchar unsigned char

// 基本参数----------------------------------------------------------------------------------------------/
typedef struct _Sample
{
	double* data;
	double* label;

	int sample_w;
	int sample_h;
	int sample_count;
} Sample;

typedef struct _Kernel
{
	double* W;
	double* dW;
} Kernel;

typedef struct _Map
{
	double* data;
	double* error;
	double  b;
	double  db;
} Map;

typedef struct _Layer
{
	int map_w;
	int map_h;
	int map_count;
	Map* map;

	int kernel_w;
	int kernel_h;
	int kernel_count;
	Kernel* kernel;

	double* map_common;
} Layer;

const int batch_size = 10;
const int classes_count = 10;
const int width = 32;
const int height = 32;
const int train_sample_count = 60000;
const int test_sample_count = 10000;

Layer input_layer, output_layer;
Layer c1_conv_layer, c3_conv_layer, c5_conv_layer;
Layer s2_pooling_layer, s4_pooling_layer;

//*-------------------------------------------------------------------------------------------------------/

// 初始化------------------------------------------------------------------------------------------------/
void init_kernel(double* kernel, int size, double weight_base)
{
	for (int i = 0; i < size; i++)
	{
		kernel[i] = (genrand_real1() - 0.5) * 2 * weight_base;//随机数生成器
	}
}

void init_layer(Layer* layer, int prevlayer_map_count, int map_count, int kernel_w, int kernel_h, int map_w, int map_h, bool is_pooling)
{
	int mem_size = 0;

	const double scale = 6.0;
	int fan_in = 0;
	int fan_out = 0;
	if (is_pooling)
	{
		fan_in = 4;
		fan_out = 1;
	}
	else
	{
		fan_in = prevlayer_map_count * kernel_w * kernel_h;
		fan_out = map_count * kernel_w * kernel_h;
	}
	int denominator = fan_in + fan_out;
	double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;

	layer->kernel_count = prevlayer_map_count * map_count;
	layer->kernel_w = kernel_w;
	layer->kernel_h = kernel_h;
	layer->kernel = (Kernel*)malloc(layer->kernel_count * sizeof(Kernel));
	mem_size = layer->kernel_w * layer->kernel_h * sizeof(double);
	for (int i = 0; i < prevlayer_map_count; i++)
	{
		for (int j = 0; j < map_count; j++)
		{
			layer->kernel[i * map_count + j].W = (double*)malloc(mem_size);
			init_kernel(layer->kernel[i * map_count + j].W, layer->kernel_w * layer->kernel_h, weight_base);
			layer->kernel[i * map_count + j].dW = (double*)malloc(mem_size);
			memset(layer->kernel[i * map_count + j].dW, 0, mem_size);
		}
	}

	layer->map_count = map_count;
	layer->map_w = map_w;
	layer->map_h = map_h;
	layer->map = (Map*)malloc(layer->map_count * sizeof(Map));
	mem_size = layer->map_w * layer->map_h * sizeof(double);
	for (int i = 0; i < layer->map_count; i++)
	{
		layer->map[i].b = 0.0;
		layer->map[i].db = 0.0;
		layer->map[i].data = (double*)malloc(mem_size);
		layer->map[i].error = (double*)malloc(mem_size);
		memset(layer->map[i].data, 0, mem_size);
		memset(layer->map[i].error, 0, mem_size);
	}
	layer->map_common = (double*)malloc(mem_size);
	memset(layer->map_common, 0, mem_size);
}

void release_layer(Layer* layer)
{
	for (int i = 0; i < layer->kernel_count; i++)
	{
		free(layer->kernel[i].W);
		free(layer->kernel[i].dW);
		layer->kernel[i].W = NULL;
		layer->kernel[i].dW = NULL;
	}
	free(layer->kernel);
	layer->kernel = NULL;

	for (int i = 0; i < layer->map_count; i++)
	{
		free(layer->map[i].data);
		free(layer->map[i].error);
		layer->map[i].data = NULL;
		layer->map[i].error = NULL;
	}
	free(layer->map_common);
	layer->map_common = NULL;
	free(layer->map);
	layer->map = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

// 读取数据----------------------------------------------------------------------------------------------/
// 高低位转换
int SwapEndien_32(int i)
{
	return ((i & 0x000000FF) << 24) | ((i & 0x0000FF00) << 8) | ((i & 0x00FF0000) >> 8) | ((i & 0xFF000000) >> 24);
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// 读训练/测试集数据
void read_mnist_data(Sample* sample, const char* file_name)
{
	FILE* fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;
	int n_rows = 0, n_cols = 0, padding = 2;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);
	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);
	fread((char*)&n_rows, sizeof(n_rows), 1, fp);
	n_rows = SwapEndien_32(n_rows);
	fread((char*)&n_cols, sizeof(n_cols), 1, fp);
	n_cols = SwapEndien_32(n_cols);

	double scale_max = 1.0;
	double scale_min = -1.0;
	unsigned char temp = 0;
	int size = width * height;
	int mem_size = size * sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].data = (double*)malloc(mem_size);

		for (int i = 0; i < size; i++)
		{
			sample[k].data[i] = scale_min;
		}

		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				fread((char*)&temp, sizeof(temp), 1, fp);
				sample[k].data[(i + padding) * width + j + padding] = ((double)temp / 255.0) * (scale_max - scale_min) + scale_min;
			}
		}
	}

	fclose(fp);
	fp = NULL;
}

// 读训练/测试集标签
void read_mnist_label(Sample* sample, const char* file_name)
{
	FILE* fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);

	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);

	uchar label = 0;
	int mem_size = classes_count * sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].label = (double*)malloc(mem_size);
		for (int i = 0; i < classes_count; i++)
		{
			sample[k].label[i] = -0.8;
		}

		fread((char*)&label, sizeof(label), 1, fp);
		sample[k].label[label] = 0.8;
	}

	fclose(fp);
	fp = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

// 损失函数----------------------------------------------------------------------------------------------/
struct loss_func
{
	inline static double mse(double y, double t)
	{
		return (y - t) * (y - t) / 2;
	}

	inline static double dmse(double y, double t)
	{
		return y - t;
	}
};
//*-------------------------------------------------------------------------------------------------------/

// 激活函数----------------------------------------------------------------------------------------------/
struct activation_func
{
	/* scale: -0.8 ~ 0.8 和label初始值对应 */
	inline static double tan_h(double val)
	{
		double ep = exp(val);
		double em = exp(-val);

		return (ep - em) / (ep + em);
	}

	inline static double dtan_h(double val)
	{
		return 1.0 - val * val;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline static double relu(double val)
	{
		return val > 0.0 ? val : 0.0;
	}

	inline static double drelu(double val)
	{
		return val > 0.0 ? 1.0 : 0.0;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline double sigmoid(double val)
	{
		return 1.0 / (1.0 + exp(-val));
	}

	double dsigmoid(double val)
	{
		return val * (1.0 - val);
	}
};
//*-------------------------------------------------------------------------------------------------------/

// 克罗内克积--------------------------------------------------------------------------------------------/
void kronecker(double* in_data, int in_map_w, int in_map_h, double* out_data, int out_map_w)
{
	for (int i = 0; i < in_map_h; i++)
	{
		for (int j = 0; j < in_map_w; j++)
		{
			for (int n = 2 * i; n < 2 * (i + 1); n++)
			{
				for (int m = 2 * j; m < 2 * (j + 1); m++)
				{
					out_data[n * out_map_w + m] = in_data[i * in_map_w + j];
				}
			}
		}
	}
}
//*-------------------------------------------------------------------------------------------------------/

// 卷积--------------------------------------------------------------------------------------------------/
void convn_valid(double* in_data, int in_w, int in_h, double* kernel, int kernel_w, int kernel_h, double* out_data, int out_w, int out_h)
{
	double sum = 0.0;
	for (int i = 0; i < out_h; i++)
	{
		for (int j = 0; j < out_w; j++)
		{
			sum = 0.0;
			for (int n = 0; n < kernel_h; n++)
			{
				for (int m = 0; m < kernel_w; m++)
				{
					sum += in_data[(i + n) * in_w + j + m] * kernel[n * kernel_w + m];
				}
			}
			out_data[i * out_w + j] += sum;
		}
	}
}
//*-------------------------------------------------------------------------------------------------------/

// 正向传播----------------------------------------------------------------------------------------------/
#define O true
#define X false
bool connection_table[6 * 16] =
{
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void conv_fprop(Layer* prev_layer, Layer* layer, bool* pconnection)
{
	int index = 0;
	int size = layer->map_w * layer->map_h;
	for (int i = 0; i < layer->map_count; i++)
	{
		memset(layer->map_common, 0, size * sizeof(double));
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			index = j * layer->map_count + i;
			if (pconnection != NULL && !pconnection[index])
			{
				continue;
			}

			convn_valid(
				prev_layer->map[j].data, prev_layer->map_w, prev_layer->map_h,
				layer->kernel[index].W, layer->kernel_w, layer->kernel_h,
				layer->map_common, layer->map_w, layer->map_h);
		}

		for (int k = 0; k < size; k++)
		{
			layer->map[i].data[k] = activation_func::tan_h(layer->map_common[k] + layer->map[i].b);
		}
	}
}

void avg_pooling_fprop(Layer* prev_layer, Layer* layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;
	const double scale_factor = 0.25;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				double sum = 0.0;
				for (int n = 2 * i; n < 2 * (i + 1); n++)
				{
					for (int m = 2 * j; m < 2 * (j + 1); m++)
					{
						sum += prev_layer->map[k].data[n * upmap_w + m] * layer->kernel[k].W[0];
					}
				}

				sum *= scale_factor;
				sum += layer->map[k].b;
				layer->map[k].data[i * map_w + j] = activation_func::tan_h(sum);
			}
		}
	}
}

void max_pooling_fprop(Layer* prev_layer, Layer* layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				double max_value = prev_layer->map[k].data[2 * i * upmap_w + 2 * j];
				for (int n = 2 * i; n < 2 * (i + 1); n++)
				{
					for (int m = 2 * j; m < 2 * (j + 1); m++)
					{
						max_value = MAX(max_value, prev_layer->map[k].data[n * upmap_w + m]);
					}
				}

				layer->map[k].data[i * map_w + j] = activation_func::tan_h(max_value);
			}
		}
	}
}

void fully_connected_fprop(Layer* prev_layer, Layer* layer)
{
	for (int i = 0; i < layer->map_count; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			sum += prev_layer->map[j].data[0] * layer->kernel[j * layer->map_count + i].W[0];
		}

		sum += layer->map[i].b;
		layer->map[i].data[0] = activation_func::tan_h(sum);
	}
}

void forward_propagation()
{
	// In-->C1
	conv_fprop(&input_layer, &c1_conv_layer, NULL);

	// C1-->S2
	max_pooling_fprop(&c1_conv_layer, &s2_pooling_layer);/*avg_pooling_fprop*/

	// S2-->C3
	conv_fprop(&s2_pooling_layer, &c3_conv_layer, connection_table);

	// C3-->S4
	max_pooling_fprop(&c3_conv_layer, &s4_pooling_layer);/*avg_pooling_fprop*/

	// S4-->C5
	conv_fprop(&s4_pooling_layer, &c5_conv_layer, NULL);

	// C5-->Out
	fully_connected_fprop(&c5_conv_layer, &output_layer);
}
//*-------------------------------------------------------------------------------------------------------/