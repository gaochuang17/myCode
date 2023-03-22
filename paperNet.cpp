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