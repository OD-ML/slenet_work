#include "slenet_params.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_CLASSES 10

#define INPUT -1.0f // 1.0f //
#define WEIGHT -1.0f // 1.0f //
#define BIAS -1.0f // 1.0f //
#define CONV_POST_ACT 1.0f //  1.0f //
#define SS_POST_ACT  0.0f // 1.0f //
#define FC_POST_ACT  (1/(1+2.71828)) // 1.0f //

#define INSIZE 28
#define FILTER_SIZE 5
#define STRIDE 1
#define CHANNEL 6

#define CONV_OUTPUT_SIZE ((INSIZE - FILTER_SIZE)/STRIDE + 1) //24
#define SS_SIZE 4
#define SS_STRIDE 4
#define SS_CHANNELS 1

#define SS_OUTPUT_SIZE ((CONV_OUTPUT_SIZE - SS_SIZE)/SS_STRIDE + 1) //6

#define N1 CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE
#define K1 64

#define N2 CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE
#define K2 8

#define N3 NUM_CLASSES * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE
#define K3 10

// Defining the mnist_data struct
typedef struct mnist_data {
	double data[INSIZE][INSIZE];
	unsigned int label;
} mnist_data;

// Defining the Layer class
class Layer {
	public:
		int M, N, O;
		float *pre_output, *output;
		float *weight, *bias;
    float *im2col_A; //for im2col

		Layer(int M, int N, int O);
		~Layer();
};

Layer::Layer(int M, int N, int O) {
	this->M = M;
	this->N = N;
	this->O = O;

	float *temp_weight, *temp_bias;

	// Initializing weights and biases
	temp_weight = (float*)malloc(sizeof(float) * M * N);
	temp_bias = (float*)malloc(sizeof(float) * N);

	for (int i = 0; i < M * N; i++)
		temp_weight[i] = WEIGHT; //1.0f;

	for (int i = 0; i < N; i++)
		temp_bias[i] = BIAS; //1.0f;

	// Allocating space for CUDA variables
	cudaMalloc(&pre_output, sizeof(float) * O);
	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&weight, sizeof(float) * M * N);
	cudaMalloc(&bias, sizeof(float) * N);

  cudaMalloc(&im2col_A, sizeof(float) *M*O/N);
  

	// Copying weights and biases to CUDA variables
	cudaMemcpy(weight, temp_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(bias, temp_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Freeing temporary weights and biases
	free(temp_weight);
	free(temp_bias);
}

Layer::~Layer() {
	// Freeing all CUDA varibles of a layer
	cudaFree(pre_output);
	cudaFree(output);
	cudaFree(weight);
	cudaFree(bias);
  cudaFree(im2col_A);
}

// Initializing a convolutional layer
Layer conv_layer(FILTER_SIZE * FILTER_SIZE, CHANNEL, CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
Layer ss_layer(SS_SIZE * SS_SIZE, SS_CHANNELS, CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
Layer fc_layer(CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, NUM_CLASSES, NUM_CLASSES);

double time_taken = 0.0;

static unsigned int mnist_bin_to_int(unsigned char *tmp) {
	// Converting the binary char value to the integer value
	unsigned int result = 0;
	short charSize = 4;
	short multiplier = 256;

	for (int i = 0; i < charSize; i++) {
		unsigned int temp = tmp[i];

		for (int j = 0; j < charSize - i - 1; j++)
			temp *= multiplier;

		result += temp;
	}

	// Returning the integer value
	return result;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count) {
	// Initializing necessary variables
	FILE *images;
	FILE *labels;

	unsigned char *imagesBuffer;
	unsigned char *labelsBuffer;

	long imagesFileSize;
	long labelsFileSize;

	short unsignedIntSize = 4;
	short unsignedByteSize = 1;

	unsigned int imageMagicNumber;
	unsigned int labelMagicNumber;
	unsigned int imageTotalNumber;
	unsigned int labelTotalNumber;
	unsigned int rows, cols;

	// Opening image and label files of the test
	images = fopen("data/t10k-images.idx3-ubyte", "rb");

	if (images == NULL) {
		printf("Error! Images file cannot be read!\n");
		return 1;
	}

	labels = fopen("data/t10k-labels.idx1-ubyte", "rb");

	if (images == NULL) {
		printf("Error! Labels file cannot be read!\n");
		return 1;
	}

	fseek(images, 0, SEEK_END);
	fseek(labels, 0, SEEK_END);

	imagesFileSize = ftell(images);
	labelsFileSize = ftell(labels);

	fseek(images, 0, SEEK_SET);
	fseek(labels, 0, SEEK_SET);

	imagesBuffer = (unsigned char*)malloc(sizeof(unsigned char) * imagesFileSize);

	if (imagesBuffer == NULL) {
		printf("Error! Memory error has occured!\n");
		return 2;
	}

	labelsBuffer = (unsigned char*)malloc(sizeof(unsigned char) * labelsFileSize);

	if (labelsBuffer == NULL) {
		printf("Error! Memory error has occured!\n");
		return 2;
	}

	// Reading a magic number
	fread(imagesBuffer, unsignedIntSize, 1, images);
	fread(labelsBuffer, unsignedIntSize, 1, labels);
	imageMagicNumber = mnist_bin_to_int(imagesBuffer);
	labelMagicNumber = mnist_bin_to_int(labelsBuffer);
	printf("Image magic number: %d\n", imageMagicNumber);
	printf("Label magic number: %d\n", labelMagicNumber);

	// Reading a number of images and label files
	fread(imagesBuffer, unsignedIntSize, 1, images);
	fread(labelsBuffer, unsignedIntSize, 1, labels);
	imageTotalNumber = mnist_bin_to_int(imagesBuffer);
	labelTotalNumber = mnist_bin_to_int(labelsBuffer);
	printf("Number of images: %d\n", imageTotalNumber);
	printf("Number of labels: %d\n", labelTotalNumber);

	// Check whether the number of images and label files is the same
	if (imageTotalNumber != labelTotalNumber) {
		printf("Error! The number of images and the number of labels are different!\n");
		return 3;
	} else {
		printf("The number of images and the number of labels are the same!\n");
	}

	// Check the number of rows and columns
	fread(imagesBuffer, unsignedIntSize, 1, images);
	rows = mnist_bin_to_int(imagesBuffer);
	fread(imagesBuffer, unsignedIntSize, 1, images);
	cols = mnist_bin_to_int(imagesBuffer);
	printf("Rows: %d\n", rows);
	printf("Cols: %d\n", cols);

	*data_set = (mnist_data*)malloc(sizeof(mnist_data) * imageTotalNumber);

	// Load image data as double type
	for (int i = 0; i < imageTotalNumber; i++) {
		fread(imagesBuffer, rows * cols, 1, images);
		fread(labelsBuffer, unsignedByteSize, 1, labels);

		for (int j = 0; j < INSIZE; j++) {
			for (int k = 0; k < INSIZE; k++) {
				(*data_set)[i].data[j][k] = imagesBuffer[j * INSIZE + k] / 255.0;
			}
		}

		(*data_set)[i].label = labelsBuffer[0];
	}

	// Closing opened files
	fclose(images);
	fclose(labels);
	free(imagesBuffer);
	free(labelsBuffer);
	*count = imageTotalNumber;
	return 0;
}

// Printing MNIST data set examples
void printExamples(mnist_data **data_set, int count) {
	for (int i = 0; i < count; i++) {
		printf("\nImage:\n");

		for (int j = 0; j < INSIZE; j++) {
			for (int k = 0; k < INSIZE; k++) {
				if ((*data_set)[i].data[j][k] > 0) {
					printf("1");
				} else {
					printf("0");
				}
			}
			printf("\n");
		}

		printf("Label: %d\n", (*data_set)[i].label);
	}
}

__global__ void kernel_conv_filter(float input[INSIZE][INSIZE], float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float weight[CHANNEL][FILTER_SIZE][FILTER_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
	float tempC = 0.0f;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			tempC += weight[channel][i][j] * input[i + output_x][j + output_y];
		}
	}
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] = tempC;
}


__global__ void kernel_conv_bias(float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float bias[CHANNEL]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] += bias[channel];
}

__global__ void kernel_conv_sigmoid(float preact[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  output[channel][output_x][output_y] = 1 / (1 + exp(-preact[channel][output_x][output_y]));
}

__global__ void kernel_ss1_filter(float input[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float weight[SS_CHANNELS][SS_SIZE][SS_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
	float tempC = 0.0f;

	for (int i = 0; i < SS_SIZE; i++) {
		for (int j = 0; j < SS_SIZE; j++) {
			tempC += weight[0][i][j] * input[channel][i + output_x * SS_STRIDE][j + output_y * SS_STRIDE];
		}
	}
  if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] = tempC;
}

__global__ void kernel_ss1_bias(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float bias[SS_CHANNELS]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
  if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] += bias[0];
}

__global__ void kernel_ss1_sigmoid(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
  if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
	  output[channel][output_x][output_y] = 1 / (1 + exp(-pre_output[channel][output_x][output_y]));
}

__global__ void kernel_fc1(float input[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float pre_output[NUM_CLASSES], 
                            float weight[NUM_CLASSES][CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	float tempC = 0.0f;

	for (int i = 0; i < CHANNEL; i++) {
		for (int j = 0; j < SS_OUTPUT_SIZE; j++) {
			for (int k = 0; k < SS_OUTPUT_SIZE; k++) {
				tempC += weight[channel][i][j][k] * input[i][j][k];
			}
		}
	}

	pre_output[channel] = tempC;
}

__global__ void kernel_fc1_bias(float pre_output[NUM_CLASSES], float bias[NUM_CLASSES]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	pre_output[channel] += bias[channel];
}

__global__ void kernel_fc1_sigmoid(float pre_output[NUM_CLASSES], float output[NUM_CLASSES]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	output[channel] = 1 / (1 + exp(-pre_output[channel]));
}

void verifyConv(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0; 
	for (int i = 0; i < CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; i++){ 
		maxError = max(abs(A[i] - val), maxError);
    if (maxError != 0)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE);
}

void verifySS(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE; i++){
		maxError = max(abs(A[i] - val), maxError);
    if (maxError > 0.0001f)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE);
}

void verifyFC(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < NUM_CLASSES; i++){
		maxError = max(abs(A[i] - val), maxError);
    if (maxError > 0.0009f)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, NUM_CLASSES);
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
__global__ void im2col_gpu_kernel_ext(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        float* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}


// Performing a forward pass using a single image
static double forward_pass(double data[INSIZE][INSIZE], bool verify) {
	// Copying a double data to a float data
	float input[INSIZE][INSIZE];
	float *verification;

	for (int i = 0; i < INSIZE; i++) {
		for (int j = 0; j < INSIZE; j++)
      input[i][j] = INPUT; //Simulated data
			//input[i][j] = data[i][j];  //MNIST data
	}

	float (*d_input)[INSIZE];
	cudaMalloc(&d_input, sizeof(float) * INSIZE * INSIZE);
	cudaMemcpy(d_input, input, sizeof(float) * INSIZE * INSIZE, cudaMemcpyHostToDevice);

  //For im2col workspace 
  //float* im2col_input;  
  //cudaMalloc(&im2col_input, sizeof(float) * INSIZE * INSIZE);

  //float* im2col_workspace;  
  //cudaMalloc(&im2col_workspace, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


  //im2col_gpu_kernel_ext<<<(N1+K1-1)/K1, K1>>>(CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE, d_input, INSIZE, INSIZE, FILTER_SIZE, FILTER_SIZE, 0, 0, STRIDE, STRIDE, 1, 1, CONV_OUTPUT_SIZE, CONV_OUTPUT_SIZE,ic_workspace);
///*
  im2col_gpu_kernel_ext<<<(N1+K1-1)/K1, K1>>>(CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE, //num_kernels, = channels * height_col * width_col; 
                                              (float *)d_input, //data_im, 
                                              INSIZE, //height, 
                                              INSIZE, //width, 
                                              FILTER_SIZE, //kernel_h, 
                                              FILTER_SIZE, //kernel_w, 
                                              0, //pad_h,
                                              0, //pad_w, 
                                              STRIDE, //stride_h, 
                                              STRIDE, //stride_w, 
                                              1, //dilation_h, 
                                              1, //dilation_w, 
                                              CONV_OUTPUT_SIZE, //height_col, 
                                              CONV_OUTPUT_SIZE, //width_col, 
                                              conv_layer.im2col_A); //data_col);
                                      
//*/

	// Verifying im2col operation
	if (verify) {
		printf("Verifying im2col: ");
		verification = (float*)malloc(sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.im2col_A, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT); //-1.0f
		free(verification);
	}

#if 0
  int n = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; //l.out_w*l.out_h
  int k = FILTER_SIZE * FILTER_SIZE; // l.size*l.size
  int m = CHANNEL; // l.n / l.groups

  //For gemm workspace 
  //float *a = (float*)malloc(sizeof(float) * M * N);

  float *a = (float*)malloc(sizeof(float)*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE*FILTER_SIZE*FILTER_SIZE); 
                  //l.weights_gpu + j*l.nweights / l.groups;
  float *b = im2col_workspace; //state.workspace
  float *c = l.output_gpu + (i*l.groups + j)*n*m;

  gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
#endif 


	// Performing Convolutional filtering
	kernel_conv_filter<<<(N1+K1-1)/K1, K1>>>(d_input, 
                                            (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output,
                                            (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight);
  
	// Verifying Convolutional filtering operation
	if (verify) {
		printf("Veri Convolutional filtering");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE); //25.0f
		free(verification);
	}

	// Performing Convolutional bias addition
	kernel_conv_bias<<<(N1+K1-1)/K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, conv_layer.bias);

	// Verifying Convolutional bias operation
	if (verify) {
		printf("Veri Convolutional bias: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE + BIAS); // 26.0f
		free(verification);
	}

	// Performing Convolutional sigmoid operation
	kernel_conv_sigmoid<<<(N1+K1-1)/K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output);

	// Verifying Convolutional sigmoid operation
	if (verify) {
		printf("Veri Convolutional sigmoid: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, CONV_POST_ACT); // 1.0f
		free(verification);
	}

	// Performing Subsampling filtering
	kernel_ss1_filter<<<(N2+K2-1)/K2, K2>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, (float(*)[SS_SIZE][SS_SIZE])ss_layer.weight);

	// Verifying Subsampling filtering operation
	if (verify) {
		printf("Veri Subsampling filtering: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, CONV_POST_ACT*WEIGHT*SS_SIZE*SS_SIZE); //16.0f
		free(verification);
	}

	// Performing Subsampling bias addition
	kernel_ss1_bias<<<(N2+K2-1)/K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, ss_layer.bias);

	// Verifying Subsampling bias operation
	if (verify) {
		printf("Veri Subsampling bias: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, CONV_POST_ACT*WEIGHT*SS_SIZE*SS_SIZE + BIAS); // 17.0f
		free(verification);
	}

	// // Performing Subsampling sigmoid operation
	kernel_ss1_sigmoid<<<(N2+K2-1)/K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output);

	// Verifying Subsampling sigmoid operation
	if (verify) {
		printf("Veri Subsampling sigmoid: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		cudaMemcpy(verification, ss_layer.output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, SS_POST_ACT); //1.0f
		free(verification);
	}

	// Performing Fully-Connected Computation
	kernel_fc1<<<(N3+K3-1)/K3, K3>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output, (float(*))fc_layer.pre_output, (float(*)[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])fc_layer.weight);

	// Verifying Fully-Connected Computation
	if (verify) {
		printf("Veri Fully-Connected: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, SS_POST_ACT*WEIGHT*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE);//216.0f
		free(verification);
	}

	// Performing Fully-Connected bias operation
	kernel_fc1_bias<<<1, K3>>>((float(*))fc_layer.pre_output, fc_layer.bias);

	// Verifying Fully-Connected bias operation
	if (verify) {
		printf("Veri Fully-Connected bias: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, SS_POST_ACT*WEIGHT*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE + BIAS); //217.0f
		free(verification);
	}

	// Performing Fully-Connected sigmoid operation
	kernel_fc1_sigmoid<<<1, K3>>>((float(*))fc_layer.pre_output, (float(*))fc_layer.output);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Verifying Fully-Connected sigmoid operation
	if (verify) {
		printf("Veri Fully-Connected sigmoid: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, FC_POST_ACT); // 1.0f
		free(verification);
	}

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	return elapsedTime;
}

void copy_trained_parameters() {
	cudaMemcpy(conv_layer.weight, c1_weight, sizeof(float) * CHANNEL * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(conv_layer.bias, c1_bias, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(ss_layer.weight, s2_weight, sizeof(float) * SS_SIZE * SS_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(ss_layer.bias, s2_bias, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fc_layer.weight, f3_weight, sizeof(float) * NUM_CLASSES * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(fc_layer.bias, f3_bias, sizeof(float) * NUM_CLASSES, cudaMemcpyHostToDevice);
}

int main() {
	int ret, i;
	mnist_data *test_set;
	static unsigned int test_cnt;

	// Calling the mnist_load() function
	if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt) != 0) {
		printf("An error occured: %d\n", ret);
	} else {
		printf("test_cnt = %d\n", test_cnt);
	}

	// Verifying the image and label data of the specified number of examples
	//printExamples(&test_set, 1);

	// Verifying the convolutional layer
	double data[INSIZE][INSIZE];

	for (i = 0; i < INSIZE; i++) {
		for (int j = 0; j < INSIZE; j++)
			data[i][j] = INPUT; //1.0f;
	}

	forward_pass(data, true);

	//copy_trained_parameters();

	// Performing forward pass
	unsigned int error = 0;
	unsigned int max = 0;
	float res[10];

	for (i = 0; i < test_cnt; i++) {
		time_taken += forward_pass(test_set[i].data, false);
		cudaMemcpy(res, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);

		for (int j = 0; j < NUM_CLASSES; j++) {
			if (res[max] < res[j])
				max = j;
		}

		if (max != test_set[i].label)
			error++;
	}

	printf("Error Rate = %f%% (%d out of 10000)\n", double(error) / double(test_cnt) * 100.0, error);
	printf("Accuracy = %.3f%% (%d out of 10000)\n", 100.0 - double(error) / double(test_cnt) * 100.0, test_cnt - error);
	printf("Execution time = %f (ms) \n", time_taken);

	free(test_set);
	return 0;
}
