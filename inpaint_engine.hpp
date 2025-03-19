#ifndef INPAINTING_MODEL_HPP
#define INPAINTING_MODEL_HPP

#include <string>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "NvInferPlugin.h"
#include <cuda_fp16.h>

class InpaintingEngine {
public:
    InpaintingEngine(const std::string& engineFilePath);
    ~InpaintingEngine();

    // Enqueue inference using the asynchronous enqueueV3 method.
    bool inferEnqueueV3(void* inputData, void* outputData, cudaStream_t stream);

    // Helper functions to compute tensor sizes.
    size_t getMemorySize() const;
    size_t getInputSize() const;
    size_t getOutputSize() const;

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    int maskedFramesIndex;
    int memoryIndex;
    int outputIndex;
    int newMemoryIndex;

    int nbIOTensors;

    void* memoryBuffer;
    void* newMemoryBuffer;
};

// The InpaintingModel class encapsulates the inpainting engine,
// its input/output buffers, and associated resource management.
class InpaintingModel {
public:
    explicit InpaintingModel(const std::string& engineFilePath = "st_360_reshape_fp16_win.engine");

    ~InpaintingModel();

    // Runs inference on the provided image blob and outputs a predicted image.
    bool runInference(const cv::Mat& imageBlob, cv::Mat& predictedImage360);

private:

    InpaintingEngine* m_inference;
    void* m_inputDeviceBuffer;
    void* m_outputDeviceBuffer;
    size_t m_totalInputSize;
    size_t m_totalOutputSize;
    cudaStream_t m_stream;

    // Image tensor constants for the inpainted output.
    static const int kBatchImg = 1;
    static const int kChannelsImg = 3;
    static const int kHeightImg = 360;
    static const int kWidthImg = 640;
    static const size_t kNumElementsImg = kBatchImg * kChannelsImg * kHeightImg * kWidthImg;
    static const size_t kSizeImg = kNumElementsImg * sizeof(uint32_t); // Using FP32 (4 bytes per element)
};

#endif // INPAINTING_MODEL_HPP
