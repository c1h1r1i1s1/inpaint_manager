#include "inpaint_engine.hpp"
#include "common.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>

// ------------------------------------------------------------------------
// Instead of globals, we encapsulate constants within this file or the class
// ------------------------------------------------------------------------
static const int kBatchImg = 1;
static const int kChannelsImg = 3;
static const int kHeightImg = 360;
static const int kWidthImg = 640;
static const size_t kNumElementsImg = kBatchImg * kChannelsImg * kHeightImg * kWidthImg;
static const size_t kSizeImg = kNumElementsImg * sizeof(uint32_t);

// Global logger instance can remain global or be encapsulated if desired.
Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };


InpaintingModel::InpaintingModel(const std::string& engineFilePath)
    : m_inference(nullptr), m_inputDeviceBuffer(nullptr), m_outputDeviceBuffer(nullptr),
    m_totalInputSize(0), m_totalOutputSize(0), m_stream(0)
{
    cudaError_t err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream.");
    }

    // Create the TRTInference instance.
    m_inference = new InpaintingEngine(engineFilePath);
    if (!m_inference) {
        throw std::runtime_error("Failed to create TRTInference instance.");
    }
    // Setup input and output sizes.
    m_totalInputSize = m_inference->getInputSize();
    m_totalOutputSize = m_inference->getOutputSize();

    // Allocate device memory.
    err = cudaMalloc(&m_inputDeviceBuffer, m_totalInputSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate input device buffer.");
    }
    err = cudaMalloc(&m_outputDeviceBuffer, m_totalOutputSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate output device buffer.");
    }
}

// Destructor releases all resources.
InpaintingModel::~InpaintingModel() {
    if (m_inference) {
        delete m_inference;
        m_inference = nullptr;
    }
    if (m_inputDeviceBuffer) {
        cudaFree(m_inputDeviceBuffer);
        m_inputDeviceBuffer = nullptr;
    }
    if (m_outputDeviceBuffer) {
        cudaFree(m_outputDeviceBuffer);
        m_outputDeviceBuffer = nullptr;
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = 0;
    }
}

// Run inference on a given input image blob and extract the predicted image.
bool InpaintingModel::runInference(const cv::Mat& imageBlob, cv::Mat& predictedImage360) {
    // Copy the input image blob to device memory.
    cudaMemcpy(m_inputDeviceBuffer, imageBlob.data, m_totalInputSize, cudaMemcpyHostToDevice);

    // Enqueue inference on the model.
    bool status = m_inference->inferEnqueueV3(m_inputDeviceBuffer, m_outputDeviceBuffer, m_stream);
    if (status) {
        return 1;
    }

    // Synchronize to ensure inference is complete.
    cudaStreamSynchronize(m_stream);

    predictedImage360.create(kHeightImg, kWidthImg, CV_32FC3);
    cudaError_t err = cudaMemcpy(predictedImage360.data, m_outputDeviceBuffer, kSizeImg, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}

InpaintingEngine::InpaintingEngine(const std::string& engineFilePath)
    : runtime(nullptr), engine(nullptr), context(nullptr),
    nbIOTensors(0), memoryIndex(-1), outputIndex(-1),
    newMemoryIndex(-1), maskedFramesIndex(-1),
    memoryBuffer(nullptr), newMemoryBuffer(nullptr)
{
    // Read engine file from disk.
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening engine file: " + engineFilePath);
    }
    file.seekg(0, std::ios::end);
    auto fsize = file.tellg();
    file.seekg(0, std::ios::beg);
    char* engineData = new char[fsize];
    file.read(engineData, fsize);
    file.close();

    // Create the runtime and deserialize the engine.
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }
    engine = runtime->deserializeCudaEngine(engineData, fsize);
    delete[] engineData;
    if (!engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    // Determine I/O tensor indices.
    nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        if (strcmp(tensorName, "masked_frames") == 0) {
            maskedFramesIndex = i;
        }
        else if (strcmp(tensorName, "memory") == 0) {
            memoryIndex = i;
        }
        else if (strcmp(tensorName, "output") == 0) {
            outputIndex = i;
        }
        else if (strcmp(tensorName, "new_memory") == 0) {
            newMemoryIndex = i;
        }
    }

    if (maskedFramesIndex == -1 || memoryIndex == -1 || outputIndex == -1 || newMemoryIndex == -1) {
        throw std::runtime_error("Could not find all required tensor names (masked_frames, memory, output, new_memory) in engine.");
    }

    // Create an execution context (using static allocation strategy). nvinfer1::ExecutionContextAllocationStrategy::kSTATIC
    context = engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC);
    if (!context) {
        throw std::runtime_error("Failed to create execution context.");
    }

    size_t memSize = getMemorySize();
    cudaError_t err = cudaMalloc(&memoryBuffer, memSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for memoryBuffer.");
    }
    err = cudaMalloc(&newMemoryBuffer, memSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for newMemoryBuffer.");
    }
}

InpaintingEngine::~InpaintingEngine() {
    if (memoryBuffer) {
        cudaFree(memoryBuffer);
    }
    if (newMemoryBuffer) {
        cudaFree(newMemoryBuffer);
    }
    if (context) {
        delete context;
        context = nullptr;
    }
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
}

bool InpaintingEngine::inferEnqueueV3(void* inputData, void* outputData, cudaStream_t stream) {
    context->setTensorAddress("masked_frames", inputData);
    context->setTensorAddress("memory", memoryBuffer);
    context->setTensorAddress("output", outputData);
    context->setTensorAddress("new_memory", newMemoryBuffer);

    bool success = context->enqueueV3(stream);
    if (!success) {
        std::cerr << "Issue with data enqueue" << std::endl;
        return 1;
    }

    size_t memSize = getMemorySize();
    cudaMemcpyAsync(memoryBuffer, newMemoryBuffer, memSize, cudaMemcpyDeviceToDevice, stream);
    return 0;
}

size_t InpaintingEngine::getMemorySize() const {
    nvinfer1::Dims dims = engine->getTensorShape("memory");
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        volume *= dims.d[i];
    //int32_t bytesPerComponent = engine->getTensorBytesPerComponent("memory");
    //bytesPerComponent = std::abs(bytesPerComponent);
    int bytesPerComponent = 4; // fp32
    return volume * bytesPerComponent;
}

size_t InpaintingEngine::getInputSize() const {
    nvinfer1::Dims dims = engine->getTensorShape("masked_frames");
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        volume *= dims.d[i];
    //int32_t bytesPerComponent = engine->getTensorBytesPerComponent("masked_frames");
    //bytesPerComponent = std::abs(bytesPerComponent);
    int bytesPerComponent = 4; // fp32
    return volume * bytesPerComponent;
}

size_t InpaintingEngine::getOutputSize() const {
    nvinfer1::Dims dims = engine->getTensorShape("output");
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        vol *= dims.d[i];
    //int32_t bytesPerComponent = engine->getTensorBytesPerComponent("output");
    //bytesPerComponent = std::abs(bytesPerComponent);
    int bytesPerComponent = 4; // fp32
    return vol * bytesPerComponent;
}