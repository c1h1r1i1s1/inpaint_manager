#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include "inpaint_engine.hpp"

InpaintingModel* inpaintingModel_L;
InpaintingModel* inpaintingModel_R;

float* pinnedOutL = nullptr;
float* pinnedOutR = nullptr;

// Define the shared memory size for an FP32 blob of shape (1, 3, 360, 640).
// FP32 is 4 bytes per element.
const size_t FP32_BLOB_SIZE = 3 * 360 * 640 * sizeof(float);

// Use wide-string names for shared memory and events.
const wchar_t* SHARED_MEMORY_NAME = L"InpaintingSharedMemory";
const wchar_t* EVENT_INPUT_READY = L"InputReadyEvent";
const wchar_t* EVENT_OUTPUT_READY = L"OutputReadyEvent";

int main() {
    inpaintingModel_L = new InpaintingModel("st_360_reshape_stage2_epoch2.engine");
    inpaintingModel_R = new InpaintingModel("st_360_reshape_stage2_epoch2.engine");

    cudaStream_t streamL = inpaintingModel_L->getStream();
    cudaStream_t streamR = inpaintingModel_R->getStream();
    // Create or open the shared memory region.
    HANDLE hMapFile = CreateFileMappingW(
        INVALID_HANDLE_VALUE,    // Use the system paging file.
        NULL,                    // Default security.
        PAGE_READWRITE,          // Read/write access.
        0,                       // Maximum object size (high-order DWORD).
        (DWORD)FP32_BLOB_SIZE*2, // Maximum object size (low-order DWORD).
        SHARED_MEMORY_NAME       // Name of the mapping object.
    );

    if (hMapFile == NULL) {
        std::wcerr << L"Could not create file mapping object (" << GetLastError() << L").\n";
        return 1;
    }

    // Map the shared memory into our address space.
    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FP32_BLOB_SIZE*2);
    if (pBuf == NULL) {
        std::wcerr << L"Could not map view of file (" << GetLastError() << L").\n";
        CloseHandle(hMapFile);
        return 1;
    }

    // Open named events for synchronization.
    HANDLE hInputEvent = CreateEventW(NULL, FALSE, FALSE, EVENT_INPUT_READY);
    HANDLE hOutputEvent = CreateEventW(NULL, FALSE, FALSE, EVENT_OUTPUT_READY);
    if (hInputEvent == NULL || hOutputEvent == NULL) {
        std::wcerr << L"Could not create events (" << GetLastError() << L").\n";
        UnmapViewOfFile(pBuf);
        CloseHandle(hMapFile);
        return 1;
    }

    // Get pBuf address
    auto* sharedBytes = static_cast<const uint8_t*>(pBuf);

    // Allocate staging and output buffers
    float* pinnedStaging = nullptr;
    cudaHostAlloc(reinterpret_cast<void**>(&pinnedStaging),
        2 * FP32_BLOB_SIZE,
        cudaHostAllocDefault);
    auto* pinnedStagingAddr = reinterpret_cast<uint8_t*>(pinnedStaging);

    cudaHostAlloc(&pinnedOutL, FP32_BLOB_SIZE, cudaHostAllocDefault);
    cudaHostAlloc(&pinnedOutR, FP32_BLOB_SIZE, cudaHostAllocDefault);
    cv::Mat outputBlobFP32_L(360, 640, CV_32FC3, pinnedOutL);
    cv::Mat outputBlobFP32_R(360, 640, CV_32FC3, pinnedOutR);

    // Create events for L->R memory communication
    cudaEvent_t memReady;
    cudaEventCreate(&memReady);
    bool firstFrame = true;

    std::wcout << L"Inpainting server process is running...\n";

    while (true) {
        // Wait for the client to signal that new FP32 input blob is available.
        DWORD dwWaitResult = WaitForSingleObject(hInputEvent, INFINITE);
        if (dwWaitResult != WAIT_OBJECT_0) {
            std::wcerr << L"Error waiting for input event.\n";
            break;
        }

        // Copy blob data from buffer into staging memory
        memcpy(
            pinnedStagingAddr + 0 * FP32_BLOB_SIZE,
            sharedBytes,
            FP32_BLOB_SIZE
        );
        memcpy(
            pinnedStagingAddr + 1 * FP32_BLOB_SIZE,
            sharedBytes + FP32_BLOB_SIZE,
            FP32_BLOB_SIZE
        );

        // Wrap dat with Mat headers for reading
        cv::Mat leftBlob(360, 640, CV_32FC3, pinnedStaging + 0 * FP32_BLOB_SIZE / sizeof(float));
        cv::Mat rightBlob(360, 640, CV_32FC3, pinnedStaging + 1 * FP32_BLOB_SIZE / sizeof(float));

        if (!firstFrame) {
            // seed streamR
            cudaMemcpyAsync(
                inpaintingModel_R->getInputBuffer(),
                inpaintingModel_L->getOutputBuffer(),
                inpaintingModel_L->getMemorySize(),
                cudaMemcpyDeviceToDevice,
                streamL
            );

            cudaEventRecord(memReady, streamL);

            // make streamR wait until streamL’s memory-pull is done
            cudaStreamWaitEvent(streamR, memReady, 0);
            inpaintingModel_R->runInference(pinnedStaging + 1 * FP32_BLOB_SIZE / sizeof(float));
        }

        // Run the inpainting inference on the FP32 blob.
        inpaintingModel_L->runInference(pinnedStaging + 0 * FP32_BLOB_SIZE / sizeof(float));

        inpaintingModel_L->getOutputs(outputBlobFP32_L);
        inpaintingModel_R->getOutputs(outputBlobFP32_R);

        cudaStreamSynchronize(streamR);
        cudaStreamSynchronize(streamL);

        // Write the output blob back into shared memory.
        // Here we assume the output blob is the same size as the input.
        if (outputBlobFP32_L.total() * outputBlobFP32_L.elemSize() != FP32_BLOB_SIZE) {
            std::wcerr << L"Output blob size mismatch L.\n";
            std::cerr << outputBlobFP32_L.total() << std::endl;
            std::cerr << outputBlobFP32_L.elemSize() << std::endl;
            return 1;
        }
        if (outputBlobFP32_R.total() * outputBlobFP32_R.elemSize() != FP32_BLOB_SIZE) {
            std::wcerr << L"Output blob size mismatch R.\n";
            std::cerr << outputBlobFP32_R.total() << std::endl;
            std::cerr << outputBlobFP32_R.elemSize() << std::endl;
            return 1;
        }

        auto* buf = reinterpret_cast<uint8_t*>(pBuf);
        memcpy(buf,
            outputBlobFP32_L.ptr<float>(),
            FP32_BLOB_SIZE);

        memcpy(buf + FP32_BLOB_SIZE,
            outputBlobFP32_R.ptr<float>(),
            FP32_BLOB_SIZE);

        // Signal that the output is ready.
        SetEvent(hOutputEvent);

        firstFrame = false;
    }

    // Cleanup
    CloseHandle(hInputEvent);
    CloseHandle(hOutputEvent);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    cudaEventDestroy(memReady);
    cudaFreeHost(pinnedStaging);
    cudaFreeHost(pinnedOutL);
    cudaFreeHost(pinnedOutR);

    delete inpaintingModel_R;
    delete inpaintingModel_L;
    return 0;
}
