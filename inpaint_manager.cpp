#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include "inpaint_engine.hpp"

InpaintingModel* inpaintingModel;

// Define the shared memory size for an FP32 blob of shape (1, 3, 360, 640).
// FP32 is 4 bytes per element.
const size_t FP32_BLOB_SIZE = 1ull * 3 * 360 * 640 * sizeof(uint32_t);

// Use wide-string names for shared memory and events.
const wchar_t* SHARED_MEMORY_NAME = L"InpaintingSharedMemory";
const wchar_t* EVENT_INPUT_READY = L"InputReadyEvent";
const wchar_t* EVENT_OUTPUT_READY = L"OutputReadyEvent";

int main() {
    inpaintingModel = new InpaintingModel("st_360_reshape_stage1_epoch15_win.engine");
    // Create or open the shared memory region.
    HANDLE hMapFile = CreateFileMappingW(
        INVALID_HANDLE_VALUE,    // Use the system paging file.
        NULL,                    // Default security.
        PAGE_READWRITE,          // Read/write access.
        0,                       // Maximum object size (high-order DWORD).
        (DWORD)FP32_BLOB_SIZE,   // Maximum object size (low-order DWORD).
        SHARED_MEMORY_NAME       // Name of the mapping object.
    );

    if (hMapFile == NULL) {
        std::wcerr << L"Could not create file mapping object (" << GetLastError() << L").\n";
        return 1;
    }

    // Map the shared memory into our address space.
    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FP32_BLOB_SIZE);
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

    std::wcout << L"Inpainting server process is running...\n";

    while (true) {
        // Wait for the client to signal that new FP32 input blob is available.
        DWORD dwWaitResult = WaitForSingleObject(hInputEvent, INFINITE);
        if (dwWaitResult != WAIT_OBJECT_0) {
            std::wcerr << L"Error waiting for input event.\n";
            break;
        }

        // Here we assume the shared memory now holds the raw FP32 blob data.
        // Create a cv::Mat header that points directly to the shared memory.
        // Since blobFromImage produces a blob in CHW format (1, 3, 360, 640), we can create a 1x(N) matrix and reshape it.
        cv::Mat inputBlobFP32(1, 3 * 360 * 640, CV_32F, pBuf);
        // Optionally, you can reshape it if needed:
        // cv::Mat inputBlobFP32 = cv::Mat(1, 3 * 360 * 640, CV_32F, pBuf).reshape(1, {1, 3, 360, 640});

        // Run the inpainting inference on the FP32 blob.
        cv::Mat outputBlobFP32;
        inpaintingModel->runInference(inputBlobFP32, outputBlobFP32);

        // Write the output blob back into shared memory.
        // Here we assume the output blob is the same size as the input.
        if (outputBlobFP32.total() * outputBlobFP32.elemSize() != FP32_BLOB_SIZE) {
            std::wcerr << L"Output blob size mismatch.\n";
            std::cerr << outputBlobFP32.total() << std::endl;
            std::cerr << outputBlobFP32.elemSize() << std::endl;
            // You might choose to handle this error differently.
            return 1;
        }
        memcpy(pBuf, outputBlobFP32.data, FP32_BLOB_SIZE);

        // Signal that the output is ready.
        SetEvent(hOutputEvent);
    }

    // Cleanup
    //delete inpaintingModel;
    CloseHandle(hInputEvent);
    CloseHandle(hOutputEvent);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
