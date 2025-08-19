# Inpaint_manager

High-performance **real-time inpainting server** used in my thesis on *diminished reality for privacy in MR meetings*.  
This process owns the inpainting model (TensorRT engine) and exposes a zero-copy(ish) IPC interface using **Windows shared memory** and **named events** for low-latency request/response.

> Input/Output: FP32 planar blob shaped **(1, 3, 360, 640)** (NCHW), transferred via shared memory.  
> Synchronization: `InputReadyEvent` → inference → `OutputReadyEvent`.

---

## Table of contents

- [Features](#features)
- [Model engine](#model-engine)
- [Dependencies](#dependencies)
- [Build (Windows)](#build-windows)
- [Run](#run)
- [IPC protocol](#ipc-protocol)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)

---

## Features

- TensorRT engine runtime with CUDA stream for async execution
- Pinned host buffers for fast H2D/D2H
- Windows shared memory (`CreateFileMappingW` / `MapViewOfFile`) for IO
- Named events for synchronization:
  - `InputReadyEvent` (client → server)
  - `OutputReadyEvent` (server → client)

---

## Model engine

The server expects an engine file named:

st_360_reshape_stage2_epoch2.engine

Download:
➡️ **[Download the TensorRT engine](https://github.com/c1h1r1i1s1/Inpaint_manager/releases/download/v1.0.0/st_360_reshape_stage2_epoch2.engine)**

---

## Dependencies

**OS / Toolchain**
- Windows 10/11 x64
- Visual Studio 2022 (v143 toolset) with MSVC C++17

**GPU Stack**
- NVIDIA GPU with recent driver
- CUDA Toolkit **12.1** (recommended)  
- cuDNN **9.6.0**
- TensorRT **10.7** runtime libraries matching the engine build

**Computer Vision**
- OpenCV **4.x** (C++), prebuilt or built from source  
  > CUDA-enabled OpenCV is optional; this app only uses core/imgproc/mat containers

**CMake**
- CMake **3.22+**

**Headers/Libraries Required at Build/Run**
- `cuda_runtime.lib`, CUDA includes
- TensorRT: `nvinfer.lib`, `nvonnxparser.lib` (as needed by your `inpaint_engine.hpp`), plus DLLs on `PATH`
- OpenCV: `opencv_world4xx.lib` (or modular libs), plus DLLs on `PATH`

---

## Build (Windows)

```powershell
# From repo root
git clone https://github.com/c1h1r1i1s1/Inpaint_manager.git
cd Inpaint_manager

# Configure (edit paths if you don’t rely on env vars)
cmake -S . -B build -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR="C:\opencv\build" `
  -DTensorRT_ROOT="C:\TensorRT-10.7.0.23" `
  -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

# Build
cmake --build build --config Release
```

Artifacts appear under build/Release/.

## Run

1. Place `st_360_reshape_stage2_epoch2.engine` next to the executable (or pass the correct path in code).
2. Launch the server executable (e.g., `Inpaint_manager.exe`). You should see:

```
Inpainting server process is running...
```

3. A client process should:
- Open the named shared memory region
- Write a **(1,3,360,640) FP32 NCHW** blob into it
- Signal **`InputReadyEvent`**
- Wait for **`OutputReadyEvent`**
- Read the FP32 result blob

---

## IPC protocol

**Shared memory**
- **Name:** `InpaintingSharedMemory` (wide string)
- **Size:** `3 * 360 * 640 * sizeof(float)` = **2,764,800 bytes** (≈ **2.64 MiB**)

**Events**
- `InputReadyEvent` (auto-reset, server waits)
- `OutputReadyEvent` (auto-reset, client waits)

**Buffer layout**
- **NCHW FP32** continuous buffer
- Channel order: `C0 (H×W) | C1 (H×W) | C2 (H×W)`
- Endianness: little-endian (Windows)
- **Color/normalization**  
The server performs **no normalization**; it assumes the blob is already in the format the engine expects (e.g., `[0,1]`, `[-1,1]`, or ImageNet stats). Prepare your input accordingly on the client.

**Lifecycle**
1. Client writes input FP32 blob → signals `InputReadyEvent`
2. Server copies shared memory → pinned staging → runs inference (TensorRT)
3. Server copies output back to shared memory → signals `OutputReadyEvent`
4. Client reads output

---

## Performance notes

- Pinned host memory (`cudaHostAlloc`) is used for staging and output.
- A dedicated CUDA stream allows `cudaMemcpyAsync`/inference overlap.
- Shared memory avoids extra sockets/serialization overhead.
- End-to-end latency primarily depends on:
- Engine precision (FP16/INT8 preferred if available)
- GPU compute capability & clocking
- Client preprocessing/postprocessing time

---

## Troubleshooting

- **Engine deserialization fails**  
Ensure TensorRT runtime **major/minor** matches the engine build. Rebuild the engine for your TRT version if needed.
- **DLL not found (OpenCV / TensorRT / CUDA)**  
Add their `bin` directories to `PATH` (System Environment Variables) or place DLLs beside the executable.
- **Wrong output / garbage values**  
Verify your input blob is **NCHW FP32**, correct normalization, and correct channel order.
- **Deadlock / events never fire**  
Confirm the client signals `InputReadyEvent`, and that both processes use the exact same **wide** event and mapping names.
- **Size mismatch warning**  
The server asserts that output size equals `FP32_BLOB_SIZE`. If your engine output shape differs, update the shared memory size and checks.

---
