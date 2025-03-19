#include "NvInferPlugin.h"
#include <iostream>

// ----------------------------------------------------------------------
// Logger Declaration
// ----------------------------------------------------------------------
class Logger : public nvinfer1::ILogger {
public:
    // Inline constructor
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {
    }

    // Inline override of log, matching the base class exactly.
    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

private:
    nvinfer1::ILogger::Severity reportableSeverity;
};

inline float fp16ToFloat(uint16_t h) {
    __half h_val = *reinterpret_cast<__half*>(&h);
    return __half2float(h_val);
}