
#pragma once

#include "view.h"

void launch_rms_norm(
    const View& input,   // [Batch, Seq, Hidden]
    const View& weight,  // [Hidden]
    View& output,        // [Batch, Seq, Hidden]
    float eps,           
    cudaStream_t stream  
); 