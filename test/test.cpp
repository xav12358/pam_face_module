#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

#include "pam_face_module/face_detector_test.h"
#include "pam_face_module/face_recognition_test.h"

TEST_F(Face_detector_test, process) { EXPECT_EQ(0, 1); }

TEST_F(Face_recognition_test, process) { EXPECT_EQ(0, 2); }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    //  TF_Graph *graph = TF_NewGraph();
    //  TF_SessionOptions *options = TF_NewSessionOptions();
    //  TF_Status *status = TF_NewStatus();
    //  TF_Session *session = TF_NewSession(graph, options, status);
    //  std::string hello = "Hello TensorFlow!";
    //  TF_Tensor *tensor = TF_AllocateTensor(
    //      TF_STRING, 0, 0, 8 + TF_StringEncodedSize(hello.length()));
    //  TF_Tensor *tensorOutput;
    //  TF_OperationDescription *operationDescription =
    //      TF_NewOperation(graph, "Const", "hello");
    //  TF_Operation *operation;
    //  struct TF_Output output;

    //  TF_StringEncode(hello.c_str(), hello.length(),
    //                  8 + (char *)TF_TensorData(tensor),
    //                  TF_StringEncodedSize(hello.length()), status);
    //  memset(TF_TensorData(tensor), 0, 8);
    //  TF_SetAttrTensor(operationDescription, "value", tensor, status);
    //  TF_SetAttrType(operationDescription, "dtype", TF_TensorType(tensor));
    //  operation = TF_FinishOperation(operationDescription, status);

    //  output.oper = operation;
    //  output.index = 0;

    //  TF_SessionRun(session, 0, 0, 0, 0,       // Inputs
    //                &output, &tensorOutput, 1, // Outputs
    //                &operation, 1,             // Operations
    //                0, status);

    //  printf("status code: %i\n", TF_GetCode(status));
    //  printf("%s\n", ((char *)TF_TensorData(tensorOutput)) + 9);

    //  TF_CloseSession(session, status);
    //  TF_DeleteSession(session, status);
    //  TF_DeleteStatus(status);
    //  TF_DeleteSessionOptions(options);

    return EXIT_SUCCESS;
}
