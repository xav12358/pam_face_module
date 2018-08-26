#include "face_module/architecture/FaceNet/facenet.h"

FaceNet::FaceNet() : Net() {}

bool FaceNet::CreateSession() {
  std::shared_ptr<TF_SessionOptions> opt(
      TF_NewSessionOptions(), std::default_delete<TF_SessionOptions>());
  TF_Session *session = TF_NewSession(graph_.get(), opt.get(), status_.get());
  if (session) {
    sess_.reset(session, std::default_delete<TF_Session>());
  } else {
    if (TF_GetCode(status_.get()) != TF_OK) {
      std::cerr << "ERROR: Unable to create session"
                << std::string(TF_Message(status_.get())) << std::endl;
    } else {
      std::cerr << "ERROR: Unable to create session" << std::endl;
    }
    return false;
  }

  return true;
}

bool FaceNet::Init(std::string const filename) {
  bool ret = false;
  // load the graph
  ret = LoadGraph(filename);
  // create the session
  ret = CreateSession();
  // indicate if init is made
  is_init_ = ret;
  // Init the operators
  is_init_ = Init();
  return is_init_;
}

bool FaceNet::LoadGraph(std::string const filename) {
  // Load graph.
  {
    std::shared_ptr<TF_Buffer> graph_def = ReadFile(filename.c_str());
    graph_.reset(TF_NewGraph(), std::default_delete<TF_Graph>());
    std::shared_ptr<TF_ImportGraphDefOptions> opts;
    opts.reset(TF_NewImportGraphDefOptions(),
               std::default_delete<TF_ImportGraphDefOptions>());
    TF_GraphImportGraphDef(graph_.get(), graph_def.get(), opts.get(),
                           status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
      std::cerr << "ERROR: Unable to import graph " << std::endl
                << std::string(TF_Message(status_.get())) << std::endl;
      return false;
    }
  }
  return true;
}

bool FaceNet::Init() {
  // Prepare input
  {
    TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), "input");
    run_inputs_[0].oper = input_op;
    run_inputs_[0].index = 0;
  }

  // Prepare outputs.
  {
    TF_Operation *feature_op =
        TF_GraphOperationByName(graph_.get(), "l2_normalize");
    run_outputs_[0].oper = feature_op;
    run_outputs_[0].index = 0;
  }
  is_init_ = true;
  return is_init_;
}

void FaceNet::Process(std::vector<cv::Mat> &image_candidates) {
  Debug(" >>>>> FaceRecognition::Process ");
  int batch = int(image_candidates.size());

  /* prepare input image data */
  int input_size = batch * kHeight_ * kWidth_ * 3;
  std::vector<float> input_buffer((size_t)(input_size));
  float *input_data = input_buffer.data();

  int patch_size = kWidth_ * kHeight_ * 3;
  for (size_t i = 0; i < image_candidates.size(); i++) {
    copy_one_image(image_candidates[i], input_data);
    input_data += patch_size;
  }

  // Generate input.
  std::shared_ptr<int64_t> raw_input_dims_;
  raw_input_dims_.reset(new int64_t[4]{batch, kHeight_, kWidth_, 3});
  std::shared_ptr<TF_Tensor> input_tensor_;
  input_tensor_.reset(
      TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4, input_buffer.data(),
                   sizeof(float) * batch * kHeight_ * kWidth_ * 3,
                   &nullDeallocator, NULL),
      &TF_DeleteTensor);

  run_inputs_tensors_[0] = input_tensor_.get();

  TF_SessionRun(sess_.get(), nullptr, run_inputs_, run_inputs_tensors_, 1,
                run_outputs_, &run_output_tensors_[0], 1, nullptr, 0, nullptr,
                status_.get());

  if (TF_GetCode(status_.get()) != TF_OK) {
    std::cerr << "ERROR: Unable to run output_op: "
              << std::string(TF_Message(status_.get())) << std::endl;
    return;
  }

  /*retrieval the forward results*/
  std::vector<Eigen::MatrixXf> image_features;
  float *feature_data =
      ( float *)(TF_TensorData(run_output_tensors_[0]));

  for(int i=0;i<batch;i++){
      image_features.push_back(  Eigen::Map<Eigen::Matrix<float,128, 1>>(feature_data));
  }

  TF_DeleteTensor(run_output_tensors_[0]);

#ifdef USEDEBUG
  for(auto feature: image_features){
      std::cout << "image_features " << feature << std::endl;
  }

#endif
}
