#include "pam_face_module/feature_manager.h"

FeatureManager::FeatureManager() {}

std::unordered_map<std::string,
                   std::unordered_map<std::string, Eigen::MatrixXf>>
FeatureManager::Read(std::string filename) {

  std::ifstream ifs(filename);
  Json::Reader reader;
  Json::Value obj;
  reader.parse(ifs, obj);

  std::unordered_map<std::string,
                     std::unordered_map<std::string, Eigen::MatrixXf>>
      person_features;

  for (Json::Value::iterator it_person = obj.begin(); it_person != obj.end();
       ++it_person) {

    for (Json::Value::iterator it_orientation = (*it_person).begin();
         it_orientation != (*it_person).end(); ++it_orientation) {
      auto iterv = (*it_orientation).begin();
      Eigen::MatrixXf feature_vector(128, 1);
      int i = 0;
      for (Json::Value::iterator it_feature = (*iterv).begin();
           it_feature != (*iterv).end(); ++it_feature, ++i) {
        feature_vector(0, i) = (*it_feature).asDouble();
      }
      person_features[it_person.key().asString()]
                     [it_orientation.key().asString()] = feature_vector;
    }
  }

  return person_features;
}

Json::Value toVec(Eigen::MatrixXf M) {
  Json::Value ret(Json::arrayValue);
  Json::Value vec(Json::arrayValue);
  for (int i = 0; i < M.cols(); i++) {
    vec.append(M(0, i));
  }
  ret.append(vec);
  return ret;
}

void FeatureManager::Write(
    std::unordered_map<std::string,
                       std::unordered_map<std::string, Eigen::MatrixXf>>
        person_features,
    std::string filename) {

  Json::Value val;
  for (auto person : person_features) {
    for (auto orientation : person.second) {
      val[person.first][orientation.first] = toVec(orientation.second);
    }
  }

  std::ofstream file_id;
  file_id.open(filename);

  Json::StyledWriter styledWriter;
  file_id << styledWriter.write(val);

}
