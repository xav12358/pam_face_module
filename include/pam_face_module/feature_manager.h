#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>

#include <eigen3/Eigen/Dense>


class FeatureManager
{
    const int kFeatureSize_ = 128;
public:
    FeatureManager();
    static std::unordered_map<std::string, std::unordered_map<std::string, Eigen::MatrixXf> > Read(std::string filename);
    static void Write(std::unordered_map<std::string,
                      std::unordered_map<std::string, Eigen::MatrixXf>> person_features, std::string filename);
};

#endif // FEATURE_MANAGER_H
