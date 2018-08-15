#ifndef PARSER_H
#define PARSER_H

#include <algorithm>
#include <unordered_map>

std::unordered_map<std::string,std::string> ParseCommand(int argc, char **argv);
bool IsNumber(const std::string& s);
#endif // PARSER_H
