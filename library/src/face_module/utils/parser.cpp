#include <face_module/utils/parser.h>

std::unordered_map<std::string,std::string> ParseCommand(int argc, char **argv) {

  std::unordered_map<std::string,std::string> parsedCommand;
  int i = 1;
  while (i < argc) {
    if (argv[i][0] == '-' && argv[i][1] == '-' && (i+1) < argc) {
      // find argument
      std::string cmd = argv[i];
      std::string argu = argv[i + 1];
      parsedCommand[cmd] = argu;
      i++;
    }
    i++;
  }
  return parsedCommand;
}

bool IsNumber(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}
