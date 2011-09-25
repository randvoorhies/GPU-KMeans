#include <vector>

struct pointdescriptor
{
  float x, y, r, g, b;
  size_t classid;
};

std::vector<size_t> kmeans(std::vector<pointdescriptor> & points, size_t k);

