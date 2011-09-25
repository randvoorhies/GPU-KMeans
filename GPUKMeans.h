#include <vector>

struct pointdescriptor
{
  float x, y, r, g, b;
  size_t classid;
};

// ######################################################################
struct ClassSummary
{
  void initialize()
  {
    n = 0;
    mean_x = 0.0;
    mean_y = 0.0;
    mean_r = 0.0;
    mean_g = 0.0;
    mean_b = 0.0;
  }

  size_t n;
  float mean_x;
  float mean_y;
  float mean_r;
  float mean_g;
  float mean_b;
};

std::vector<ClassSummary> kmeans(std::vector<pointdescriptor> & points, size_t k);

