#include "CImg.h"
#include <iostream>
#include "GPUKMeans.h"

using namespace cimg_library;

int main(int argc, char* argv[]) 
{

  if(argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " IMAGE_FILE" << std::endl;
    exit(-1);
  }

  CImg<unsigned char> image(argv[1]);
  CImg<unsigned char> kmeans_image(image.width(), image.height(), 1, 3);

  std::vector<PointDescriptor> points(image.width() * image.height());
  size_t idx = 0;
  for(int y = 0; y < image.height(); ++y)
    for(int x = 0; x < image.width(); ++x)
    {
      points[idx].x = x;
      points[idx].y = y;
      points[idx].r = image.atXYZC(x,y,0,0);
      points[idx].g = image.atXYZC(x,y,0,1);
      points[idx].b = image.atXYZC(x,y,0,2);
      ++idx;
    }

  std::cout << "Computing kmeans" << std::endl;
  std::vector<ClassSummary> classes = kmeans(points, 5);
  std::cout << "Done" << std::endl;

  std::cout << "Points: " << points.size() << std::endl;
  for(size_t i=0; i<points.size(); ++i)
  {
    kmeans_image(points[i].x, points[i].y, 0, 0) = classes[points[i].classid].mean_r; 
    kmeans_image(points[i].x, points[i].y, 0, 1) = classes[points[i].classid].mean_g; 
    kmeans_image(points[i].x, points[i].y, 0, 2) = classes[points[i].classid].mean_b; 
  }

  CImgDisplay kmeans_disp(kmeans_image,"KMeans Image");
  CImgDisplay main_disp(image,"Input Image");
  while (!main_disp.is_closed() && ! kmeans_disp.is_closed())
  {
    kmeans_disp.wait();
    main_disp.wait();
  }
  return 0;
}

