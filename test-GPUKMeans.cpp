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
  CImg<unsigned char> kmeans_image(argv[1]);

  CImg<unsigned char>::const_iterator imgit = image.begin();
  std::vector<pointdescriptor> points(image.width() * image.height());
  for(int y = 0; y < image.height(); ++y)
    for(int x = 0; x < image.width(); ++x)
    {
      const size_t idx = x + y*image.width();
      points[idx].x = x;
      points[idx].x = y;
      points[idx].r = *imgit++;
      points[idx].g = *imgit++;
      points[idx].b = *imgit++;
    }

  std::cout << "Computing kmeans" << std::endl;
  std::vector<size_t> classes = kmeans(points, 30);
  std::cout << "Done" << std::endl;

  //std::cout << "Classes: " << std::endl;
  //for(size_t i=0; i<classes.size(); ++i)
  //  std::cout << classes[i] << std::endl;


  CImgDisplay main_disp(image,"Input Image");
  while (!main_disp.is_closed())
  {
    main_disp.wait();
  }
  return 0;
}

