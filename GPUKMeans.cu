#include "GPUKMeans.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>

// ######################################################################
struct ClassSummary_unary_op
{
  __device__
    ClassSummary operator()(const pointdescriptor& point) const
    {
      ClassSummary result;

      result.n = 1;

      result.mean_x = point.x;
      result.mean_y = point.y;
      result.mean_r = point.r;
      result.mean_g = point.g;
      result.mean_b = point.b;

      return result;
    }
};

// ######################################################################
struct ClassSummary_binary_op : 
  public thrust::binary_function<const ClassSummary&, const ClassSummary&, ClassSummary>
{
  __host__ __device__
    ClassSummary operator()(const ClassSummary & p1, const ClassSummary & p2) const
    {
      ClassSummary result;

      size_t n = p1.n + p2.n;
      float const delta_x = p2.mean_x - p1.mean_x;
      float const delta_y = p2.mean_y - p1.mean_y;
      float const delta_r = p2.mean_r - p1.mean_r;
      float const delta_g = p2.mean_g - p1.mean_g;
      float const delta_b = p2.mean_b - p1.mean_b;

      result.n = n;
      result.mean_x = p1.mean_x + delta_x * p2.n / n;
      result.mean_y = p1.mean_y + delta_y * p2.n / n;
      result.mean_r = p1.mean_r + delta_r * p2.n / n;
      result.mean_g = p1.mean_g + delta_g * p2.n / n;
      result.mean_b = p1.mean_b + delta_b * p2.n / n;

      return result;
    }
};

// ######################################################################
struct ClassSorter :
  public thrust::binary_function<pointdescriptor const&, pointdescriptor const&, bool>
{
  __host__ __device__
    bool operator()(pointdescriptor const& p1, pointdescriptor const& p2)
    {
      return p1.classid < p2.classid;
    }
};

// ######################################################################
struct ClassFinder
{
  size_t classid;

  void setClass(size_t classid_)  { classid = classid_; }

  __host__ __device__
    bool operator()(pointdescriptor const& p)
    {
      return p.classid == classid;
    }
};

// ######################################################################
struct ClassAssigner
{
  ClassSummary * classesPtr;
  size_t const num_classes;
  float const float_max;

  __device__ __host__
  ClassAssigner(thrust::device_vector<ClassSummary> class_means_) :
    num_classes(class_means_.size()),
    float_max(1000000000000000)
  { 
  
    classesPtr = thrust::raw_pointer_cast(&class_means_[0]); 
  
  }

  __device__
    pointdescriptor operator()(pointdescriptor const& p) const
    {
      size_t min_idx = 0;
      float min_dist = float_max;
      for(size_t i=1; i<num_classes; ++i)
      {
        float dist = sqrt(pow(classesPtr[i].mean_x - p.x, 2) +
                          pow(classesPtr[i].mean_y - p.y, 2) +
                          pow(classesPtr[i].mean_r - p.r, 2) +
                          pow(classesPtr[i].mean_g - p.g, 2) +
                          pow(classesPtr[i].mean_b - p.b, 2));

        if(min_dist > dist)
        { min_dist = dist; min_idx = i; }
      }

      pointdescriptor ret = p;
      ret.classid = min_idx;
      return ret;
    }
};

// ######################################################################
std::vector<ClassSummary> kmeans(std::vector<pointdescriptor> & points, size_t k)
{
  // Randomly assign classes to each point descriptor
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, k);
  std::vector<pointdescriptor>::iterator pointsit;
  for(pointsit = points.begin(); pointsit != points.end(); ++pointsit)
    pointsit->classid = dist(rng);

  // Create a device copy of the point descriptors
  thrust::device_vector<pointdescriptor> device_points(points.size());
  thrust::copy(points.begin(), points.end(), device_points.begin());

  thrust::device_vector<ClassSummary> device_class_means(k);

  for(int i=0; i<10; ++i)
  {
    // Sort the point descriptors by their class id
    ClassSorter sorter;
    thrust::sort(device_points.begin(), device_points.end(), sorter);

    // Find all of the class means
    thrust::device_vector<pointdescriptor>::iterator classBegin = device_points.begin();
    for(size_t classid = 0; classid < k; ++classid)
    {
      ClassFinder finder;
      finder.setClass(classid);
      thrust::device_vector<pointdescriptor>::iterator classEnd =
        thrust::find_if_not(classBegin, device_points.end(), finder);

      //Compute class means (cluster centers)
      ClassSummary init_summary;
      init_summary.initialize();
      ClassSummary_unary_op unary_summary_op;
      ClassSummary_binary_op binary_summary_op;

      device_class_means[classid] = thrust::transform_reduce(classBegin, classEnd,
          unary_summary_op, init_summary, binary_summary_op);
    }

    // Reassign all point descriptors to their nearest class
    ClassAssigner assigner(device_class_means);
    thrust::transform(device_points.begin(), device_points.end(), device_points.begin(), assigner);
  }

  thrust::host_vector<ClassSummary> host_class_means(device_class_means.size());
  thrust::copy(device_class_means.begin(), device_class_means.end(), host_class_means.begin());
  std::vector<ClassSummary> class_means(host_class_means.size());
  std::copy(host_class_means.begin(), host_class_means.end(), class_means.begin());
  return class_means;
}

