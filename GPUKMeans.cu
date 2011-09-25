#include "GPUKMeans.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>

// ######################################################################
struct ClusterCompBinary :
  public thrust::binary_function<const ClassSummary&, const ClassSummary&, ClassSummary>
{
  size_t const clusterIdx;
  ClusterCompBinary(size_t const clusterIdx_) : clusterIdx(clusterIdx_) {}

  __host__ __device__
    ClassSummary operator()(const ClassSummary & p1, const ClassSummary & p2) const
    {
      ClassSummary result;

      if(p1.id == clusterIdx && p2.id == clusterIdx)
      {
        result.n = p1.n + p2.n;
        result.id = clusterIdx;
        result.mean_x = p1.mean_x + (p2.mean_x - p1.mean_x) * p2.n / result.n;
        result.mean_y = p1.mean_y + (p2.mean_y - p1.mean_y) * p2.n / result.n;
        result.mean_r = p1.mean_r + (p2.mean_r - p1.mean_r) * p2.n / result.n;
        result.mean_g = p1.mean_g + (p2.mean_g - p1.mean_g) * p2.n / result.n;
        result.mean_b = p1.mean_b + (p2.mean_b - p1.mean_b) * p2.n / result.n;
      }
      else if(p1.id == clusterIdx)
      {
        result.n = p1.n;
        result.id = p1.id;
        result.mean_x = p1.mean_x;
        result.mean_y = p1.mean_y;
        result.mean_r = p1.mean_r;
        result.mean_g = p1.mean_g;
        result.mean_b = p1.mean_b;
      }
      else if(p2.id == clusterIdx)
      {
        result.n = p2.n;
        result.id = p2.id;
        result.mean_x = p2.mean_x;
        result.mean_y = p2.mean_y;
        result.mean_r = p2.mean_r;
        result.mean_g = p2.mean_g;
        result.mean_b = p2.mean_b;
      }
      else
      {
        result.id = p1.id;
      }

      return result;
    }


};

// ######################################################################
struct ClusterCompUnary
{
  __device__
    ClassSummary operator()(const PointDescriptor& point) const
    {
      ClassSummary result;

      result.n = 1;
      result.id = point.classid;
      result.mean_x = point.x;
      result.mean_y = point.y;
      result.mean_r = point.r;
      result.mean_g = point.g;
      result.mean_b = point.b;

      return result;
    }
};

// ######################################################################
struct MinCluster
{
  __host__ __device__
    void operator()(thrust::tuple<PointDescriptor, float, ClassSummary> t)
    {
      PointDescriptor const & p = thrust::get<0>(t);

      float dist = sqrt(pow(thrust::get<2>(t).mean_x - p.x, 2) +
                        pow(thrust::get<2>(t).mean_y - p.y, 2) +
                        pow(thrust::get<2>(t).mean_r - p.r, 2) +
                        pow(thrust::get<2>(t).mean_g - p.g, 2) +
                        pow(thrust::get<2>(t).mean_b - p.b, 2));

      if(dist < thrust::get<1>(t))
      {
        thrust::get<1>(t) = dist;
        thrust::get<0>(t).classid = thrust::get<2>(t).id;
      }
    }
};

// ######################################################################
std::vector<ClassSummary> kmeans(std::vector<PointDescriptor> & points, size_t const k)
{
  // Randomly assign classes to each point descriptor
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, k-1);
  std::vector<PointDescriptor>::iterator pointsit;
  for(pointsit = points.begin(); pointsit != points.end(); ++pointsit)
    pointsit->classid = dist(rng);

  // Create a device copy of the point descriptors
  thrust::device_vector<PointDescriptor> device_points(points.size());
  thrust::copy(points.begin(), points.end(), device_points.begin());

  // Create a device copy of the class means
  thrust::device_vector<ClassSummary> device_class_means(k);

  for(int iteration=0; iteration<10; ++iteration)
  {
    std::cout << "Iteration " << iteration << " --------------------------------" << std::endl;

    //Compute class means (cluster centers)
    for(size_t clusterIdx=0; clusterIdx<k; ++clusterIdx)
    {
      ClassSummary init;
      init.initialize();

      // Compute the mean for this cluster
      device_class_means[clusterIdx] =
        thrust::transform_reduce(device_points.begin(), device_points.end(), ClusterCompUnary(), init, ClusterCompBinary(clusterIdx));

      ClassSummary summary = device_class_means[clusterIdx];
      std::cout << "Cluster " << clusterIdx << " n: " << summary.n << 
        " x: " <<summary.mean_x << " y: " << summary.mean_y << " n: " << summary.n << 
        " r: " << summary.mean_r << " g: " << summary.mean_g << " b: " << summary.mean_b << std::endl;
    }

    // Assign each point to its new cluster
    thrust::device_vector<float>  distances(device_points.size(), std::numeric_limits<float>::max());
    for(size_t clusterIdx=0; clusterIdx<k; ++clusterIdx)
    {
      thrust::device_vector<float> clusterDistances(device_points.size());
      thrust::constant_iterator<ClassSummary> currentCluster(device_class_means[clusterIdx]);

      thrust::for_each(
          thrust::make_zip_iterator(thrust::make_tuple(device_points.begin(), distances.begin(), currentCluster)),
          thrust::make_zip_iterator(thrust::make_tuple(device_points.end(),   distances.end(),   currentCluster)),
          MinCluster());
    }
  }

  thrust::host_vector<ClassSummary> host_class_means(device_class_means.size());
  thrust::copy(device_class_means.begin(), device_class_means.end(), host_class_means.begin());
  std::vector<ClassSummary> class_means(host_class_means.size());
  std::copy(host_class_means.begin(), host_class_means.end(), class_means.begin());
  return class_means;
}

