#include "HPR.h"

using namespace Eigen;
using namespace pcl;

void HPR(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in, std::vector<float> camera_pos, int param, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out) // Hidden Point Removal
{
	int dim = 3;
	int numPts = cloud_in->size();

	ArrayXXd pts(numPts, dim);
	for (int i = 0; i < cloud_in->size(); ++i)
	{
		pts(i, 0) = pow(cloud_in->at(i).x - camera_pos[0], 2);
		pts(i, 1) = pow(cloud_in->at(i).y - camera_pos[1], 2);
		pts(i, 2) = pow(cloud_in->at(i).z - camera_pos[2], 2);
	}

	// ArrayXd C(3);
	// C << camera_pos[0], camera_pos[1], camera_pos[2];
	// p = p - C.transpose().replicate(numPts, 1);
	// ArrayXd normp = p.rowwise().sum().sqrt();
	double maxVal = -100000;
	// ArrayXd normp(numPts);
	std::vector<double> normp;
	for (int j = 0; j < numPts; j++)
	{
		normp.push_back(sqrt(pts(j, 0) + pts(j, 1) +pts(j, 2)));
		if (maxVal<normp[j]){
			maxVal = normp[j];
		}
	}
	// std::cout << maxVal << "compute normal ... !"<< normp.maxCoeff() << std::endl;
	// ArrayXd maxNormp(1);
	// maxNormp << normp.maxCoeff() * pow(10, param);
	// double val = maxVal * pow(10, param);
	// ArrayXd R = maxNormp.replicate(numPts, 1);
	// ArrayXd shit = (R - normp);
	// ArrayXXd P = p + 2 * shit.replicate(1, dim) * p / normp.replicate(1, dim);
	maxVal *= pow(10, param);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_P(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < pts.rows(); ++i)
	{
		double t = maxVal - normp[i];
		pcl::PointXYZ point;
		point.x = pts(i, 0) + 2 * t * pts(i, 0) / normp[i];
		point.y = pts(i, 1) + 2 * t * pts(i, 1) / normp[i];
		point.z = pts(i, 2) + 2 * t * pts(i, 2) / normp[i];
		cloud_P->push_back(point);
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<pcl::Vertices> polygons;
	pcl::ConvexHull<pcl::PointXYZ> chull;
	chull.setDimension(3);
	chull.setInputCloud(cloud_P);
	chull.reconstruct(*cloud_hull, polygons);

	pcl::PointIndices::Ptr indices(new pcl::PointIndices);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_P);

	for (int i = 0; i < cloud_hull->size(); ++i)
	{
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(cloud_hull->at(i), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			indices->indices.push_back(pointIdxNKNSearch[0]);
		}
	}

	// Extract the inliers
	pcl::ExtractIndices<PointXYZ> extract;
	extract.setInputCloud(cloud_in);
	extract.setIndices(indices);
	extract.setNegative(false);
	extract.filter(*cloud_out);
}