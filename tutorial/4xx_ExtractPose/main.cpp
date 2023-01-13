#include <igl/bbw.h>
#include <igl/boundary_conditions.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/deform_skeleton.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/normalize_row_sums.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/readTGF.h>
#include <igl/remove_unreferenced.h>
#include <igl/writeDMAT.h>
#include <igl/writeOFF.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

using AffineList =
    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>;
using RotationList = std::vector<Eigen::Quaterniond,
                                 Eigen::aligned_allocator<Eigen::Quaterniond>>;

void gram_schmidt(const Eigen::Vector3d &v, Eigen::Vector3d &a, Eigen::Vector3d &b) {
  if (std::abs(v(0)) < 0.75) {
    // cross product between this and x-axis
    a(0) = 0.0;
    a(1) = v(2);
    a(2) = -v(1);
  } else {
    // cross product between this and y-axis
    a(0) = -v(2);
    a(1) = 0.0;
    a(2) = v(0);
  }
  a.normalize();
  b = v.cross(a);
}

Eigen::AngleAxisd rotation(Eigen::Vector3d from, Eigen::Vector3d to) {
  from.normalize();
  to.normalize();
  Eigen::Vector3d n = from.cross(to);
  if (n.norm() > 1e-6) {
    n.normalize();
    double cos_angle = from.dot(to);
    if (cos_angle > 1.0)
      cos_angle = 1.0;
    else if (cos_angle < -1.0)
      cos_angle = -1.0;
    const double angle = std::acos(cos_angle);
    return Eigen::AngleAxisd(angle, n);
  }
  if (to.dot(from) < 0.0) {
    Eigen::Vector3d tmp;
    gram_schmidt(to, n, tmp);
    constexpr double pi = 3.14159265358979323846;
    return Eigen::AngleAxisd(pi, n);
  }
  return Eigen::AngleAxisd::Identity();
}

void relative_transforms(const Eigen::VectorXi &P, const RotationList &vQ,
                         RotationList &dQ) {
  const auto m = P.size();
  std::vector<bool> computed(m, false);
  dQ.resize(m);

  while (std::count(computed.begin(), computed.end(), false) != 0) {
    for (Eigen::Index b = 0; b < m; b++) {
      if (!computed[b]) {
        const int p = P(b);
        if (p < 0) {
          // FK: vQ[b] = dQ[b];
          dQ[b] = vQ[b];
          computed[b] = true;
        } else if (computed[p]) {
          // FK: vQ[b] = vQ[p] * dQ[b];
          dQ[b] = vQ[p].inverse() * vQ[b];
          computed[b] = true;
        }
      }
    }
  }
}

void relative_transforms(const Eigen::VectorXi &P, const AffineList &vA,
                         RotationList &dQ) {
  RotationList vQ(vA.size());
  for (size_t i = 0; i < vA.size(); i++)
    vQ[i] = vA[i].rotation();

  relative_transforms(P, vQ, dQ);
}

int main(int argc, char *argv[]) {
  using namespace Eigen;
  using namespace std;
  namespace fs = std::filesystem;

  MatrixXd C0, Ci;
  MatrixXi BE, BEtmp;
  VectorXi P;
  RotationList pose;

  fs::path pose_dir("F:/Dropbox/Work/Data/Fats_obese_dancing/pooh");
  igl::readTGF("F:/Dropbox/Work/Data/Fats_obese_dancing/fats.tgf", C0, BE);

  // retrieve parents for forward kinematics
  igl::directed_edge_parents(BE, P);

  std::vector<std::string> frames = { "F:/Dropbox/Work/Data/Fats_obese_dancing/fats.tgf" };
  for (const auto& f : fs::directory_iterator(pose_dir)) {
      if (fs::is_regular_file(f))
      {
          frames.push_back(f.path().string());
      }
  }
  std::sort(frames.begin(), frames.end());

  const int dim = C0.cols();
  const int num_steps = 40;
  const int num_frames = frames.size();
  MatrixXd Tout(BE.rows() * (dim + 1) * dim, num_frames * num_steps);
  RotationList pose_last(BE.rows());
  for (auto& p : pose_last)
      p.setIdentity();

  for (int frame = 0; frame < num_frames; ++frame)
  {
      igl::readTGF(frames[frame], Ci, BEtmp);

      AffineList rots(BE.rows());
      for (Index i = 0; i < BE.rows(); ++i) 
      {
        Vector3d l0 = (C0.row(BE(i, 1)) - C0.row(BE(i, 0))).transpose();
        Vector3d li = (Ci.row(BE(i, 1)) - Ci.row(BE(i, 0))).transpose();
        rots[i] = rotation(l0, li);
      }

	  RotationList pose;
	  relative_transforms(P, rots, pose);

      for (int step = 0; step < num_steps; ++step)
      {
          const double anim_t = step * (1.0 / num_steps);
		  RotationList anim_pose(pose.size());
		  for (int e = 0; e < pose.size(); e++)
		  {
			  anim_pose[e] = pose[e].slerp(1.0-anim_t, pose_last[e]);
		  }

		  RotationList vQ;
		  vector<Vector3d> vT;
		  igl::forward_kinematics(C0, BE, P, anim_pose, vQ, vT);

		  MatrixXd T(BE.rows() * (dim + 1), dim);
		  for (int e = 0; e < BE.rows(); e++)
          {
			  Affine3d a = Affine3d::Identity();
			  a.translate(vT[e]);
			  a.rotate(vQ[e]);
			  T.block(e * (dim + 1), 0, dim + 1, dim) =
				  a.matrix().transpose().block(0, 0, dim + 1, dim);
		  }
		  VectorXd c = T.reshaped();
		  Tout.col(frame * num_steps + step) = c;
      }

      pose_last = pose;
  }

  igl::writeDMAT("F:/Dropbox/Work/Data/Fats_obese_dancing/fats-anim-pooh.dmat", Tout);

  return EXIT_SUCCESS;
}
