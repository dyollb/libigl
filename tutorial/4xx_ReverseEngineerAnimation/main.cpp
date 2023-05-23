#include <igl/boundary_conditions.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/normalize_row_sums.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/readTGF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/bbw.h>
#include <igl/writeDMAT.h>
#include <igl/writeOFF.h>
#include <igl/remove_unreferenced.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>


using AffineList = std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>;
using RotationList = std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >;


void relative_transforms(
	const Eigen::VectorXi& P,
	const RotationList& vQ,
	RotationList& dQ)
{
	const auto m = P.size();
	std::vector<bool> computed(m, false);
	dQ.resize(m);

	while (std::count(computed.begin(), computed.end(), false) != 0)
	{
		for (Eigen::Index b = 0; b < m; b++)
		{
			if (!computed[b])
			{
				const int p = P(b);
				if (p < 0)
				{
					// FK: vQ[b] = dQ[b];
					dQ[b] = vQ[b];
					computed[b] = true;
				}
				else if (computed[p])
				{
					// FK: vQ[b] = vQ[p] * dQ[b];
					dQ[b] = vQ[p].inverse() * vQ[b];
					computed[b] = true;
				}
			}
		}
	}
}

void relative_transforms(
	const Eigen::VectorXi& P,
	const AffineList& vA,
	RotationList& dQ)
{
	RotationList vQ(vA.size());
	for (size_t i = 0; i < vA.size(); i++)
		vQ[i] = vA[i].rotation();

	relative_transforms(P, vQ, dQ);
}


int main(int argc, char* argv[])
{
	using namespace Eigen;
	using namespace std;

	MatrixXd Cin, Cout, Tin, Tout;
	MatrixXi BEin, BEout;
	VectorXi P;
	RotationList pose;

	igl::readTGF("F:/Data/_DirectDeltaMush/elephant/elephant.tgf", Cin, BEin);
	igl::readDMAT("F:/Data/_DirectDeltaMush/elephant/elephant-anim.dmat", Tin);

	igl::readTGF("F:/Data/_DirectDeltaMush/fats-muscle/fats.tgf", Cout, BEout);

	// retrieve parents for forward kinematics
	igl::directed_edge_parents(BEin, P);

	Tout.resize(Tin.rows(), Tin.cols());

	for (Index t = 0; t < Tin.cols(); ++t)
	{
		const int dim = Cin.cols();

		for (int frame = 0; frame < Tin.cols(); frame++)
		{
			const Map<MatrixXd> Tf(Tin.data() + frame * Tin.rows(), 4 * BEin.rows(), 3);

			AffineList T_list(BEin.rows());
			for (int e = 0; e < BEin.rows(); e++)
			{
				T_list[e] = Affine3d::Identity();
				T_list[e].matrix().block(0, 0, 3, 4) = Tf.block(e * 4, 0, 4, 3).transpose();
			}

			AffineList T_list_out(T_list);

			RotationList pose;
			relative_transforms(P, T_list, pose);

			RotationList vQ;
			vector<Vector3d> vT;
			igl::forward_kinematics(Cout, BEout, P, pose, vQ, vT);

			MatrixXd T(BEin.rows() * (dim + 1), dim);
			for (int e = 0; e < BEout.rows(); e++)
			{
				Affine3d a = Affine3d::Identity();
				a.translate(vT[e]);
				a.rotate(vQ[e]);
				T.block(e * (dim + 1), 0, dim + 1, dim) =
					a.matrix().transpose().block(0, 0, dim + 1, dim);
			}
			VectorXd c = T.reshaped();
			Tout.col(frame) = c;
		}
	}

	igl::writeDMAT("F:/Data/_DirectDeltaMush/fats-muscle/fats-anim.dmat", Tout);

	return EXIT_SUCCESS;
}
