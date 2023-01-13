#include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/normalize_row_sums.h>
#include <igl/direct_delta_mush.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Geometry>
#include <vector>

int main(int argc, char* argv[])
{
	struct ModelData
	{
		Eigen::MatrixXd V, U, T, M;
		Eigen::MatrixXi F;
		Eigen::RowVector3d offset;
		Eigen::Index num_handles = 0;
		size_t id = -1;
	};

	std::vector<std::string> paths = {
		"F:/Data/_DirectDeltaMush/ella/ella",
		"F:/Data/_DirectDeltaMush/fats/fats",
		"F:/Data/_DirectDeltaMush/fats-muscle/fats",
		"F:/Data/_DirectDeltaMush/ella-base/ella",
		"F:/Data/_DirectDeltaMush/fats-base/fats",
	};

	// load & pre-compute
	std::vector<ModelData> models;
	for (const auto& prefix : paths)
	{
		auto& m = models.emplace_back();
		Eigen::MatrixXi T;
		Eigen::MatrixXd W;

		igl::readMESH(prefix + ".mesh", m.V, T, m.F);
		igl::readDMAT(prefix + "-weights.dmat", W);
		igl::readDMAT(prefix + "-anim.dmat", m.T);

		const Eigen::MatrixXi _F2 = m.F.col(2);
		m.F.col(2) = m.F.col(1);
		m.F.col(1) = _F2;

		igl::normalize_row_sums(W, W);
		igl::lbs_matrix(m.V, W, m.M);

		m.num_handles = W.cols();
		m.offset << 0, 0, 0;
	}

	// setup layout
	const auto& m1 = models.front();
	Eigen::RowVector3d offset_x(1.1 * (m1.V.col(0).maxCoeff() - m1.V.col(0).minCoeff()), 0, 0);
	Eigen::RowVector3d offset_z(0, 0, 1.1 * (m1.V.col(2).maxCoeff() - m1.V.col(2).minCoeff()));
	models[1].offset -= offset_x;
	models[2].offset += offset_x;
	models[3].offset += 0.5 * offset_x + offset_z;
	models[4].offset += 1.5 * offset_x + offset_z;

	igl::opengl::glfw::Viewer viewer;
	viewer.core().background_color << 0.2f, 0.2f, 0.2f, 1.0f;
	models[0].id = viewer.selected_data_index;
	for (size_t i = 1; i < models.size(); ++i)
	{
		viewer.append_mesh();
		models[i].id = viewer.selected_data_index;
	}

	const int num_frames = m1.T.cols();
	int frame = 0;
	viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool
	{
		if (viewer.core().is_animating)
		{
			for (auto& m : models)
			{
				const Eigen::Map<Eigen::MatrixXd> Tf(
					m.T.data() + frame * m.T.rows(), 4 * m.num_handles, 3);
				m.U = (m.M * Tf).rowwise() - m.offset;

				viewer.data(m.id).set_vertices(m.U);
				viewer.data(m.id).compute_normals();
			}

			frame++;
			if (frame == num_frames)
			{
				frame = 0;
				viewer.core().is_animating = false;
			}
		}
		return false;
	};
	viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int mod)
	{
		switch (key)
		{
		case ' ':
			viewer.core().is_animating = !viewer.core().is_animating;
			return true;
		}
		return false;
	};

	for (auto& m : models)
	{
		viewer.data(m.id).set_mesh((m.V.rowwise() - m.offset).eval(), m.F);
		viewer.data(m.id).set_colors(Eigen::RowVector3d(214. / 255., 170. / 255., 148. / 255.));

		viewer.data(m.id).show_lines = false;
		viewer.data(m.id).set_face_based(true);
		viewer.data(m.id).show_overlay_depth = false;
	}
	viewer.core().is_animating = false;
	viewer.core().animation_max_fps = 24.;


	viewer.launch_init();
	viewer.core().align_camera_center(m1.V);

	viewer.launch_rendering(true);
	viewer.launch_shut();
}
