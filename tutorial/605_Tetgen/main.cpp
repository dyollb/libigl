#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/readTGF.h>
#include <igl/writeMESH.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/matrix_to_list.h>
#include <igl/list_to_matrix.h>


// Input polygon
Eigen::MatrixXd V, VA;
Eigen::MatrixXi F;
Eigen::MatrixXd B;

// Tetrahedralized interior
Eigen::MatrixXd TV;
Eigen::MatrixXi TT;
Eigen::MatrixXi TF;

Eigen::MatrixXd C;
Eigen::MatrixXi BE;

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  using namespace std;
  using namespace Eigen;

  if (key >= '1' && key <= '9')
  {
    double t = double((key - '1')+1) / 9.0;

    VectorXd v = B.col(2).array() - B.col(2).minCoeff();
    v /= v.col(0).maxCoeff();

    vector<int> s;

    for (unsigned i=0; i<v.size();++i)
      if (v(i) < t)
        s.push_back(i);

    MatrixXd V_temp(s.size()*4,3);
    MatrixXi F_temp(s.size()*4,3);

    for (unsigned i=0; i<s.size();++i)
    {
      V_temp.row(i*4+0) = TV.row(TT(s[i],0));
      V_temp.row(i*4+1) = TV.row(TT(s[i],1));
      V_temp.row(i*4+2) = TV.row(TT(s[i],2));
      V_temp.row(i*4+3) = TV.row(TT(s[i],3));
      F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
      F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
      F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
      F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
    }

    viewer.data().clear();
    viewer.data().set_mesh(V_temp,F_temp);
    viewer.data().set_face_based(true);
  }


  return false;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a surface mesh
  igl::readOBJ("F:/Data/_DirectDeltaMush/ella/ella.obj",V,F);
  igl::readTGF("F:/Data/_DirectDeltaMush/ella/ella.tgf", C, BE);

  VA.resize(V.rows() + C.rows(), V.cols());
  VA << V, C;

  std::vector<std::vector<double> > vV;
  std::vector<std::vector<int> > vF;
  std::vector<std::vector<int> > vBE;
  igl::matrix_to_list(VA, vV);
  igl::matrix_to_list(F, vF);

  //MatrixXi I; I.resize(C.rows(), 1);
  //for (Index i = 0; i < C.rows(); ++i)
  //    I(i) = V.rows() + i;
  BE += V.rows() * MatrixXi::Ones(BE.rows(), BE.cols());
  igl::matrix_to_list(BE, vBE);
  vF.insert(vF.end(), vBE.begin(), vBE.end());


  std::vector<std::vector<double> > vTV;
  std::vector<std::vector<int> > vTT;
  std::vector<std::vector<int> > vTF;

  // Tetrahedralize the interior
  igl::copyleft::tetgen::tetrahedralize(vV,vF,"pq1.6", vTV,vTT,vTF);

  igl::list_to_matrix(vTV, TV);
  igl::list_to_matrix(vTT, TT);
  igl::list_to_matrix(vTF, TF);

  for (int p = 0; p < C.rows(); p++)
  {
	  VectorXd pos = C.row(p);
	  // loop over domain vertices

	  for (int i = 0; i < TV.rows(); i++)
	  {
		  VectorXd vi = TV.row(i);
		  double sqrd = (vi - pos).squaredNorm();
		  if (sqrd <= 1e-8)
		  {
              std::cout << p << ": " << sqrd << "\n";
		  }
	  }
  }


  igl::writeMESH("F:/Data/_DirectDeltaMush/ella/ella.mesh", TV, TT, TF);

  // Compute barycenters
  igl::barycenter(TV,TT,B);

  // Plot the generated mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.callback_key_down = &key_down;
  key_down(viewer,'5',0);
  viewer.launch();
}
