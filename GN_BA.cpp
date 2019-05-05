#include<Eigen/Core>
#include<Eigen/Dense>

using namespace Eigen;

#include<vector>
#include<fstream>
#include<iostream>
#include<iomanip>

#include"/home/sunmb/lib/Sophus/sophus/se3.h"

using namespace std;
using namespace Sophus;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d> > VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d> > VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";

ifstream ifp3d, ifp2d;
string sp3d, sp2d;

int main(int argc, char **argv)
{
    VecVector3d p3d;
    VecVector2d p2d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    ifp3d.open(p3d_file.c_str());
    if(!ifp3d.is_open())
    {
        cerr<<"p3d_file is not open! p3d_file: "<<p3d_file.c_str()<<endl;
        return -1;
    }
    ifp2d.open(p2d_file.c_str());
    if(!ifp2d.is_open())
    {
        cerr<<"p2d_file is not open! p2d_file: "<<p2d_file.c_str()<<endl;
        return -1;
    }

    while(getline(ifp3d,sp3d) && !sp3d.empty())
    {
        istringstream issP3d(sp3d);
        Vector3d vp3d;
        issP3d >> vp3d[0] >> vp3d[1] >> vp3d[2];
        p3d.push_back(vp3d);
    }

    while (getline(ifp2d,sp2d) && !sp2d.empty())
    {
        istringstream issp2d(sp2d);
        Vector2d vp2d;
        issp2d >> vp2d[0] >> vp2d[1];
        p2d.push_back(vp2d);
    }

    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastcost = 0;
    int npoints = p3d.size();

    Matrix3d R = Matrix3d::Identity();
    Vector3d t = Vector3d::Zero();
    Sophus::SE3 T(R, t);
    for(int iter = 0; iter < iterations; iter++)
    {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        Vector2d e;
        cost = 0;
        for(int i = 0; i< npoints; i++)
        {
            Vector3d Pc = T * p3d[i];
            Vector3d erro = Vector3d(p2d[i][0], p2d[i][1], 1) - K * Pc / Pc[2];
            e[0] = erro[0];
            e[1] = erro[1];
            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2];
            double x2 = x * x;
            double y2 = y * y;
            double z2 = z * z;

            Matrix<double, 2, 6> J = Matrix<double, 2, 6>::Zero();
            J(0,0) = -fx / z;
            J(0,2) = fx * x / z2;
            J(0,3) = fx * x * y / z2;
            J(0,4) = -fx - fy * x2 / z2;
            J(0,5) = fx * y / z;
            J(1,1) = -fy / z;
            J(1,2) = fy * y / z2;
            J(1,3) = fy + fy * y2 / z2;
            J(1,4) = -fy * x * y / z2;
            J(1,5) = -fy * x / z;

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += 0.5 * e.transpose() * e;
        }
        Vector6d dx;
        dx = H.ldlt().solve(b);
        cout<<"iter: " << iter <<"dx: "<<dx.transpose()<<endl;

        if(isnan(dx[0]))
        {
            cerr<<"result is nan" <<endl;
            break;
        }

        if(iter > 0 && cost >= lastcost)
        {
            cerr << "cost:" << cost << ", last cost: " << lastcost <<endl;
            break;
        }

        T = Sophus::SE3::exp(dx) * T;

        lastcost = cost;

        cout<<"iteration "<< iter << "cost= " << cout.precision(12) << cost << endl;

    }
    cout << "estimated pose : \n" << T.matrix() <<endl;
    return 0;
}
