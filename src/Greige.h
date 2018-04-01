#ifndef GREIGE_H
#define GREIGE_H

// This class provides the x/y/z coordinates of a rectilinear grid
class Greige{
  private:
    double x0, y0, z0;
    double dx, dy, dz;
  public:
    Greige() : x0(0.0), y0(0.0), z0(0.0), dx(1.0), dy(1.0), dz(1.0) {};
    Greige(double x0_, double y0_, double z0_, double dx_, double dy_, double dz_) :
            x0(x0_), y0(y0_), z0(z0_), dx(dx_), dy(dy_), dz(dz_) {};
    double x(const double ix) const {return x0+ix*dx;}
    double y(const double jy) const {return y0+jy*dy;}
    double z(const double kz) const {return z0+kz*dz;}
};

#endif
