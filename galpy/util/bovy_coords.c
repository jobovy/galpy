#include <math.h>
#include <bovy_coords.h>
void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
}
