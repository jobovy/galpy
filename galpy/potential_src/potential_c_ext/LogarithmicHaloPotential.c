
double Rforce(double R,double z,double *args, int nargs){
  //Get args
  double q= *args;
  args++;
  double c= *args;
  //Calculate Rforce
  double zq= z/q;
  return -R/(R*R+zq*zq+c);
}
double zforce(double R,double z,double *args, int nargs){
  //Get args
  double q= *args;
  args++;
  double c= *args;
  //Calculate zforce
  double zq= z/q;
  return -z/q/q/(R*R+zq*zq+c);
}
