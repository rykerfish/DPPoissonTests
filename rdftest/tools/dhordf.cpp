#include<cmath>
#include<iostream>
#include<iomanip>

using namespace std;
using scalar = double;
int main(int argc, char*argv[]){
  
  scalar r;
  scalar sign=std::stod(argv[1]);
  scalar u0=std::stod(argv[2]);
  scalar perm=std::stod(argv[3]);
  scalar r_m=std::stod(argv[4]);
  scalar p=std::stod(argv[5]);
  scalar gw=std::stod(argv[6]);
  scalar pi=M_PI;
  scalar dl=std::stod(argv[7]);
  scalar sigma=std::stod(argv[8]);
  int pot = std::atoi(argv[9]); //LJ:0, HS:1
    while(cin>>r){
      if(r<=0.1){
	std::cout<<"0 0"<<std::endl;
	continue;
      }
      scalar Usteric;
      if(pot==0){
	scalar uljclose = -4*u0*(pow(sigma/r_m,p) - pow(sigma/r_m, 2*p) -0.25) +
	  4*u0*p*(pow(sigma/r_m, p)/r_m - 2*pow(sigma/r_m, 2*p)/r_m)*(r-r_m);
	scalar lj = (((4.0*u0*(pow(sigma/r,2.0*p) - pow(sigma/r,p))+u0))*(r>r_m) + uljclose*(r<=r_m))*(r<(sigma*pow(2,1.0/p)));
	Usteric = lj;
      }
      else {
        Usteric = (r<sigma)?(1e30):(0);
      }
      scalar ugaus = sign*erf(0.5*r/(gw))/(4*pi*perm*r);
      scalar u_yuka = ugaus*exp(-r/dl);
      scalar g_dho=exp(-Usteric-u_yuka);
      std::cout<<setprecision(2*sizeof(scalar))<<r<<" "<<g_dho<<std::endl;
    }
  return 0;
}
