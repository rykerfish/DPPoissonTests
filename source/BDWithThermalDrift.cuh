/*Raul P. Pelaez 2021. A modification of the Leimkuhler Integrator in UAMMD.
the LeimkuhlerMobility Integrator allows to set a self mobility dependent on the height, taking into acocunt the thermal drift that arises in this case.

The specific self mobility is read and interpolated from a file.

This file must have two columns with a list of normalized heights (so Z must go from -1 to 1) and normalized mobilities (i.e. 6*pi*eta*a*M0) in X, Y and Z. The values for each particle will be linearly interpolated from the data provided in the file. The order of the values does not matter. Example:
--- mobility.dat---
-1.0 1.0 1.0 1.0
 0.0 1.0 1.0 1.0
 1.0 1.0 1.0 1.0
-------------------
The above example is equivalent to calling uammd's Leimkuhler integrator.

Small modifications to the SelfMobility class can be made (in particular to its () operator) to make the self mobility vary also in X and/or Y.


*/
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"misc/TabulatedFunction.cuh"
#include"external/spline.h"
#include <cmath>
#include <fstream>
using namespace uammd;

//The () operator of this struct must return the normalized self mobility and its derivative when given a position,
// i.e returning 6*pi*eta*a*{M0(x,y,z), \nabla M0(x,y,z)}
class SelfMobility{

  //Reads the height vs the three self mobilities from the file
  auto readMobilityFile(std::string fileName){
    std::ifstream in(fileName);
    std::istream_iterator<double4> begin(in), end;
    std::vector<double4> data{begin, end};
    //The spline library needs input data in ascending order
    std::sort(data.begin(), data.end(),[](double4 a, double4 b){return a.x<b.x;});
    std::vector<double> Z(data.size());
    auto Mx = Z;
    auto My = Mx;
    auto Mz = My;
    std::transform(data.begin(), data.end(), Z.begin(), [](double4 a){return a.x;});
    std::transform(data.begin(), data.end(), Mx.begin(), [](double4 a){return a.y;});
    std::transform(data.begin(), data.end(), My.begin(), [](double4 a){return a.z;});
    std::transform(data.begin(), data.end(), Mz.begin(), [](double4 a){return a.w;});
    return std::make_tuple(Z,Mx,My,Mz);
  }

  //Returns a functor that returns the derivative of "y" at any point via spline interpolation
  template<class Functor> 
  auto computeDerivative(Functor y, int N){
    std::vector<double> derivative(N,0);
    auto x = derivative;
    double h = 2.0/(N-1);
    x[0] = -1;
    derivative[0] = (y(-1+h)-y(-1))/h;
    for(int i = 1; i<N-1; i++){
      real z = -1+h*i;
      x[i] = z;
      derivative[i] = (y(z+h)-y(z-h))/(2.0*h);
    }
    x[N-1] = 1;
    derivative[N-1] = (y(1)-y(1-h))/h;
    tk::spline sy;
    sy.set_points(x, derivative);
    return sy;
  }

  TabulatedFunction<real4> mobilityAndDerivative;
  real Lz;
public:

  SelfMobility(std::string fileName, real Lz):Lz(Lz){
    System::log<System::MESSAGE>("[SelfMobility] Initialized, Lz=%g", Lz);
    auto data = this->readMobilityFile(fileName);
    tk::spline mobilityx, mobilityy, mobilityz;
    mobilityx.set_points(std::get<0>(data), std::get<1>(data));
    mobilityy.set_points(std::get<0>(data), std::get<2>(data));
    mobilityz.set_points(std::get<0>(data), std::get<3>(data));
    const real hdiff = 1e-3;
    const int ntablePoints = std::max(int(Lz/hdiff), 1<<20);
    //const int derivativePoints = std::max(int(Lz/hdiff), ntablePoints);
    //tk::spline derivative = this->computeDerivative(mobilityz, derivativePoints);
    auto allM =[&](real z){return make_real4(mobilityx(z),mobilityy(z),mobilityz(z), mobilityz.deriv(1, z));};
    // std::ofstream out("mob.test");
    // int nsam = 1000;
    // fori(0, nsam){
    //   real z = -1 + i*2.0/nsam;
    //   auto mob = allM(z);
    //   out<<z<<" "<<mob<<std::endl;
    // }
    mobilityAndDerivative = TabulatedFunction<real4>(ntablePoints, -1, 1, allM);
  }

  //The position received by this function will be folded to the main cell.
  __device__ auto operator()(real3 pos){
    real z = (pos.z - floor(pos.z/Lz + real(0.5))*Lz)/Lz; //Height in the [-0.5,0.5] interval
    const auto MandDiffM = mobilityAndDerivative(real(2.0)*z);
    real3 M = make_real3(MandDiffM);
    real diffMz = MandDiffM.w;
    // const real mz = 2+sin(4*M_PI*z);
    // const real dmz = 2*M_PI*cos(4*M_PI*z);
    // M = M*0+mz;
    // diffMz = dmz;
    return thrust::make_pair(M, make_real3(0, 0, real(2.0)*diffMz/Lz));
  }

};

namespace BDWithThermalDrift_ns{
  enum class update_rules{leimkuhler, euler_maruyama};
}
//Implements the algorithm in eq. 45 of [2]. Allows for a position-dependent mobility
class BDWithThermalDrift: public BD::BaseBrownianIntegrator{
  std::shared_ptr<SelfMobility> selfMobilityFactor;
  thrust::device_vector<real3> noisePrevious;
  BDWithThermalDrift_ns::update_rules brownian_rule;
public:
  
  BDWithThermalDrift(shared_ptr<ParticleData> pd,
		     Parameters par,
		     std::string BrownianUpdateRule,
		     std::shared_ptr<SelfMobility> selfMobilityFactor):
    BaseBrownianIntegrator(pd, par),
    selfMobilityFactor(selfMobilityFactor){
    this->seed = sys->rng().next32();
    this->steps = 0;
    if(BrownianUpdateRule == "EulerMaruyama"){
      this->brownian_rule = BDWithThermalDrift_ns::update_rules::euler_maruyama;
    }
    else if(BrownianUpdateRule == "Leimkuhler"){
      this->brownian_rule = BDWithThermalDrift_ns::update_rules::leimkuhler;
    }
    else{
      throw std::runtime_error("[BDWithThermalDrift] Invalid update rule, only EulerMaruyama or Leimkuhler are available");
    }
    sys->log<System::MESSAGE>("[BDWithThermalDrift] Initialized with seed %u in %s mode", this->seed, BrownianUpdateRule.c_str());
  }

  void forwardTime() override;

private:
  void updatePositions();
};


namespace BDWithThermalDrift_ns{
  __device__ real3 genNoise(int i, uint stepNum, uint seed){
    Saru rng(i, stepNum, seed);
    return make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
  }
  
  //This integration scheme allows for a self mobility depending on the position.
  //With the associated non-zero thermal drift term.
  template<update_rules rule>
  __global__ void integrateGPU(real4* pos,
			       ParticleGroup::IndexIterator indexIterator,
			       const int* originalIndex,
			       const real4* force,
			       real selfMobility,
			       SelfMobility selfMobilityFactor,
			       real3* noisePrevious,
			       real* radius,
			       real dt,
			       real temperature,
			       int N,
			       uint stepNum, uint seed){
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    int i = indexIterator[id];
    real3 R = make_real3(pos[i]);
    const real3 F = make_real3(force[i]);
    const auto factor = selfMobilityFactor(R);
    const real m0 = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
    const real3 M = m0*factor.first;
    R += dt*(M*F);
    if(temperature > 0){
      int ori = originalIndex[i];
      const auto Bn = sqrt(real(2.0)*temperature*M*dt);
      const auto dWn = genNoise(ori, stepNum, seed);      
      if(rule == update_rules::euler_maruyama){
	R += Bn*dWn + temperature*dt*m0*factor.second;
      }
      else if(rule ==update_rules::leimkuhler){	
	R += real(0.5)*(Bn*dWn+noisePrevious[ori]) + temperature*dt*m0*factor.second;
	noisePrevious[ori] = Bn*dWn;
      }
    }
    pos[i].x = R.x;
    pos[i].y = R.y;
    pos[i].z = R.z;
  }

}

void BDWithThermalDrift::forwardTime(){
  steps++;
  sys->log<System::DEBUG1>("[BD::Leimkuhler] Performing integration step %d", steps);
  updateInteractors();
  computeCurrentForces();
  updatePositions();
}

void BDWithThermalDrift::updatePositions(){
  int numberParticles = pg->getNumberParticles();
  noisePrevious.resize(numberParticles);
  if(steps==1)
    thrust::fill(noisePrevious.begin(), noisePrevious.end(), real3());
  int BLOCKSIZE = 128;
  uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
  uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
  real * d_radius = getParticleRadiusIfAvailable();
  auto groupIterator = pg->getIndexIterator(access::location::gpu);
  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
  auto force = pd->getForce(access::location::gpu, access::mode::read);
  auto originalIndex = pd->getIdOrderedIndices(access::location::gpu);
  using namespace BDWithThermalDrift_ns;
  auto foo =  integrateGPU<update_rules::euler_maruyama>;
  if(brownian_rule == update_rules::leimkuhler){
    foo =  integrateGPU<update_rules::leimkuhler>;
  }
  foo<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
				    groupIterator,
				    originalIndex,
				    force.raw(),
				    selfMobility,
				    *selfMobilityFactor,
				    thrust::raw_pointer_cast(noisePrevious.data()),
				    d_radius,
				    dt,
				    temperature,
				    numberParticles,
				    steps, seed);
}
