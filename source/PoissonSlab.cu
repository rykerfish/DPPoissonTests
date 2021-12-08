
/*Raul P. Pelaez 2020-2021. DP Poisson test

This file encodes the following simulation:

A group of Gaussian sources of charge with width gw evolve via Brownian Dynamics inside a doubly periodic domain of dimensions Lxy, Lxy, H (periodic in XY and open in Z). 
Optionally, a flag can be passed in the input to enable triply periodic electrostatics.

Sources interact via an electrostatic potential and repell each other via a repulsive LJ-like potential.

There are two (potentially) charged walls on z=-H/2 and z=H/2, this walls have the same charge equal to half of the sum of all charges in the system but with opposite sign (so the overall system is always electroneutral).
Charges are repelled by the wall via the same repulsive potential they have between each other.
The three domains demarcated by the walls (above, between and below) may have different, arbitrary permittivities.

The repulsive potential can be found in RepulsivePotential.cuh (should be available along this file) and has the following form:
U_{LJ}(r) = 4*U0* ( (sigma/r)^{2p} - (sigma/r)^p ) + U0
The repulsive force is then defined by parts using F_{LJ}=-\partial U_{LJ}/\partial r 
F(r<=rm) = F_{LJ}(r=rm);
F(rm<r<=sigma*2^1/p) = F_{LJ}(r);
F(r>sigma*2^1/p) = 0;

USAGE:
This code expects to find a file called data.main in the folder where it is executed.
data.main contains a series of parameters that allow to customize the simulation and must have the following  format:

----data.main starts
#Lines starting with # are ignored
option [argument] #Anything after the argument is ignored
flag #An option that does not need arguments
----end of data.main

The following options are available:

 numberParticles: The number of charges in the simulation
 gw: The Gaussian width of the charges
 
 H: The width of the domain
 Lxy: The dimensions of the box in XY
 permitivity: Permittivity inside the slab (only one used in triply periodic mode)
 permitivityBottom: Below z=-H/2
 permitivityTop: Above z=H/2

 temperature: Temperature for the Brownian Dynamics integrator, the diffusion coefficient will be D=T/(6*pi*viscosity*hydrodynamicRadius)
 viscosity: For BD
 hydrodynamicRadius: For BD
 dt: Time step for the BD integrator
 
 U0, sigma, r_m, p: Parameters for the repulsive interaction. If U0=0 the steric repulsion is turned off. 
 
 numberSteps: The simulation will run for this many steps
 printSteps: If greater than 0, the positions and forces will be printed every printSteps steps
 relaxSteps: The simulation will run without printing for this many steps.
 
 outfile: Positions and charge will be written to this file, each snapshot is separated by a #, each line will contain X Y Z Charge. Can be /dev/stdout to print to screen.
 forcefile: Optional, if present forces acting on particles will written to this file.
 readFile: Optional, if present charge positions will be read from this file with the format X Y Z Charge. numberParticles lines will be read. Can be /dev/stdin to read from pipe.
 
 noWall: Optional, if this flag is present particles will not be repelled by the wall.
 triplyPeriodic: Optional, if this flag is present electrostatics will be solved with a triply periodic spectral ewald solver. Notice that many parameters are not needed in this mode and will be ignored.

 split: The Ewald splitting parameter. It is mandatory if triply periodic mode is enabled.
 Nxy: The number of cells in XY. If this option is present split must NOT be present, it will be computed from this. Nxy can be provided instead of split for doubly periodic mode.

 mobilityFile: Optional, if this option is present, the mobility will depend on the height of the particle according to the data in this file. This file must have two columns with a list of normalized heights (so Z must go from -1 to 1) and normalized mobilities (i.e. 6*pi*eta*a*M0). The values for each particle will be linearly interpolated from the data provided in the file. The order of the values does not matter. Example:
--- mobility.dat---
-1.0 0.0
 0.0 1.0
 1.0 0.0
-------------------

 The following accuracy options are optional, the defaults provide a tolerance of 5e-4:
 support: Number of support cells for the interpolation kernel. Default is 10.
 numberStandardDeviations: Gaussian truncation. Default is 4
 tolerance: In doubly periodic, determines the cut off for the near field section of the algortihm. In triply periodic mode this represents the overall accuracy of the solver. Default is 1e-4. 
 upsampling: The relation between the grid cell size and gt=sqrt(gw^2+1/(4*split^2)). h_xy= gt/upsampling. default is 1.2


 
*/
#include"uammd.cuh"
#include"RepulsivePotential.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/SpectralEwaldPoisson.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"Leimkuhler.cuh"
#include"utils/InputFile.h"
#include"Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include <fstream>
#include<random>
using namespace uammd;
using std::make_shared;
using std::endl;

class RepulsiveWall{
  RepulsivePotentialFunctor::PairParameters params;
  real H;
public:
  RepulsiveWall(real H, RepulsivePotentialFunctor::PairParameters ip):H(H),params(ip){}

  __device__ real3 force(real4 pos){
    real distanceToImage = abs(abs(pos.z) - H * real(0.5))*real(2.0);
    real fz = RepulsivePotentialFunctor::force(distanceToImage * distanceToImage, params) * distanceToImage;
    return make_real3(0, 0, fz*(pos.z<0?real(-1.0):real(1.0)));
  }

  auto getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }
};
  
struct Parameters{
  int numberParticles;
  real Lxy, H;
  int Nxy = -1;
  int support = 10;
  real numberStandardDeviations = 4;
  real upsampling = 1.2;
  real tolerance = 1e-4;
  real temperature;
  real permitivity, permitivityBottom, permitivityTop;
  
  int numberSteps, printSteps, relaxSteps;
  real dt, viscosity, hydrodynamicRadius;

  real gw;
  real split = -1;
  real U0, sigma, r_m, p, cutOff;
  
  std::string outfile, readFile, forcefile;

  bool noWall = false;
  bool triplyPeriodic=false;

  std::string mobilityFile;
  bool idealParticles;
};

struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<thrust::device_vector<real4>> savedPositions;
  Parameters par;
};

Parameters readParameters(std::string fileName, shared_ptr<System> sys);

void initializeParticles(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::write);
  if(sim.par.readFile.empty()){
    std::generate(pos.begin(), pos.end(),
		  [&](){
		    real Lxy = sim.par.Lxy;
		    real H = sim.par.H;
		    real3 p;
		    real pdf;
		    do{
		      p = make_real3(sim.sys->rng().uniform3(-0.5, 0.5))*make_real3(Lxy, Lxy, H-2*sim.par.gw);
		      pdf = 1.0;
		    }while(sim.sys->rng().uniform(0, 1) > pdf);
		    return make_real4(p, 0);
		  });
    fori(0, sim.par.numberParticles){
      charge[i] = ((i%2)-0.5)*2;
    }
  }
  else{
    std::ifstream in(sim.par.readFile);
    fori(0, sim.par.numberParticles){
      in>>pos[i].x>>pos[i].y>>pos[i].z>>charge[i];
      pos[i].w = 0; 
    }
  }
  thrust::copy(pos.begin(), pos.end(), sim.savedPositions->begin());
}

UAMMD initialize(int argc, char *argv[]){
  UAMMD sim;
  sim.sys = std::make_shared<System>(argc, argv);
  std::random_device r;
  auto now = static_cast<long long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  sim.sys->rng().setSeed(now);
  std::string datamain = argv[1]; 
  if(datamain.empty()){
    datamain = "data.main";
  }
  sim.par = readParameters(datamain, sim.sys); 
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles, sim.sys);
  sim.savedPositions = std::make_shared<thrust::device_vector<real4>>();
  sim.savedPositions->resize(sim.par.numberParticles);
  initializeParticles(sim); 
  return sim;
}

auto createIntegrator(UAMMD sim){
  typename BD::Parameters par;
  par.temperature = sim.par.temperature;
  par.viscosity = sim.par.viscosity;
  par.hydrodynamicRadius = sim.par.hydrodynamicRadius;
  par.dt = sim.par.dt;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  std::shared_ptr<Integrator> integrator;
  if(sim.par.mobilityFile.empty()){
    using BDMethod = BD::Leimkuhler;
    integrator =  std::make_shared<BDMethod>(sim.pd, pg, sim.sys, par);
  }
  else{
    auto selfMobilityFactor = std::make_shared<SelfMobility>(sim.par.mobilityFile, sim.par.H);
    integrator = std::make_shared<LeimkuhlerWithMobility>(sim.pd, pg, sim.sys, par, selfMobilityFactor);
  }
  return integrator;
}

auto createDoublyPeriodicElectrostaticInteractor(UAMMD sim){  
  DPPoissonSlab::Parameters par;
  par.Lxy = make_real2(sim.par.Lxy);
  par.H = sim.par.H;
  DPPoissonSlab::Permitivity perm;
  perm.inside = sim.par.permitivity;
  perm.top = sim.par.permitivityTop;
  perm.bottom = sim.par.permitivityBottom;
  par.permitivity = perm;
  par.gw = sim.par.gw;
  par.tolerance = sim.par.tolerance;
  if(sim.par.split>0){
    par.split = sim.par.split;
  }
  if(sim.par.upsampling > 0){
    par.upsampling=sim.par.upsampling;
  }
  if(sim.par.numberStandardDeviations > 0){
    par.numberStandardDeviations=sim.par.numberStandardDeviations;
  }
  if(sim.par.support > 0){
    par.support=sim.par.support;
  }
  if(sim.par.Nxy > 0){
    if(sim.par.split>0){
      System::log<System::CRITICAL>("ERROR: Cannot set both Nxy and split at the same time");
    }
    par.Nxy = sim.par.Nxy;
  }
  par.support = sim.par.support;
  par.numberStandardDeviations = sim.par.numberStandardDeviations;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<DPPoissonSlab>(sim.pd, pg, sim.sys, par);
}

auto createTriplyPeriodicElectrostaticInteractor(UAMMD sim){
  Poisson::Parameters par;
  par.box = Box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  par.epsilon = sim.par.permitivity;
  par.gw = sim.par.gw;
  par.tolerance = sim.par.tolerance;
  if(sim.par.split<0){
    System::log<System::CRITICAL>("ERROR: Triply periodic mode needs an splitting parameter (Nxy wont do)");
  }
  par.split = sim.par.split;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<Poisson>(sim.pd, pg, sim.sys, par);
}

auto createWallRepulsionInteractor(UAMMD sim){
  RepulsivePotentialFunctor::PairParameters potpar;
  potpar.cutOff2 = sim.par.cutOff*sim.par.cutOff;
  potpar.sigma = sim.par.sigma;
  potpar.U0 = sim.par.U0;
  potpar.r_m = sim.par.r_m;
  potpar.p = sim.par.p;
  return make_shared<ExternalForces<RepulsiveWall>>(sim.pd, sim.sys, make_shared<RepulsiveWall>(sim.par.H, potpar));
}

auto createPotential(UAMMD sim){
  auto pot = std::make_shared<RepulsivePotential>(sim.sys);
  RepulsivePotential::InputPairParameters ppar;
  ppar.cutOff = sim.par.cutOff;
  ppar.U0 = sim.par.U0;
  ppar.sigma = sim.par.sigma;
  ppar.r_m = sim.par.r_m;
  ppar.p = sim.par.p;
  sim.sys->log<System::MESSAGE>("Repulsive rcut: %g", ppar.cutOff);
  pot->setPotParameters(0, 0, ppar);
 return pot;
}

template<class UsePotential> auto createShortRangeInteractor(UAMMD sim){
  auto pot = createPotential(sim);
  using SR = PairForces<UsePotential>;
  typename SR::Parameters params;
  real Lxy = sim.par.Lxy;
  real H = sim.par.H;
  params.box = Box(make_real3(Lxy, Lxy, H));
  if(not sim.par.triplyPeriodic){
    params.box.setPeriodicity(1,1,0);
  }
  auto pairForces = std::make_shared<SR>(sim.pd, sim.sys, params, pot);
  return pairForces;
}

void writeSimulation(UAMMD sim){ 
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::read);
  auto force = sim.pd->getForce(access::location::cpu, access::mode::read);
  static std::ofstream out(sim.par.outfile);
  static std::ofstream outf(sim.par.forcefile);
  Box box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  box.setPeriodicity(1,1,sim.par.triplyPeriodic?1:0);
  real3 L = box.boxSize;
  out<<"#Lx="<<L.x*0.5<<";Ly="<<L.y*0.5<<";Lz="<<L.z*0.5<<";"<<std::endl;
  outf<<"#"<<std::endl;
  fori(0, sim.par.numberParticles){
    real3 p = box.apply_pbc(make_real3(pos[i]));
    p.z = pos[i].z;
    real q = charge[i];
    out<<std::setprecision(2*sizeof(real))<<p<<" "<<q<<"\n";
    if(outf.good()){
      outf<<std::setprecision(2*sizeof(real))<<force[i]<<"\n";
    }
  }
  out<<std::flush;
}

struct CheckOverlap {
  real H;
  CheckOverlap(real H):H(H){
    
  }

  __device__ bool operator()(real4 p){
    return abs(p.z) >= (real(0.5)*H);
  }  
  
};

bool checkWallOverlap(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  //int overlappingCharges = thrust::count_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  //return overlappingCharges > 0;
  auto overlappingPos = thrust::find_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  return overlappingPos != pos.end();
}

void restoreLastSavedConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::write);  
  thrust::copy(thrust::cuda::par, sim.savedPositions->begin(), sim.savedPositions->end(), pos.begin());  
}

void saveConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  thrust::copy(thrust::cuda::par, pos.begin(), pos.end(), sim.savedPositions->begin()); 
}

int main(int argc, char *argv[]){  
  auto sim = initialize(argc, argv);
  auto bd = createIntegrator(sim);
  if(not sim.par.idealParticles){
    if(sim.par.triplyPeriodic){
      bd->addInteractor(createTriplyPeriodicElectrostaticInteractor(sim));
    }
    else{
      bd->addInteractor(createDoublyPeriodicElectrostaticInteractor(sim));
    }
    if(sim.par.U0 > 0){
      bd->addInteractor(createShortRangeInteractor<RepulsivePotential>(sim));
    }
  }
  if(not sim.par.noWall){
    bd->addInteractor(createWallRepulsionInteractor(sim));
  }
  int numberRetries=0;
  int numberRetriesThisStep=0;
  int lastStepSaved=0;
  constexpr int saveRate = 100;
  constexpr int maximumRetries = 1e6;
  constexpr int maximumRetriesPerStep=1e4;
  forj(0, sim.par.relaxSteps){
    bd->forwardTime();
    if(not sim.par.triplyPeriodic){
      if(checkWallOverlap(sim)){
	numberRetries++;
	if(numberRetries>maximumRetries){
	  throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
	}
	numberRetriesThisStep++;
	if(numberRetriesThisStep>maximumRetriesPerStep){
	  throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
	}
	j=lastStepSaved;
	restoreLastSavedConfiguration(sim);
	continue;
      }
      if(j%saveRate==0){
	numberRetriesThisStep = 0;
	lastStepSaved=j;
	saveConfiguration(sim);
      }
    }
  }
  Timer tim;
  tim.tic();
  lastStepSaved=0;
  forj(0, sim.par.numberSteps){
    bd->forwardTime();
    if(not sim.par.triplyPeriodic){
      if(checkWallOverlap(sim)){
	numberRetries++;
	if(numberRetries>maximumRetries){
	  throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
	}
	numberRetriesThisStep++;
	if(numberRetriesThisStep>maximumRetriesPerStep){
	  throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
	}
	j=lastStepSaved;
	restoreLastSavedConfiguration(sim);
	continue;
      }
      if(j%saveRate==0){
	numberRetriesThisStep=0;
	lastStepSaved=j;
	saveConfiguration(sim);
      }
    }
    if(sim.par.printSteps > 0 and j%sim.par.printSteps==0){
      writeSimulation(sim);
      numberRetriesThisStep=0;
      lastStepSaved=j;
      saveConfiguration(sim);
    }
  }
  System::log<System::MESSAGE>("Number of rejected configurations: %d (%g%% of total)", numberRetries, (double)numberRetries/(sim.par.numberSteps + sim.par.relaxSteps)*100.0);
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f", sim.par.numberSteps/totalTime);
  return 0;
}

Parameters readParameters(std::string datamain, shared_ptr<System> sys){
  InputFile in(datamain, sys);
  Parameters par;
  if(in.getOption("triplyPeriodic", InputFile::Optional)){
    par.triplyPeriodic= true;
  }

  in.getOption("Lxy", InputFile::Required)>>par.Lxy;
  in.getOption("H", InputFile::Required)>>par.H;
  in.getOption("numberSteps", InputFile::Required)>>par.numberSteps;
  in.getOption("printSteps", InputFile::Required)>>par.printSteps;
  in.getOption("relaxSteps", InputFile::Required)>>par.relaxSteps;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>par.hydrodynamicRadius;
  in.getOption("outfile", InputFile::Required)>>par.outfile;
  in.getOption("forcefile", InputFile::Optional)>>par.forcefile;
  in.getOption("U0", InputFile::Required)>>par.U0;
  in.getOption("r_m", InputFile::Required)>>par.r_m;
  in.getOption("p", InputFile::Required)>>par.p;
  in.getOption("sigma", InputFile::Required)>>par.sigma;
  in.getOption("readFile", InputFile::Optional)>>par.readFile;
  in.getOption("gw", InputFile::Required)>>par.gw;
  in.getOption("tolerance", InputFile::Optional)>>par.tolerance;
  in.getOption("permitivity", InputFile::Required)>>par.permitivity;
  if(not par.triplyPeriodic){
    in.getOption("permitivityTop", InputFile::Required)>>par.permitivityTop;
    in.getOption("permitivityBottom", InputFile::Required)>>par.permitivityBottom;
  }
  if(not par.triplyPeriodic){
    in.getOption("split", InputFile::Optional)>>par.split;
  }
  else{
    in.getOption("split", InputFile::Required)>>par.split;
  }
  in.getOption("Nxy", InputFile::Optional)>>par.Nxy;
  in.getOption("support", InputFile::Optional)>>par.support;
  in.getOption("numberStandardDeviations", InputFile::Optional)>>par.numberStandardDeviations;
  in.getOption("upsampling", InputFile::Optional)>>par.upsampling;
  if(in.getOption("noWall", InputFile::Optional)){
    par.noWall= true;
  }  
  if(par.split < 0 and par.Nxy < 0){
    System::log<System::CRITICAL>("ERROR: I need either Nxy or split");    
  }
  par.cutOff = par.sigma*pow(2,1.0/par.p);
  in.getOption("useMobilityFromFile", InputFile::Optional)>>par.mobilityFile;
  in.getOption("idealParticles", InputFile::Optional)>>par.idealParticles;
  return par;
}



