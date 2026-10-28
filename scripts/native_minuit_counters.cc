// Read-only call instrumentation for the installed ROOT/Combine libraries.
// Each wrapper forwards the original ABI call and returns its result unchanged.
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumParameters.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <utility>
#include <unistd.h>

namespace {
using Gradient = ROOT::Minuit2::FunctionGradient;
using Parameters = ROOT::Minuit2::MinimumParameters;
std::atomic<unsigned long> nll_total{0}, gradient_total[2]{{0},{0}};
std::atomic<unsigned long> phase_nll{0}, phase_gradient[2]{{0},{0}};
std::atomic<unsigned long> entries[5]{{0},{0},{0},{0},{0}}, episode{0};
thread_local unsigned gradient_depth = 0;
thread_local bool in_minimize = false;

template<class Function> Function original(char const* symbol) {
  auto pointer = dlsym(RTLD_NEXT, symbol);
  if (!pointer) { std::fprintf(stderr,"Cannot instrument %s: %s\n",symbol,dlerror()); std::abort(); }
  return reinterpret_cast<Function>(pointer);
}
struct GradientScope {
  GradientScope(unsigned kind, unsigned entry) {
    ++entries[entry];
    if (gradient_depth++ == 0) {
      ++gradient_total[kind];
      if(in_minimize) ++phase_gradient[kind];
    }
  }
  ~GradientScope() { --gradient_depth; }
};
void write_event(char const* type, bool success) {
  auto prefix=std::getenv("COMBINE_JAX_NATIVE_COUNTER_PREFIX");
  if(!prefix) return;
  std::ofstream out(std::string(prefix)+"."+std::to_string(getpid())+".jsonl",std::ios::app);
  out << "{\"event\":\"" << type << "\",\"pid\":" << getpid()
      << ",\"episode\":" << episode << ",\"success\":" << (success?"true":"false")
      << ",\"nll_total\":" << nll_total << ",\"numerical_gradient_total\":" << gradient_total[0]
      << ",\"hessian_gradient_total\":" << gradient_total[1]
      << ",\"phase_nll\":" << phase_nll << ",\"phase_numerical_gradients\":" << phase_gradient[0]
      << ",\"phase_hessian_gradients\":" << phase_gradient[1]
      << ",\"gradient_entries\":[" << entries[0] << ',' << entries[1] << ',' << entries[2] << ',' << entries[3]
      << ',' << entries[4] << "]}\n";
}
__attribute__((destructor)) void finish() {
  if(episode || gradient_total[0] || gradient_total[1] || nll_total) write_event("process_exit",true);
}
}

extern "C" bool counted_minimize(void*) asm("_ZN4ROOT7Minuit216Minuit2Minimizer8MinimizeEv");
extern "C" bool counted_minimize(void* self) {
  static auto call=original<bool(*)(void*)>("_ZN4ROOT7Minuit216Minuit2Minimizer8MinimizeEv");
  ++episode; phase_nll=0; phase_gradient[0]=0; phase_gradient[1]=0; in_minimize=true;
  try {
    bool result=call(self); in_minimize=false; write_event("minimize_return",result); return result;
  } catch(...) { in_minimize=false; write_event("minimize_exception",false); throw; }
}

extern "C" double counted_nll(void const*) asm("_ZNK10cacheutils13CachingSimNLL8evaluateEv");
extern "C" double counted_nll(void const* self) {
  static auto call=original<double(*)(void const*)>("_ZNK10cacheutils13CachingSimNLL8evaluateEv");
  ++nll_total; if(in_minimize) ++phase_nll; return call(self);
}

#define GRADIENT_ONE(NAME, SYMBOL, KIND, ENTRY) \
extern "C" Gradient NAME(void const*,Parameters const&) asm(SYMBOL); \
extern "C" Gradient NAME(void const* self,Parameters const& point) { \
  static auto call=original<Gradient(*)(void const*,Parameters const&)>(SYMBOL); \
  GradientScope scope(KIND,ENTRY); return call(self,point); }
#define GRADIENT_TWO(NAME, SYMBOL, KIND, ENTRY) \
extern "C" Gradient NAME(void const*,Parameters const&,Gradient const&) asm(SYMBOL); \
extern "C" Gradient NAME(void const* self,Parameters const& point,Gradient const& gradient) { \
  static auto call=original<Gradient(*)(void const*,Parameters const&,Gradient const&)>(SYMBOL); \
  GradientScope scope(KIND,ENTRY); return call(self,point,gradient); }

GRADIENT_ONE(numerical_one,"_ZNK4ROOT7Minuit229Numerical2PGradientCalculatorclERKNS0_17MinimumParametersE",0,0)
GRADIENT_TWO(numerical_two,"_ZNK4ROOT7Minuit229Numerical2PGradientCalculatorclERKNS0_17MinimumParametersERKNS0_16FunctionGradientE",0,1)
GRADIENT_ONE(hessian_one,"_ZNK4ROOT7Minuit225HessianGradientCalculatorclERKNS0_17MinimumParametersE",1,2)
GRADIENT_TWO(hessian_two,"_ZNK4ROOT7Minuit225HessianGradientCalculatorclERKNS0_17MinimumParametersERKNS0_16FunctionGradientE",1,3)

using Delta = std::pair<Gradient,ROOT::Minuit2::MnAlgebraicVector>;
extern "C" Delta hessian_delta(void const*,Parameters const&,Gradient const&)
  asm("_ZNK4ROOT7Minuit225HessianGradientCalculator13DeltaGradientERKNS0_17MinimumParametersERKNS0_16FunctionGradientE");
extern "C" Delta hessian_delta(void const* self,Parameters const& point,Gradient const& gradient) {
  static auto call=original<Delta(*)(void const*,Parameters const&,Gradient const&)>(
    "_ZNK4ROOT7Minuit225HessianGradientCalculator13DeltaGradientERKNS0_17MinimumParametersERKNS0_16FunctionGradientE");
  GradientScope scope(1,4); return call(self,point,gradient);
}
