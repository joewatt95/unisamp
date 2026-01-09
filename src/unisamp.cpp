/*
 ApproxMC

 Copyright (c) 2019-2020, Mate Soos and Kuldeep S. Meel. All rights reserved
 Copyright (c) 2009-2018, Mate Soos. All rights reserved.
 Copyright (c) 2015, Supratik Chakraborty, Daniel J. Fremont,
 Kuldeep S. Meel, Sanjit A. Seshia, Moshe Y. Vardi
 Copyright (c) 2014, Supratik Chakraborty, Kuldeep S. Meel, Moshe Y. Vardi

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <iostream>
#include <memory>

#include "GitSHA1.h"
#include "config.h"
#include "sampler.h"

using namespace ApproxMC;

using std::cout;
using std::endl;

#if defined _WIN32
#define DLL_PUBLIC __declspec(dllexport)
#else
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
#endif

#define set_get_macro(TYPE, NAME)                                         \
  DLL_PUBLIC void UniS::set_##NAME(TYPE NAME) { data->conf.NAME = NAME; } \
  DLL_PUBLIC TYPE UniS::get_##NAME() const { return data->conf.NAME; }

namespace UniSamp {
struct UniSampPrivateData {
  Sampler sampler;
  AppMC* appmc;
  Config conf;
};
}  // namespace UniSamp

using namespace UniSamp;

DLL_PUBLIC UniS::UniS(AppMC* appmc) {
  data = std::make_unique<UniSampPrivateData>();
  data->sampler.appmc = appmc;
}

DLL_PUBLIC UniS::~UniS() {}

DLL_PUBLIC void UniS::set_callback(UniSamp::callback _callback_func,
                                   void* _callback_func_data) {
  data->sampler.callback_func = _callback_func;
  data->sampler.callback_func_data = _callback_func_data;
}

DLL_PUBLIC void UniS::setup_randbits(string rand_file_name)
{
    data->conf.rand_file_name = rand_file_name;
}

DLL_PUBLIC void UniS::setup_cert(string cert_file_name)
{
    data->conf.unisamp_cert_file_name = cert_file_name;
}

DLL_PUBLIC void UniS::sample(const SolCount* sol_count, uint32_t num_samples) {
  if (!data->sampler.callback_func) {
    cout << "ERROR! You must set the callback function or your samples will be "
            "lost"
         << endl;
    exit(-1);
  }
  data->sampler.sample(data->conf, *sol_count, num_samples);
}

DLL_PUBLIC string UniS::get_version_sha1() {
  return UnisampIntNS::get_version_sha1();
}

set_get_macro(double, epsilon) set_get_macro(double, r_thresh_pivot)
    set_get_macro(bool, force_sol_extension)
        set_get_macro(const std::vector<uint32_t>&, full_sampling_vars)
            set_get_macro(bool, verb_sampler_cls)

                DLL_PUBLIC void UniS::set_verbosity(uint32_t verb) {
  data->conf.verb = verb;
}
