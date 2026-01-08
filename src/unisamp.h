/*
 UniSamp

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

#pragma once

#include <cryptominisat5/cryptominisat.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ApproxMC {
class AppMC;
class SolCount;
}  // namespace ApproxMC

namespace UniSamp {

typedef std::function<void(const std::vector<int>& solution, uint32_t num_tries,
                           void* data)>
    callback;

struct UniSampPrivateData;
#ifdef _WIN32
class __declspec(dllexport) UniS
#else
class UniS
#endif
{
 public:
  UniS(ApproxMC::AppMC* appmc);
  ~UniS();
  static std::string get_version_sha1();

  void sample(const ApproxMC::SolCount* sol_count, uint32_t num_samples);

  // For certification
  void setup_randbits(std::string log_file_name);
  void setup_cert(std::string cert_file_name);

  // Misc options -- do NOT to change unless you know what you are doing!
  void set_epsilon(double epsilon);
  void set_r_thresh_pivot(double r_thresh_pivot);
  void set_only_indep_samples(bool only_indep_samples);
  void set_verb_sampler_cls(bool verb_sampler_cls);
  void set_force_sol_extension(bool force_sol_extension);
  void set_verbosity(uint32_t verb);
  void set_callback(UniSamp::callback f, void* data = nullptr);
  void set_full_sampling_vars(const std::vector<uint32_t>& vars);
  void set_empty_sampling_vars(const std::vector<uint32_t>& vars);

  // Querying default values
  double get_epsilon() const;
  double get_r_thresh_pivot() const;
  bool get_only_indep_samples() const;
  bool get_verb_sampler_cls() const;
  bool get_force_sol_extension() const;
  bool get_verb_banning_cls() const;
  const std::vector<uint32_t>& get_full_sampling_vars() const;
  const std::vector<uint32_t>& get_empty_sampling_vars() const;

 private:
  ////////////////////////////
  // Do not bother with this, it's private
  ////////////////////////////
  std::unique_ptr<UniSampPrivateData> data;
};

}  // namespace UniSamp
