/*
Sampler

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

#include <approxmc/approxmc.h>
#include <cryptominisat5/cryptominisat.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <random>

#include "config.h"
#include "unisamp.h"

using ApproxMC::SolCount;
using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;
using namespace CMSat;
using namespace ApproxMC;

struct SavedModel {
  SavedModel(uint32_t _hash_num, const vector<lbool>& _model)
      : model(_model), hash_num(_hash_num) {}

  vector<lbool> model;
  uint32_t hash_num;
};

struct Hash {
  Hash(uint32_t _act_var, vector<uint32_t>& _hash_vars, bool _rhs)
      : act_var(_act_var), hash_vars(_hash_vars), rhs(_rhs) {}

  Hash() {}

  uint32_t act_var;
  vector<uint32_t> hash_vars;
  bool rhs;
};

struct HashesModels {
  map<uint64_t, Hash> hashes;
  vector<SavedModel> glob_model;  // global table storing models
};

struct SolNum {
  SolNum(uint64_t _solutions, uint64_t _repeated)
      : solutions(_solutions), repeated(_repeated) {}
  uint64_t solutions = 0;
  uint64_t repeated = 0;
};

struct SparseData {
  explicit SparseData(int _table_no) : table_no(_table_no) {}

  uint32_t next_index = 0;
  double sparseprob = 0.5;
  int table_no = -1;
};

class Sampler {
 public:
  void sample(const Config conf, const SolCount sol_count,
              const uint32_t num_samples);
  AppMC* appmc;

  // --- Dynamic Heuristic Fields ---
  int current_window_size;
  double current_slowdown_threshold;

  // --- Heuristic Constants (these are your new tuning knobs) ---
  static constexpr int WINDOW_MIN =
      10;  // Start with a hyper-reactive 10-sample window
  static constexpr int WINDOW_MAX = 200;  // Don't let the window grow forever
  static constexpr double WINDOW_GROW_FACTOR =
      1.5;  // How fast to grow the window (e.g., 10, 15, 22, 33...)

  static constexpr double THRESHOLD_MIN =
      1.8;  // Start with a very strict 1.8x slowdown
  static constexpr double THRESHOLD_MAX =
      3.0;  // Don't let the threshold become too lenient
  static constexpr double THRESHOLD_RELAX_STEP =
      0.2;  // How much to relax the threshold

  // This defines the line between a "slip-up" and a "disaster".
  // 1.5x means a failure is "Major" if it's 50% worse than the threshold.
  static constexpr double CATASTROPHE_FACTOR = 1.5;

  SATSolver* solver;

  std::unique_ptr<SATSolver> base_solver;
  std::unique_ptr<SATSolver> working_solver;

  SATSolver* appmc_solver;
  bool is_using_appmc_solver;

  std::optional<double> baseline_time;
  double current_window_total_time;
  int samples_in_window;

  /// What to call on samples
  UniSamp::callback callback_func;
  void* callback_func_data;

 private:
  uint32_t startiter;
  uint32_t thresh;
  double thresh_sampler_gen;

  Config conf;
  string gen_rnd_bits(const uint32_t size, const uint32_t numhashes);
  uint32_t sols_to_return(uint32_t numSolutions);
  void add_sampler_options();
  bool gen_rhs();
  uint32_t gen_n_samples(const uint32_t num_samples_needed);
  Hash add_hash(uint32_t total_num_hashes);
  string binary(const uint32_t x, const uint32_t length);
  void generate_samples(const uint32_t num_samples);

  // For dynamic backoff
  // void reset_heuristic_params();
  // void load_and_initialize();
  // void reset_working_solver();
  // void check_and_perform_reset();

  SolNum bounded_sol_count(uint32_t maxSolutions, const vector<Lit>* assumps,
                           const uint32_t hashCount, uint32_t minSolutions = 1,
                           HashesModels* hm = nullptr,
                           vector<vector<int>>* out_solutions = nullptr);
  bool bounded_sol_count_unisamp(const vector<Lit>* assumps,
                                 const uint32_t hashCount,
                                 const uint32_t num_tries = 1,
                                 HashesModels* hm = nullptr,
                                 vector<vector<int>>* out_solutions = nullptr);
  vector<Lit> set_num_hashes(uint32_t num_wanted, map<uint64_t, Hash>& hashes);

  void simplify();

  // For certification
  void open_rand_file();
  void open_cert_file();

  ////////////////
  // Helper functions
  ////////////////
  void print_xor(const vector<uint32_t>& vars, const uint32_t rhs);
  vector<int> get_solution_ints(const vector<lbool>& model);
  void ban_one(const uint32_t act_var, const vector<lbool>& model);
  void check_model(const vector<lbool>& model, const HashesModels* const hm,
                   const uint32_t hashCount);
  bool check_model_against_hash(const Hash& h, const vector<lbool>& model);
  uint64_t add_glob_banning_cls(
      const HashesModels* glob_model = nullptr,
      const uint32_t act_var = std::numeric_limits<uint32_t>::max(),
      const uint32_t num_hashes = std::numeric_limits<uint32_t>::max());

  void readInAFile(SATSolver* solver2, const string& filename);
  void readInStandardInput(SATSolver* solver2);

  // Data so we can output temporary count when catching the signal
  vector<uint64_t> numHashList;
  vector<int64_t> numCountList;
  template <class T>
  T findMedian(vector<T>& numList);
  template <class T>
  T findMin(vector<T>& numList);

  ////////////////
  // internal data
  ////////////////
  double startTime;
  std::mt19937 randomEngine;
  uint32_t orig_num_vars;
  double total_inter_simp_time = 0;
  uint32_t threshold;  // precision, it's computed

  // For certification
  uint32_t base_rand = 0;
  std::ifstream rand_file;
  std::ofstream cert_file;
};