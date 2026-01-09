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

#include "sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "time_mem.h"

#define verb_print(a, b) \
  if (conf.verb >= a) cout << "c o " << b << endl

using std::cerr;
using std::cout;
using std::endl;
using std::map;
using std::optional;
using std::ios;

// Hash Sampler::add_hash(uint32_t hash_index) {
//   const auto randomBits =
//       gen_rnd_bits(appmc->get_sampling_set().size(), hash_index);

//   vector<uint32_t> vars;
//   uint32_t i = 0;
//   for (const auto& sampling_var : appmc->get_sampling_set()) {
//     if (randomBits[i] == '1') vars.push_back(sampling_var);
//     i++;
//   }

//   solver->new_var();
//   const uint32_t act_var = solver->nVars() - 1;
//   const bool rhs = gen_rhs();
//   Hash h(act_var, vars, rhs);

//   vars.push_back(act_var);
//   solver->add_xor_clause(vars, rhs);
//   if (conf.verb_sampler_cls) print_xor(vars, rhs);
//   return h;
// }

void Sampler::open_rand_file() {
  if (!conf.rand_file_name.empty()) {
    rand_file.open(conf.rand_file_name.c_str(), ios::binary | ios::in);
    if (!rand_file.is_open()) {
      cout << "[unis] Cannot open random bits file '"
           << conf.rand_file_name << "' for reading." << endl;
      exit(1);
    }
  }
}

void Sampler::open_cert_file() {
  if (!conf.unisamp_cert_file_name.empty()) {
    cert_file.open(conf.unisamp_cert_file_name.c_str());
    if (!cert_file.is_open()) {
      cout << "[unis] Cannot open certification file '"
           << conf.unisamp_cert_file_name << "' for writing." << endl;
      exit(1);
    }
  }
}

// ---------------------------------------------------------
// The Dice Roll Function
// ---------------------------------------------------------
uint32_t Sampler::dice_roll(BitReader& bits, uint32_t n) {
  double l = 0.0;
  double h = static_cast<double>(n);

  while (true) {
    // Termination condition: Interval is inside a single integer
    const double floor_l = std::floor(l);
    const double ceil_end = std::ceil(l + h);
    if (floor_l == ceil_end - 1.0) return static_cast<uint32_t>(floor_l);

    h /= 2.0;
    if (bits.next_bit()) l = l + h;
  }
}

/* TODO:
  Copied directly from approxmc-cert.
  This may need adjusting, for instance base_rand is adjusted in approxmc-cert
  based on the current iteration, iter, but here, we don't have that concept.
*/
Hash Sampler::add_hash(uint32_t hash_index) {
  string random_bits;

  if (rand_file.is_open()) {
    // set read position of random bits
    uint32_t pos_rand = base_rand + hash_index * (conf.full_sampling_vars.size() + 1);
    rand_file.seekg(pos_rand / 8, ios::beg);

    // read random bits
    char ch;
    uint32_t cnt = 0;
    while (cnt < conf.full_sampling_vars.size() + 1 && rand_file.get(ch)) {
      for (int i = (cnt ? 7 : 7 - pos_rand % 8);
           cnt < conf.full_sampling_vars.size() + 1 && i >= 0; i--) {
        random_bits += '0' + ((ch >> i) & 1);
        cnt++;
      }
    }
    if (random_bits.size() != conf.full_sampling_vars.size() + 1) {
      cout << "[unis] Cannot read " << conf.full_sampling_vars.size() + 1
           << " random bits from file '" << conf.rand_file_name << "'." << endl;
      exit(1);
    }
  } else {
    random_bits = gen_rnd_bits(conf.full_sampling_vars.size(), hash_index);
  }

  vector<uint32_t> vars;
  for (uint32_t j = 0; j < conf.full_sampling_vars.size(); j++) {
    if (random_bits[j] == '1') vars.push_back(conf.full_sampling_vars[j]);
  }

  solver->new_var();
  const uint32_t act_var = solver->nVars() - 1;
  bool rhs;
  if (rand_file.is_open()) {
    rhs = random_bits.back() == '1';
  } else {
    rhs = gen_rhs();
  }
  auto h = Hash(act_var, vars, rhs);

  vars.push_back(act_var);
  solver->add_xor_clause(vars, rhs);
  if (conf.verb_sampler_cls) print_xor(vars, rhs);

  return h;
}

void Sampler::ban_one(const uint32_t act_var, const vector<lbool>& model) {
  vector<Lit> lits;
  lits.push_back(Lit(act_var, false));
  for (const uint32_t var : appmc->get_sampling_set()) {
    lits.push_back(Lit(var, model[var] == l_True));
  }
  solver->add_clause(lits);
}

/// adding banning clauses for repeating solutions
uint64_t Sampler::add_glob_banning_cls(const HashesModels* hm,
                                       const uint32_t act_var,
                                       const uint32_t num_hashes) {
  if (!hm) return 0;
  assert(act_var != std::numeric_limits<uint32_t>::max());
  assert(num_hashes != std::numeric_limits<uint32_t>::max());

  uint64_t repeat = 0;
  vector<Lit> lits;
  for (const auto& sm : hm->glob_model) {
    // Model was generated with 'sm.hash_num' active
    // We will have 'num_hashes' hashes active

    if (sm.hash_num >= num_hashes) {
      ban_one(act_var, sm.model);
      repeat++;
    } else if (static_cast<int>(num_hashes) - static_cast<int>(sm.hash_num) <
               9) {
      // Model has to fit all hashes
      bool ok = true;
      for (const auto& h : hm->hashes) {
        // This hash is number: h.first
        // Only has to match hashes below current need
        // note that "h.first" is numbered from 0, so this is a "<" not "<="
        if (h.first < num_hashes) {
          ok &= check_model_against_hash(h.second, sm.model);
          if (!ok) break;
        }
      }
      if (ok) {
        // cout << "Found repeat model, had to check " << checked << " hashes"
        // << endl;
        ban_one(act_var, sm.model);
        repeat++;
      }
    }
  }
  return repeat;
}

SolNum Sampler::bounded_sol_count(uint32_t maxSolutions,
                                  const vector<Lit>* assumps,
                                  const uint32_t hashCount,
                                  uint32_t minSolutions, HashesModels* hm,
                                  vector<vector<int>>* out_solutions) {
  verb_print(1,
             "[unis] "
             "[ " << std::setw(7)
                  << std::setprecision(2) << std::fixed
                  << (cpuTimeTotal() - startTime) << " ]"
                  << " bounded_sol_count looking for " << std::setw(4)
                  << maxSolutions << " solutions"
                  << " -- hashes active: " << hashCount);

  // Set up things for adding clauses that can later be removed
  vector<Lit> new_assumps;
  if (assumps) {
    assert(assumps->size() == hashCount);
    new_assumps = *assumps;
  } else
    assert(hashCount == 0);
  solver->new_var();
  const uint32_t sol_ban_var = solver->nVars() - 1;
  new_assumps.push_back(Lit(sol_ban_var, true));

  if (appmc->get_simplify() >= 2) {
    verb_print(1, "[unis] inter-simplifying");
    double myTime = cpuTime();
    solver->simplify(&new_assumps);
    solver->set_verbosity(0);
    total_inter_simp_time += cpuTime() - myTime;
    verb_print(1, "[unis] inter-simp finished, total simp time: "
                      << total_inter_simp_time);
  }

  const uint64_t repeat = add_glob_banning_cls(hm, sol_ban_var, hashCount);
  uint64_t solutions = repeat;
  double last_found_time = cpuTimeTotal();
  vector<vector<lbool>> models;
  while (solutions < maxSolutions) {
    lbool ret = solver->solve(&new_assumps, false);
    assert(ret == l_False || ret == l_True);

    if (conf.verb >= 2) {
      cout << "c o [unis] bounded_sol_count ret: " << std::setw(7) << ret;
      if (ret == l_True)
        cout << " sol no.  " << std::setw(3) << solutions;
      else
        cout << " No more. " << std::setw(3) << "";
      cout << " T: " << std::setw(7) << std::setprecision(2) << std::fixed
           << (cpuTimeTotal() - startTime) << " -- hashes act: " << hashCount
           << " -- T since last: " << std::setw(7) << std::setprecision(2)
           << std::fixed << (cpuTimeTotal() - last_found_time) << endl;
      if (conf.verb >= 3) solver->print_stats();
    }
    last_found_time = cpuTimeTotal();
    if (ret != l_True) break;

    // Add solution to set
    solutions++;
    const vector<lbool> model = solver->get_model();
    check_model(model, hm, hashCount);
    models.push_back(model);
    if (out_solutions) out_solutions->push_back(get_solution_ints(model));

    // ban solution
    vector<Lit> lits;
    lits.push_back(Lit(sol_ban_var, false));
    for (const uint32_t var : appmc->get_sampling_set()) {
      assert(solver->get_model()[var] != l_Undef);
      lits.push_back(Lit(var, solver->get_model()[var] == l_True));
    }
    if (conf.verb_sampler_cls)
      cout << "c o [unis] Adding banning clause: " << lits << endl;
    solver->add_clause(lits);
  }

  // Log solutions to certificate file.
  if (cert_file.is_open()) {
    cert_file << solutions;
    for (const auto& solution : *out_solutions) {
      for (const auto& val : solution) cert_file << val << " ";
      cert_file << "0" << endl;
    }
  }

  if (solutions < maxSolutions) {
    // Sampling -- output a random sample of N solutions
    if (solutions >= minSolutions) {
      assert(minSolutions > 0);
      vector<size_t> modelIndices(models.size());
      std::iota(modelIndices.begin(), modelIndices.end(), 0);
      std::shuffle(modelIndices.begin(), modelIndices.end(), randomEngine);

      for (uint32_t i = 0; i < sols_to_return(solutions); i++) {
        const auto& model = models.at(modelIndices.at(i));
        callback_func(get_solution_ints(model), 1, callback_func_data);
      }
    }
  }

  // Save global models
  if (hm && appmc->get_reuse_models()) {
    for (const auto& model : models) {
      hm->glob_model.push_back(SavedModel(hashCount, model));
    }
  }

  // Remove solution banning
  vector<Lit> cl_that_removes;
  cl_that_removes.push_back(Lit(sol_ban_var, false));
  solver->add_clause(cl_that_removes);

  return SolNum(solutions, repeat);
}

// NOTE: This performs bsat AND ALSO samples a solution from the solution set
// obtained from bsat.
bool Sampler::bounded_sol_count_unisamp(const vector<Lit>* assumps,
                                        const uint32_t hashCount,
                                        const uint32_t num_tries,
                                        HashesModels* hm,
                                        vector<vector<int>>* out_solutions) {
  verb_print(1,
             "[unis] "
             "[ " << std::setw(7)
                  << std::setprecision(2) << std::fixed
                  << (cpuTimeTotal() - startTime) << " ]"
                  << " bounded_sol_count looking for " << std::setw(4)
                  << thresh + 1 << " solutions"
                  << " -- hashes active: " << hashCount);

  // Set up things for adding clauses that can later be removed
  vector<Lit> new_assumps;
  if (assumps) {
    assert(assumps->size() == hashCount);
    new_assumps = *assumps;
  } else
    assert(hashCount == 0);
  solver->new_var();
  const uint32_t sol_ban_var = solver->nVars() - 1;
  new_assumps.push_back(Lit(sol_ban_var, true));

  if (appmc->get_simplify() >= 2) {
    verb_print(1, "[unis] inter-simplifying");
    double myTime = cpuTime();
    solver->simplify(&new_assumps);
    solver->set_verbosity(0);
    total_inter_simp_time += cpuTime() - myTime;
    verb_print(1, "[unis] inter-simp finished, total simp time: "
                      << total_inter_simp_time);
  }

  const uint64_t repeat = add_glob_banning_cls(hm, sol_ban_var, hashCount);
  uint64_t solutions = repeat;
  double last_found_time = cpuTimeTotal();
  vector<vector<lbool>> models;
  while (solutions < thresh + 1) {
    lbool ret = solver->solve(&new_assumps, false);
    assert(ret == l_False || ret == l_True);

    if (conf.verb >= 2) {
      cout << "c o [unis] bounded_sol_count ret: " << std::setw(7) << ret;
      if (ret == l_True)
        cout << " sol no.  " << std::setw(3) << solutions;
      else
        cout << " No more. " << std::setw(3) << "";
      cout << " T: " << std::setw(7) << std::setprecision(2) << std::fixed
           << (cpuTimeTotal() - startTime) << " -- hashes act: " << hashCount
           << " -- T since last: " << std::setw(7) << std::setprecision(2)
           << std::fixed << (cpuTimeTotal() - last_found_time) << endl;
      if (conf.verb >= 3) solver->print_stats();
    }
    last_found_time = cpuTimeTotal();
    if (ret != l_True) break;

    // Add solution to set
    solutions++;
    const vector<lbool> model = solver->get_model();
    check_model(model, hm, hashCount);
    models.push_back(model);
    if (out_solutions) out_solutions->push_back(get_solution_ints(model));

    // ban solution
    vector<Lit> lits;
    lits.push_back(Lit(sol_ban_var, false));
    for (const uint32_t var : appmc->get_sampling_set()) {
      assert(solver->get_model()[var] != l_Undef);
      lits.push_back(Lit(var, solver->get_model()[var] == l_True));
    }
    if (conf.verb_sampler_cls)
      cout << "c o [unis] Adding banning clause: " << lits << endl;
    solver->add_clause(lits);
  }

  // Log solutions to certificate file.
  if (cert_file.is_open()) {
    cert_file << solutions;
    for (const auto& solution : *out_solutions) {
      for (const auto& val : solution) cert_file << val << " ";
      cert_file << "0" << endl;
    }
  }

  bool ok = false;

  verb_print(1, "[unis] Number of solutions BSAT found: " << solutions);

  if (1 <= solutions && solutions <= thresh) {
    const size_t index =
        std::uniform_int_distribution<size_t>(0, thresh)(randomEngine);
    if (index < models.size()) {
      const auto& model = models.at(index);
      callback_func(get_solution_ints(model), num_tries, callback_func_data);
      ok = true;
    }
  }

  // Save global models
  if (hm && appmc->get_reuse_models())
    for (const auto& model : models)
      hm->glob_model.push_back(SavedModel(hashCount, model));

  // Remove solution banning
  vector<Lit> cl_that_removes;
  cl_that_removes.push_back(Lit(sol_ban_var, false));
  solver->add_clause(cl_that_removes);

  return ok;
}

void Sampler::sample(Config _conf, const ApproxMC::SolCount solCount,
                     const uint32_t num_samples) {
  conf = _conf;
  solver = appmc->get_solver();
  orig_num_vars = solver->nVars();
  startTime = cpuTimeTotal();
  randomEngine.seed(appmc->get_seed());

  open_cert_file();

  // Our optimised pivot
  thresh_sampler_gen = 1.0 / (pow(conf.r_thresh_pivot - 1, 2) * conf.epsilon);

  // Original pivot from paper
  // thresh_sampler_gen = std::max(200.0, 2 / conf.epsilon);

  verb_print(1, "[appmc] Approximate count: 2^" << solCount.hashCount << " * "
                                                << solCount.cellSolCount);

  if (solCount.hashCount == 0 && solCount.cellSolCount == 0) {
    cout << "c o [unis] The input formula is unsatisfiable." << endl;
    exit(-1);
  }

  int m = floor(solCount.hashCount + log2(solCount.cellSolCount) -
                log2(thresh_sampler_gen) + 0.5);

  // cout << "c o hashCount: " << solCount.hashCount << endl;

  // cout << "c o cellSolCount: " << solCount.cellSolCount << endl;
  // cout << "c o log2(cellSolCount): " << log2(solCount.cellSolCount) << endl;

  // cout << "c o pivot: " << thresh_sampler_gen << endl;
  // cout << "c o log2(pivot): " << log2(thresh_sampler_gen) << endl;

  if (conf.verb > 3) cout << "c o m: " << m << endl;
  if (m > 0)
    startiter = m;
  else
    startiter = 0; /* Indicate ideal sampling case */

  generate_samples(num_samples);

  if (rand_file.is_open()) rand_file.close();
  if (cert_file.is_open()) cert_file.close();
}

vector<Lit> Sampler::set_num_hashes(uint32_t num_wanted,
                                    map<uint64_t, Hash>& hashes) {
  vector<Lit> assumps;
  for (uint32_t i = 0; i < num_wanted; i++) {
    if (hashes.find(i) != hashes.end()) {
      assumps.push_back(Lit(hashes[i].act_var, true));
    } else {
      Hash h = add_hash(i);
      assumps.push_back(Lit(h.act_var, true));
      hashes[i] = h;
    }
  }
  assert(num_wanted == assumps.size());

  return assumps;
}

void Sampler::simplify() {
  verb_print(1, "[unis] simplifying");
  solver->set_sls(1);
  solver->set_intree_probe(1);
  solver->set_full_bve_iter_ratio(appmc->get_var_elim_ratio());
  solver->set_full_bve(1);
  solver->set_distill(1);
  solver->set_scc(1);

  solver->simplify();

  solver->set_sls(0);
  solver->set_intree_probe(0);
  solver->set_full_bve(0);
  solver->set_distill(0);
  // solver->set_scc(0);
}

void Sampler::generate_samples(uint32_t num_samples_needed) {
  double genStartTime = cpuTimeTotal();

  // Our optimised thresh
  thresh = ceil(conf.r_thresh_pivot * (1.0 + 2.0 * thresh_sampler_gen));

  // Original thresh from paper
  // thresh = 2 + 4 * ceil(thresh_sampler_gen);

  // cout << "epsilon = " << conf.epsilon << endl;
  // cout << "r_thresh_pivot = " << conf.r_thresh_pivot << endl;

  // cout << "delta = " << appmc->get_delta() << endl;
  // cout << "pivot = " << thresh_sampler_gen << endl;
  // cout << "thresh = " << thresh << endl;

  // cout << "m = " << startiter << endl;

  verb_print(1, "[unis] Samples requested: " << num_samples_needed);

  verb_print(1, "[unis] starting sample generation."
                    << " pivot: " << thresh_sampler_gen
                    << ", threshold: " << thresh << ", m: " << startiter);

  uint32_t num_samples = 0;
  if (startiter > 0) {
    verb_print(1, "[unis] non-ideal sampling case");
    while (num_samples < num_samples_needed)
      num_samples += gen_n_samples(num_samples_needed);
  } else {
    verb_print(1, "[unis] ideal sampling case");
    vector<vector<int>> out_solutions;
    const uint32_t count =
        bounded_sol_count(
            std::numeric_limits<uint32_t>::max()  // max no. solutions
            ,
            nullptr  // assumps is empty
            ,
            0  // number of hashes (information only)
            ,
            1  // min num. solutions
            ,
            nullptr  // gobal model (would be banned)
            ,
            &out_solutions)
            .solutions;
    assert(count > 0);

    std::uniform_int_distribution<unsigned> uid{0, count - 1};
    for (uint32_t i = 0; i < num_samples_needed; ++i) {
      auto it = out_solutions.begin();
      for (uint32_t j = uid(randomEngine); j > 0; --j) ++it;
      ++num_samples;
      callback_func(*it, 1, callback_func_data);
    }
  }

  verb_print(1, "[unis] Time to sample: "
                    << cpuTimeTotal() - genStartTime << " s"
                    << " -- Time count+samples: " << cpuTimeTotal() << " s");
  verb_print(1, "[unis] Samples generated: " << num_samples);
}

// Helper to reset the heuristics to aggressive state.
// void Sampler::reset_heuristic_params() {
//   baseline_time = std::nullopt;
//   current_window_total_time = 0.0;
//   samples_in_window = 0;
//   current_window_size = WINDOW_MIN;
//   current_slowdown_threshold = THRESHOLD_MIN;
// }

// void Sampler::load_and_initialize() {
//   // ... (all the solver borrowing logic is the same) ...
//   appmc_solver = appmc->get_solver();
//   is_using_appmc_solver = true;

//   base_solver = std::make_unique<SATSolver>();
//   my_copy_solver_to_solver(appmc_solver, base_solver.get());

//   // --- Initialize Heuristics ---
//   reset_heuristic_params();
// }

// void Sampler::reset_working_solver() {
//   // 1. Create a new, "cold" solver that *we* will own
//   auto new_solver = std::make_unique<SATSolver>();

//   new_solver->set_seed(appmc->get_seed());

//   // approxmc sets these options.
//   // new_solver->set_up_for_scalmc();
//   // new_solver->set_allow_otf_gauss();

//   // 2. Perform the FAST "simplified" copy.
//   // This is fast, but we know it's "dumb" - it's missing
//   // the '1 0' clause AND the '1 = TRUE' assignment.
//   my_copy_solver_to_solver(base_solver.get(), new_solver.get());

//   // 3. --- THE FIX for Unit Propagation ---
//   // We manually "prime" the new solver with the level-0
//   // assignments it missed.
//   // std::vector<CMSat::Lit> units = base_solver->get_zero_assigned_lits();

//   // This adds the '[1 = TRUE]' assignment back in.
//   // for (const CMSat::Lit& lit : units) new_solver->add_clause({lit});

//   // 4. Take ownership. The old `working_solver` (if any) is auto-deleted.
//   working_solver = std::move(new_solver);

//   // solver = working_solver.get();
//   // simplify();

//   // 5. SWITCH THE FLAG! We are now done with the ApproxMC solver.
//   is_using_appmc_solver = false;

//   // 5. (Good practice) Forget the borrowed pointer so we don't use it by
//   // mistake. The ApproxMC object is still responsible for deleting it at
//   // program exit.
//   appmc_solver = nullptr;

//   // 6. --- Reset Heuristics to Aggressive Default ---
//   reset_heuristic_params();
// }

// void Sampler::check_and_perform_reset() {
//   // 1. Check if the *current* window is full
//   if (samples_in_window < current_window_size) return;  // Window not full
//   yet

//   // 2. The window is full. Calculate this window's average time.
//   double recent_avg = current_window_total_time / samples_in_window;

//   // 3. Reset the counters for the *next* window
//   current_window_total_time = 0.0;
//   samples_in_window = 0;

//   // 4. Set the baseline on the very first run.
//   if (!baseline_time.has_value()) {
//     baseline_time.emplace(recent_avg);
//     return;  // Don't check on the first run, just set the baseline
//   }

//   const double current_fail_threshold =
//       baseline_time.value() * current_slowdown_threshold;

//   // --- The Hybrid Check ---
//   if (recent_avg <= current_fail_threshold) {
//     // --- PASSED: Reward the solver ---

//     int new_window_size =
//         static_cast<int>(current_window_size * WINDOW_GROW_FACTOR);
//     current_window_size = std::clamp(new_window_size, WINDOW_MIN,
//     WINDOW_MAX);

//     double new_threshold = current_slowdown_threshold + THRESHOLD_RELAX_STEP;
//     current_slowdown_threshold =
//         std::clamp(new_threshold, THRESHOLD_MIN, THRESHOLD_MAX);
//   } else {
//     // --- FAILED: We are too slow ---

//     // Is this solver still in the "Nursery"?
//     // We define "Nursery" as any solver that hasn't been "promoted" at least
//     // once.
//     const bool is_in_nursery = (current_window_size <= WINDOW_MIN);
//     if (is_in_nursery)
//       // "Nursery" failure = Hard Reset
//       // This solver failed its very first test.
//       // It's a "simple formula" case. It gets NO leniency.
//       reset_working_solver();
//     else {
//       // "Trusted" failure = Check Minor/Major failure
//       // // This solver was *proven* (it passed at least one check).
//       // NOW we grant it the "Minor vs. Major failure" logic.
//       const double catastrophe_limit =
//           current_fail_threshold * CATASTROPHE_FACTOR;
//       if (recent_avg < catastrophe_limit) {
//         // "MINOR" FAILURE (Soft Reset)
//         int new_window_size = static_cast<int>(current_window_size / 2.0);
//         current_window_size =
//             std::clamp(new_window_size, WINDOW_MIN, WINDOW_MAX);
//       } else
//         // "MAJOR" FAILURE (Hard Reset)
//         reset_working_solver();
//     }
//   }
// }

uint32_t Sampler::gen_n_samples(const uint32_t num_samples_needed) {
  SparseData sparse_data(-1);
  // Total number of samples obtained.
  uint32_t num_samples = 0;
  // Number of tries so far for current sample.
  uint32_t num_tries = 0;

  while (num_samples < num_samples_needed) {
    // if (num_samples > 0 && num_samples % 100 == 0) reset_working_solver();

    // 1. Check if the current window is full and if we need to reset.
    // check_and_perform_reset();

    // 2. Decide which solver to use for this iteration.
    // solver = is_using_appmc_solver ? appmc_solver : working_solver.get();

    // 3. Start the timer.
    // auto start = std::chrono::high_resolution_clock::now();

    // 4. Do the actual work.
    map<uint64_t, Hash> hashes;
    // For unisamp, startiter represents m, which we directly use as our hash
    // count
    const vector<Lit> assumps = set_num_hashes(startiter, hashes);
    num_tries++;
    const bool ok = bounded_sol_count_unisamp(&assumps, startiter, num_tries);
    if (ok) {
      num_samples++;

      cout << "c o [unis] Found solution " << num_samples << " (out of "
           << num_samples_needed << ") after a total of " << num_tries
           << " tries" << endl;

      num_tries = 0;
    }

    // 5. Stop the timer.
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;

    // 6. Update the timers for the *next* check
    // current_window_total_time += elapsed.count();
    // samples_in_window++;

    if (appmc->get_simplify() >= 1) simplify();
  }

  return num_samples;
}

vector<int> Sampler::get_solution_ints(const vector<lbool>& model) {
  vector<int> solution;
  // std::bernoulli_distribution dist(0.5);
  for (const uint32_t var : conf.full_sampling_vars) {
    assert(model[var] != l_Undef);
    solution.push_back(((model[var] != l_True) ? -1 : 1) *
                       (static_cast<int>(var) + 1));
  }
  return solution;
}

bool Sampler::gen_rhs() {
  std::bernoulli_distribution dist(0.5);
  bool rhs = dist(randomEngine);
  // cout << "rnd rhs:" << (int)rhs << endl;
  return rhs;
}

string Sampler::gen_rnd_bits(const uint32_t size,
                             const uint32_t /*hash_index*/) {
  string randomBits;
  std::uniform_int_distribution<uint32_t> dist{0, 1000};
  uint32_t cutoff = 500;

  while (randomBits.size() < size) {
    bool val = dist(randomEngine) < cutoff;
    randomBits += '0' + val;
  }
  assert(randomBits.size() >= size);

  // cout << "rnd bits: " << randomBits << endl;
  return randomBits;
}

void Sampler::print_xor(const vector<uint32_t>& vars, const uint32_t rhs) {
  cout << "c o [unis] Added XOR ";
  bool first = true;
  for (const auto& var : vars) {
    if (!first) cout << " + ";
    cout << var + 1;
    first = false;
  }
  cout << " = " << (rhs ? "True" : "False") << endl;
}

/* Number of solutions to return from one invocation of gen_n_samples. */
uint32_t Sampler::sols_to_return(uint32_t numSolutions) {
  if (startiter == 0)
    return numSolutions;
  else
    return 1;
}

void Sampler::check_model(const vector<lbool>& model,
                          const HashesModels* const hm,
                          const uint32_t hashCount) {
  for (uint32_t var : appmc->get_sampling_set()) assert(model[var] != l_Undef);

  if (!hm) return;

  bool ok = true;
  for (const auto& h : hm->hashes) {
    // This hash is number: h.first
    // Only has to match hashes at & below
    // Notice that "h.first" is numbered from 0, so it's a "<" not "<="
    if (h.first < hashCount) {
      // cout << "Checking model against hash" << h.first << endl;
      ok &= check_model_against_hash(h.second, model);
      if (!ok) break;
    }
  }
  assert(ok);
}

bool Sampler::check_model_against_hash(const Hash& h,
                                       const vector<lbool>& model) {
  bool rhs = h.rhs;
  for (const uint32_t var : h.hash_vars) {
    assert(model[var] != l_Undef);
    rhs ^= model[var] == l_True;
  }

  // If we started with rhs=FALSE and we XOR-ed in only FALSE
  // rhs is FALSE but we should return TRUE

  // If we started with rhs=TRUE and we XOR-ed in only one TRUE
  // rhs is FALSE but we should return TRUE

  // hence return !rhs
  return !rhs;
}
