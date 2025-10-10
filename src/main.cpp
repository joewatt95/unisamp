/*
 UniGen

 Copyright (c) 2019-2020, Mate Soos and Kuldeep S. Meel. All rights reserved
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

#include <approxmc/approxmc.h>
#include <arjun/arjun.h>
#include <cryptominisat5/dimacsparser.h>
#include <cryptominisat5/streambuffer.h>

#include <argparse/argparse.hpp>

#if defined(__GNUC__) && defined(__linux__)
#include <fenv.h>
#endif

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "time_mem.h"
#include "unigen.h"
#include "unisamp.h"

using namespace CMSat;
using namespace UniGen;
using namespace UniSamp;
using std::cout;
using std::endl;
using std::set;
using std::string;
using std::vector;

ApproxMC::AppMC* appmc = NULL;
UniG* unigen = NULL;
UniS* unisamp = NULL;
argparse::ArgumentParser program =
    argparse::ArgumentParser("unigen", UniGen::UniG::get_version_sha1(),
                             argparse::default_arguments::help);
std::unique_ptr<CMSat::FieldGen> fg;

uint32_t verb = 1;
uint32_t seed;
double epsilon = 0.3;
double r_thresh_pivot = 1.5;
double delta;
uint32_t verb_banning_cls = 0;
uint32_t simplify;
double var_elim_ratio;
uint32_t reuse_models = 1;
uint32_t sparse;

// Arjun
int do_arjun = 0;
int arjun_gates = 0;
ArjunNS::SimpConf simp_conf;
ArjunNS::Arjun::ElimToFileConf etof_conf;
int with_e = 0;

// UniGen
uint32_t num_samples = 500;
int multisample;
string sample_fname;
double kappa =
    0.638; /* Corresponds to UniGen's epsilon=16 in the TACAS-15 paper */
bool verb_sampler_cls;

#define myopt(name, var, fun, hhelp)                             \
  program.add_argument(name)                                     \
      .action([&](const auto& a) { var = std::fun(a.c_str()); }) \
      .default_value(var)                                        \
      .help(hhelp)
#define myopt2(name1, name2, var, fun, hhelp)                    \
  program.add_argument(name1, name2)                             \
      .action([&](const auto& a) { var = std::fun(a.c_str()); }) \
      .default_value(var)                                        \
      .help(hhelp)

void print_version() {
  std::stringstream ss;
  cout << "c o UniGen SHA1: " << UniGen::UniG::get_version_sha1() << endl;
  cout << "c o CMS SHA1: " << CMSat::SATSolver::get_version_sha1() << endl;
  cout << "c o Arjun SHA1: " << ArjunNS::Arjun ::get_version_sha1() << endl;
  cout << "c o Arjun SBVA SHA1: " << ArjunNS::Arjun::get_sbva_version_sha1()
       << endl;
  cout << "c o ApproxMC SHA1: " << ApproxMC::AppMC::get_version_sha1() << endl;
  cout << CMSat::SATSolver::get_thanks_info("c o ");
  cout << ArjunNS::Arjun::get_thanks_info("c o ");
}

void SIGINT_handler(int) {
  return;  // Perhaps we should output all the generated samples so far
}

void add_unigen_options() {
  ApproxMC::AppMC tmp(fg);
  // epsilon = tmp.get_epsilon();
  delta = tmp.get_delta();
  simplify = tmp.get_simplify();
  var_elim_ratio = tmp.get_var_elim_ratio();
  sparse = tmp.get_sparse();
  seed = tmp.get_seed();

  myopt2("-v", "--verb", verb, atoi, "Verbosity");
  program.add_argument("-v", "--version")
      .action([&](const auto&) {
        print_version();
        exit(0);
      })
      .flag()
      .help("Print version and exit");
  myopt2("-s", "--seed", seed, atoi, "Seed");
  myopt2("-e", "--epsilon", epsilon, stod,
         "Tolerance parameter, i.e. how close is the count from the correct "
         "count? "
         "Count output is within bounds of (exact_count/(1+e)) < count < "
         "(exact_count*(1+e)). "
         "So e=0.8 means we'll output at most 180%% of exact count and at "
         "least 55%% of exact count. "
         "Lower value means more precise.");
  myopt2("-r", "--r-thresh-pivot", r_thresh_pivot, atof,
         "Ratio between threshold and pivot. Default of 1.5.");
  myopt2(
      "-d", "--delta", delta, stod,
      "Confidence parameter, i.e. how sure are we of the result? "
      "(1-d) = probability the count is within range as per epsilon parameter. "
      "So d=0.2 means we are 80%% sure the count is within range as specified "
      "by epsilon. "
      "The lower, the higher confidence we have in the count.");
  myopt("--kappa", kappa, atof, "Uniformity parameter (see TACAS-15 paper)");

  myopt("--arjun", do_arjun, atoi, "Use arjun to minimize sampling set");
  myopt("--sparse", sparse, atoi, "Generate sparse XORs when possible");
  myopt("--reusemodels", reuse_models, atoi,
        "Reuse models while counting solutions");
  myopt("--verbanbcls", verb_banning_cls, atoi,
        "Print banning clause + xor clauses. Highly verbose.");
  myopt("--simplify", simplify, atoi, "Simplify agressiveness");
  myopt("--velimratio", var_elim_ratio, atof,
        "Variable elimination ratio for each simplify run");
  myopt("--samples", num_samples, atoi, "Number of random samples to generate");
  myopt("--multisample", multisample, atoi,
        "Return multiple samples from each call");
  myopt("--sampleout", sample_fname, string, "Write samples to this file");
  myopt("--verbsamplercls", verb_sampler_cls, atoi,
        "Print XOR constraints added for sampling");

  program.add_argument("inputfile").remaining().help("input CNF");
}

void parse_supported_options(int argc, char** argv) {
  add_unigen_options();
  try {
    program.parse_args(argc, argv);
    if (program.is_used("--help")) {
      cout << "Probilistic Uniform Sample Generator" << endl
           << endl
           << "approxmc [options] inputfile" << endl;
      cout << program << endl;
      exit(0);
    }
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    exit(-1);
  }
}

template <class T>
void parse_file(const std::string& filename, T* reader) {
#ifndef USE_ZLIB
  FILE* in = fopen(filename.c_str(), "rb");
  DimacsParser<StreamBuffer<FILE*, CMSat::FN>, T> parser(reader, nullptr, 0,
                                                         fg);
#else
  gzFile in = gzopen(filename.c_str(), "rb");
  DimacsParser<StreamBuffer<gzFile, CMSat::GZ>, T> parser(reader, nullptr, 0,
                                                          fg);
#endif
  if (in == nullptr) {
    std::cout << "ERROR! Could not open file '" << filename
              << "' for reading: " << strerror(errno) << endl;
    std::exit(-1);
  }
  if (!parser.parse_DIMACS(in, true)) exit(-1);
#ifndef USE_ZLIB
  fclose(in);
#else
  gzclose(in);
#endif

  if (!reader->get_sampl_vars_set()) {
    vector<uint32_t> tmp;
    for (uint32_t i = 0; i < reader->nVars(); i++) tmp.push_back(i);
    reader->set_sampl_vars(tmp);
  } else {
    // Check if CNF has all vars as indep. Then its's all_indep
    set<uint32_t> tmp;
    for (auto const& s : reader->get_sampl_vars()) {
      if (s >= reader->nVars()) {
        cout << "ERROR: Sampling var " << s + 1
             << " is larger than number of vars in formula: " << reader->nVars()
             << endl;
        exit(-1);
      }
      tmp.insert(s);
    }
    if (tmp.size() == reader->nVars()) etof_conf.all_indep = true;
    if (!reader->get_opt_sampl_vars_set()) {
      reader->set_opt_sampl_vars(reader->get_sampl_vars());
    }
  }
}

void mycallback(const std::vector<int>& solution, void* file) {
  std::ostream* os = (std::ostream*)file;
  for (uint32_t i = 0; i < solution.size(); i++) {
    (*os) << solution[i] << " ";
  }
  (*os) << "0" << endl;
}

inline double stats_line_percent(double num, double total) {
  if (total == 0) {
    return 0;
  } else {
    return num / total * 100.0;
  }
}

void print_final_indep_set(const vector<uint32_t>& indep_set,
                           uint32_t orig_sampling_set_size) {
  cout << "c o ind ";
  for (const uint32_t s : indep_set) cout << s + 1 << " ";
  cout << "0" << endl;

  cout << "c o [arjun] final set size:      " << std::setw(7)
       << indep_set.size() << " percent of original: " << std::setw(6)
       << std::setprecision(4)
       << stats_line_percent(indep_set.size(), orig_sampling_set_size) << " %"
       << endl;
}

template <class T>
void check_sanity_sampling_vars(T vars, const uint32_t nvars) {
  for (const auto& v : vars)
    if (v >= nvars) {
      cout << "ERROR: sampling set provided is incorrect, it has a variable in "
              "it: "
           << v + 1
           << " that is larger than the total number of variables: " << nvars
           << endl;
      exit(-1);
    }
}

// int main() {
//   // --- Example 1: Basic Sampling and Failure ---
//   std::cout << "## 1. Basic Sampling & Failure ##\n";
//   {
//     // using StringMeasure = SubProbMeasure<std::string, std::string, Rng>;
//     std::vector<std::string> planets = {"Mercury", "Venus", "Earth", "Mars"};
//     std::vector<std::string> empty = {};

//     // Create a measure that samples uniformly from the 'planets' vector
//     auto pick_planet = sub_prob_measure::uniform_range(planets);

//     std::cout << "Sampling a planet: " << *pick_planet() << std::endl;
//     std::cout << "Sampling another planet: " << *pick_planet() << std::endl;

//     // Create a measure from an empty vector, which will always fail
//     auto pick_from_empty = sub_prob_measure::uniform_range(empty);
//     if (!pick_from_empty()) {
//       std::cout << "Sampling from an empty vector correctly resulted in "
//                    "failure (nullopt).\n";
//     }
//   }
//   std::cout << "\n" << std::string(40, '-') << "\n\n";

//   // --- Example 2: Transforming a Result (`map`) ---
//   std::cout << "## 2. Transforming a Result (`map`) ##\n";
//   {
//     // 1. Start with a measure that produces an integer (a d20 roll)
//     auto roll_d20 = sub_prob_measure::uniform_int(1, 20);

//     // 2. Define a pure function to transform the integer into a string
//     auto describe_roll = [](int roll) {
//       if (roll == 20) return "Critical Hit! (" + std::to_string(roll) + ")";
//       if (roll == 1) return "Critical Fail! (" + std::to_string(roll) + ")";
//       return "Normal Roll. (" + std::to_string(roll) + ")";
//     };

//     // 3. Use `transform` to create a new measure that produces a string
//     auto described_roll_measure = roll_d20.transform(describe_roll);

//     std::cout << "Executing the transformed measure: "
//               << *described_roll_measure() << std::endl;
//   }
//   std::cout << "\n" << std::string(40, '-') << "\n\n";

//   // --- Example 3: Chaining Probabilistic Choices (`>>=`) ---
//   std::cout << "## 3. Chaining Probabilistic Choices (`>>=`) ##\n";
//   {
//     // Define two categories of items
//     std::vector<std::string> tools = {"Hammer", "Wrench", "Screwdriver"};
//     std::vector<std::string> fruits = {"Apple", "Banana", "Cherry"};
//     std::vector<std::vector<std::string>> categories = {tools, fruits};

//     // 1. First measure: probabilistically pick a category (a vector of strings)
//     auto pick_category = sub_prob_measure::uniform_range(categories);

//     // 2. Second measure factory: a function that takes a category and creates a
//     // measure to pick an item from it
//     auto pick_item_from = [](const std::vector<std::string>& category) {
//       return sub_prob_measure::uniform_range(category);
//     };

//     // 3. Chain them with and_then (>>=): first pick a category, then pick an
//     // item from it
//     auto pick_random_item = pick_category.and_then(pick_item_from);

//     std::cout << "Picking a random item from a random category...\n";
//     for (int i = 0; i < 5; ++i) {
//       std::cout << "  - Sampled: " << *pick_random_item() << std::endl;
//     }
//   }
//   std::cout << "\n" << std::string(40, '-') << "\n\n";

//   // --- Example 4: Scaling a Measure's Probability (`scale`) ---
//   std::cout << "## 4. Scaling a Measure's Probability (`scale`) ##\n";
//   {
//     auto measure =
//         sub_prob_measure::pure(42);  // A measure that always produces 42

//     // Scale it so it only succeeds with a 20% probability
//     auto scaled_measure = measure.scale(0.2);

//     int successes = 0;
//     int trials = 100;
//     for (int i = 0; i < trials; ++i) {
//       if (scaled_measure()) {
//         successes++;
//       }
//     }
//     std::cout << "Scaled measure succeeded " << successes << " out of "
//               << trials << " times.\n";
//   }
//   std::cout << "\n" << std::string(40, '-') << "\n\n";

//   std::cout << "## 5. Input stream of numbers ##\n";

//   std::istringstream input_stream("1 3 5 9 11");
//   auto number_stream = std::ranges::istream_view<int>(input_stream);

//   // auto number_stream = {1, 3, 5, 8, 11};

//   std::cout << "Uniformly sampling a number..." << std::endl;
//   sub_prob_measure::uniform_range(number_stream).transform([](int n) {
//     std::cout << "Sampled number: " << n << std::endl;
//     return n;
//   })();
//   // std::cout << "Sampled number: " << std::to_string(number.value()) << std::endl;

//   auto process_number = [](int current_sum, int n) {
//     std::cout << "Processing number: " << n << "..." << std::endl;
//     return sub_prob_measure::guard(n % 2 != 0)
//         .transform([=](const std::monostate&) { return current_sum + n; });
//   };

//   std::istringstream input_stream_("1 3 5 9 11");
//   auto number_stream_ = std::ranges::istream_view<int>(input_stream_);

//   auto computation = sub_prob_measure::foldl_m(0, number_stream_, process_number);

//   std::cout << "Starting the monadic fold over the integer stream..." << std::endl;
//   std::optional<int> result = computation();

//   std::cout << "\n--- Computation Finished ---" << std::endl;
//   if (result) {
//     std::cout << "Success! The fold processed the entire stream." << std::endl;
//     std::cout << "Final sum of odd numbers: " << *result << std::endl;
//   } else {
//     std::cout << "Failure. The fold was terminated because an even number was "
//                  "encountered."
//               << std::endl;
//   }
//   std::cout << "\n" << std::string(40, '-') << "\n\n";

//   return 0;
// }

int main(int argc, char** argv) {
#if defined(__GNUC__) && defined(__linux__)
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif

  // Reconstruct the command line so we can emit it later if needed
  string command_line;
  for (int i = 0; i < argc; i++) {
    command_line += string(argv[i]);
    if (i + 1 < argc) {
      command_line += " ";
    }
  }

  fg = std::make_unique<ArjunNS::FGenMpz>();
  appmc = new ApproxMC::AppMC(fg);
  unigen = new UniG(appmc);
  unisamp = new UniS(appmc);
  simp_conf.appmc = true;
  simp_conf.oracle_sparsify = false;
  simp_conf.iter1 = 2;
  simp_conf.iter2 = 0;
  etof_conf.do_bce = false;
  etof_conf.do_extend_indep = false;
  parse_supported_options(argc, argv);
  if (verb) {
    print_version();
    cout << "c o executed with command line: " << command_line << endl;
  }

  appmc->set_verbosity(verb);
  appmc->set_seed(seed);
  appmc->set_reuse_models(reuse_models);
  appmc->set_sparse(sparse);
  appmc->set_simplify(simplify);
  appmc->set_var_elim_ratio(var_elim_ratio);
  vector<uint32_t> sampling_vars_orig;

  const auto& files = program.get<std::vector<std::string>>("inputfile");
  if (files.empty()) {
    cout << "ERROR: you provided --inputfile but no file. Strange. Exiting. "
         << endl;
    exit(-1);
  }
  const string fname(files[0]);
  ArjunNS::SimplifiedCNF cnf(fg);
  if (do_arjun) {
    parse_file(fname, &cnf);
    sampling_vars_orig = cnf.sampl_vars;
    const auto orig_sampl_vars = cnf.sampl_vars;
    double my_time = cpuTime();
    ArjunNS::Arjun arjun;
    arjun.set_verb(verb);
    arjun.set_or_gate_based(arjun_gates);
    arjun.set_xor_gates_based(arjun_gates);
    arjun.set_ite_gate_based(arjun_gates);
    arjun.set_irreg_gate_based(arjun_gates);
    arjun.standalone_minimize_indep(cnf, etof_conf.all_indep);
    if (with_e) arjun.standalone_elim_to_file(cnf, etof_conf, simp_conf);
    appmc->new_vars(cnf.nVars());
    appmc->set_sampl_vars(cnf.sampl_vars);
    for (const auto& c : cnf.clauses) appmc->add_clause(c);
    for (const auto& c : cnf.red_clauses) appmc->add_red_clause(c);
    appmc->set_multiplier_weight(cnf.multiplier_weight);
    print_final_indep_set(cnf.sampl_vars, orig_sampl_vars.size());
    cout << "c o [arjun] Arjun finished. T: " << (cpuTime() - my_time) <<
    endl;
  } else {
    parse_file(fname, appmc);
    sampling_vars_orig = appmc->get_sampl_vars();
    print_final_indep_set(appmc->get_sampl_vars(),
                          appmc->get_sampl_vars().size());
  }

  unisamp->set_epsilon(epsilon);
  unisamp->set_r_thresh_pivot(r_thresh_pivot);

  // Hack to set approxmc parameters for unisamp.
  appmc->set_epsilon(sqrt(2) - 1);
  // appmc->set_delta(0.01202);

  appmc->set_delta((pow(r_thresh_pivot - 1, 2) * epsilon) /
                   (3.19899995 * r_thresh_pivot * (1 + epsilon)));

  // delta from original unisamp paper, when eps = 0.3
  // appmc->set_delta(0.3 / 4);

  auto sol_count = appmc->count();
  unisamp->set_verbosity(verb);
  unisamp->set_verb_sampler_cls(verb_banning_cls);
  // unigen->set_kappa(kappa);
  unisamp->set_full_sampling_vars(sampling_vars_orig);

  void* myfile = &std::cout;
  std::ofstream sample_out;
  if (sample_fname != "") {
    sample_out.open(sample_fname.c_str());
    if (!sample_out.is_open()) {
      cout << "[Sampler] Cannot open samples file '" << sample_fname
           << "' for writing." << endl;
      exit(-1);
    }
    myfile = &sample_out;
  }
  unisamp->set_callback(mycallback, myfile);
  // unigen->sample(&sol_count, num_samples);
  unisamp->sample(&sol_count, num_samples);

  delete unigen;
  delete unisamp;
  delete appmc;

  return 0;
}
