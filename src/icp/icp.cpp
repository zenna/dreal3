/*********************************************************************
Author: Soonho Kong <soonhok@cs.cmu.edu>

dReal -- Copyright (C) 2013 - 2015, Soonho Kong, Sicun Gao, and Edmund Clarke

dReal is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dReal is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with dReal. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************/

#include <tuple>
#include <random>
#include <vector>
#include <map>

#include "icp/icp.h"
#include "util/logging.h"
#include "util/stat.h"

namespace dreal {

void output_solution(box const & b, SMTConfig & config, unsigned i) {
    if (i > 0) {
        cout << i << "-th ";
    }
    cout << "Solution:" << endl;
    cout << b << endl;
    if (!config.nra_model_out.is_open()) {
        config.nra_model_out.open(config.nra_model_out_name.c_str(), std::ofstream::out | std::ofstream::trunc);
        if (config.nra_model_out.fail()) {
            cout << "Cannot create a file: " << config.nra_model_out_name << endl;
            exit(1);
        }
    }
    display(config.nra_model_out, b, false, true);
}

box naive_icp::solve(box b, contractor const & ctc, SMTConfig & config) {
    vector<box> solns;
    vector<box> box_stack;
    box_stack.push_back(b);
    do {
        DREAL_LOG_INFO << "icp_loop()"
                       << "\t" << "box stack Size = " << box_stack.size();
        b = box_stack.back();
        box_stack.pop_back();
        try {
            b = ctc.prune(b, config);
            if (config.nra_use_stat) { config.nra_stat.increase_prune(); }
        } catch (contractor_exception & e) {
            // Do nothing
        }
        if (!b.is_empty()) {
            tuple<int, box, box> splits = b.bisect(config.nra_precision);
            if (config.nra_use_stat) { config.nra_stat.increase_branch(); }
            int const i = get<0>(splits);
            if (i >= 0) {
                box const & first  = get<1>(splits);
                box const & second = get<2>(splits);
                if (second.is_bisectable()) {
                    box_stack.push_back(second);
                    box_stack.push_back(first);
                } else {
                    box_stack.push_back(first);
                    box_stack.push_back(second);
                }
                if (config.nra_proof) {
                    config.nra_proof_out << "[branched on "
                                         << b.get_name(i)
                                         << "]" << endl;
                }
            } else {
                config.nra_found_soln++;
                if (config.nra_found_soln >= config.nra_multiple_soln) {
                    break;
                }
                if (config.nra_multiple_soln > 1) {
                    // If --multiple_soln is used
                    output_solution(b, config, config.nra_found_soln);
                }
                solns.push_back(b);
            }
        }
    } while (box_stack.size() > 0);
    if (config.nra_multiple_soln > 1 && solns.size() > 0) {
        return solns.back();
    } else {
        assert(!b.is_empty() || box_stack.size() == 0);
        // cerr << "BEFORE ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";
        b.adjust_bound(box_stack);
        // cerr << "AFTER  ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";
        return b;
    }
}

inline float log1pexp(float x)
{   return x<-88.029691931? 0.: log1p(exp(x));
}
inline float sum_log_prob(float a, float b)
{   return a>b? a+log1pexp(b-a):  b+log1pexp(a-b);
}
inline double log1pexp(double x)
{   return x<-709.089565713? 0.: log1p(exp(x));
}
inline double sum_log_prob(double a, double b)
{   return a>b? a+log1pexp(b-a):  b+log1pexp(a-b);
}
inline long double log1pexp(long double x)
{   return x<-11355.8302591? 0.: log1p(exp(x));
}
inline long double sum_log_prob(long double a, long double b)
{   return a>b? a+log1pexp(b-a):  b+log1pexp(a-b);
}

double logprob_depth(unsigned int depth) {
  const double loghalf = log(0.5);
  double logone = 0.0;
  for (unsigned int i = 0; i < depth; ++i) {
    logone += loghalf;
  }
  return logone;
}

double proposal_prob(unsigned int depth, const std::map<unsigned int, unsigned int> &nempty) {
  double q = logprob_depth(depth);

  for (const auto depthnum : nempty) {
    double logprob = logprob_depth(depthnum.first);
    for (unsigned int i = 0; i < depthnum.second; ++i) {
      q = sum_log_prob(q, logprob);
    }
  }
  return q;
}

bool random_icp::random_bool() {
    static thread_local std::mt19937_64 rg(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> m_dist(0, 1);
    return m_dist(rg) >= 0.5;
}

class BoxComparator
{
public:
    bool operator()(const box& b1, const box& b2){
        return b1.depth < b2.depth;
    }
};

box sample_icp::solve(box init_b, contractor const & ctc, SMTConfig & config ) {
    vector<box> solns;
    vector<box> box_stack;
    box b = init_b;
    box_stack.push_back(b);
    std::map<unsigned int, unsigned int> nempty; // Number of empty boxes at particular depth
    int nconflicts = 0;

    // Random Restarts
    // I need a container for my stack that will allow me to add elements to the end, i.e lower depth means its at the end and pop from either end
    // std::set<box, BoxComparator> box_stack;
    do {
        DREAL_LOG_INFO << "icp_loop()"
                       << "\t" << "box stack Size = " << box_stack.size();
        b = box_stack.back();
        box_stack.pop_back();
        // std::cout << "Depth: " << b.depth << std::endl;
        try {
            b = ctc.prune(b, config);
        } catch (contractor_exception & e) {
            // Do nothing
        }
        if (!b.is_empty()) {
            tuple<int, box, box> splits = b.bisect(config.nra_precision);
            int const i = get<0>(splits);
            // std::cout << "i is" << i << std::endl;
            if (i >= 0) {
                box & first  = get<1>(splits);
                box & second = get<2>(splits);
                first.depth = b.depth + 1;
                second.depth = b.depth + 1;
                bool leftright = random_icp::random_bool();
                // std::cout << "left or right? " << leftright << std::endl;
                if (leftright) {
                    box_stack.push_back(second);
                    box_stack.push_back(first);
                } else {
                    box_stack.push_back(first);
                    box_stack.push_back(second);
                }
                if (config.nra_proof) {
                    config.nra_proof_out << "[branched on "
                                         << b.get_name(i)
                                         << "]" << endl;
                }
            } else {
                config.nra_found_soln++;
                if (config.nra_found_soln >= config.nra_multiple_soln) {
                    break;
                }
                if (config.nra_multiple_soln > 1) {
                    // If --multiple_soln is used
                    output_solution(b, config, config.nra_found_soln);
                }
                solns.push_back(b);
            }
        }
        else {
          // std::cout << "Backtracking at Depth: " << b.depth << std::endl;
          nempty[b.depth] += 1;
          nconflicts += 1;
          int nconflicts_before_restart = 200;
          if (nconflicts > nconflicts_before_restart) {
            std::cout << "Restarting" << std::endl;
            box_stack.empty();
            box_stack.push_back(init_b);
            nempty.empty();
            nconflicts = 0;
            continue;
          }
        }
    } while (box_stack.size() > 0);
    if (config.nra_multiple_soln > 1 && solns.size() > 0) {
        return solns.back();
    } else {
        assert(!b.is_empty() || box_stack.size() == 0);
        // cerr << "BEFORE ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";
        b.adjust_bound(box_stack);
        // cerr << "AFTER  ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";

        double q = proposal_prob(b.depth, nempty);
        b.logprob = q;
        return b;
    }
}


box ncbt_icp::solve(box b, contractor const & ctc, SMTConfig & config) {
    static unsigned prune_count = 0;
    vector<box> box_stack;
    vector<int> bisect_var_stack;
    box_stack.push_back(b);
    bisect_var_stack.push_back(-1);  // Dummy var
    do {
        // Loop Invariant
        assert(box_stack.size() == bisect_var_stack.size());
        DREAL_LOG_INFO << "new_icp_loop()"
                       << "\t" << "box stack Size = " << box_stack.size();
        b = box_stack.back();
        try {
            b = ctc.prune(b, config);
            if (config.nra_use_stat) { config.nra_stat.increase_prune(); }
        } catch (contractor_exception & e) {
            // Do nothing
        }
        prune_count++;
        box_stack.pop_back();
        bisect_var_stack.pop_back();
        if (!b.is_empty()) {
            // SAT
            tuple<int, box, box> splits = b.bisect(config.nra_precision);
            if (config.nra_use_stat) { config.nra_stat.increase_branch(); }
            int const index = get<0>(splits);
            if (index >= 0) {
                box const & first    = get<1>(splits);
                box const & second   = get<2>(splits);
                if (second.is_bisectable()) {
                    box_stack.push_back(second);
                    box_stack.push_back(first);
                } else {
                    box_stack.push_back(first);
                    box_stack.push_back(second);
                }
                bisect_var_stack.push_back(index);
                bisect_var_stack.push_back(index);
            } else {
                break;
            }
        } else {
            // UNSAT
            while (box_stack.size() > 0) {
                assert(box_stack.size() == bisect_var_stack.size());
                int bisect_var = bisect_var_stack.back();
                ibex::BitSet const & input = ctc.input();
                DREAL_LOG_DEBUG << ctc;
                if (!input[bisect_var]) {
                    box_stack.pop_back();
                    bisect_var_stack.pop_back();
                } else {
                    break;
                }
            }
        }
    } while (box_stack.size() > 0);
    DREAL_LOG_DEBUG << "prune count = " << prune_count;
    b.adjust_bound(box_stack);
    return b;
}

box random_icp::solve(box b, contractor const & ctc, SMTConfig & config ) {
    vector<box> solns;
    vector<box> box_stack;
    box_stack.push_back(b);
    do {
        DREAL_LOG_INFO << "icp_loop()"
                       << "\t" << "box stack Size = " << box_stack.size();
        b = box_stack.back();
        box_stack.pop_back();
        try {
            b = ctc.prune(b, config);
        } catch (contractor_exception & e) {
            // Do nothing
        }
        if (!b.is_empty()) {
            tuple<int, box, box> splits = b.bisect(config.nra_precision);
            int const i = get<0>(splits);
            if (i >= 0) {
                box const & first  = get<1>(splits);
                box const & second = get<2>(splits);
                if (random_bool()) {
                    box_stack.push_back(second);
                    box_stack.push_back(first);
                } else {
                    box_stack.push_back(first);
                    box_stack.push_back(second);
                }
                if (config.nra_proof) {
                    config.nra_proof_out << "[branched on "
                                         << b.get_name(i)
                                         << "]" << endl;
                }
            } else {
                config.nra_found_soln++;
                if (config.nra_found_soln >= config.nra_multiple_soln) {
                    break;
                }
                if (config.nra_multiple_soln > 1) {
                    // If --multiple_soln is used
                    output_solution(b, config, config.nra_found_soln);
                }
                solns.push_back(b);
            }
        }
    } while (box_stack.size() > 0);
    if (config.nra_multiple_soln > 1 && solns.size() > 0) {
        return solns.back();
    } else {
        assert(!b.is_empty() || box_stack.size() == 0);
        // cerr << "BEFORE ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";
        b.adjust_bound(box_stack);
        // cerr << "AFTER  ADJUST_BOUND\n==================\n" << b << "=========================\n\n\n";
        return b;
    }
}
}  // namespace dreal
