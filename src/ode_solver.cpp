/*********************************************************************
Author: Soonho Kong <soonhok@cs.cmu.edu>
        Sicun Gao <sicung@cs.cmu.edu>
        Edmund Clarke <emc@cs.cmu.edu>

dReal -- Copyright (C) 2013, Soonho Kong, Sicun Gao, and Edmund Clarke

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

#include "ode_solver.h"
#include <limits>
#include <boost/algorithm/string/join.hpp>
// #include <boost/lambda/lambda.hpp>
// #include <boost/lambda/bind.hpp>
// #include <boost/lambda/if.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/zip_iterator.hpp>

using namespace capd;
using boost::tuple;
using boost::make_tuple;
using boost::make_zip_iterator;
using boost::algorithm::join;

ode_solver::ode_solver(int group,
                       SMTConfig& c,
                       set < Enode* > & ode_vars,
                       rp_box b,
                       std::map<Enode*, int>& enode_to_rp_id,
                       bool & ode_result
    ) :
    _group(group),
    _config(c),
    _ode_vars(ode_vars),
    _b(b),
    _enode_to_rp_id(enode_to_rp_id),
    ODEresult(ode_result),
    stepControl(c.nra_ODE_step)
{
}

ode_solver::~ode_solver()
{

}

void ode_solver::print_datapoint(ostream& out, const interval& t, const interval& v) const {
    out << "{ "
        << "\"time\": " << t << ", "
        << "\"enclosure\": " << v;
    out << "}";
}

void ode_solver::print_trace(ostream& out,
            const string key,
            const int idx,
            const list<pair<const interval, const IVector> > & trajectory) const
{
    out << "{" << endl;
    out << "\t" << "\"key\": \"" << key << "\"," << endl;
    out << "\t" << "\"group\": \"" << _group << "\"," << endl;
    out << "\t" << "\"values\": [" << endl;

    list<pair<const interval, const IVector> >::const_iterator iter = trajectory.begin();
    print_datapoint(out, iter->first, iter->second[idx]);

    for(++iter; iter != trajectory.end(); iter++) {
        out << ", " << endl;
        print_datapoint(out, iter->first, iter->second[idx]);
    }
    out << endl;
    out << "\t" << "]" << endl;
    out << "}" << endl;
}

void ode_solver::print_trajectory(ostream& out) const
{
    out.precision(12);
    out << ",[" << endl;
    print_trace(out, var_list[0], 0, trajectory);

    for(size_t i = 1; i < var_list.size(); i++) {
        out << ", " << endl;
        print_trace(out, var_list[i], i, trajectory);
    }
    out << endl << "]" << endl;
}

void ode_solver::prune_trajectory(const interval& time)
{
    list<pair<const capd::interval, const capd::IVector> >::iterator ite =
        find_if(trajectory.begin(),
             trajectory.end(),
             [&time](const pair<const capd::interval, const capd::IVector>& item)
             {
                 return item.first.leftBound() > time.rightBound();
             });
    trajectory.erase(ite, trajectory.end());
}


string ode_solver::create_diffsys_string(set < Enode* > & ode_vars,
                                         vector<Enode*> & _0_vars,
                                         vector<Enode*> & _t_vars)
{
    vector<string> ode_list;
    // 1. partition ode_vars into _0_vars and _t_vars by their
    // ODE_vartype
    var_list.clear();
    for_each(ode_vars.begin(),
             ode_vars.end(),
             [&] (Enode* ode_var) {
                 _0_vars.push_back(ode_var);
                 _t_vars.push_back(ode_var->getODEopposite());
                 var_list.push_back(ode_var->getODEvarname());
                 string ode = ode_var->getODE();
                 ode_list.push_back(ode);
             });

    // 3. join var_list to make diff_var, ode_list to diff_fun
    string diff_var = "var:" + join(var_list, ", ") + ";";
    string diff_fun = "fun:" + join(ode_list, ", ") + ";";

    // 4. construct diff_sys (string to CAPD)
    string diff_sys = diff_var + diff_fun;
    if(_config.nra_verbose) {
        cerr << "diff_var : " << diff_var << endl;
        cerr << "diff_fun : " << diff_fun << endl;
        cerr << "diff_sys : " << diff_sys << endl;
    }

    return diff_sys;
}


IVector ode_solver::varlist_to_IVector(vector<Enode*> vars)
{
    IVector intvs (vars.size());

    /* Assign current interval values */
    for_each(make_zip_iterator(boost::make_tuple(vars.begin(), intvs.begin())),
             make_zip_iterator(boost::make_tuple(vars.end(), intvs.end())),
             [&] (tuple<Enode*, interval&> items) {
                 Enode* var     = items.get<0>();
                 interval& intv = items.get<1>();
                 double lb = get_lb(var);
                 double ub = get_ub(var);
                 intv = interval(lb, ub);
                 if(_config.nra_verbose) {
                     cerr << "The interval on "
                          << var->getCar()->getName()
                          << ": "<< intv <<endl;
                 }
             });

    // intvs.begin();

    // for_each(vars.begin(),
    //          vars.end(),
    //          [&] (Enode* var) {
    //              double lb = get_lb(var);
    //              double ub = get_ub(var);
    //              intvs[i] = interval(lb, ub);
    //              if(_config.nra_verbose) {
    //                  cerr << "The interval on "
    //                       << var->getCar()->getName()
    //                       << " is "<< intvs[i] <<endl;
    //              };
    //          });
    return intvs;
}

IVector ode_solver::extract_invariants(vector<Enode*> vars)
{
    IVector ret (vars.size());

    /* Assign current interval values */
    int i = 0;
    for (vector<Enode*>::iterator var_ite = vars.begin();
         var_ite != vars.end();
         var_ite++, i++)
    {
        pair<double, double> inv = (*var_ite)->getODEinvariant();
        ret[i] = interval(inv.first, inv.second);

        if(_config.nra_verbose) {
            cerr << "Invariant extracted from "
                 << (*var_ite)->getCar()->getName()
                 << " is "<< ret[i] <<endl;
        }
    }
    return ret;
}


void ode_solver::IVector_to_varlist(IVector & v, vector<Enode*> & vars)
{
    for(int i = 0; i < v.size(); i++)
    {
        double lb = get_lb(vars[i]);
        double ub = get_ub(vars[i]);
        if (lb < v[i].leftBound())
            set_lb(vars[i], v[i].leftBound());
        if (ub > v[i].rightBound())
            set_ub(vars[i], v[i].rightBound());
    }
}

void ode_solver::prune(vector<Enode*>& _t_vars,
                       IVector v,
                       interval dt,
                       vector<IVector> & out_v_list,
                       vector<interval> & out_time_list,
                       interval time
    )
{
    bool candidate = true;
    for(int i = 0; candidate && i < v.size(); i++)
    {
        if(_config.nra_verbose) {
            cerr << endl
                 << "  v[" << i << "] = "
                 << "[" << v[i].leftBound() << ", " << v[i].rightBound() << "]"
                 << endl;
            cerr << "x_t[" << i << "] = "
                 << "[" << get_lb(_t_vars[i]) << ", " << get_ub(_t_vars[i]) << "]"
                 << endl;
        }

        /*
          [         t_vars[i]        ]
                                           [         v[i]             ]
          ----------------------------------------------------------------------
                                           [         t_vars[i]        ]
          [         v[i]             ]
         */
        if (dt.leftBound() > time.rightBound() ||
            time.leftBound() > dt.rightBound()) {
            candidate = false;
        }

        if (v[i].leftBound() > get_ub(_t_vars[i]) ||
            v[i].rightBound() < get_lb(_t_vars[i]))
        {
            candidate = false;
        }
    }
    if(_config.nra_verbose) {
        cerr << "IS " << (candidate ? "CANDIDATE" : "NOT CANDIDATE") << endl;
    }

    if (candidate) {
        out_v_list.push_back(v);
        out_time_list.push_back(dt);
    }
}

bool ode_solver::solve_forward()
{
    if(_config.nra_verbose) {
        cerr << "ODE_Solver::solve_forward()" << endl;
    }
    cerr.precision(12);
    bool ret = true;
    try {
        // 1. Construct diff_sys, which are the ODE
        vector<Enode*> _0_vars;
        vector<Enode*> _t_vars;

        string diff_sys = create_diffsys_string(_ode_vars,
                                                _0_vars,
                                                _t_vars);

        //pass the problem with variables
        IMap vectorField(diff_sys);

        //initialize the solver
        // The solver uses high order enclosure method to verify the existence of the solution.
        // The order will be set to 20.
        // The time step (when step control is turned off) will be 0.1.
        // The time step control is turned on by default but the solver must know if we want to
        // integrate forwards or backwards (then put negative number).
        ITaylor solver(vectorField, _config.nra_ODE_taylor_order, .1);  /* HERE */
        ITimeMap timeMap(solver);

        //initial conditions
        IVector start = varlist_to_IVector(_0_vars); /* TODO */
        IVector inv = extract_invariants(_t_vars);
        IVector end = varlist_to_IVector(_t_vars);   /* TODO */

        //end = start; //set the initial comparison

        // define a doubleton representation of the interval vector start
        C0Rect2Set s(start);

        //time range
        Enode* time = (*_0_vars.begin())->getODEtimevar();
        interval T = interval(get_lb(time), get_ub(time));  /* TODO
                                                             * (sign) */

        if(_config.nra_verbose) {
            cerr << "interval T = " << T << endl;
        }

        timeMap.stopAfterStep(true);

        bool fastForward = true;
        if (stepControl == 0.0) {
            timeMap.turnOnStepControl();
        } else {
            timeMap.turnOffStepControl();
            timeMap.setStep(stepControl); /* TODO (sign) */
        }

        interval prevTime(0.);

        /* TODO (only in forward) */
        if(_config.nra_json) {
            trajectory.clear();
            trajectory.push_back(make_pair(timeMap.getCurrentTime(), IVector(s)));
        }

        vector<IVector> out_v_list;
        vector<interval> out_time_list;
        bool invariantViolated = false;
        do
        {
            if(stepControl != 0) {
                timeMap.setStep(stepControl);  /* TODO (sign) */
            }

            timeMap(T.rightBound(),s);         /* TODO direction */

            interval stepMade = solver.getStep();
            if(_config.nra_verbose) {
                cerr << "step made: " << stepMade << endl;
            }

            if (!fastForward || T.leftBound() <= timeMap.getCurrentTime().rightBound()) {  /*  TODO (sym) */
                // This is how we can extract an information about the
                // trajectory between time steps. The type CurveType
                // is a function defined on the interval [0,stepMade].
                // It can be evaluated at a point (or interval). The
                // curve can be also differentiated wrt to time. We
                // can also extract from it the 1-st order derivatives
                // wrt.
                const ITaylor::CurveType& curve = solver.getCurve();
                interval domain = interval(0,1)*stepMade;
                /* TODO (sign) */
                double domainWidth = domain.rightBound() - domain.leftBound();

                // Here we use a uniform grid of last time step made
                // to enclose the trajectory between time steps.
                // You can use your own favourite subdivision, perhaps nonuniform,
                // depending on the problem you want to solve.

                for(unsigned i = 0; i < _config.nra_ODE_grid_size; i++)
                {
                    interval subsetOfDomain = domain / _config.nra_ODE_grid_size
                        + (domainWidth / _config.nra_ODE_grid_size) * i;

                    // The above interval does not need to be a subset of domain.
                    // This is due to rounding to floating point numbers.
                    // We take the intersection with the domain.
                    intersection(domain,subsetOfDomain,subsetOfDomain);

                    // Here we evaluated curve at the interval subsetOfDomain.
                    // v will contain rigorous bound for the trajectory for this time interval.
                    IVector v = curve(subsetOfDomain);

                    if(_config.nra_verbose) {
                        cerr << "enclosure for t=" << prevTime + subsetOfDomain << ":  " << v << endl;
                        cerr << "diam(enclosure): " << diam(v) << endl;
                    }

                    IVector v_intersected;
                    if(!intersection(v, inv, v_intersected)) {
                        invariantViolated = true;
                        // cerr << "invariant violated (2)!" << endl;
                        break;
                    }

                    if(_config.nra_verbose) {
                        cerr << "enclosure for t intersected with inv =" << prevTime + subsetOfDomain << ":  " << v << endl;
                    }
                    if(_config.nra_json) {
                        trajectory.push_back(make_pair(prevTime + subsetOfDomain, v_intersected));
                    }
                    prune(_t_vars, v_intersected, prevTime + subsetOfDomain, out_v_list, out_time_list, T);
                }
            }
            else {
                if(_config.nra_verbose) {
                    cerr << "Fast-forward:: " << prevTime << " ===> " << timeMap.getCurrentTime() << endl;
                    cerr << "enclosure for t=" << timeMap.getCurrentTime() << ":  " << IVector(s) << endl;
                }
                if(_config.nra_json) {
                    trajectory.push_back(make_pair(timeMap.getCurrentTime(), IVector(s)));
                }
            }
            prevTime = timeMap.getCurrentTime();
            if(_config.nra_verbose) {
                cerr << "current time: " << prevTime << endl;
            }
            // cerr << "ODEresult        : " << ODEresult << endl
            //      << "InvViolated      : " << invariantViolated << endl
            //      << "timeMap.completed: " << timeMap.completed() << endl;
        }
        while (ODEresult && !invariantViolated && !timeMap.completed());

        // 1. Union all the out_v_list and intersect with end
        IVector vector_union;
        bool end_empty = false;
        if(_config.nra_verbose) {
            cerr << "Union and intersect V" << endl;
        }
        if(out_v_list.size() == 0) {
            if(_config.nra_verbose) {
                cerr << "There is nothing to collect for V" << endl;
            }
            end_empty = true;
        } else {
            vector_union = *(out_v_list.begin());
            for(vector<IVector>::iterator ite = ++(out_v_list.begin());
                ite != out_v_list.end();
                ite++)
            {
                if(_config.nra_verbose) {
                    cerr << "U(" << vector_union << ", " << *ite << ") = ";
                }
                vector_union = intervalHull (vector_union, *ite);
                if(_config.nra_verbose) {
                    cerr << vector_union << endl;
                }
            }
            // end = intersection \cap end;

            if(_config.nra_verbose) {
                cerr << "Intersect(" << vector_union << ", " << end << ") = ";
            }
            if(intersection(vector_union, end, end))
            {
                IVector_to_varlist(end, _t_vars);   /* TODO */
                if(_config.nra_verbose) {
                    cerr << end << endl;
                }
            }
            else {
                // intersection is empty!!
                end_empty = true;

                if(_config.nra_verbose) {
                    cerr << "empty" << endl;
                }
            }
        }

        if(_config.nra_verbose) {
            cerr << endl << "Union and intersect time" << endl;
        }
        bool time_empty = false;
        // 2. Union all the out_time_list and intersect with T
        interval time_union;

        if(out_time_list.size() == 0) {
            if(_config.nra_verbose) {
                cerr << "There is nothing to collect for time" << endl;
            }
            time_union = true;
        } else {
            time_union = *out_time_list.begin();
            for(vector<interval>::iterator ite = ++(out_time_list.begin());
                ite != out_time_list.end();
                ite++)
            {
                if(_config.nra_verbose) {
                    cerr << "U(" << time_union << ", " << *ite << ") = ";
                }
                time_union = intervalHull(time_union, *ite);

                if(_config.nra_verbose) {
                    cerr << time_union << endl;
                }
            }

            /* T = \cap (time_union, T) */

            if(_config.nra_verbose) {
                cerr << "Intersect(" << time_union << ", " << T << ") = ";
            }

            if(intersection(time_union, T, T))
            {
                set_lb(time, T.leftBound());
                set_ub(time, T.rightBound());
                if(_config.nra_verbose) {
                    cerr << T << endl;
                }
            }
            else {
                /* there is no intersection, use empty interval [+oo, -oo] */
                time_empty = true;
                if(_config.nra_verbose) {
                    cerr << "empty" << endl;
                }
            }
        }

        if(end_empty) {
            for(vector<Enode*>::iterator ite = _t_vars.begin();
                ite != _t_vars.end();
                ite++)
            {
                set_empty_interval(*ite);
            }
            ret = false;
        } else {
            IVector_to_varlist(end, _t_vars);
        }

        if(time_empty) {
            set_empty_interval(time);
            ret = false;
        }

        prune_trajectory(T);
    }
    catch(std::exception& e)
    {
        if(_config.nra_verbose) {
            cerr << endl
                 << endl
                 << "Exception caught!" << endl
                 << e.what() << endl << endl;
        }
    }
    return ret;
}

bool ode_solver::solve_backward()
{
    if(_config.nra_verbose) {
        cerr << "ODE_Solver::solve_backward()" << endl;
    }
    cerr.precision(12);
    bool ret = true;
    try {
        // 1. Construct diff_sys, which are the ODE
        vector<Enode*> _0_vars;
        vector<Enode*> _t_vars;

        string diff_sys = create_diffsys_string(_ode_vars,
                                                _0_vars,
                                                _t_vars);

        //pass the problem with variables
        IMap vectorField(diff_sys);

        //initialize the solver
        // The solver uses high order enclosure method to verify the existence of the solution.
        // The order will be set to 20.
        // The time step (when step control is turned off) will be 0.1.
        // The time step control is turned on by default but the solver must know if we want to
        // integrate forwards or backwards (then put negative number).
        ITaylor solver(vectorField,_config.nra_ODE_taylor_order,-.1);
        ITimeMap timeMap(solver);

        //initial conditions
        IVector start = varlist_to_IVector(_t_vars);
        IVector inv = extract_invariants(_t_vars);
        IVector end = varlist_to_IVector(_0_vars);

        //end = start; //set the initial comparison

        // define a doubleton representation of the interval vector start
        C0Rect2Set s(start);

        //time range
        Enode* time = (*_0_vars.begin())->getODEtimevar();
        interval T = - interval(get_lb(time), get_ub(time));

        if(_config.nra_verbose) {
            cerr << "interval T = " << T << endl;
        }

        timeMap.stopAfterStep(true);

        bool fastForward = true;
        if (stepControl == 0.0) {
            timeMap.turnOnStepControl();
        } else {
            timeMap.turnOffStepControl();
            timeMap.setStep(-stepControl);  /* TODO */
        }

        interval prevTime(0.);

        vector<IVector> out_v_list;
        vector<interval> out_time_list;
        bool invariantViolated = false;
        do
        {
            if(stepControl != 0) {
                timeMap.setStep(- stepControl);
            }
            timeMap(T.leftBound(),s);

            //timeMap(T,e);
            interval stepMade = solver.getStep();
            if(_config.nra_verbose) {
                cerr << "step made: " << stepMade << endl
                     << "T : " << T << endl
                     << "currentTime : " << timeMap.getCurrentTime() << endl;
            }

            if (!fastForward || T.rightBound() >= timeMap.getCurrentTime().leftBound()) {
                // This is how we can extract an information about the
                // trajectory between time steps. The type CurveType
                // is a function defined on the interval [0,stepMade].
                // It can be evaluated at a point (or interval). The
                // curve can be also differentiated wrt to time. We
                // can also extract from it the 1-st order derivatives
                // wrt.
                const ITaylor::CurveType& curve = solver.getCurve();
                interval domain = interval(0,1)*stepMade;
                double domainWidth = domain.rightBound() - domain.leftBound();

                // Here we use a uniform grid of last time step made
                // to enclose the trajectory between time steps.
                // You can use your own favourite subdivision, perhaps nonuniform,
                // depending on the problem you want to solve.

                for(unsigned i = 0; i < _config.nra_ODE_grid_size; i++)
                {
                    interval subsetOfDomain = domain / _config.nra_ODE_grid_size
                        - (domainWidth / _config.nra_ODE_grid_size) * i;
                    // The above interval does not need to be a subset of domain.
                    // This is due to rounding to floating point numbers.
                    // We take the intersection with the domain.
                    intersection(domain,subsetOfDomain,subsetOfDomain);

                    // Here we evaluated curve at the interval subsetOfDomain.
                    // v will contain rigorous bound for the
                    // trajectory for this time interval.
                    IVector v = curve(subsetOfDomain);
                    if(_config.nra_verbose) {
                        cerr << "subsetOfDomain: " << subsetOfDomain << endl;
                        cerr << "enclosure for t=" << prevTime + subsetOfDomain << ":  " << v << endl;
                        cerr << "diam(enclosure): " << diam(v) << endl;
                    }

                    if(!intersection(v, inv, v)) {
                        invariantViolated = true;
                        // cerr << "invariant violated(2)!" << endl;
                        break;
                    }
                    if(_config.nra_verbose) {
                        cerr << "enclosure for t intersected with inv =" << prevTime + subsetOfDomain << ":  " << v << endl;
                    }
                    prune(_0_vars, v, prevTime + subsetOfDomain, out_v_list, out_time_list, T);
                }
            }
            else {
                if(_config.nra_verbose) {
                    cerr << "Fast-forward:: " << prevTime << " ===> " << timeMap.getCurrentTime() << endl;
                }
            }
            if(_config.nra_verbose) {
                cerr << "=============================================" << endl;
            }

            prevTime = timeMap.getCurrentTime();
            if(_config.nra_verbose) {
                cerr << "current time: " << prevTime << endl;
            }
            // cerr << "ODEresult        : " << ODEresult << endl
            //      << "InvViolated      : " << invariantViolated << endl
            //      << "timeMap.completed: " << timeMap.completed() << endl;
        }
        while (ODEresult && !invariantViolated && !timeMap.completed());

        // 1. Union all the out_v_list and intersect with end
        IVector vector_union;
        if(_config.nra_verbose) {
            cerr << "Union and intersect V" << endl;
        }
        if(out_v_list.size() == 0) {
            if(_config.nra_verbose) {
                cerr << "There is nothing to collect for V" << endl;
            }
        } else {
            vector_union = *(out_v_list.begin());
            for(vector<IVector>::iterator ite = ++(out_v_list.begin());
                ite != out_v_list.end();
                ite++)
            {
                if(_config.nra_verbose) {
                    cerr << "U(" << vector_union << ", " << *ite << ") = ";
                }
                vector_union = intervalHull (vector_union, *ite);
                if(_config.nra_verbose) {
                    cerr << vector_union << endl;
                }
            }
            // end = intersection \cap start;
            if(_config.nra_verbose) {
                cerr << "Intersect(" << vector_union << ", " << end << ") = ";
            }
            if(intersection(vector_union, end, end))
            {
                IVector_to_varlist(end, _0_vars);
                if(_config.nra_verbose) {
                    cerr << end << endl;
                }
            }
            else {
                // intersection is empty!!
                if(_config.nra_verbose) {
                    cerr << "empty" << endl;
                }
            }
        }
    }
    catch(std::exception& e)
    {
        if(_config.nra_verbose) {
            cerr << endl
                 << endl
                 << "Exception caught!" << endl
                 << e.what() << endl << endl;
        }
    }
    return ret;
}
