#include <iostream>
#include <alglib/optimization.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
using namespace alglib;
void function1_grad(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    //
    // this callback calculates f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4
    // and its derivatives df/d0 and df/dx1
    //
    func = 100*pow(x[0]+3,4) + pow(x[1]-3,4);
    grad[0] = 400*pow(x[0]+3,3);
    grad[1] = 4*pow(x[1]-3,3);
}

int main(int argc, char **argv)
{
    //
    // This example demonstrates minimization of
    //
    //     f(x,y) = 100*(x+3)^4+(y-3)^4
    //
    real_1d_array x = "[0,0]";
    real_1d_array s = "[1,1]";
    double epsg = 0;
    double epsf = 0;
    double epsx = 0.0000000001;
    ae_int_t maxits = 0;
    minlbfgsstate state;
    minlbfgscreate(1, x, state);
    minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    minlbfgssetscale(state, s);
    minlbfgsoptguardsmoothness(state);
    minlbfgsoptguardgradient(state, 0.001);
    minlbfgsreport rep;
    alglib::minlbfgsoptimize(state, function1_grad);
    minlbfgsresults(state, x, rep);
    printf("%s\n", x.tostring(2).c_str()); // EXPECTED: [-3,3]
    optguardreport ogrep;
    minlbfgsoptguardresults(state, ogrep);
    printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
    printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
    printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false
    return 0;
}

