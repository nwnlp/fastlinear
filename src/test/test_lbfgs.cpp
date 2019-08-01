#include <stdio.h>
#include <lbfgs.h>
#include <vector>
using namespace std;
class objective_function
{
protected:
    lbfgsfloatval_t *m_x;

public:
    objective_function() : m_x(NULL)
    {
    }

    virtual ~objective_function()
    {
        if (m_x != NULL) {
            lbfgs_free(m_x);
            m_x = NULL;
        }
    }

    int run(int N)
    {
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *m_x = lbfgs_malloc(N);

        if (m_x == NULL) {
            printf("ERROR: Failed to allocate a memory block for variables.\n");
            return 1;
        }

        /* Initialize the variables. */
        for (int i = 0;i < N;i += 2) {
            m_x[i] = -1.2;
            m_x[i+1] = 1.0;
        }

        /*
            Start the L-BFGS optimization; this will invoke the callback functions
            evaluate() and progress() when necessary.
         */
        int ret = lbfgs(N, m_x, &fx, _evaluate, _progress, this, NULL);

        /* Report the result. */
        printf("L-BFGS optimization terminated with status code = %d\n", ret);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);

        return ret;
    }

protected:
    static lbfgsfloatval_t _evaluate(
            void *instance,
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
    )
    {
        return reinterpret_cast<objective_function*>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
    )
    {
        lbfgsfloatval_t fx = 0.0;

        for (int i = 0;i < n;i += 2) {
            lbfgsfloatval_t t1 = 1.0 - x[i];
            lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
            g[i+1] = 20.0 * t2;
            g[i] = -2.0 * (x[i] * g[i+1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;
    }

    static int _progress(
            void *instance,
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls
    )
    {
        return reinterpret_cast<objective_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls
    )
    {
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
        return 0;
    }
};



#define N   100



typedef vector<pair<int,double>> FEATURE_VALUE;
typedef vector<FEATURE_VALUE> DATA_MAT;
typedef vector<double> LABEL_VEC;
typedef vector<double> W_VEC;
/*
# logistic function
idx = t > 0
out = np.empty(t.size, dtype=np.float)
out[idx] = 1. / (1 + np.exp(-t[idx]))
exp_t = np.exp(t[~idx])
out[~idx] = exp_t / (1. + exp_t)
*/
#include <cmath>

double sigmoid_func(double t){
    if(t > 0.0){
        return 1.0 / (1+std::exp(-t));
    }else{
        double exp_t = std::exp(t);
        return exp_t / (1+exp_t);
    }
}
/*
    z = X.dot(w)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
*/

void gradient(DATA_MAT& X, LABEL_VEC& y, W_VEC& w){
    vector<double> Z_0;
    Z_0.resize(X.size());
    for (int data_index = 0; data_index < X.size(); ++data_index) {
        FEATURE_VALUE& k_v = X[data_index];
        double z = 0.0;
        for (int index = 0; index < k_v.size(); ++index) {
            z += w[k_v[index].first] * k_v[index].second;
        }
        double z_0 = (sigmoid_func(z*y[data_index])-1.0)*y[data_index];
        Z_0[data_index] = z_0;
    }


}

int main(int argc, char **argv)
{
    DATA_MAT X;
    LABEL_VEC y;
    FEATURE_VALUE x1;
    x1.push_back(make_pair(1,0.45));
    x1.push_back(make_pair(5,0.5));
    x1.push_back(make_pair(10,0.4));
    x1.push_back(make_pair(19,0.25));
    X.push_back(x1);
    FEATURE_VALUE x2;
    x2.push_back(make_pair(3,0.45));
    x2.push_back(make_pair(4,0.5));
    x2.push_back(make_pair(11,0.4));
    x2.push_back(make_pair(14,0.25));
    X.push_back(x2);

    y.push_back(1.0);
    y.push_back(-1.0);
    W_VEC w;
    w.resize(20);
    gradient(X, y, w)
    objective_function obj;
    return obj.run(N);
}