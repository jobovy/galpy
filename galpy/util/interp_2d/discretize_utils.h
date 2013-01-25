#ifndef __DISCRETIZE_UTILS_H__
#define __DISCRETIZE_UTILS_H__

#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_blas.h>

#define TSIM_AR1_TEST 1001000
#define T0SIM_AR1_TEST 1000
#define TSIM_STAT_DIST 1000
#define STAT_DIST_TOL 10.0e-8

typedef struct
{	
	int nstates;
	int curr_state;
	const gsl_matrix * trans_mat;
	gsl_ran_discrete_t ** prob_lookup_tables;		
}markov_chain;

markov_chain * markov_chain_alloc(int nstates, int init_state, const gsl_matrix * trans_mat);
void markov_chain_free(markov_chain * m);		
int get_state_ind(markov_chain * m);
void get_probs(markov_chain * m, gsl_vector * v);
void set_state_ind(markov_chain * m, int s);
double gen_state_realiz(markov_chain * m, gsl_rng * r);
int gen_ind_realiz(markov_chain * m, gsl_rng * r);
void get_stat_dist(markov_chain * m, gsl_vector * stat_dist, int * iter);


void linspace(double a, double b, int n, gsl_vector *v);
void expspace(double a, double b, int n, double gamma, gsl_vector * v);
void tauchen(double persistence, double innov_std_dev, int m, int nstates, gsl_vector * states, gsl_matrix * pi);
void rouwenhorst(double persistence, double innov_std_dev, int nstates, gsl_vector * states, gsl_matrix * pi);
void pi_rouw(int n, double p, double q, gsl_matrix * pi);
void test_acc_ar1(double persistence, double std_dev_innov, const gsl_vector * states, markov_chain * m, gsl_rng * r);

#endif
