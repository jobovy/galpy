#include "discretize_utils.h"

markov_chain * markov_chain_alloc(int nstates, int init_state, const gsl_matrix *trans_mat)
{
	markov_chain * m = (markov_chain *)malloc(sizeof(markov_chain));
	m->nstates = nstates;
	m->curr_state = init_state;
	m->trans_mat = trans_mat;
	m->prob_lookup_tables = (gsl_ran_discrete_t **)malloc(nstates * sizeof(gsl_ran_discrete_t *));
	
	int i;
	for(i=0;i<nstates;i++)
	{
		gsl_vector_const_view trans_mat_row=gsl_matrix_const_row(trans_mat,i);
		(m->prob_lookup_tables)[i]=gsl_ran_discrete_preproc(nstates,(&trans_mat_row.vector)->data);	
	}	

	return m;

}

void markov_chain_free(markov_chain * m)
{
	int i;
	for(i=0;i<m->nstates;i++)
	{
		gsl_ran_discrete_free((m->prob_lookup_tables)[i]);
	}

	free(m->prob_lookup_tables);
	free(m);

	return;	
}

int get_state_ind(markov_chain * m)
{
	return m->curr_state;
}

void set_state_ind(markov_chain * m, int s)
{
	m->curr_state=s;

	return;
}

void get_probs(markov_chain * m, gsl_vector * probs)
{
	gsl_matrix_get_row(probs, m->trans_mat, m->curr_state);
}

int gen_ind_realiz(markov_chain * m, gsl_rng * r)
{
	m->curr_state=gsl_ran_discrete(r,m->prob_lookup_tables[m->curr_state]);
	return m->curr_state;
}

void get_stat_dist(markov_chain * m, gsl_vector * stat_dist, int * iter)
{
	gsl_vector_set_basis(stat_dist,0);
	gsl_vector* tmp=gsl_vector_alloc(m->nstates);
	
	(*iter)=0;
	double err;
	do
	{
		(*iter)++;
		gsl_vector_set_zero(tmp);
		gsl_blas_dgemv(CblasTrans, 1.0, m->trans_mat, stat_dist, 1.0, tmp);
		gsl_blas_daxpy(-1.0, tmp, stat_dist);
		err=gsl_vector_max(stat_dist);
		gsl_vector_memcpy(stat_dist,tmp);		
	}
	while(err>STAT_DIST_TOL && (*iter)<TSIM_STAT_DIST);

	gsl_vector_free(tmp);

	return;
}	


/**************************************************************************************************************/
void linspace(double a, double b, int n, gsl_vector * v)
{
	double d=(b-a)/(n-1.0);
	
	gsl_vector_set(v,0,a);
	int i;
	for(i=1;i<n;i++)
	{
		gsl_vector_set(v,i,gsl_vector_get(v,i-1)+d);
	}

	return;
}
/**************************************************************************************************************/
void expspace(double a, double b, int n, double exp, gsl_vector * v)
{
	linspace(0.0,pow(b-a,1.0/exp),n,v);
	int i;
	for(i=0;i<n;i++)
	{
		gsl_vector_set(v,i,pow(gsl_vector_get(v,i),exp)+a);
	}

	return;
}
/**************************************************************************************************************/
void tauchen(double persistence, double innov_std_dev, int m, int nstates, gsl_vector * states, gsl_matrix * pi)
{
	double sigma_z=innov_std_dev/sqrt(1-pow(persistence,2.0));
	linspace(-m*sigma_z,m*sigma_z,nstates,states);
	double tmp, d=gsl_vector_get(states,1)-gsl_vector_get(states,0);
	
	int i;
	for(i=0;i<nstates;i++)
	{
		tmp=gsl_cdf_gaussian_P((gsl_vector_get(states,0)+d/2.0-persistence*gsl_vector_get(states,i))/innov_std_dev,1.0);
		gsl_matrix_set(pi,i,0,tmp);
		int j;
		for(j=1;j<nstates-1;j++)
		{
			tmp=gsl_cdf_gaussian_P((gsl_vector_get(states,j)+d/2.0-persistence*gsl_vector_get(states,i))/innov_std_dev,1.0);
			tmp=tmp-gsl_cdf_gaussian_P((gsl_vector_get(states,j)-d/2.0-persistence*gsl_vector_get(states,i))/innov_std_dev,1.0);
			gsl_matrix_set(pi,i,j,tmp);
		}
		tmp=1.0-gsl_cdf_gaussian_P((gsl_vector_get(states,nstates-1)-d/2.0-persistence*gsl_vector_get(states,i))/innov_std_dev,1.0);
		gsl_matrix_set(pi,i,nstates-1,tmp);
	}

	return;
}
/**************************************************************************************************************/
void rouwenhorst(double persistence, double innov_std_dev, int nstates, gsl_vector * states, gsl_matrix * pi)
{
	double psi, p, q, sigma_z;

	sigma_z=innov_std_dev/sqrt(1.0-pow(persistence,2.0));
	psi=sqrt(nstates-1.0)*sigma_z;
	int i;
	for(i=0; i<nstates; i++)
	{
		gsl_vector_set(states,i,-psi+2.0*psi/(nstates-1.0)*(i-1.0));
	}

	p=(1.0+persistence)/2.0;
	q=p;
	
	pi_rouw(nstates, p, q, pi);
}

void pi_rouw(int n, double p, double q, gsl_matrix * pi)
{
	if(n==2)
	{
		gsl_matrix_set(pi, 0, 0, p);
		gsl_matrix_set(pi, 0, 1, 1.0-p);
		gsl_matrix_set(pi, 1, 0, 1.0-q);
		gsl_matrix_set(pi, 1, 1, q);
	}
	else
	{
		gsl_matrix *tmp=gsl_matrix_alloc(n-1, n-1);
		pi_rouw(n-1, p, q, tmp);

		int i, j;
		for(j=1;j<=n;j++)
		{
			gsl_matrix_set(pi,0,j-1,gsl_sf_fact(n-1)/(gsl_sf_fact(j-1)*gsl_sf_fact((n-1)-(j-1)))*pow(p,(n-j))*pow((1-p),(j-1)));
			gsl_matrix_set(pi,n-1,j-1,gsl_sf_fact(n-1)/(gsl_sf_fact(j-1)*gsl_sf_fact((n-1)-(j-1)))*pow((1-q),(n-j))*pow(q,(j-1)));
		}
		for(i=2;i<=n-1;i++)
		{
			gsl_matrix_set(pi,i-1,0,p*gsl_matrix_get(tmp,i-1,0));
			for(j=2;j<=n-1;j++)
			{
				gsl_matrix_set(pi,i-1,j-1,p*gsl_matrix_get(tmp,i-1,j-1)+(1-p)*gsl_matrix_get(tmp,i-1,j-2));
			}
			gsl_matrix_set(pi,i-1,n-1,(1-p)*gsl_matrix_get(tmp,i-1,n-2));
		}		
		
		gsl_matrix_free(tmp);
	}
}
/******************************************************************************************************/
void test_acc_ar1(double persistence, double innov_std_dev, const gsl_vector * states, markov_chain * m, gsl_rng * r)
{
	gsl_vector * realiz_discrete = gsl_vector_alloc(TSIM_AR1_TEST);
	gsl_vector_set(realiz_discrete,0,gsl_vector_get(states,0));

	//generate realizations
	int t;
	for(t=1;t<TSIM_AR1_TEST; t++)
	{
		gsl_vector_set(realiz_discrete,t,gsl_vector_get(states,gen_ind_realiz(m,r)));
	}

	//set up vector views for analysis
	gsl_vector_view discrete_view;
	discrete_view=gsl_vector_subvector(realiz_discrete,T0SIM_AR1_TEST, TSIM_AR1_TEST-T0SIM_AR1_TEST);
	
	//statistical comparison
	double discrete_autocorr, discrete_std, discrete_skew, discrete_kurt;

	discrete_autocorr=gsl_stats_lag1_autocorrelation((&discrete_view.vector)->data,1,TSIM_AR1_TEST-T0SIM_AR1_TEST);
	discrete_std=gsl_stats_sd((&discrete_view.vector)->data,1,TSIM_AR1_TEST-T0SIM_AR1_TEST);
	discrete_skew=gsl_stats_skew((&discrete_view.vector)->data,1,TSIM_AR1_TEST-T0SIM_AR1_TEST);
	discrete_kurt=gsl_stats_kurtosis((&discrete_view.vector)->data,1,TSIM_AR1_TEST-T0SIM_AR1_TEST);
	
	printf("\nTrue process has autocorrelation of rho = %f and std dev of sigma = %f\n",persistence,innov_std_dev/sqrt(1-pow(persistence,2.0)));
	printf("Discretized proccess has autocorrelation of rho hat = %f and std dev of sigma hat = %f\n",discrete_autocorr,discrete_std);

	printf("\nlog(1-rho hat)/log(1-rho): %f\n",log(1.0-discrete_autocorr)/log(1.0-persistence));
	printf("sigma hat/sigma: %f\n\n",discrete_std/(innov_std_dev/sqrt(1-pow(persistence,2.0))));

	gsl_vector_free(realiz_discrete);
		
}

