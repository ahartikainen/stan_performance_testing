data{
  int M;
  int J;
  int T;
  int E;
  int G;
  int N[G];
  int ii[M];
  int jj[M];
  int gg[M];
  int g_all[sum(N)];
  int y[M];
  matrix[J,J] obs_corr[G];
}

transformed data{
  matrix[E,E] eta_identity = diag_matrix(rep_vector(1.0,E));
  matrix[J,J] y_identity = diag_matrix(rep_vector(1.0,J));
  vector[G] ldet_obs;
  int N_all = sum(N);

  for(g in 1:G)
    ldet_obs[g] = log_determinant(obs_corr[g]);
}

parameters{
  ordered[T] thresholds_raw[G,J];
  matrix<multiplier=5>[E,J] lam[G];
  matrix[N_all,E] eta;
  matrix[N_all,J] ystar_raw;
}

transformed parameters {
  ordered[T] thresholds[G,J];

  for(g in 1:G)
    for(j in 1:J)
      thresholds[g,j] = thresholds_raw[g,j] * 5;
}


model{
  matrix[N_all,J] ystar;
  int pos = 1;

  to_vector(eta) ~ std_normal();
  to_vector(ystar_raw) ~ std_normal();

  for(g in 1:G){
    int g_ids[N[g]] = segment(g_all,pos,N[g]);

    for(j in 1:J) {
      thresholds_raw[g,j] ~ std_normal();
      lam[g,,j] ~ normal(0,5);
    }

    ystar[g_ids,] = eta[g_ids,] * lam[g] + ystar_raw[g_ids,];
    pos += N[g];
  }

  for(m in 1:M)
    y[m] ~ ordered_logistic(ystar[ii[m],jj[m]], thresholds[gg[m],jj[m]]);
}

generated quantities {
  int yrep[M];
  vector[M] log_lik;
  matrix[E,J] lam_corr[G];
  matrix[J,J] sigma_hat[G];
  vector[G] ldet_sigma;
  matrix[J,J] inv_sigma[G];
  vector[G] obs_d;
  matrix[N_all,E] eta_corr;

  {
    int pos = 1;
    matrix[N_all,J] ystar_corr;

    for(g in 1:G){
      int g_ids[N[g]] = segment(g_all,pos,N[g]);
  
      if(lam[g,1,1] < 0) {
        lam_corr[g,1,] = -1 * lam[g,1,];
        eta_corr[g_ids,1] = -1 * eta[g_ids,1]; 
      } else {
        lam_corr[g,1,] = lam[g,1,];
        eta_corr[g_ids,1] = eta[g_ids,1]; 
      }

      sigma_hat[g] = quad_form(eta_identity, lam_corr[g]) + y_identity;
      ldet_sigma[g] = log_determinant(sigma_hat[g]);
      inv_sigma[g] = inverse(sigma_hat[g]);
      obs_d[g] = ldet_sigma[g] - ldet_obs[g] + trace(obs_corr[g] * inv_sigma[g]) - J;

      ystar_corr[g_ids,] = eta_corr[g_ids,] * lam_corr[g];

      pos += N[g];
    }

    ystar_corr += ystar_raw;

    for(m in 1:M) {
      yrep[m] = ordered_logistic_rng(ystar_corr[ii[m],jj[m]], thresholds[gg[m],jj[m]]);
      log_lik[m] = ordered_logistic_lpmf(y[m] | ystar_corr[ii[m],jj[m]], thresholds[gg[m],jj[m]]);
    }
  }
}
