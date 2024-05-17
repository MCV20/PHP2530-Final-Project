data {
  int<lower = 0> N;                // number of observations
  int<lower = 1> K;                // number of categories in the outcome
  int<lower = 1, upper = K> y[N];  // outcome variable with categories 1 through K
  int<lower = 1> P;                // number of predictors after encoding
  matrix[N, P] x;                  // predictor matrix
}

parameters {
  matrix[P, K-1] beta;             // coefficients for predictors, omitting the baseline category
}

transformed parameters {
  matrix[N, K-1] x_beta;           // Predictor contributions for each non-baseline category

  // Constructing the x_beta matrix for non-baseline categories
  for (n in 1:N) {
    for (k in 1:(K-1)) {
      x_beta[n, k] = x[n] * beta[, k];
    }
  }
}

model {
  // Priors
  to_vector(beta) ~ normal(0, 5);

  // Likelihood
  for (n in 1:N) {
    vector[K] eta;
    for (k in 1:(K-1)) {
      eta[k] = x_beta[n, k];
    }
    eta[K] = 0;  // Baseline category has eta = 0
    y[n] ~ categorical_logit(eta);
  }
}

generated quantities {
  matrix[N, K] predicted_probabilities;

  // Calculating predicted probabilities using x_beta from transformed parameters
  for (n in 1:N) {
    vector[K] eta;
    vector[K] exp_eta;
    for (k in 1:(K-1)) {
      eta[k] = x_beta[n, k];
    }
    eta[K] = 0;  // Baseline category has eta = 0
    exp_eta = exp(eta);
    predicted_probabilities[n] = to_row_vector(exp_eta / sum(exp_eta));  // Normalize to get probabilities and convert to row_vector
  }
}
