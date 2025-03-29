logistic_regression_NR <- function(X, y, tol = 1e-6, max_iter = 100) {
  X <- as.matrix(cbind(1, X))  # Tambahkan intercept
  beta <- rep(0, ncol(X))
  iter <- 0
  log_likelihood_old <- -Inf

  while (iter < max_iter) {
    eta <- X %*% beta
    p <- 1 / (1 + exp(-eta))
    W <- diag(as.vector(p * (1 - p)), nrow = length(y))
    z <- eta + (y - p) / (p * (1 - p))

    beta_new <- solve(t(X) %*% W %*% X) %*% (t(X) %*% W %*% z)
    log_likelihood_new <- sum(y * log(p) + (1 - y) * log(1 - p))

    if (abs(log_likelihood_new - log_likelihood_old) < tol) {
      break
    }

    beta <- beta_new
    log_likelihood_old <- log_likelihood_new
    iter <- iter + 1
  }

  aic <- -2 * log_likelihood_new + 2 * length(beta)

  return(list(method = "Newton-Raphson", beta = beta, fit = p, log_likelihood = log_likelihood_new, aic = aic))
}

logistic_regression_IWLS <- function(X, y, tol = 1e-6, max_iter = 100) {
  X <- as.matrix(cbind(1, X))  # Tambahkan intercept
  beta <- rep(0, ncol(X))
  iter <- 0
  log_likelihood_old <- -Inf

  while (iter < max_iter) {
    eta <- X %*% beta
    p <- 1 / (1 + exp(-eta))
    W <- diag(as.vector(p * (1 - p)), nrow = length(y))
    z <- eta + (y - p) / (p * (1 - p))

    beta_new <- solve(t(X) %*% W %*% X) %*% (t(X) %*% W %*% z)
    log_likelihood_new <- sum(y * log(p) + (1 - y) * log(1 - p))

    if (abs(log_likelihood_new - log_likelihood_old) < tol) {
      break
    }

    beta <- beta_new
    log_likelihood_old <- log_likelihood_new
    iter <- iter + 1
  }

  aic <- -2 * log_likelihood_new + 2 * length(beta)

  return(list(method = "IWLS", beta = beta, fit = p, log_likelihood = log_likelihood_new, aic = aic))
}
