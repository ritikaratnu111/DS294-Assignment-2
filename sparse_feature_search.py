
import logging
import numpy as np
import random
log = logging.getLogger("feature_sign")
log.setLevel(logging.INFO)

y=np.random.rand(3,1)
A=np.random.rand(3,2)
x=np.ones((2,1))
gamma=1

def feature_sign_search(A, y, gamma, solution=None):

    effective_zero = 1e-18
    # precompute matrices for speed.
    gram_matrix = np.dot(A.T, A)
    target_correlation = np.dot(A.T, y)
    # initialization goes here.
    if solution is None:
        solution = np.zeros(gram_matrix.shape[0])
    else:
        assert solution.ndim == 1, "solution must be 1-dimensional"
        assert solution.shape[0] == A.shape[1], (
            "solution.shape[0] does not match A.shape[1]"
        )
        # Initialize all elements to be zero.
        solution[...] = 0.
    signs = np.zeros(gram_matrix.shape[0], dtype=np.int8)
    active_set = set()
    z_opt = np.inf
    # Used to store max(abs(grad[nzidx] + gamma * signs[nzidx])).
    # Set to 0 here to trigger a new feature activation on first iteration.
    nz_opt = 0
    # second term is zero on initialization.
    grad = - 2 * target_correlation  # + 2 * np.dot(gram_matrix, solution)
    max_grad_zero = np.argmax(np.abs(grad))
    # Just used to compute exact cost function.
    sds = np.dot(y.T, y)
    while z_opt > gamma or not np.allclose(nz_opt, 0):
        if np.allclose(nz_opt, 0):
            candidate = np.argmax(np.abs(grad) * (signs == 0))
            log.debug("candidate feature: %d" % candidate)
            if grad[candidate] > gamma:
                signs[candidate] = -1.
                solution[candidate] = 0.
                log.debug("added feature %d with negative sign" %
                          candidate)
                active_set.add(candidate)
            elif grad[candidate] < -gamma:
                signs[candidate] = 1.
                solution[candidate] = 0.
                log.debug("added feature %d with positive sign" %
                          candidate)
                active_set.add(candidate)
            if len(active_set) == 0:
                break
        else:
            log.debug("Non-zero coefficient optimality not satisfied, "
                      "skipping new feature activation")
        indices = np.array(sorted(active_set))
        restr_gram = gram_matrix[np.ix_(indices, indices)]
        restr_corr = target_correlation[indices]
        restr_sign = signs[indices]
        rhs = restr_corr - gamma * restr_sign / 2

        new_solution = np.linalg.solve(np.atleast_2d(restr_gram), rhs)
        new_signs = np.sign(new_solution)
        restr_oldsol = solution[indices]
        sign_flips = np.where(abs(new_signs - restr_sign) > 1)[0]
        if len(sign_flips) > 0:
            best_obj = np.inf
            best_curr = None
            best_curr = new_solution
            best_obj = (sds + (np.dot(new_solution,
                                      np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr))
                        + gamma * abs(new_solution).sum())
            if log.isEnabledFor(logging.DEBUG):

                ocost = (sds + (np.dot(restr_oldsol,
                                       np.dot(restr_gram, restr_oldsol))
                        - 2 * np.dot(restr_oldsol, restr_corr))
                        + gamma * abs(restr_oldsol).sum())
                cost = (sds + np.dot(new_solution,
                                     np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr)
                        + gamma * abs(new_solution).sum())
                log.debug("Cost before linesearch (old)\t: %e" % ocost)
                log.debug("Cost before linesearch (new)\t: %e" % cost)
            else:
                ocost = None
            for idx in sign_flips:
                a = new_solution[idx]
                b = restr_oldsol[idx]
                prop = b / (b - a)
                curr = restr_oldsol - prop * (restr_oldsol - new_solution)
                cost = sds + (np.dot(curr, np.dot(restr_gram, curr))
                              - 2 * np.dot(curr, restr_corr)
                              + gamma * abs(curr).sum())
                log.debug("Line search coefficient: %.5f cost = %e "
                          "zero-crossing coefficient's value = %e" %
                          (prop, cost, curr[idx]))
                if cost < best_obj:
                    best_obj = cost
                    best_prop = prop
                    best_curr = curr
            log.debug("Lowest cost after linesearch\t: %e" % best_obj)
            if ocost is not None:
                if ocost < best_obj and not np.allclose(ocost, best_obj):
                    log.debug("Warning: objective decreased from %e to %e" %
                              (ocost, best_obj))
        else:
            log.debug("No sign flips, not doing line search")
            best_curr = new_solution
        solution[indices] = best_curr
        zeros = indices[np.abs(solution[indices]) < effective_zero]
        solution[zeros] = 0.
        signs[indices] = np.int8(np.sign(solution[indices]))
        active_set.difference_update(zeros)
        grad = - 2 * target_correlation + 2 * np.dot(gram_matrix, solution)
        z_opt = np.max(abs(grad[signs == 0]))
        nz_opt = np.max(abs(grad[signs != 0] + gamma * signs[signs != 0]))
    return solution

x=feature_sign_search(A, y, gamma)