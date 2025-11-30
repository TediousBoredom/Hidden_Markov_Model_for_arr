"""
Hidden Markov Model for Human Activity Recognition
Using UCI HAR Dataset with EM (Baum-Welch) Algorithm
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==================== Data Loading ====================
def load_uci_har_data(data_dir):
    """Load UCI HAR dataset"""
    # Load training data
    X_train = np.loadtxt(f"{data_dir}/train/X_train.txt")
    y_train = np.loadtxt(f"{data_dir}/train/y_train.txt", dtype=int)
    subject_train = np.loadtxt(f"{data_dir}/train/subject_train.txt", dtype=int)
    
    # Load test data
    X_test = np.loadtxt(f"{data_dir}/test/X_test.txt")
    y_test = np.loadtxt(f"{data_dir}/test/y_test.txt", dtype=int)
    subject_test = np.loadtxt(f"{data_dir}/test/subject_test.txt", dtype=int)
    
    return (X_train, y_train, subject_train), (X_test, y_test, subject_test)

# Load data
data_dir = "data/human+activity+recognition+using+smartphones/UCI_HAR_Dataset"
(X_train, y_train, subject_train), (X_test, y_test, subject_test) = load_uci_har_data(data_dir)

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"Activity classes: {np.unique(y_train)}")

# ==================== Preprocessing: PCA ====================
# Apply PCA to reduce dimensionality from 561 to 10
D = 10  # Reduced dimension
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=D)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nPCA: {X_train.shape[1]} -> {D} dimensions")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ==================== Create Sequences ====================
# Group by subject to create sequences (each subject's activities form a sequence)
def create_sequences(X, y, subjects):
    """Group data by subject to create sequences"""
    sequences = []
    labels = []
    for subj_id in np.unique(subjects):
        mask = subjects == subj_id
        seq_data = X[mask]
        seq_labels = y[mask] - 1  # Convert to 0-indexed
        sequences.append(seq_data)
        labels.append(seq_labels)
    return sequences, labels

train_seqs, train_labels = create_sequences(X_train_pca, y_train, subject_train)
test_seqs, test_labels = create_sequences(X_test_pca, y_test, subject_test)

print(f"\nTraining: {len(train_seqs)} sequences")
print(f"Test: {len(test_seqs)} sequences")
print(f"Sequence lengths: {[len(s) for s in train_seqs[:5]]}")

# ==================== HMM Implementation ====================
def logsumexp(a, axis=None):
    """Numerically stable log-sum-exp"""
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return res.squeeze()
    return res.squeeze()

def gaussian_logpdf_vec(x, means, covs):
    """Compute log PDF of multivariate Gaussian for each state"""
    T, D = x.shape
    K = means.shape[0]
    res = np.zeros((T, K))
    for k in range(K):
        diff = x - means[k]
        inv = np.linalg.inv(covs[k])
        logdet = np.linalg.slogdet(covs[k])[1]
        res[:, k] = -0.5 * (np.sum(diff @ inv * diff, axis=1) + D * np.log(2 * np.pi) + logdet)
    return res

def forward_log(obs, pi, A, means, covs):
    """Forward algorithm in log-space"""
    T = obs.shape[0]
    K = pi.shape[0]
    log_alpha = np.zeros((T, K))
    logB = gaussian_logpdf_vec(obs, means, covs)
    log_alpha[0] = np.log(pi + 1e-10) + logB[0]
    for t in range(1, T):
        tmp = log_alpha[t-1][:, None] + np.log(A + 1e-10)
        log_alpha[t] = logB[t] + logsumexp(tmp, axis=0)
    return log_alpha

def backward_log(obs, A, means, covs):
    """Backward algorithm in log-space"""
    T = obs.shape[0]
    K = A.shape[0]
    log_beta = np.zeros((T, K))
    logB = gaussian_logpdf_vec(obs, means, covs)
    for t in range(T-2, -1, -1):
        tmp = np.log(A + 1e-10) + logB[t+1][None, :] + log_beta[t+1][None, :]
        log_beta[t] = logsumexp(tmp, axis=1)
    return log_beta

def baum_welch(seqs, K, D, max_iter=30, tol=1e-3):
    """Baum-Welch (EM) algorithm for HMM training"""
    # Initialize parameters
    pi = np.ones(K) / K
    A = np.ones((K, K)) / K
    all_obs = np.vstack(seqs)
    means = all_obs[np.random.choice(all_obs.shape[0], K, replace=False)].copy()
    covs = np.array([np.cov(all_obs.T) + 1e-2 * np.eye(D) for _ in range(K)])
    
    prev_ll = -np.inf
    for it in range(max_iter):
        pi_acc = np.zeros(K)
        A_num = np.zeros((K, K))
        A_den = np.zeros(K)
        means_num = np.zeros((K, D))
        means_den = np.zeros(K)
        covs_num = np.zeros((K, D, D))
        total_ll = 0.0
        
        for obs in seqs:
            T = obs.shape[0]
            log_alpha = forward_log(obs, pi, A, means, covs)
            log_beta = backward_log(obs, A, means, covs)
            loglik = logsumexp(log_alpha[-1])
            total_ll += loglik
            
            # Compute gamma (posterior marginals)
            log_gamma = log_alpha + log_beta - logsumexp(log_alpha + log_beta, axis=1)[:, None]
            gamma = np.exp(log_gamma)
            pi_acc += gamma[0]
            
            # Compute xi (posterior transitions)
            logB = gaussian_logpdf_vec(obs, means, covs)
            for t in range(T-1):
                log_xi = (log_alpha[t][:, None] + np.log(A + 1e-10) + 
                          logB[t+1][None, :] + log_beta[t+1][None, :])
                log_xi -= logsumexp(log_xi)
                xi = np.exp(log_xi)
                A_num += xi
                A_den += gamma[t]
            
            # Update means and covariances
            for k in range(K):
                wk = gamma[:, k][:, None]
                means_num[k] += (wk * obs).sum(axis=0)
                means_den[k] += wk.sum()
                diff = obs - means[k]
                covs_num[k] += (wk * diff).T @ diff
        
        # M-step: update parameters
        pi = pi_acc / pi_acc.sum()
        A = A_num / (A_den[:, None] + 1e-10)
        A = np.maximum(A, 1e-8)
        A = A / A.sum(axis=1, keepdims=True)
        
        for k in range(K):
            if means_den[k] > 1e-8:
                means[k] = means_num[k] / means_den[k]
            covs[k] = covs_num[k] / np.maximum(means_den[k], 1e-8) + 1e-6 * np.eye(D)
        
        if abs(total_ll - prev_ll) < tol:
            print(f"Converged at iteration {it+1}")
            break
        prev_ll = total_ll
        if (it + 1) % 5 == 0:
            print(f"Iteration {it+1}: log-likelihood = {total_ll:.2f}")
    
    return pi, A, means, covs, total_ll

# ==================== Training ====================
K = 6  # Number of hidden states (matching 6 activity classes)
print(f"\nTraining HMM with K={K} states, D={D} dimensions...")
pi_est, A_est, means_est, covs_est, ll = baum_welch(train_seqs, K, D, max_iter=30)
print(f"Final training log-likelihood: {ll:.2f}")

# ==================== State Label Matching ====================
# Match HMM states to true activity labels using Hungarian algorithm
# Use class centroids for matching
class_centroids = []
for label in range(6):
    mask = (y_train - 1) == label
    centroid = X_train_pca[mask].mean(axis=0)
    class_centroids.append(centroid)
class_centroids = np.array(class_centroids)

cost = np.linalg.norm(means_est[:, None, :] - class_centroids[None, :, :], axis=2)
row_ind, col_ind = linear_sum_assignment(cost)
perm = col_ind  # HMM state i maps to activity class perm[i]
print(f"\nState-to-activity mapping: {dict(zip(range(K), perm + 1))}")

def relabel(states, perm):
    """Relabel states according to permutation"""
    mapping = {i: perm[i] for i in range(len(perm))}
    return np.array([mapping[s] for s in states])

# ==================== Inference Functions ====================
def forward_marginals(obs, pi, A, means, covs):
    """Filtering: P(z_t | y_{1:t})"""
    log_alpha = forward_log(obs, pi, A, means, covs)
    log_marg = log_alpha - logsumexp(log_alpha, axis=1)[:, None]
    return np.exp(log_marg)

def backward_marginals(obs, pi, A, means, covs):
    """Smoothing: P(z_t | y_{1:T})"""
    log_alpha = forward_log(obs, pi, A, means, covs)
    log_beta = backward_log(obs, A, means, covs)
    log_smooth = log_alpha + log_beta - logsumexp(log_alpha + log_beta, axis=1)[:, None]
    return np.exp(log_smooth)

def viterbi(obs, pi, A, means, covs):
    """Viterbi algorithm for MAP sequence estimation"""
    T = obs.shape[0]
    K = pi.shape[0]
    log_delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    logB = gaussian_logpdf_vec(obs, means, covs)
    log_delta[0] = np.log(pi + 1e-10) + logB[0]
    for t in range(1, T):
        tmp = log_delta[t-1][:, None] + np.log(A + 1e-10)
        psi[t] = np.argmax(tmp, axis=0)
        log_delta[t] = logB[t] + np.max(tmp, axis=0)
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(log_delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states

# ==================== Evaluation ====================
print("\nEvaluating on test set...")
results = []

for obs, true_labels in zip(test_seqs, test_labels):
    # Filtering
    filt = forward_marginals(obs, pi_est, A_est, means_est, covs_est)
    filt_pred = np.argmax(filt, axis=1)
    filt_pred_relabeled = relabel(filt_pred, perm) + 1  # Convert back to 1-indexed
    filt_acc = (filt_pred_relabeled == (true_labels + 1)).mean()
    
    # Smoothing
    smooth = backward_marginals(obs, pi_est, A_est, means_est, covs_est)
    smooth_pred = np.argmax(smooth, axis=1)
    smooth_pred_relabeled = relabel(smooth_pred, perm) + 1
    smooth_acc = (smooth_pred_relabeled == (true_labels + 1)).mean()
    
    # Viterbi (MAP)
    vpath = viterbi(obs, pi_est, A_est, means_est, covs_est)
    vpath_relabeled = relabel(vpath, perm) + 1
    viterbi_acc = (vpath_relabeled == (true_labels + 1)).mean()
    
    # One-step prediction
    pred_dists = filt[:-1] @ A_est
    pred_states = np.argmax(pred_dists, axis=1)
    pred_states_relabeled = relabel(pred_states, perm) + 1
    pred_acc = (pred_states_relabeled == (true_labels[1:] + 1)).mean()
    
    results.append([filt_acc, smooth_acc, viterbi_acc, pred_acc])

# ==================== Results Summary ====================
df = pd.DataFrame(results, columns=['Filtering', 'Smoothing', 'Viterbi (MAP)', 'One-step Prediction'])
summary = df.mean().to_dict()

print("\n" + "="*60)
print("Test Set Results (Average Accuracy)")
print("="*60)
for method, acc in summary.items():
    print(f"{method:25s}: {acc:.4f}")
print("="*60)

# Save results
results_df = pd.DataFrame([summary])
print(f"\nResults saved to summary:")
print(results_df.to_string(index=False))
