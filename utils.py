import numpy as np
import ot
import torch
from torch.optim import Adam
from tqdm import tqdm

# Stage 2: Find heuristics via Wasserstein barycenters
def get_cost_matrix(item, prefs, nagents):
  # Construct cost matrix M
  a = np.zeros((nagents,nagents))
  # Populate individual allocation measures
  for i in range(nagents):
    a[i][i] = 1

  A = a.T

  envy = np.zeros((nagents,nagents,nagents))
  M = np.zeros((nagents,nagents,))

  # Get envy
  for i in range(nagents):
    p = A[i]
    for j in range(nagents):
      for k in range(nagents):
        envy[i][j][k] = prefs[j][item] * p[k] / (k + 1) - prefs[j][item] * p[j] / (j + 1)

  print("Average envy for n extreme allocations", np.mean(np.sum(envy**2, axis=(1, 2))))

  # Get cost
  for i in range(nagents):
    for j in range(i, nagents):
      M[j][i] = M[i][j] = np.sum((envy[i] - envy[j])**2, axis=(0, 1))

  M /= M.max()
  return A, M

def find_barycenter(A, M, reg=1e-2, numItermax=100000):
  weights = np.array([1 / A.shape[1]] * A.shape[1])
#   print(A.shape[1], weights)
  bary_wass = ot.bregman.barycenter(A, M, reg, weights=weights, numItermax=numItermax)
  return bary_wass

def compute_loss(ps, aten, nagents, alpha=0.01):
  prs = 1 - ps.sum(axis=1)
  all_ps = torch.concat([ps, prs.unsqueeze(-1)], axis=-1)

  E = torch.zeros((nagents,nagents))

  for j in range(nagents):
    for k in range(nagents):
      E[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * all_ps[:, k]) / (k + 1) - sum(aten[j] * all_ps[:, j]) / (j + 1))

  V = torch.zeros(nagents)

  for i in range(nagents):
    V[i] = sum(aten[i] * all_ps[:, i]) / (i + 1)

  reg = sum([torch.pow(_p, 2) if _p < 0.5 else torch.pow(1 - _p, 2) for _ps in all_ps for _p in _ps])

  J = torch.sum(E**2,axis=[0,1]) + alpha * sum(V)**2 + reg

  return J

def get_WEF1(intps, n_agents, aten, is_heur=True):
  with torch.no_grad():
    intE = torch.zeros((n_agents, n_agents), requires_grad=False)

    for j in range(n_agents):
        for k in range(n_agents):
            intE[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * intps[:, k]) / (k + 1) - sum(aten[j] * intps[:, j]) / (j + 1))

    max_approx = -torch.inf
    for i in range(n_agents):
        for j in range(n_agents):
            if intE[i][j] > 0:
                if max(aten[i] * intps[:, j]) / (j + 1) < intE[i][j]:
                    approx = intE[i][j] / (max(aten[i] * intps[:, j]) / (j + 1))
                    if approx != torch.inf and max_approx < approx:
                        max_approx = approx
    # print(f"                 Approx = {max_approx}")
    if is_heur and max_approx == -torch.inf:
        return 1.00 
    return max_approx


def check_PO(intps, n_agents, n_items, aten):
  with torch.no_grad():
    saved_PO = True
    for j in range(n_agents):
        for i in range(n_items):
            if aten[j, i] * intps[i, j] < 0:
                if torch.any(aten[:,i] >= 0):
                    saved_PO = False

    return saved_PO