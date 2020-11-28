"""
Wrapper for neural network tearing model.
"""
import os

import numpy as np
import torch

class NNTearingModel(object):

    def __init__(
            self,
            tearing_model_path,
            data_header,
            cuda_device='',
    ):
        # Load in the tearing model, the standardizers, and the header.
        device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
        self.model = torch.load(os.path.join(tearing_model_path, 'model.pt'),
                                map_location=device)
        self.data_means = np.load(os.path.join(tearing_model_path,
                                               'data_means.npy'))
        self.data_means = torch.Tensor(self.data_means.reshape(1,-1))
        self.data_stds = np.load(os.path.join(tearing_model_path,
                                              'data_stds.npy'))
        self.data_stds = torch.Tensor(self.data_stds.reshape(1, -1))
        # Get permutation needed.
        data_header = [dh.lower() for dh in data_header]
        with open(os.path.join(tearing_model_path, 'headers.txt'), 'r') as f:
            model_header = f.readlines()[0].split(',')
        perm = []
        for sig in data_header:
            perm.append(model_header.index(sig))
        self.perm = np.array(perm)

    def multi_predict(self, obs):
        """Call the tearing predictor.
        Args:
            obs: ndarray of shape (num_guesses, x, num_signals)
                 or (x, num_signals) where signals are in the correct order,
                 the first column is observations 100ms ago, and the last
                 column is current observations.
        """
        # Make sure right shape format.
        if len(obs.shape) == 2:
            obs = obs[np.newaxis, ...]
        num_in = obs.shape[0]
        num_sigs = obs.shape[2]
        obs = obs[:, :, self.perm]
        new_obs = []
        for ob in obs:
            new_obs.append(np.vstack([
                ob[0, :].reshape(1, -1),
                ob[-1, :].reshape(1, -1),
            ]))
        obs = np.asarray(new_obs)
        net_in = torch.cat([
            torch.Tensor(np.mean(obs, axis=1)).reshape(num_in, num_sigs),
            torch.Tensor(obs[:, 1, :] - obs[:, 0, :]).reshape(num_in, num_sigs),
        ], dim=1)
        # Standardize the inputs.
        net_in = (net_in - self.data_means) / self.data_stds
        # Make predictions.
        with torch.no_grad():
            tearability = self.model(net_in).cpu().numpy().flatten()
        return tearability
