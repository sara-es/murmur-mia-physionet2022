import numpy as np
from scipy.stats import multivariate_normal
from .duration_distributions import get_duration_distributions

def viterbi_decode_recording(observation_sequence,
                             pi_vector,
                             models,
                             total_obs_distribution,
                             heart_rate,
                             systolic_time,
                             recording_frequency):
   """

   Parameters
   ----------
   observation_sequence
   pi_vector
   total_obs_distribution
   heart_rate
   systolic_time
   recording_frequency
   models

   Returns
   -------

   """
   seq_len = observation_sequence.shape[0]
   num_states = 4
   max_duration_D = int(np.round((60./ heart_rate) * recording_frequency))
   delta = np.ones((seq_len + max_duration_D - 1, num_states)) * -np.inf
   # Note that we don't actually need to calculate this: since you can only cycle through the states in sequence,
   # the state you are most likely to have come from is ((current state - 1) % 4)
   psi = np.zeros((seq_len + max_duration_D - 1, num_states)).astype(int)
   psi_duration = np.zeros((seq_len + max_duration_D - 1, num_states))
   observation_probs = np.zeros((seq_len, num_states))

   for state_i in range(num_states):
      # We get p(state | obs) from the logistic regression
      # and use Bayes to get p(obs | state) = p(state | obs) * p(obs) / p(states)
      # p(obs) is derived from multivariate normal distribution using observation statistics
      # p(pstates) is just taken from the pi vector
      pihat = models[state_i].predict_proba(observation_sequence)
      # To make things less confusing, I switch the order of pihat columns
      pihat = pihat[:, ::-1]

      for time_ in range(seq_len):
         Po_correction = multivariate_normal.pdf(observation_sequence[time_, :].reshape(1, -1),
                                                 mean=total_obs_distribution[0],
                                                 cov=total_obs_distribution[1])

         observation_probs[time_, state_i] = (pihat[time_, 1] * Po_correction) / pi_vector[state_i]

   observation_probs = np.where(observation_probs == 0, np.finfo(float).tiny, observation_probs)
   d_distributions, max_S1, min_S1,\
   max_S2, min_S2, max_systole, min_systole,\
   max_diastole, min_diastole = get_duration_distributions(heart_rate, systolic_time)

   # line 170 in the matlab code suggests we might get some index errors in the code below, since matlab autoextends vectors
   duration_probs = np.zeros((num_states, 3 * recording_frequency))
   duration_sum = np.zeros(num_states)

   for state_j in [1, 2, 3, 4]:
      for duration in range(1, max_duration_D):
         if state_j == 1:
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(duration,
                                                                     mean=d_distributions[state_j - 1, 0],
                                                                     cov=d_distributions[state_j, 1])
            if duration < min_S1 or duration > max_S1:
               duration_probs[state_j - 1, duration] = 0 # np.finfo(float).tiny

         elif state_j == 3:
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(duration,
                                                                     mean=d_distributions[state_j - 1, 0],
                                                                     cov=d_distributions[state_j - 1, 1])
            if duration < min_S2 or duration > max_S2:
               duration_probs[state_j - 1, duration] = 0 # np.finfo(float).tiny

         elif state_j == 2:
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(duration,
                                                                     mean=d_distributions[state_j - 1, 0],
                                                                     cov=d_distributions[state_j - 1, 1])
            if duration < min_systole or duration > max_systole:
               duration_probs[state_j - 1, duration] = 0 # np.finfo(float).tiny

         elif state_j == 4:
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(duration,
                                                                     mean=d_distributions[state_j - 1 , 0],
                                                                     cov=d_distributions[state_j - 1, 1])
            if duration < min_diastole or duration > max_diastole:
               duration_probs[state_j - 1, duration] = 0 #np.finfo(float).tiny

      duration_sum[state_j - 1] = np.sum(duration_probs[state_j - 1, :])

   assigned_states = np.zeros(delta.shape[0])
   delta[0, :] = np.log(pi_vector) + np.log(observation_probs[0, :])
   psi[0, :] = -1

   a_matrix = np.array([ [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
 
   # a_matrix generates a bunch of errors because we're dividing by zero
   # replace 0s with float.tiny as a naive workaround to get it to shut up
   a_matrix_nonzero =  np.where(a_matrix == 0, np.finfo(float).tiny, a_matrix)

   for time_ in range(2, seq_len + max_duration_D):
      end_time = time_
      if time_ > seq_len:
         end_time = seq_len
      old_start_time = end_time

      probs = observation_probs[end_time-1, :]
      for duration in range(1, max_duration_D + 1):
         start_time = time_ - duration
         if start_time < 1:
            start_time = 1
         if start_time > seq_len-1:
            start_time = seq_len-1


         # deltas = delta[start_t - 1, :] + np.log(a_matrix[:, j])
         deltas = delta[start_time - 1, :] + np.log(a_matrix_nonzero.T)
         max_index = np.argmax(deltas, axis=1)
         max_delta = deltas[np.arange(4), max_index]
         # max_delta = deltas[max_index]
         # max_index = np.argmax(delta[start_t - 1, :] + np.log(a_matrix_nonzero[:, j]))
         # max_delta = (delta[start_t - 1, :] + np.log(a_matrix_nonzero[:, j]))[max_index]

         if start_time < old_start_time:
            assert start_time == old_start_time - 1
            probs = probs * observation_probs[start_time - 1]
            old_start_time = start_time
         # probs = np.prod(observation_probs[start_time - 1 : end_time], axis=0)
         # assert np.allclose(probs, iterated_product)

         probs = np.where(probs == 0, np.finfo(float).tiny, probs)
         emission_probs = np.log(probs)

         emission_probs = np.where(emission_probs == 0, np.finfo(float).tiny, emission_probs)

         duration_probs_normalised = duration_probs[: , duration - 1] / duration_sum
         duration_probs_normalised = np.where(duration_probs_normalised == 0,
                                              np.finfo(float).tiny,
                                              duration_probs_normalised)
         delta_temp = max_delta + emission_probs + np.log(duration_probs_normalised)
         # delta_temp = max_delta + emission_probs + np.log(duration_probs_nonzero[j, d - 1] / duration_sum[j])

         for state_j in [1, 2, 3, 4]:
            if delta_temp[state_j - 1] > delta[time_ - 1, state_j - 1]:
               delta[time_ - 1, state_j - 1] = delta_temp[state_j - 1]
               psi[time_ - 1 , state_j - 1] = max_index[state_j - 1] + 1
               psi_duration[time_ - 1, state_j - 1] = duration

   temp_delta = delta[seq_len:, :]
   idx = np.argmax(temp_delta)
   pos, _ = np.unravel_index(idx, temp_delta.shape)

   pos = pos + seq_len + 1

   state = int(np.argmax(delta[pos - 1, :])) + 1

   offset = pos
   preceding_state = psi[offset - 1, state - 1]
   onset = int(offset - psi_duration[offset - 1, state - 1] + 1)

   assigned_states[onset-1:offset] = state

   state = int(preceding_state)

   count = 0

   while onset > 2:

      offset = int(onset - 1)
      preceding_state = psi[offset-1 , state-1]

      onset = int(offset - psi_duration[offset-1 , state-1] + 1)

      if onset < 2:
         onset = 1

      assigned_states[onset-1:offset] = state
      state = int(preceding_state)
      count = count + 1

      if count > 1000:
         break

   assigned_states = assigned_states[:seq_len+1]

   return delta, psi, assigned_states