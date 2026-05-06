# exercise 1

def f(x, s, n, k, a, b, flag):
    from math import sqrt, pi, exp, comb
    if flag == "normal":
        return (1/(sqrt(2*pi)*s)) * exp(-0.5*((x-a)/s)**2)
    elif flag == "binomial":
        return comb(n, k) * (x**k) * ((1-x)**(n-k))
    elif flag == "beta":
        from math import gamma
        return (x**(a-1)*(1-x)**(b-1))/(gamma(a)*gamma(b)/gamma(a+b))
    
# bad naming and crowded signature

from math import comb, exp, gamma, pi, sqrt

def normal_pdf(x, sigma, mean):
    return exp(-0.5 * ((x - mean) / sigma) ** 2) / (sqrt(2 * pi) * sigma)

def binomial_pmf(p_success, n_trials, n_successes):
    return (
        comb(n_trials, n_successes) * \
            (p_success ** n_successes) * \
                ((1 - p_success) ** (n_trials - n_successes))
                )

def beta_pdf(x, alpha, beta):
    beta_fn = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / beta_fn



# exercise 2

def run(data, conds, sbjs, dprimes, criteria, mdl, fit, r, p, out):
    r = []
    for s in sbjs:
        sd = [data[i] for i in range(len(data)) if conds[i] == s]
        dp = sum([x[0] for x in sd]) / len(sd)
        cr = sum([x[1] for x in sd]) / len(sd)
        dprimes.append(dp)
        criteria.append(cr)
    fit = mdl(dprimes, criteria)
    out = "significant" if fit.pvalue < 0.05 else "not significant"
    return out

# too many parameters

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class SubjectFitAnalysis:
    data: Sequence[tuple[float, float]]
    subject_ids: Sequence[str]
    subjects: Sequence[str]
    model: Callable[[list[float], list[float]], object]

    def get_subject_data(self, subject_id: str) -> list[tuple[float, float]]:
        return [
            trial
            for trial, current_subject_id in zip(self.data, self.subject_ids)
            if current_subject_id == subject_id
        ]

    def mean_dprime(self, subject_id: str) -> float:
        subject_data = self.get_subject_data(subject_id)
        return sum(trial[0] for trial in subject_data) / len(subject_data)

    def mean_criterion(self, subject_id: str) -> float:
        subject_data = self.get_subject_data(subject_id)
        return sum(trial[1] for trial in subject_data) / len(subject_data)

    def fit(self) -> str:
        dprimes = [self.mean_dprime(subject_id) for subject_id in self.subjects]
        criteria = [self.mean_criterion(subject_id) for subject_id in self.subjects]
        fit_result = self.model(dprimes, criteria)
        if fit_result.pvalue < 0.05:
            return "significant"
        return "not significant"


# exercise 3
def c(h, f):
    from scipy.stats import norm
    return norm.ppf(h) - norm.ppf(f)

def b(h, f):
    from scipy.stats import norm
    return -0.5 * (norm.ppf(h) + norm.ppf(f))

# unclear naming and short identifiers

from scipy.stats import norm

def z_scores(hit_rate, false_alarm_rate):
    z_hit = norm.ppf(hit_rate)
    z_false_alarm = norm.ppf(false_alarm_rate)
    return z_hit, z_false_alarm

def dprime(hit_rate, false_alarm_rate):
    z_hit, z_false_alarm = z_scores(hit_rate, false_alarm_rate)
    return z_hit - z_false_alarm

def criterion(hit_rate, false_alarm_rate):
    z_hit, z_false_alarm = z_scores(hit_rate, false_alarm_rate)
    return -0.5 * (z_hit + z_false_alarm)



# exercise 4

def compute_sensitivity_index_for_signal_detection_analysis(hr, far):
    from scipy.stats import norm
    # Compute d-prime using the standard SDT formula
    # hr is hit rate, far is false alarm rate
    z_hit = norm.ppf(hr)   # z-score for hit rate
    z_far = norm.ppf(far)  # z-score for false alarm rate
    dp = z_hit - z_far     # d-prime is the difference
    return dp              # return the result

# excessive comments and long function name 

def dprime(hit_rate, false_alarm_rate):
    """Compute d-prime from hit and false-alarm rates."""
    z_hit = norm.ppf(hit_rate)
    z_false_alarm = norm.ppf(false_alarm_rate)
    return z_hit - z_false_alarm



# exercise 5

import numpy as np

def analyze(results):
    results = [r for r in results if r is not None]
    results = [r * 1000 for r in results]
    results = np.array(results)
    results = results[results < np.percentile(results, 95)]
    results = results - results.mean()
    return results

# variable mutation 
import numpy as np

def analyze(results):
    valid_results = [result for result in results if result is not None]
    results_ms = np.array([result * 1000 for result in valid_results])
    cutoff = np.percentile(results_ms, 95)
    trimmed_results = results_ms[results_ms < cutoff]
    centered_results = trimmed_results - trimmed_results.mean()
    return centered_results



# execise 6
def run_experiment(participant_id, session_number,
                   condition_label, stimulus_list,
                   response_key_map, timeout_ms,
                   practice_trials, fixation_duration_ms,
                   feedback_enabled, log_file_path):
    ...

# too many parameters

class ExperimentSettings:
    def __init__(
            self, timeout_ms, practice_trials,
            fixation_duration_ms, feedback_enabled
            ):
        self.timeout_ms = timeout_ms
        self.practice_trials = practice_trials
        self.fixation_duration_ms = fixation_duration_ms
        self.feedback_enabled = feedback_enabled

class ExperimentSession:
    def __init__(
            self, participant_id, session_number,
            condition_label, stimulus_list,
            response_key_map, log_file_path,
            settings: ExperimentSettings
            ):
        self.participant_id = participant_id
        self.session_number = session_number
        self.condition_label = condition_label
        self.stimulus_list = stimulus_list
        self.response_key_map = response_key_map
        self.log_file_path = log_file_path
        self.settings = settings

    def run(self):
        ...





# exercise 7
hits_a  = sum(1 for t in block_a if t.correct)
total_a = len(block_a)
hr_a    = hits_a / total_a

hits_b  = sum(1 for t in block_b if t.correct)
total_b = len(block_b)
hr_b    = hits_b / total_b

# duplicated code 

def hit_rate(block):
    hits = sum(1 for trial in block if trial.correct)
    total = len(block)
    return hits / total
hit_rate_a = hit_rate(block_a)
hit_rate_b = hit_rate(block_b)



# exercise 8
def integrand_uniform(self, p):
    return self.likelihood(p) * self.uniform_prior(p)

def integrand_beta(self, p):
    return self.likelihood(p) * self.beta_prior(p)

def integrand_jeffreys(self, p):
    return self.likelihood(p) * self.jeffreys_prior(p)

# shotgun surgery
def integrand(self, p, prior_type):
    if prior_type == "uniform":
        prior = self.uniform_prior(p)
    elif prior_type == "beta":
        prior = self.beta_prior(p)
    elif prior_type == "jeffreys":
        prior = self.jeffreys_prior(p)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")
    return self.likelihood(p) * prior


# exercise 9
def compute_median(trial_list):
    trial_list.sort()
    n = len(trial_list)
    return trial_list[n // 2]

rts = [0.42, 0.31, 0.58, 0.29]
m   = compute_median(rts)
print(rts)  # [0.29, 0.31, 0.42, 0.58] — caller's list was mutated


# mutating input list

def compute_median(tial_list):
    sorted_list = sorted(trial_list)  # create a sorted copy
    n = len(sorted_list)
    return sorted_list[n // 2]  


# exercise 10
class SignalDetection:

    # Set the d-prime value
    def set_dprime(self, value):
        self.dprime = value  # store d-prime

    # Get the d-prime value
    def get_dprime(self):
        return self.dprime  # return the threshold
    
# lazy class
class  SignalDetection:
    def __init__(self, dprime):
        self.dprime = dprime

# or not using a class
dprime = 1.5  




# exercise 11
summary = {
    s: sum(t.rt for t in trials if t.subject==s)
       / len([t for t in trials if t.subject==s])
    for s in subjects
}

# dense expression and not legible 

summary = {}

for subject in subjects:
    subject_trials = [trial for trial in trials if trial.subject == subject]
    mean_rt = sum(trial.rt for trial in subject_trials) / len(subject_trials)
    summary[subject] = mean_rt


# exercise 12
from scipy.stats import norm

def analyse(sid, hit, far, mrt, srt, age, grp):
    if not (0 <= hit <= 1):
        raise ValueError("hit out of range")
    dp = norm.ppf(hit) - norm.ppf(far)
    return sid, dp, mrt / srt


# Speculative generality and unclear variable names

def analyse(
        subject_id, hit_rate, false_alarm_rate, 
        mean_reaction_time, std_reaction_time
        ):
    if not (0 <= hit_rate <= 1):
        raise ValueError("hit_rate out of range")
    d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    ratio = mean_reaction_time / std_reaction_time

    return subject_id, d_prime, ratio
