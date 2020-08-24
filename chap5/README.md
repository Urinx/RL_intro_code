# Chapter 5<br>Monte Carlo Methods

### Pseudocode
![](chap5.1-first-visit_MC.png)

![](chap5.3-monte_carlo_ES.png)

![](chap5.4-on_policy_MC_control.png)

![](chap5.6-off_policy_MC_prediction.png)

![](chap5.7-off_policy_MC_control.png)

### Figure
![](fig_5.1_state_value_for_balckjack.png)
**Figure 5.1:** Approximate state-value functions for the blackjack policy that sticks only on 20 or 21, computed by Monte Carlo policy evaluation.

![](fig_5.2_balckjack_MCES.png)
Figure 5.2: The optimal policy and state-value function for blackjack, found by Monte Carlo ES. The state-value function shown was computed from the action-value function found by Monte Carlo ES.

![](fig_5.3_ordinary_and_weighted_importance_sampling.png)
Figure 5.3: Weighted importance sampling produces lower error estimates of the value of a single blackjack state from off-policy episodes (see Example 5.3).

![](fig_5.4_infinite_variance.png)
Figure 5.4: Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP shown inset (Example 5.5). The correct estimate here is 1 (γ = 1), and, even though this is the expected value of a sample return (after importance sampling), the variance of the samples is infinite, and the estimates do not converge to this value. These results are for off-policy first-visit MC.