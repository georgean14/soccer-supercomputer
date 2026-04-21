# soccer-supercomputer

A predictive model for Premier League matches using poisson distribution and maximum likelihood estimation(MLE)

Technical Details:

Language: C++
Algorithms: MLE using gradient descent
Data: CSV sourced from https://www.football-data.co.uk/

Limitations and future improvements:

Does not account for injuries, weather, or scheduling(how compact the team's schedule is). Additionally, time decay is not implemented, so games early on in the season are weighted just as highly as recent games. 

The reliance on poisson distribution assumes goals are independent, which is not true. Games may change depending on the scoreline, time left, where the team is in the table, and also the opponent. This oversimplifies many matches.

-----------------------Techincal Explanation-----------------------

Each team consists of two variables:

Attack score, a
Defense score, d

When a game is played: 

λ(number of goals scored that game) = e^(attack - defense + μ(baseline) + (H(home advantage))(0 or 1))

λ_home = e^(attack_home - defense_away + μ + H)

λ_away = e^(attack_away - defense_home + μ)



42 Unknowns:

a_1, a_2, …, a_20   (attack scores)
d_1, d_2, …, d_20  (defense scores)
μ                              (baseline)
H                             (home advantage)


Maximum Likelihood Estimation 
MLE = choose parameters that make reality least surprising under your model. 

Total likelihood = Π P(actual | parameters)

Take log of both sides:

log(total likelihood) = ∑ log(P(actual | parameters))

Maximize: 

With P(g | λ), probability of scoring g goals with lambda = λ

log(total likelihood) = ∑ log(λ^g) + log(e^-λ) - log(g!)

∑(g log λ - λ)

(g_h)(attack_h-defense_a+base+H) - e^(attack_h-defense_a+base+H) + (g_a)(attack_a-defense_h+base) - e^(attack_a-defense_h+base)

Gradient Ascent Loop

For each “gradient”
Run through all 380 games.
For each game:
Compute λ_home and λ_away
residue_home = goals_h - λ_home
residue_away = goals_a - λ_away

gradient[attack_home] += residue_home
gradient[attack_away] += residue_away
gradient[defense_home] -= residue_away
gradient[defense_away] -= residue_home
gradient[μ] += residue_home + residue_away
gradient[H] += residue_home

After all 380 games:
update: params += learning_rate * gradient

Keep one team’s attack to 0
Keep one team’s defense to 0

(to prevent the model from drifting,since adding the same value to everyone's attack chnages nothing)
