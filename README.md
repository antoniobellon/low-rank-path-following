# low-rank-path-following
Low-rank path-following code

The file main.py contains an example of usage of the low-rank path-following algorithm.
It first asks the user to enter the dimension of an SDP problem, whose data functions are then generated by the class _ProblemCreator.
It then solve the SDP with MOSEK (mosek_ipm_solver) at the given initial_time in order to return an accurate initial point for the algorithm.
Then, it runs the path follownig algorithm atarting from the intial poknt obtained at the step before and using the parameter chosen from the user.
Finally, it prints the runtime and the residual accuracy of the procedure.
The solutions for the TV-SDP are contained in the list attribute _primal_solutions_list of the _PathFollowing class.

For a detailed explaination of the path-following algorithm we refer to https://arxiv.org/pdf/2210.08387.pdf .
