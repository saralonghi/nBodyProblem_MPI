# nBodyProblem_MPI

N-Body Problem – Description and Implemented Solution
The N-body problem involves simulating the motion of N bodies that interact with each other through forces (typically gravitational). This is a classic problem in physics and computational science, with complexity increasing rapidly as the number of bodies grows.

In this project, a simplified version of the problem was implemented by restricting the simulation to two dimensions (x and y) instead of three.

To compute the interactions and update the positions/velocities of the bodies over time, two programs were developed in C, both using the MPI (Message Passing Interface) library to enable distributed computation across multiple processes:

Blocking communication version: processes communicate synchronously, waiting for the communication to complete before continuing.

Non-blocking communication version: processes initiate communication and proceed with computation in parallel, potentially improving performance through better concurrency.

These two versions allow for a comparison of the performance and behavior of blocking vs non-blocking communication approaches in the context of a computationally intensive problem.

