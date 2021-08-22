# Slime

A real-time slime simulation written with CUDA kernels and freeglut OpenGl.

The slime algorithm consists of 3 parts:
1. Agents scatter on the screen.
2. Each agent move, leaving a trace behind it, heading in the direction with the most traces density in front of it. A little randomness is also applied to the agent's movements.
3. A trace dissolves over time.
