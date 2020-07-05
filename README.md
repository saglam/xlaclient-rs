## xlaclient-rs

In reinforcement learning (RL), one often designs a *policy network* 
for an *agent*, which outputs a probability distribution over the actions the 
agent can take, given the state of the world, hopefully enabling the agent to 
achieve certain goals.

Any such policy network induces a probability distribution over the "path space"
of the world: starting from an initial state, the agent samples an action
according to the network and takes this action, and keeps repeating this until
the simulation is over. 

Suppose now we are given a policy network <img alt="\inline W" src="https://latex.codecogs.com/png.latex?%5Cinline%20W" align="center"/>, with the induced path space
distribution <img alt="\inline P" src="https://latex.codecogs.com/png.latex?%5Cinline%20P" align="center"/>. We may run many many parallel simulations by letting the
agent behave according to <img alt="\inline W" src="https://latex.codecogs.com/png.latex?%5Cinline%20W" align="center"/>, thereby drawing many samples from <img alt="\inline P" src="https://latex.codecogs.com/png.latex?%5Cinline%20P" align="center"/>, say
obtaining

<p align=center><img alt="\displaystyle{\{p_1, p_2,\ldots, p_N\}}" src="https://latex.codecogs.com/png.latex?%5Cdisplaystyle%7B%5C%7Bp_1%2C%20p_2%2C%5Cldots%2C%20p_N%5C%7D%7D"/></p>


where <img alt="\inline p_i\in\mathrm{supp}(P)" src="https://latex.codecogs.com/png.latex?%5Cinline%20p_i%5Cin%5Cmathrm%7Bsupp%7D%28P%29" align="center"/> is a particular "path" the agent took in 
simulation number <img alt="\inline i" src="https://latex.codecogs.com/png.latex?%5Cinline%20i" align="center"/>. Suppose now we can assign a quality score <img alt="\inline q(p_i)" src="https://latex.codecogs.com/png.latex?%5Cinline%20q%28p_i%29" align="center"/> to
each path denoting how well the agent did to achieve the particular goal we
want.

Now consider the "empirical distribution" on the path space obtained as follows:
Sample <img alt="\inline i\in\{1,2,\ldots, N\}" src="https://latex.codecogs.com/png.latex?%5Cinline%20i%5Cin%5C%7B1%2C2%2C%5Cldots%2C%20N%5C%7D" align="center"/> with probability proportional to <img alt="\inline q(p_i)" src="https://latex.codecogs.com/png.latex?%5Cinline%20q%28p_i%29" align="center"/> and
output <img alt="\inline p_i" src="https://latex.codecogs.com/png.latex?%5Cinline%20p_i" align="center"/>. Call this distribution over the paths <img alt="\inline Q" src="https://latex.codecogs.com/png.latex?%5Cinline%20Q" align="center"/>. In words, <img alt="\inline Q" src="https://latex.codecogs.com/png.latex?%5Cinline%20Q" align="center"/> is a
reweigting (of an empirical sample) of <img alt="\inline P" src="https://latex.codecogs.com/png.latex?%5Cinline%20P" align="center"/>.

(One interesting thing is that while <img alt="\inline P" src="https://latex.codecogs.com/png.latex?%5Cinline%20P" align="center"/> was Markovian with respect to world
state, <img alt="\inline Q" src="https://latex.codecogs.com/png.latex?%5Cinline%20Q" align="center"/> is not necessarily so. It is mainly the sampling that breaks
the Markov property; it we just re-weighted the true <img alt="\inline P" src="https://latex.codecogs.com/png.latex?%5Cinline%20P" align="center"/> according to the final
state of the world, the resulting distribution would still be Markovian.)

In a certain sense, this simulation we made revealed to us
<img alt="\inline \mathbf{D}(Q\,\|\,P)" src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cmathbf%7BD%7D%28Q%5C%2C%5C%7C%5C%2CP%29" align="center"/> bits of information about the shortcomings of our policy
network <img alt="\inline N" src="https://latex.codecogs.com/png.latex?%5Cinline%20N" align="center"/>, where <img alt="\inline \mathbf{D}(\cdot\|\cdot)" src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cmathbf%7BD%7D%28%5Ccdot%5C%7C%5Ccdot%29" align="center"/> is the Kullback-Leibler
divergence.
For simplicity in notation, I will assume that in our world, all 
paths are the same length; though we can easily remove this assumption.
Writing

<p align=center><img alt="\displaystyle{ \mathbf{D}(Q\,\|\,P) =
    \sum_{i=1}^{|p_0|} \mathbf{D}(Q_i\,|\,Q_{\lt  i}\,\|\,P_i\,|\,P_{\lt  i}),}" src="https://latex.codecogs.com/png.latex?%5Cdisplaystyle%7B%20%5Cmathbf%7BD%7D%28Q%5C%2C%5C%7C%5C%2CP%29%20%3D%0A%20%20%20%20%5Csum_%7Bi%3D1%7D%5E%7B%7Cp_0%7C%7D%20%5Cmathbf%7BD%7D%28Q_i%5C%2C%7C%5C%2CQ_%7B%3C%20i%7D%5C%2C%5C%7C%5C%2CP_i%5C%2C%7C%5C%2CP_%7B%3C%20i%7D%29%2C%7D"/></p>


this experiment gives us <img alt="\inline N\times |p_0|" src="https://latex.codecogs.com/png.latex?%5Cinline%20N%5Ctimes%20%7Cp_0%7C" align="center"/> data points with which we can improve
<img alt="\inline W" src="https://latex.codecogs.com/png.latex?%5Cinline%20W" align="center"/>. Given these sample paths <img alt="\inline \{p_1,\ldots,p_N\}" src="https://latex.codecogs.com/png.latex?%5Cinline%20%5C%7Bp_1%2C%5Cldots%2Cp_N%5C%7D" align="center"/>, our goal is to fit 
<img alt="\inline W" src="https://latex.codecogs.com/png.latex?%5Cinline%20W" align="center"/>s output to <img alt="\inline Q_i\,|\,Q_{\lt  i}" src="https://latex.codecogs.com/png.latex?%5Cinline%20Q_i%5C%2C%7C%5C%2CQ_%7B%3C%20i%7D" align="center"/>, say via SGD, so that each term in the above
summation approaches zero, therefore the entire divergence on the path space
approaches 0.

Doing this simulation and then SGD training and simulation with improved network
and so on alternately gives us a powerful way to discover extremely competitive
strategies.

### Why an XLA client in Rust
In a certain sense in RL we develop a better understanding about the world
by making many simulations with a more competent agent. This results
in a self-reinforcing cycle, which leads to the name RL.
Making many
parallel simulations quickly involves two things

1. Feeding the world state to the network <img alt="\inline W" src="https://latex.codecogs.com/png.latex?%5Cinline%20W" align="center"/> and getting back a distribution
on actions

2. Being able to run a time step of the "world" and our agent in the simulation.

While Python is extremely well suited for (1),
it is the second item where Python makes things very slow, difficult, or
in some cases impossible.

This repo is my attempt at writing an XLA client in Rust so that one can 
perform (2) efficiently while still being able to do (1) quickly by offloading the
network inference to powerful accelerators such as Google Cloud GPUs or TPUs.

**Note:** DeepMind's
<a href="//github.com/deepmind/open_spiel">`open_spiel`</a> is a RL platform,
and thus needs to solve the same fast inference problem I outlined.
They solve this by bringing up a full TensorFlow session and running the XLA
compilation through the `@tf.function` mechanism. For the purposes of this
project, that route is not ideal as TensorFlow session is a ginormous dependency
that would require a build cluster; further `@tf.function` interface currently
does not seem to expose all the compilation options XLA provides.
