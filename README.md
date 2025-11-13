# ???

Prøvede forskellige environments, såsom FrozenLake og variationer af Cliff og WindyRooms, men nogle af dem lader til at fucked det helt op.

### QR 10k on-policy:
![](images/QR_vs_MC.png)

Value function approx error:  
![](images/qr_vf_approx.png)

Value distribution approx error:  
![](images/qr_vd_approx.png)

Even setting the amount of episodes up won't ever make the distributions any closer to the MC distribution. Both the wasserstein metric and the squared error are off by a factor of 10 compared to the paper.

### QR 40k episodes on-policy:  
![](images/qr_on_policy.png)


### Simple QR
![](images/simple_env_converged.png)

It does not make sense why none of the 3 distributions can be fully modeled, even with any configuration of decaying learning rate.