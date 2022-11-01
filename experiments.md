## DCGAN

| name                     | lr for SGD | SGD for D | hard label for G | beta1 for G | LeakyReLU for G | post norm | Notes            |
|--------------------------|------------|-----------|------------------|-------------|-----------------|-----------|------------------|
|                          |            | F         | F                | 0.9         | F               | F         |                  |
| SGD_LeakyReLU            | 0.01       | T         | F                | 0.9         | T               | F         |                  |
| beta1_SGD_LeakyReLU      | 0.01       | T         | T                | 0.5         | T               | F         |                  |
| soft_beta1_SGD_LeakyReLU | 0.01       | T         | F                | 0.5         | T               | F         |                  |
| slow_soft                | 0.01       | T         | F                | 0.5         | T               | T         | repeated         |
| fast_SGD                 | 0.05       | T         | T                | 0.5         | T               | T         |                  |
| fast_SGD_new_norm        | 0.05       | T         | T                | 0.5         | T               | T         | on going         |
| fast_soft                | 0.05       | T         | F                | 0.5         | T               | T         |                  |
| fast_soft_new_norm       | 0.05       | T         | F                | 0.5         | T               | T         | on going in copy |