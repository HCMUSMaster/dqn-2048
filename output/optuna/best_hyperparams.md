```
Best trial for dqn: #22
  Score: 2704.814
  avg_return: 2852.6
  std_return: 1477.9
  avg_length: 242.6
  avg_max_tile: 230.4
  avg_illegal_attempts: 63.3
  best_params:
    buffer_size: 200000
    batch_size: 128
    gamma: 0.9556791110090774
    lr: 0.00014394430404112647
    target_sync_every: 250
    learn_start: 1000
    learn_every: 8
    eps_end: 0.1488425606123552
    eps_decay_steps: 20000
    grad_clip: 4.3076689475049506
```

```
Best trial for double_dqn: #22
  Score: 2464.240
  avg_return: 2594.8
  std_return: 1305.6
  avg_length: 223.4
  avg_max_tile: 217.6
  avg_illegal_attempts: 55.2
  best_params:
    buffer_size: 20000
    batch_size: 256
    gamma: 0.9798426351600259
    lr: 0.0009768958892956907
    target_sync_every: 100
    learn_start: 1000
    learn_every: 2
    eps_end: 0.17004595587593185
    eps_decay_steps: 50000
    grad_clip: 6.668923925628828
```

```
Best trial for dueling_double_dqn: #6
  Score: 2799.005
  avg_return: 2958.4
  std_return: 1594.0
  avg_length: 245.9
  avg_max_tile: 246.4
  avg_illegal_attempts: 76.6
  best_params:
    buffer_size: 200000
    batch_size: 64
    gamma: 0.9654707747743687
    lr: 0.0029222928006765354
    target_sync_every: 250
    learn_start: 2000
    learn_every: 8
    eps_end: 0.19214154949975376
    eps_decay_steps: 10000
    grad_clip: 1.1189121985227757
```

```
Best trial for qr_dqn: #16
  Score: 3038.938
  avg_return: 3169.4
  std_return: 1304.6
  avg_length: 265.6
  avg_max_tile: 249.6
  avg_illegal_attempts: 79.2
  best_params:
    buffer_size: 200000
    batch_size: 256
    gamma: 0.9922439589929455
    lr: 0.0001017167470236331
    target_sync_every: 1000
    learn_start: 5000
    learn_every: 2
    eps_end: 0.16221083409453396
    eps_decay_steps: 20000
    grad_clip: 15.05157300548717
```

```
Best trial for h_dqn: #6
  Score: 2255.863
  avg_return: 2407.0
  std_return: 1511.4
  avg_length: 207.7
  avg_max_tile: 206.4
  avg_illegal_attempts: 56.9
  best_params:
    buffer_size: 20000
    batch_size: 256
    gamma: 0.9974851723474096
    lr: 0.00017931204091428885
    target_sync_every: 250
    learn_start: 5000
    learn_every: 8
    eps_end: 0.04539376238997539
    eps_decay_steps: 20000
    grad_clip: 7.010189344276133
```

```
Best trial Mamba2 double_dqn: #24
  Score: 2580.963
  avg_return: 2693.6
  std_return: 1126.4
  avg_length: 233.3
  avg_max_tile: 212.8
  avg_illegal_attempts: 73.5
  best_params:
    buffer_size: 200000
    batch_size: 64
    seq_len: 12
    gamma: 0.958278580825883
    lr: 5.1152218086166416e-05
    target_sync_every: 100
    learn_start: 5000
    learn_every: 4
    eps_end: 0.09461215273949933
    eps_decay_steps: 5000
    grad_clip: 5.6534348547385
    hidden_dim: 256
    mamba_layers: 2
    mamba_state_dim: 64
    mamba_conv_dim: 2
    mamba_expand: 1
```

```
Best trial for mamba dqn: #56
  Score: 2912.702
  avg_return: 3052.6
  std_return: 1399.0
  avg_length: 255.8
  avg_max_tile: 243.2
  avg_illegal_attempts: 79.3
  best_params:
    buffer_size: 200000
    batch_size: 64
    seq_len: 16
    gamma: 0.9739846568718238
    lr: 4.1433825788519225e-05
    target_sync_every: 500
    learn_start: 2000
    learn_every: 8
    eps_end: 0.08453161403381923
    eps_decay_steps: 5000
    grad_clip: 14.823449461719127
    hidden_dim: 256
    mamba_layers: 4
    mamba_state_dim: 64
    mamba_conv_dim: 4
    mamba_expand: 2
```