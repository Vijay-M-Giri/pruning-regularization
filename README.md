# Zero optimizer and regularizer

## Results

### MNIST

| Test, Run number | Dataset | Optimizer, epoch, activation |                            Layers                            | Train error rate | Test error rate | Remarks                                                      |
| :--------------: | ------- | :--------------------------: | :----------------------------------------------------------: | :--------------: | :-------------: | ------------------------------------------------------------ |
|                  | MNIST   |      SGD-0.1, 50, relu       |  Input-28x28, Dense-1024, Dense-1024, Dense-1024, Dense-10   |        0         |      1.56       |                                                              |
|                  |         |      SGD-0.1, 50, relu       | Input-28x28, Dropout-0.2, Dense-1024, Dropout-0.5, Dense-1024, Dropout-0.5, Dense-1024, Dense-10 |       1.21       |      1.27       |                                                              |
|                  |         |       ZO-0.1, 50, relu       |  Input-28x28, Dense-1024, Dense-1024, Dense-1024, Dense-10   |        0         |      1.60       | #of 0weights = 742k/2.9M                                     |
|                  |         |     ZO-0.1-0.9, 10, relu     | Input-28x28, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-10-ZR-8e-7 |       0.16       |       1.7       | 1 initial epoch with SGD, #of 0weights = 1.95M/2.9M          |
|                  |         |     ZO-0.1-0.9, 6, relu      | Input-28x28, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-10-ZR-8e-7 |       0.37       |      1.71       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 2.57M/2.9M |
|                  |         |     ZO-0.1-0.9, 8, relu      | Input-28x28, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-10-ZR-8e-7 |       0.24       |      1.66       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 2.6M/2.9M |
|                  |         |     ZO-0.1-0.9, 18, relu     | Input-28x28, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-10-ZR-8e-7 |      0.002       |      1.44       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 2.7M/2.9M |
|                  |         |     ZO-0.1-0.9, 18, relu     | Input-28x28, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-1024-ZR-8e-7, Dense-10-ZR-8e-7 |      0.003       |      1.40       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 2.71M/2.9M |
|                  |         |     ZO-0.1-0.9, 6, relu      | Input-28x28, Dense-2024-ZR-5e-8, Dense-2024-ZR-5e-8, Dense-2024-ZR-5e-8, Dense-10 |       0.28       |      1.74       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 = 6.13M/9.8M |
|                  |         |     ZO-0.1-0.9, 18, relu     | Input-28x28, Dense-2024-ZR-8e-8, Dense-2024-ZR-8e-8, Dense-2024-ZR-8e-8, Dense-10-ZR-8e-8 |      0.002       |      1.28       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 = 8.8M/9.8M |
|                  |         |      SGD-0.1, 15, relu       |         Input-28x28, Dense-300, Dense-200, Dense-10          |       0.01       |      1.65       |                                                              |
|                  |         |     ZO-0.1-0.9, 13, relu     | Input-28x28, Dense-300-ZR-2e-6, Dense-200-ZR-2e-6, Dense-10-ZR-2e-6 |       0.01       |      1.63       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 223k/266k |
|                  |         |     ZO-0.1-0.9, 13, relu     | Input-28x28, Dense-300-ZR-2e-6, Dense-200-ZR-2e-6, Dense-10-ZR-2e-6 |       0.06       |      1.63       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 223k/266k |
|                  |         |     ZO-0.1-0.9, 13, relu     | Input-28x28, Dense-300-ZR-1e-5, Dense-200-ZR-1e-5, Dense-10-ZR-1e-5 |       0.05       |      1.64       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 242k/266k |
|                  |         |     ZO-0.1-0.9, 20, relu     | Input-28x28, Dense-300-ZR-5e-5, Dense-200-ZR-5e-5, Dense-10-ZR-5e-5 |       0.06       |      1.86       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 weights = 253k/266k |

