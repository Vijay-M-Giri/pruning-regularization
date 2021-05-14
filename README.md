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
|                  |         |     ZO-0.1-0.9, 6, relu      | Input-28x28, Dense-2024-ZR-5e-8, Dense-2024-ZR-5e-8, Dense-2024-ZR-5e-8, Dense-10 |       0.28       |      1.74       | 1 initial epoch and 1 final epoch with SGD, ZR-f = 50, #of 0 = 6.13M/9.8M |
|                  |         |                              |                                                              |                  |                 |                                                              |
|                  |         |                              |                                                              |                  |                 |                                                              |
|                  |         |                              |                                                              |                  |                 |                                                              |

