from solution import NN
import matplotlib.pyplot as plt
import time
#from  import
from joblib import dump



n_epochs = 50
seedR=[3000]
for seed_available in seedR:
    print('\n\n########################## SEED : ', seed_available, '##########################')
    t1 = time.time()
    cifarNN = NN(hidden_dims=(512,120, 120,120,120,120,120), lr=3e-3, batch_size=100, seed=seed_available)
    cifarNN.train_loop(n_epochs)

    t2=time.time()
    print(f"time : {t2-t1}")

    x = [i for i in range(n_epochs)]

    dump(cifarNN.train_logs["train_accuracy"], f"train_accuracy_seed_{seed_available}.val")
    dump(cifarNN.train_logs["validation_accuracy"], f"validation_accuracy_seed_{seed_available}.val")
    dump(cifarNN.train_logs["train_loss"], f"train_loss_seed_{seed_available}.val")
    dump(cifarNN.train_logs["validation_loss"], f"validation_loss_seed_{seed_available}.val")