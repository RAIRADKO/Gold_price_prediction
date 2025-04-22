import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

data = pd.read_csv('gold.csv', parse_dates=['Date'])

data.sort_values('Date', inplace=True)
data['target_class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data['target_reg'] = data['Close'].shift(-1)
data.dropna(inplace=True)

feature_df = data.drop(['Date', 'target_class', 'target_reg'], axis=1)
X = feature_df.select_dtypes(include=[np.number])
y_class = data['target_class']
y_reg = data['target_reg']

test_size = 0.2
random_state = 42
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=test_size, random_state=random_state, shuffle=False
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=test_size, random_state=random_state, shuffle=False
)

det_scaler = StandardScaler()
X_train_c_scaled = det_scaler.fit_transform(X_train_c)
X_test_c_scaled = det_scaler.transform(X_test_c)
X_train_r_scaled = det_scaler.fit_transform(X_train_r)
X_test_r_scaled = det_scaler.transform(X_test_r)

POP_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
TOURNAMENT_SIZE = 3

def initialize_population(pop_size, chrom_length):
    return np.random.uniform(-1, 1, (pop_size, chrom_length))

def tournament_selection(pop, fitness, k):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), size=k, replace=False)
        winner = idx[np.argmax(fitness[idx])]
        selected.append(pop[winner])
    return np.array(selected)

def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1)-1)
        return (
            np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent2[:point], parent1[point:]])
        )
    return parent1.copy(), parent2.copy()

def mutate(chrom):
    mask = np.random.rand(len(chrom)) < MUTATION_RATE
    noise = np.random.normal(0, 0.1, size=len(chrom))
    return np.where(mask, chrom + noise, chrom)

print("=== Training Model Klasifikasi ===")
dim_c = X_train_c_scaled.shape[1] + 1
genomic_c = initialize_population(POP_SIZE, dim_c)
Xc_train = np.hstack([np.ones((X_train_c_scaled.shape[0],1)), X_train_c_scaled])
for gen in range(GENERATIONS):
    fitness_c = []
    for chrom in genomic_c:
        logits = Xc_train.dot(chrom)
        preds = (1/(1+np.exp(-logits)) >= 0.5).astype(int)
        fitness_c.append(accuracy_score(y_train_c, preds))
    fitness_c = np.array(fitness_c)
    print(f"Generasi {gen+1}/{GENERATIONS} - Best Fitness: {fitness_c.max():.4f}")
    parents_c = tournament_selection(genomic_c, fitness_c, TOURNAMENT_SIZE)
    offspring_c = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = parents_c[i], parents_c[(i+1)%POP_SIZE]
        c1, c2 = crossover(p1, p2)
        offspring_c.append(mutate(c1))
        offspring_c.append(mutate(c2))
    genomic_c = np.array(offspring_c)
best_c = genomic_c[np.argmax(fitness_c)]
Xc_test = np.hstack([np.ones((X_test_c_scaled.shape[0],1)), X_test_c_scaled])
preds_c_test = (1/(1+np.exp(-Xc_test.dot(best_c))) >= 0.5).astype(int)
print(f"Akurasi Test: {accuracy_score(y_test_c, preds_c_test):.4f}\n")

print("=== Training Model Regresi ===")
dim_r = X_train_r_scaled.shape[1] + 1
genomic_r = initialize_population(POP_SIZE, dim_r)
Xr_train = np.hstack([np.ones((X_train_r_scaled.shape[0],1)), X_train_r_scaled])
for gen in range(GENERATIONS):
    fitness_r = [-mean_squared_error(y_train_r, Xr_train.dot(ch)) for ch in genomic_r]
    fitness_r = np.array(fitness_r)
    print(f"Generasi {gen+1}/{GENERATIONS} - Best MSE: {-fitness_r.max():.4f}")
    parents_r = tournament_selection(genomic_r, fitness_r, TOURNAMENT_SIZE)
    offspring_r = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = parents_r[i], parents_r[(i+1)%POP_SIZE]
        c1, c2 = crossover(p1, p2)
        offspring_r.append(mutate(c1))
        offspring_r.append(mutate(c2))
    genomic_r = np.array(offspring_r)
best_r = genomic_r[np.argmax(fitness_r)]
Xr_test = np.hstack([np.ones((X_test_r_scaled.shape[0],1)), X_test_r_scaled])
preds_r_test = Xr_test.dot(best_r)
print(f"MSE Test: {mean_squared_error(y_test_r, preds_r_test):.4f}\n")
print("Contoh Prediksi vs Aktual:")
for i in range(5):
    print(f"Prediksi: {preds_r_test[i]:.2f}, Aktual: {y_test_r.iloc[i]:.2f}")
