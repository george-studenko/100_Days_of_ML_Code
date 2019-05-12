from sklearn import model_selection
import pandas as pd

rent_prices_evolution_2018 = ['datasets/2018_lloguer_preu_trim.csv', ',', 'utf-8']

nationalities = ['datasets/2018_POBLACIO_NACIONALITAT.csv',';', 'latin1']
csv = pd.read_csv(nationalities[0], sep=nationalities[1], encoding =nationalities[2])
print(csv.head)

