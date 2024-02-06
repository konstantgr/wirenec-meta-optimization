import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_opt = pd.read_csv(r"C:\Users\mikzu\Downloads\Wire_1e3_Opt.txt", sep='\t', encoding='utf-8')
df_base = pd.read_csv(r"C:\Users\mikzu\Downloads\Wire_non_opt.txt", sep='\t', encoding='utf-8')

for col in list(df_opt)[1:]:
    plt.plot(df_opt.freq, df_opt[col], label=f'scattering angle = {col.split("_")[-1]}$\degree$')
    plt.plot(df_base.freq, df_base[col], color='k', alpha=0.8, ls=(2, (2, 2)))

plt.xlabel('Frequency, MHz')
plt.ylabel(('Backward Scattering, $m^2$'))
plt.legend()
plt.xlim(2000, 10000)
plt.show()
