import pandas as pd
import matplotlib.pyplot as plt
import ast


hist_names = ['L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTRank','L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank','L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/HTTRank','L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METRank']

df = pd.read_csv('./HLTPhysics_PCA_NEW_chi2.csv')

for i in range(len(hist_names)):
    df['Original_'+str(i)] = df[hist_names[i]+'_original'].apply(ast.literal_eval)
    df['Prediction_'+str(i)] = df[hist_names[i]+'_prediction'].apply(ast.literal_eval)
    print(df[hist_names[i]+'_prediction'])
    df['Integral_'+str(i)] = df[hist_names[i]+'_integral']
    df['X2tol1_'+str(i)] = df[hist_names[i]+'_chi2_tol1']
    df = df.drop(columns=[hist_names[i]+'_original',hist_names[i]+'_prediction',hist_names[i]+'_integral',hist_names[i]+'_chi2_tol1'])

runs_of_interest = [361365]
selected_df = df[df['run_number'].isin(runs_of_interest)]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#for i, name in enumerate(runs_of_interest):
for i in range(len(hist_names)):
    row = i // 2  # Determine the row index for the subplot
    col = i % 2  # Determine the column index for the subplot
    array1 = selected_df[selected_df['run_number'] == runs_of_interest[0]]['Original_'+str(i)].iloc[0]
    array2 = selected_df[selected_df['run_number'] == runs_of_interest[0]]['Prediction_'+str(i)].iloc[0]
    chi2_val = selected_df[selected_df['run_number'] == runs_of_interest[0]]['X2tol1_'+str(i)].iloc[0]
    array2scaled = [val * 0.01 for val in array2]

    axs[row, col].plot(array1, label='Original')
    print(array2scaled)
    axs[row, col].plot(array2scaled, label=r'Reco $\chi^{2}$ = ' + str(chi2_val.round(2)))
    axs[row, col].set_title('Run ' + str(runs_of_interest[0]))
    axs[row, col].set_xlabel(hist_names[i].split("/")[-1])
    axs[row, col].set_ylabel('Counts')
    axs[row, col].legend()
    axs[row, col].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

print(selected_df)
selected_df.to_csv("./EnergySums1OrigRecoArrays.csv",index=False)

