import pandas as pd
import matplotlib.pyplot as plt
import ast

df = pd.read_csv('./ugmt_MuEta.csv')
df['Original'] = df['L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_original'].apply(ast.literal_eval)
df['Prediction'] = df['L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_prediction'].apply(ast.literal_eval)
df['Integral'] = df['L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_integral']
df = df.drop(columns=['L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_original','L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_prediction','L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_integral'])
runs_of_interest = [361303,361105,361052,360893]
selected_df = df[df['run_number'].isin(runs_of_interest)]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

for i, name in enumerate(runs_of_interest):
    row = i // 2  # Determine the row index for the subplot
    col = i % 2   # Determine the column index for the subplot
    array1 = selected_df[selected_df['run_number'] == name]['Original'].iloc[0]
    array2 = selected_df[selected_df['run_number'] == name]['Prediction'].iloc[0]
    chi2_val = selected_df[selected_df['run_number'] == name]['L1T//Run summary/L1TStage2uGMT/ugmtMuonEta_chi2_tol1'].iloc[0]

    array2scaled = [val * 0.01 for val in array2]
    
    axs[row, col].plot(array1, label='Original')
    axs[row, col].plot(array2scaled, label=r'Reco $\chi^{2}$ = ' + str(chi2_val.round(2)))
    axs[row, col].set_title('Run ' + str(name))
    axs[row, col].set_xlabel('uGMT Muon Eta')
    axs[row, col].set_ylabel('Counts')
    axs[row, col].legend()
    axs[row, col].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

print(selected_df)
selected_df.to_csv("./MuEtaOrigRecoArrays.csv",index=False)

