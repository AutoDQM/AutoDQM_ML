import pandas as pd

#df1 = pd.read_csv("HLTPhysics_AE_060324_runs_and_sse_scores.csv")
#df1 = df1.drop(columns=["algo","year"])
df1 = pd.read_csv("HLTPhysics_AE_180724_myfinalassessment.csv")
df1 = df1.drop(columns=["year"])
#df2 = pd.read_csv("HLTPhysics_PCA_060324_runs_and_sse_scores.csv")
#df2 = df2.drop(columns=["algo","year"])
df2 = pd.read_csv("HLTPhysics_PCA_180724_myfinalassessment.csv")
df2 = df2.drop(columns=["year"])
df3 = pd.read_csv("HLTPhysics_test_betab_runs_and_sse_scores_pull.csv")
df3 = df3.drop(columns=["algo"])
df4 = pd.read_csv("HLTPhysics_test_betab_runs_and_sse_scores_chi2.csv")
df4 = df4.drop(columns=["algo"])

df1.columns = [col + "_AE" if "L1T" in col else col for col in df1.columns]
df2.columns = [col + "_PCA" if "L1T" in col else col for col in df2.columns]
df3.columns = [col + "_BETAB_PULL" if "L1T" in col else col for col in df3.columns]
df4.columns = [col + "_BETAB_CHI2" if "L1T" in col else col for col in df4.columns]

print(df2.loc[df2['label'] == -1])
df2['label'] = df2['label'].replace(-1,0)

#df = df1.merge(df2, on=["run_number","label"], how='inner').merge(df3, on=["run_number","label"], how='inner').merge(df4, on=["run_number","label"], how='inner')
df = df2.merge(df3, on=["run_number","label"], how='inner').merge(df4, on=["run_number","label"], how='inner')
df_betab = df3.merge(df4, on=["run_number","label"], how='inner')
#print(df)
#df = df.loc[:, ~df.columns.duplicated()]

df.to_csv("./merged_df.csv", index=False)

df_betab.to_csv("./betab_df.csv", index=False)
