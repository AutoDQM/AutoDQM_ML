import pandas as pd
import numpy as np

df2 = pd.read_csv("HLTPhysics_PCA_131224_modified_chi2_values.csv")
df2 = df2.drop(columns=["year","algo"])
df3 = pd.read_csv("HLTPhysics_8_REF.csv")
#df4 = pd.read_csv("HLTPhysics_8_REF.csv")

cols_to_keep = ["run_number","label"]

df2cols = [col for col in df2.columns if "_chi2prime" in col] + cols_to_keep
df3cols = [col for col in df3.columns if "_score_beta" in col] + cols_to_keep

df2 = df2[df2cols]
df3 = df3[df3cols]

#df2['label'] = df2['label'].replace(-1,0)
#df3['label'] = df3['label'].replace(-1,0)

df = df2.merge(df3, on=["run_number","label"], how='inner')
df.to_csv("./merged_df.csv", index=False)

df3.to_csv("./betab_df.csv", index=False)
