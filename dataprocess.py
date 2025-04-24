# import libraries
import pandas as pd
import numpy as np

df = pd.read_csv(r"IMDb Movies India.csv", encoding="latin1") # load dataset and encode it to make it understand to the system.
# later start cleaning the data so that we can utilise it more efficiently.
df["Votes"] = (df["Votes"]
    .astype(str).str.replace(",", "", regex=True)
    .str.extract(r"(\d+\.?\d*)")[0].astype(float))
df["Year"]     = pd.to_numeric(df["Year"].str.extract(r"(\d{4})")[0], errors="coerce")
df["Duration"] = pd.to_numeric(df["Duration"].str.extract(r"(\d+)")[0], errors="coerce")
df.dropna(subset=["Votes","Year","Duration"], inplace=True)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

"""
here introduced genre success rate, director successrate, year success rate,actor success rate,duration success rate and similarmovie avg rating 
based on all these factors we predict the movie rating
"""
df["Genre_Success_Rate"]    = df.groupby("Genre")["Rating"].transform("mean")
df["Director_Success_Rate"] = df.groupby("Director")["Rating"].transform("mean")
df["Year_Success_Rate"]     = df.groupby("Year")["Rating"].transform("mean")
for col in ["Actor 1","Actor 2","Actor 3"]:
    df[f"{col}_Success"] = df.groupby(col)["Rating"].transform("mean")
df["Actor_Success_Rate"] = df[["Actor 1_Success","Actor 2_Success","Actor 3_Success"]].mean(axis=1)
df["Duration_Range"] = pd.cut(df["Duration"], bins=[0,90,120,150,np.inf], labels=["Short","Medium","Long","Very Long"])
df["Duration_Success_Rate"] = df.groupby("Duration_Range", observed=False)["Rating"].transform("mean")
df["Similar_Avg_Rating"] = df.groupby(["Genre","Year"])["Rating"].transform("mean")

# here we do imputing of our data frame by just avg of remaining available features to handle the missing values.
for c in ["Genre_Success_Rate","Director_Success_Rate","Year_Success_Rate",
          "Actor 1_Success","Actor 2_Success","Actor 3_Success","Actor_Success_Rate",
          "Duration_Success_Rate","Similar_Avg_Rating"]:
    df[c] = df[c].fillna(df[c].mean())

# here we are categorising the variables into the numerical formats.
cat_df = pd.get_dummies(df[["Genre","Duration_Range"]], drop_first=True, prefix=["G","D"])
df = pd.concat([df, cat_df], axis=1)
