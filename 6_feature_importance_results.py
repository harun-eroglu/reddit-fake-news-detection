
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

feature_cols = [
    'has_hyperbole', 'has_emotion', 'has_vague_source', 'has_clickbait', 'has_hedging', 'has_bias_word',
    'has_negation', 'has_superlative', 'has_certainty', 'has_quantifier',
    'char_length', 'word_count', 'avg_word_length',
    'modal_count', 'adj_noun_pairs', 'passive_voice_count'
]

# ISOT
df_isot = pd.read_csv("4_isot_dataset_preprocessed.csv")
X_isot = df_isot[feature_cols]
y_isot = df_isot['label']
X_train_i, _, y_train_i, _ = train_test_split(X_isot, y_isot, test_size=0.2, random_state=42)

# ISOT + RF
rf_isot = RandomForestClassifier(random_state=42)
rf_isot.fit(X_train_i, y_train_i)
rf_isot_importance = rf_isot.feature_importances_

# ISOT + XGB
xgb_isot = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_isot.fit(X_train_i, y_train_i)
xgb_isot_importance = xgb_isot.feature_importances_

# Reddit
df_reddit = pd.read_csv("2_reddit_dataset_preprocessed.csv")
X_reddit = df_reddit[feature_cols]
y_reddit = df_reddit['label']
X_train_r, _, y_train_r, _ = train_test_split(X_reddit, y_reddit, test_size=0.2, random_state=42)

# Reddit + RF
rf_reddit = RandomForestClassifier(random_state=42)
rf_reddit.fit(X_train_r, y_train_r)
rf_reddit_importance = rf_reddit.feature_importances_

# Reddit + XGB
xgb_reddit = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_reddit.fit(X_train_r, y_train_r)
xgb_reddit_importance = xgb_reddit.feature_importances_

# Combine everything
combined_df = pd.DataFrame({
    'Feature': feature_cols,
    'ISOT_RF': rf_isot_importance,
    'ISOT_XGB': xgb_isot_importance,
    'REDDIT_RF': rf_reddit_importance,
    'REDDIT_XGB': xgb_reddit_importance
})

combined_df.sort_values(by='ISOT_RF', ascending=False, inplace=True)

combined_df.to_csv("6_feature_importance_combined.csv", index=False)

# Terminal output
print("\n--- Combined Feature Importances ---")
print(combined_df)

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("6_feature_importance_combined.csv")

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.set_index("Feature"), annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Importance'})
plt.title("Feature Importance Heatmap – All Models")
plt.tight_layout()

# PNG
plt.savefig("6_feature_importance_combined_heatmap.png")
plt.close()
