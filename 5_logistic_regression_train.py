
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

marker_cols = [
    'has_hyperbole', 'has_emotion', 'has_vague_source', 'has_clickbait', 'has_hedging', 'has_bias_word',
    'has_negation', 'has_superlative', 'has_certainty', 'has_quantifier',
    'char_length', 'word_count', 'avg_word_length',
    'modal_count', 'adj_noun_pairs', 'passive_voice_count'
]

# Reddit
df_reddit = pd.read_csv("2_reddit_dataset_preprocessed.csv")
X_r = df_reddit[marker_cols]
y_r = df_reddit['label']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

lr_r = LogisticRegression(max_iter=1000)
lr_r.fit(X_train_r, y_train_r)
y_pred_r = lr_r.predict(X_test_r)

report_r = classification_report(y_test_r, y_pred_r, output_dict=True)
df_r = pd.DataFrame(report_r).transpose().loc[['0', '1']]
df_r['Dataset'] = 'Reddit'
df_r.reset_index(inplace=True)

# ISOT
df_isot = pd.read_csv("4_isot_dataset_preprocessed.csv")
X_i = df_isot[marker_cols]
y_i = df_isot['label']
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i, y_i, test_size=0.2, random_state=42)

scaler_i = StandardScaler()
X_train_i = scaler_i.fit_transform(X_train_i)
X_test_i = scaler_i.transform(X_test_i)

lr_i = LogisticRegression(max_iter=1000)
lr_i.fit(X_train_i, y_train_i)
y_pred_i = lr_i.predict(X_test_i)

report_i = classification_report(y_test_i, y_pred_i, output_dict=True)
df_i = pd.DataFrame(report_i).transpose().loc[['0', '1']]
df_i['Dataset'] = 'ISOT'
df_i.reset_index(inplace=True)

# Combine
df_all = pd.concat([df_r, df_i], ignore_index=True)
df_all = df_all.rename(columns={'index': 'Class'})
df_all.to_csv("5_logistic_regression_results.csv", index=False)

print("\nIt was saved as 'logistic_regression_classwise_metrics_scaled.xlsx'")
