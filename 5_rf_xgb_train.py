
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

feature_cols = [
    'has_hyperbole', 'has_emotion', 'has_vague_source', 'has_clickbait', 'has_hedging', 'has_bias_word',
    'has_negation', 'has_superlative', 'has_certainty', 'has_quantifier',
    'char_length', 'word_count', 'avg_word_length',
    'modal_count', 'adj_noun_pairs', 'passive_voice_count'
]

def process_dataset(path, dataset_name):
    df = pd.read_csv(path)
    X = df[feature_cols]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    for name, model in {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose().loc[['0', '1']]
        df_report['Model'] = name
        df_report['Dataset'] = dataset_name
        df_report.reset_index(inplace=True)
        results.append(df_report)

    return pd.concat(results, ignore_index=True)

# For both datasets
df_reddit = process_dataset("2_reddit_dataset_preprocessed.csv", "Reddit")
df_isot = process_dataset("4_isot_dataset_preprocessed.csv", "ISOT")

# Combine
df_combined = pd.concat([df_reddit, df_isot], ignore_index=True)
df_combined = df_combined.rename(columns={'index': 'Class'})
df_combined.to_csv("5_rf_xgb_results.csv", index=False)

print("\nIt was saved as rf_xgb_classwise_metrics.xlsx.")
