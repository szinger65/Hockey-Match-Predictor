import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def evaluate_full_metrics(target_year):
    #the all_teams.csv is too big for github, the data comes from MoneyPuck.com
    df = pd.read_csv('all_teams.csv')
    df = df[(df['situation'] == 'all') & (df['playoffGame'] == 0)].copy()

    def get_pts(row):
        if row['goalsFor'] > row['goalsAgainst']: return 2
        elif row['iceTime'] > 3600: return 1
        return 0

    df['game_points'] = df.apply(get_pts, axis=1)

    season_stats = df.groupby(['team', 'season']).agg({
        'game_points': 'sum',
        'gameId': 'count',
        'xGoalsPercentage': 'mean',
        'corsiPercentage': 'mean',
        'fenwickPercentage': 'mean'
    }).reset_index()

    season_stats['point_pct'] = season_stats['game_points'] / (season_stats['gameId'] * 2)
    season_stats = season_stats.sort_values(['team', 'season'])
    season_stats['next_season_pct'] = season_stats.groupby('team')['point_pct'].shift(-1)

    features = ['xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage', 'point_pct']
    
    train_data = season_stats[season_stats['season'] < target_year].dropna(subset=['next_season_pct'])
    test_data = season_stats[season_stats['season'] == (target_year - 1)].copy()
    
    if test_data.empty or train_data.empty:
        print(f"Data missing for {target_year} or its preceding year.")
        return

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train_data[features], train_data['next_season_pct'])

    test_data['predicted_pct'] = model.predict(test_data[features])
    actuals = season_stats[season_stats['season'] == target_year][['team', 'point_pct']]
    actuals.columns = ['team', 'actual_pct']
    
    results = test_data.merge(actuals, on='team')
    results['Pred_Pts'] = (results['predicted_pct'] * 164).round(1)
    results['Actual_Pts'] = (results['actual_pct'] * 164).round(1)
    results['Error'] = (results['Pred_Pts'] - results['Actual_Pts']).round(1)
    
    results['Accuracy_Pct'] = (1 - (abs(results['Pred_Pts'] - results['Actual_Pts']) / results['Actual_Pts'])) * 100
    median_actual = results['Actual_Pts'].median()
    median_pred = results['Pred_Pts'].median()
    results['Correct_Bracket'] = (results['Pred_Pts'] > median_pred) == (results['Actual_Pts'] > median_actual)
    mae = mean_absolute_error(results['Actual_Pts'], results['Pred_Pts'])
    avg_acc = results['Accuracy_Pct'].mean()
    bracket_acc = (results['Correct_Bracket'].sum() / len(results)) * 100
    print_results = results[['team', 'Pred_Pts', 'Actual_Pts', 'Error', 'Accuracy_Pct']].sort_values('Actual_Pts', ascending=False)
    
    print(f"--- {target_year} Standings, Accuracy & Error ---")
    print(print_results.to_string(index=False))
    print("-" * 50)
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"Average Point Accuracy:    {avg_acc:.2f}%")
    print(f"Bracket Match Accuracy:    {bracket_acc:.2f}%")

evaluate_full_metrics(2024)
