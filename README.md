# NHL Standings Predictor 🏒 

An hockey analytics engine that uses **Random Forest Regression** to predict future NHL standings. Unlike traditional models that only look at points, this project leverages underlying play-driving metrics (Corsi, Fenwick, and Expected Goals) to identify teams that are overperforming or underperforming.

## 🚀 Key Features
* **Predictive Modeling:** Uses a Random Forest Regressor trained on 15+ years of NHL game data.
* **Expansion-Aware:** Handles the entry of new teams (Vegas, Seattle) dynamically.
* **Backtesting Engine:** Includes a built-in evaluator to test model accuracy against historical seasons.
* **Advanced Metrics:** Incorporates `xG%`, `Corsi%`, and `Fenwick%` to account for puck luck and regression to the mean.

## 📊 Accuracy Metrics
The model is evaluated using:
* **Mean Absolute Error (MAE):** Average point deviation from actual results.
* **Bracket Accuracy:** Success rate in predicting if a team finishes in the top or bottom half of the league.
* **Point Accuracy %:** Precision of predicted points vs. actual points.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForestRegressor)
