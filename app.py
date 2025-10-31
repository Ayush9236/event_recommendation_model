from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved models
model = joblib.load("recommendation_model.pkl")
event_similarity_df = joblib.load("event_similarity.pkl")
user_event_matrix = joblib.load("user_event_matrix.pkl")
df = pd.read_csv("event_management_with_clicks_v2_consistent_ids.csv")

def recommend_for_existing_user(user_id, n=5):
    all_events = user_event_matrix.columns
    known_events = user_event_matrix.loc[user_id]
    unseen_events = all_events[known_events == 0]
    predictions = [model.predict(user_id, eid) for eid in unseen_events]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:n]
    event_info = df.drop_duplicates('event_id').set_index('event_id')[['event_type', 'event_sub_event']].to_dict('index')
    results = []
    for pred in top_preds:
        info = event_info.get(pred.iid, {'event_type': 'Unknown', 'event_sub_event': 'Unknown'})
        results.append({'event_id': pred.iid, 'event_type': info['event_type'], 'event_sub_event': info['event_sub_event']})
    return results

def recommend_similar_event(event_id, n=5):
    if event_id not in event_similarity_df.index:
        return []
    similar_events = event_similarity_df[event_id].sort_values(ascending=False)[1:n+1]
    event_info = df.drop_duplicates('event_id').set_index('event_id')[['event_type', 'event_sub_event']].to_dict('index')
    results = []
    for eid, sim in similar_events.items():
        info = event_info.get(eid, {'event_type': 'Unknown', 'event_sub_event': 'Unknown'})
        results.append({'event_id': eid, 'event_type': info['event_type'], 'event_sub_event': info['event_sub_event']})
    return results

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id')
    event_id = request.form.get('event_id')
    if user_id and user_id in user_event_matrix.index:
        recommendations = recommend_for_existing_user(user_id)
    elif event_id:
        recommendations = recommend_similar_event(event_id)
    else:
        recommendations = []
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

df = pd.read_csv("event_management_with_clicks_v2_consistent_ids.csv")

