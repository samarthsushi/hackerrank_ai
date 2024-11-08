import pandas as pd
from collections import defaultdict

training_file = r"C:\codes\hackerrank_ai\dota\data\training_data.txt"
data = pd.read_csv(training_file, header=None)

heroes = data.iloc[:, :10].values
all_heroes = heroes.flatten().tolist()
unique_heroes = sorted(set(all_heroes))
hero_to_id = {hero: idx for idx, hero in enumerate(unique_heroes)}

print(hero_to_id)
def encode_heroes(hero_array, hero_to_id):
    return [[hero_to_id[hero] for hero in row] for row in hero_array]

encoded_heroes = encode_heroes(heroes, hero_to_id)

results = data.iloc[:, 10].values
encoded_matches = [encoded_heroes[i] + [results[i]] for i in range(len(encoded_heroes))]

hero_pairs = defaultdict(int)
pair_scores = defaultdict(float)

for match in encoded_matches:
    team1 = match[:5]  
    team2 = match[5:10]  
    winner = match[10] 

    for hero1 in team1:
        for hero2 in team2:
            pair = tuple(sorted([hero1, hero2]))  
            hero_pairs[pair] += 1 
            if winner == 1:
                pair_scores[pair] += 1 
            else:
                pair_scores[pair] -= 1  

correlation_weights = {
        pair: pair_scores[pair] / hero_pairs[pair]
        for pair in hero_pairs
    }


def calculate_team_score(team1, team2, correlation_weights):
    score = 0
    for hero1 in team1:
        for hero2 in team2:
            pair = tuple(sorted([hero1, hero2])) 
            score += correlation_weights.get(pair, 0)  
    return score

def predict_matches(matches_to_predict, hero_to_id, correlation_weights):
    predictions = []
    for match in matches_to_predict:
        team1 = [hero_to_id[hero] for hero in match[:5]]
        team2 = [hero_to_id[hero] for hero in match[5:]]
        
        team1_score = calculate_team_score(team1, team2, correlation_weights)
        team2_score = -team1_score  
        
        winner = 1 if team1_score > team2_score else 2
        predictions.append(winner)
    return predictions

input_data = pd.read_csv(r"C:\codes\hackerrank_ai\dota\data\sample_input.txt", header = None)
predictions = predict_matches(input_data.values, hero_to_id, correlation_weights)

for prediction in predictions:
    print(prediction)