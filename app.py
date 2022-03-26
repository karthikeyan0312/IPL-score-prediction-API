import gzip
import numpy as np
import pickle as pk
import gzip
from flask import Flask, jsonify
from flask_restful import Resource, Api,reqparse,request
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

parser = reqparse.RequestParser()
parser.add_argument('over', type=int,help= " over not entered")
parser.add_argument('wickets', type=int,help= " wickets not entered")
parser.add_argument('runs', type=int,help= " runs not entered")
parser.add_argument('last_5_over_wickets', type=int,help= " last_5_over_wickets not entered")
parser.add_argument('last_5_over_runs', type=int,help= " last_5_over_runs not entered")
parser.add_argument('batting_team',help= " batting_team not entered")
parser.add_argument('bowling_team',help= " bowling_team not entered")
parser.add_argument('venue',help= " venue not entered")

teams = {'Chennai Super Kings': 6,
 'Delhi Capitals': 1,
 'Kolkata Knight Riders': 5,
 'Mumbai Indians': 7,
 'Punjab Kings': 4,
 'Rajasthan Royals': 2,
 'Royal Challengers Bangalore': 3,
 'Sunrisers Hyderabad': 0}
columns = np.array(['over', 'wickets', 'runs', 'last_5_over_wickets',
       'last_5_over_runs', 'batting_team', 'bowling_team',
       'Brabourne Stadium', 'Buffalo Park', 'De Beers Diamond Oval',
       'Dr DY Patil Sports Academy',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Dubai International Cricket Stadium', 'Eden Gardens',
       'Feroz Shah Kotla', 'Himachal Pradesh Cricket Association Stadium',
       'Holkar Cricket Stadium', 'JSCA International Stadium Complex',
       'Kingsmead', 'M Chinnaswamy Stadium',
       'MA Chidambaram Stadium, Chepauk',
       'Maharashtra Cricket Association Stadium', 'New Wanderers Stadium',
       'Newlands', 'OUTsurance Oval',
       'Punjab Cricket Association Stadium, Mohali',
       'Rajiv Gandhi International Stadium, Uppal',
       'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium',
       'Shaheed Veer Narayan Singh International Stadium',
       'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium',
       "St George's Park", 'Subrata Roy Sahara Stadium',
       'SuperSport Park', 'Wankhede Stadium'])

def load_model():
    global model
    global scaler
    global columns
    global teams

    with gzip.open(r"/app/model.pickle.gz", "rb") as f:
        model = pk.load(f)

    with open(r"/app/scaler(1).pickle", "rb") as f:
        scaler = pk.load(f)

    return list(teams.keys()), list(columns[7:]) + ["Barabati Stadium"]

def predict_score(overs, wickets, runs, wickets_last_5, runs_last_5, bat_team, bowl_team, venue):
    try:
        X_pred = np.zeros(columns.size)

        X_pred[0:7] = [overs, wickets, runs, wickets_last_5, runs_last_5, teams[bat_team], teams[bowl_team]]

        if venue != "Barabati Stadium":
            # because i removed first columns for prevent dummy variable trap
            # and first column of venue was Barabati Stadium
            venue_index = np.where(columns == venue)[0][0]

        X_pred = scaler.transform([X_pred])

        result = model.predict(X_pred)[0]
        return result
    except :
        return 1 # error code 1

class Randomforest(Resource):

    def post(self):
        data = parser.parse_args()
        #over,wickets,runs,last_5_over_wickets,last_5_over_runs,batting_team,bowling_team,venue = dict(data).values()
        over= data["over"]
        wickets = data["wickets"]
        runs = data["runs"]
        last_5_over_wickets = data["last_5_over_wickets"]
        last_5_over_runs = data["last_5_over_runs"]
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        venue = data["venue"]
        score =  predict_score(over, wickets, runs, last_5_over_wickets, last_5_over_runs, batting_team, bowling_team, venue)

        
        #if score == 1:
            #return jsonify({"Error":"Invalid Data"})
            #return jsonify({"Score" : str(score)})

        return jsonify({"Score" : str(score)})
        #return [over, wickets, runs, last_5_over_wickets, last_5_over_runs, batting_team, bowling_team, venue]

    
    
class status(Resource):    
    def get(self):
        try:
            return {'data': 'Api running'}
        except(error): 
            return {'data': error}

api.add_resource(Randomforest, '/v1/model')
api.add_resource(status,'/')
if __name__ == "__main__":
    load_model()
    app.run(debug=True)
    

