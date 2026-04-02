#include<bits/stdc++.h>

using namespace std;

map<string, int> teams={
    {"Arsenal", 0},
    {"Aston Villa", 1},
    {"Bournemouth", 2},
    {"Brentford", 3},
    {"Brighton", 4},
    {"Chelsea", 5},
    {"Crystal Palace", 6},
    {"Everton", 7},
    {"Fulham", 8},
    {"Ipswich", 9},
    {"Leicester", 10},
    {"Liverpool", 11},
    {"Man City", 12},
    {"Man United", 13},
    {"Newcastle", 14},
    {"Nott'm Forest", 15},
    {"Southampton", 16},
    {"Tottenham", 17},
    {"West Ham", 18},
    {"Wolves", 19},
};
vector<string> teamsIdx(20);

struct Game{
    int homeTeam, awayTeam, goalsHome, goalsAway;
};
const int N= 20;
struct Variables{
    vector<double> attack, defense;
    double mu; // baseline
    double H ; // home advantage

    Variables() : attack(20, 0.0), defense(20, 0.0), mu(0.0), H(0.0) {}
};

struct Stats{
    vector<double> avgGoals, totalGoals;

    Stats() : avgGoals(20,0.0), totalGoals(20,0.0) {}
};

vector<Game> loadData(const string& filename) { 
    vector<Game> data;
    string line;
    ifstream file(filename);

    if(!file.is_open()){
        printf("Failed to open file\n");
        return {};
    }
    getline(file, line); // skip header


    while(getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<string> cols;
        while(getline(ss, token, ',')) cols.push_back(token);

        if(cols.size() < 7) continue; // skip malformed rows

        string home = cols[3];
        string away = cols[4];
        int gh = stoi(cols[5]);
        int ga = stoi(cols[6]);

        data.push_back({teams[home], teams[away], gh, ga});
    }
    return data;
}

double testVars(Stats vars, vector<Game>& data, int startIdx=0, int endIdx=379,int N=20){
    double log_likelihood=0;
    for(int i=startIdx;i<=endIdx;i++){
        Game game = data[i];
        int h = game.homeTeam;
        int a = game.awayTeam;
        double lambda_home = vars.avgGoals[h];
        double lambda_away = vars.avgGoals[a];

        log_likelihood += game.goalsHome * log(lambda_home) - lambda_home;
        log_likelihood += game.goalsAway * log(lambda_away) - lambda_away;
    }
    return log_likelihood;
}

int main(){

    const int N = 20;
    const double learning_rate = 1e-4;

    vector<Game> data=loadData("24-25EPL.csv");
    for(auto e: teams) teamsIdx[e.second]=e.first;

    Stats vars;
    int numGames=0;
    for(Game game : data){
        // printf("Home(%s) %d - %d Away(%s)\n", teamsIdx[game.homeTeam].c_str(), game.goalsHome, game.goalsAway, teamsIdx[game.awayTeam].c_str());
        vars.totalGoals[game.homeTeam]+=game.goalsHome;
        vars.totalGoals[game.awayTeam]+=game.goalsAway;
        numGames++;
        if(numGames==300) break;
    }

    for(int i=0;i<N;i++){
        vars.avgGoals[i] = vars.totalGoals[i]/38.0;
    }
    

    cout<<"\n\n-------------------\n\n";

    cout<<"train LL/game on first 300 games: "<<(testVars(vars, data, 0, 299))/(300.0)<<'\n';
    cout<<"train LL/game on last 80 games: "<<(testVars(vars, data, 300))/(80.0)<<'\n';


    // Variables vars = mle(data);

    // cout<<"ATT scores:\n";
    // for(int i=0;i<N;i++){
    //     cout<<vars.attack[i]<<" | "<<teamsIdx[i]<<'\n';
    // }
    // cout<<"DEF scores:\n";
    // for(int i=0;i<N;i++){
    //     cout<<vars.defense[i]<<" | "<<teamsIdx[i]<<'\n';
    // }

    // cout<<"Mu: "<<vars.mu<<'\n';
    // cout<<"Home Advantage: "<<vars.H<<'\n';
    
}