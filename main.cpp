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

struct Game{
    int homeTeam, awayTeam, goalsHome, goalsAway;
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

int main(){

    const int N = 20;
    const double learning_rate = 1e-4;

    vector<Game> data=loadData("24-25EPL.csv");
    vector<string> teamsIdx(20);
    for(auto e: teams) teamsIdx[e.second]=e.first;

    // for(Game game : data){
    //     printf("Home(%s) %d - %d Away(%s)\n", teamsIdx[game.homeTeam].c_str(), game.goalsHome, game.goalsAway, teamsIdx[game.awayTeam].c_str());
    // }

    vector<double> attack(N,0), defense(N,0);
    double mu=0; // baseline
    double H =0; // home advantage

    double last_log_likelihood=1;
    double cur_log_likelihood=0;

    int counter=0;

    while(abs(cur_log_likelihood-last_log_likelihood)>1e-6 && counter<250000){
        cout<<"counter = "<<counter<<'\n';
        counter++;


        last_log_likelihood=cur_log_likelihood;
        cur_log_likelihood=0;
        vector<double> grad_att(N,0), grad_def(N,0);
        double grad_mu = 0;
        double grad_H = 0;
        for(Game game : data){
            int h = game.homeTeam;
            int a = game.awayTeam;
            double lambda_home = exp(attack[h]-defense[a]+mu+H);
            double lambda_away = exp(attack[a]-defense[h]+mu);

            cur_log_likelihood += game.goalsHome * log(lambda_home) - lambda_home;
            cur_log_likelihood += game.goalsAway * log(lambda_away) - lambda_away;

            double residue_home = game.goalsHome - lambda_home;
            double residue_away = game.goalsAway - lambda_away;

            grad_att[h] += residue_home;
            grad_att[a] += residue_away;
            grad_def[h] -= residue_away;
            grad_def[a] -= residue_home;
            grad_mu += residue_home + residue_away;
            grad_H += residue_home;
        }

        for(int i = 0; i < N; i++){
            attack[i] += learning_rate * grad_att[i];
            defense[i] += learning_rate * grad_def[i];
        }
        mu += learning_rate*grad_mu;
        H += learning_rate*grad_H;

        double avg_att = 0;
        for(int i = 0; i < N; i++) avg_att += attack[i];
        avg_att /= N;
        for(int i = 0; i < N; i++) attack[i] -= avg_att;

        double avg_def = 0;
        for(int i = 0; i < N; i++) avg_def += defense[i];
        avg_def /= N;
        for(int i = 0; i < N; i++) defense[i] -= avg_def;

    }


    cout<<"ATT scores:\n";
    for(int i=0;i<N;i++){
        cout<<attack[i]<<" | "<<teamsIdx[i]<<'\n';
    }
    cout<<"DEF scores:\n";
    for(int i=0;i<N;i++){
        cout<<defense[i]<<" | "<<teamsIdx[i]<<'\n';
    }

    cout<<"Mu: "<<mu<<'\n';
    cout<<"Home Advantage: "<<H<<'\n';
}