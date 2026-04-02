#include<bits/stdc++.h>

using namespace std;

map<string, int> teams={
    {"Arsenal", 0}, // 70
    {"Aston Villa", 1}, //54
    {"Bournemouth", 2}, // 42
    {"Brentford", 3}, // 46 
    {"Brighton", 4}, // 43
    {"Chelsea", 5}, // 48
    {"Crystal Palace", 6}, // 39
    {"Everton", 7}, // 46
    {"Fulham", 8}, // 44
    {"Sunderland", 9}, // 43
    {"Burnley", 10}, // 20
    {"Liverpool", 11},// 49
    {"Man City", 12}, //61
    {"Man United", 13},//55
    {"Newcastle", 14}, // 42
    {"Nott'm Forest", 15}, //32
    {"Leeds", 16}, // 33
    {"Tottenham", 17}, // 30
    {"West Ham", 18}, // 29
    {"Wolves", 19}, // 17
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

Variables mle(vector<Game>& data, int numGames=380, double learning_rate=1e-4, int N=20){
    if(numGames<0 || numGames>data.size()){
        cout<<"NUM GAMES INVALID\n";
        numGames=data.size();
    }
    
    Variables vars;

    double last_log_likelihood=1;
    double cur_log_likelihood=0;

    int counter=0;

    while(abs(cur_log_likelihood-last_log_likelihood)>1e-6 && counter<250000){
        if(counter%50==0) cout<<"iterations = "<<counter<<", log likelihood = "<<cur_log_likelihood<<'\n';
        counter++;


        last_log_likelihood=cur_log_likelihood;
        cur_log_likelihood=0;
        vector<double> grad_att(N,0), grad_def(N,0);
        double grad_mu = 0;
        double grad_H = 0;
        for(int i=0;i<numGames;i++){
            Game game = data[i];
            int h = game.homeTeam;
            int a = game.awayTeam;
            double lambda_home = exp(vars.attack[h]-vars.defense[a]+vars.mu+vars.H);
            double lambda_away = exp(vars.attack[a]-vars.defense[h]+vars.mu);

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

        for(int i = 1; i < N; i++){
            vars.attack[i] += learning_rate * grad_att[i];
            vars.defense[i] += learning_rate * grad_def[i];
        }
        vars.mu += learning_rate*grad_mu;
        vars.H += learning_rate*grad_H;

        // double avg_att = 0;
        // for(int i = 0; i < N; i++) avg_att += vars.attack[i];
        // avg_att /= N;
        // for(int i = 0; i < N; i++) vars.attack[i] -= avg_att;

        // double avg_def = 0;
        // for(int i = 0; i < N; i++) avg_def += vars.defense[i];
        // avg_def /= N;
        // for(int i = 0; i < N; i++) vars.defense[i] -= avg_def;

    }


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

    return vars;
}

double testVars(Variables vars, vector<Game>& data, int startIdx=0, int endIdx=379,int N=20){
    double log_likelihood=0;
    for(int i=startIdx;i<=endIdx;i++){
        Game game = data[i];
        int h = game.homeTeam;
        int a = game.awayTeam;
        double lambda_home = exp(vars.attack[h]-vars.defense[a]+vars.mu+vars.H);
        double lambda_away = exp(vars.attack[a]-vars.defense[h]+vars.mu);

        log_likelihood += game.goalsHome * log(lambda_home) - lambda_home;
        log_likelihood += game.goalsAway * log(lambda_away) - lambda_away;
    }
    return log_likelihood;
}

void endResults(Variables vars, vector<Game>& remainingGames, vector<int> _table, int N=20, int numSimulations=10000){
    std::mt19937 rng(14);
    vector<int> mcmc(N, 0);
    vector<double> xTable(N, 0);
    for(int xx = 0; xx < numSimulations; xx++){
        vector<int> table = _table;
        for(Game game : remainingGames){
            int h = game.homeTeam;
            int a = game.awayTeam;
            double lambda_home = exp(vars.attack[h]-vars.defense[a]+vars.mu+vars.H);
            double lambda_away = exp(vars.attack[a]-vars.defense[h]+vars.mu);
            poisson_distribution<int> home_dist(lambda_home);
            poisson_distribution<int> away_dist(lambda_away);
            int home_goals = home_dist(rng);
            int away_goals = away_dist(rng);

            if(away_goals==home_goals) {table[a]+=1; table[h]+=1;}
            if(away_goals>home_goals) table[a]+=3;
            if(away_goals<home_goals) table[h]+=3;
        }
        int winner=0;
        for(int i=1;i<table.size();i++) if(table[i]>=table[winner]) winner=i;
        vector<int> winners;
        for(int i=0;i<table.size();i++) if(table[i]==table[winner]) winners.push_back(i);
        for(auto ii : winners) mcmc[ii]++;
        for(int i=0;i<table.size();i++) xTable[i]+=table[i];
    }

    for(int i=0;i<N;i++){
        cout<<teamsIdx[i]<<": "<<mcmc[i]<<" times win the prem, "<<(double)mcmc[i]/numSimulations<<'\n';
    }
    cout<<"-----------------\n";
    vector<pair<double,string>> sortedxTable;
    for(int i=0;i<N;i++) sortedxTable.push_back({xTable[i]/numSimulations, teamsIdx[i]});
    sort(sortedxTable.begin(), sortedxTable.end(), greater<pair<double,string>>());

    cout<<"x table:\n";
    for(int i=0;i<N;i++){
        cout<<sortedxTable[i].second<<": "<<sortedxTable[i].first<<'\n';
    }
}

int main(){

    const int N = 20;
    const double learning_rate = 1e-4;

    vector<Game> data=loadData("25-26EPL.csv");
    vector<int> curTable(20,0);
    for(auto e: teams) teamsIdx[e.second]=e.first;

    vector<vector<bool>> gamesNeeded(N, vector<bool>(N,0));
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            gamesNeeded[i][j]=1;
            gamesNeeded[j][i]=1;
        }
    }

    for(Game game : data){
        if(game.goalsHome==game.goalsAway) { curTable[game.homeTeam]++; curTable[game.awayTeam]++; }
        if(game.goalsHome>game.goalsAway) { curTable[game.homeTeam]+=3; }
        if(game.goalsHome<game.goalsAway) { curTable[game.awayTeam]+=3; }
        gamesNeeded[game.homeTeam][game.awayTeam]=false;
        // printf("Home(%s) %d - %d Away(%s)\n", teamsIdx[game.homeTeam].c_str(), game.goalsHome, game.goalsAway, teamsIdx[game.awayTeam].c_str());
    }

    Variables vars = mle(data, 309);

    cout<<"ATT scores:\n";
    for(int i=0;i<N;i++){
        cout<<vars.attack[i]<<" | "<<teamsIdx[i]<<'\n';
    }
    cout<<"DEF scores:\n";
    for(int i=0;i<N;i++){
        cout<<vars.defense[i]<<" | "<<teamsIdx[i]<<'\n';
    }

    cout<<"Mu: "<<vars.mu<<'\n';
    cout<<"Home Advantage: "<<vars.H<<'\n';

    cout<<"\n\n-------------------\n\n";

    cout<<"train LL/game on first 309 games: "<<(testVars(vars, data, 0, 308))/(309.0)<<'\n';
    // cout<<"train LL/game on last 80 games: "<<(testVars(vars, data, 300))/(80.0)<<'\n';

    vector<Game> remainingGames;
    
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            if(gamesNeeded[i][j]){
                remainingGames.push_back({i,j,-1,-1});
            }
            if(gamesNeeded[j][i]){
                remainingGames.push_back({j,i,-1,-1});
            }
        }
    }
    endResults(vars, remainingGames, curTable, 20);

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