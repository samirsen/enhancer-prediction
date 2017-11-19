#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>

#include <sys/types.h>
#include <dirent.h>

using namespace std;

void read_directory(const string& name, vector<string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

struct Enhancer {
    string track;
    int start;
    int end;
    int label;
    int id;
};

struct BedLine {
    string track;
    int start;
    int end;
    float value;
};

vector<Enhancer> read_csv() {
    ifstream infile("enhancers.csv");
    string line;
    vector<Enhancer> result;
    int count = 0;
    while (getline(infile, line)) {
        // load the data into it's parts
        Enhancer curr;
        stringstream ss(line);
        string item;
        getline(ss, item, ',');
        curr.track = item;
        getline(ss, item, ',');
        curr.start = atoi(item.c_str());
        getline(ss, item, ',');
        curr.end = atoi(item.c_str());
        getline(ss, item, ',');
        curr.label = atoi(item.c_str());
        curr.id = count;
        result.push_back(curr);
        count++;
    }
    infile.close();
    return result;
}

map<string, vector<Enhancer> > splitEnhancers(vector<Enhancer> enhancers) {
    map<string, vector<Enhancer> > enhancer_map;
    for (int i = 0; i < enhancers.size(); i++) {
        Enhancer enhancer = enhancers[i];
        if (!(enhancer_map.count(enhancer.track))) {
            /* We need to add this track to the map */
            enhancer_map[enhancer.track] = vector<Enhancer>();
        }
        enhancer_map[enhancer.track].push_back(enhancer);
    }
    return enhancer_map;
}

void update_enhancers(vector<Enhancer>& enhancers, BedLine bl, vector<float>& result) {
   
    for (int i = 0; i < enhancers.size(); i++) {
        Enhancer enhancer = enhancers[i];
        int idx = enhancer.id;
        if (enhancer.start >= bl.start && enhancer.start < bl.end) {
            int region_len = min(enhancer.end, bl.end) - enhancer.start;
            result[idx] += (bl.value / region_len);
            
        } else if (enhancer.start <= bl.start && enhancer.end >= bl.end) {
            int region_len = bl.end - bl.start;
            result[idx] += (bl.value / region_len);
           
        } else if (enhancer.start <= bl.start && enhancer.end > bl.start && enhancer.end <= bl.end) {
            int region_len = min(bl.end, enhancer.end) - bl.start;
            result[idx] += (bl.value / region_len);
        }
    }
}

vector<float> processFile(string filename, map<string,
        vector<Enhancer> > enhancer_map,
        int num_enhancers) {
    ifstream data_file(filename.c_str());
    string line;
    vector<float> result(num_enhancers);
    string track;
    BedLine bl;
    while (getline(data_file, line)) {
        stringstream ss(line);
        ss >> track;
        if (!(enhancer_map.count(track)))
            break;
        ss >> bl.start;
        ss >> bl.end;
        ss >> bl.value;
        // cout << track << ": " << start << ", " << end << ", " << value << endl;
        update_enhancers(enhancer_map[track], bl, result);
    }
    data_file.close();
    return result;
}

void writeToFile(string filename, vector<float> result) {
    ofstream outfile(filename.c_str());
    outfile << setprecision(6) << result[0];
    for (int i = 1; i < result.size(); i++) {
        outfile << "," << result[i];
    }
    outfile.close();
}

void processAllFiles(string directory) {
    vector<string> files;
    vector<Enhancer> enhancers = read_csv();
    read_directory(directory, files);
    for (int i = 0; i < files.size(); i++) {
        if (files[i][0] == '.')
            continue;
        // process the file
        cout << "processing " << files[i] << "....." << flush;
        size_t last_idx = files[i].find_last_of(".");
        string rawname = files[i].substr(0, last_idx);
        map<string, vector<Enhancer> > enhancer_map = splitEnhancers(enhancers);
        vector<float> result =
                processFile(directory + "/" + files[i],
                enhancer_map, enhancers.size());
        writeToFile("processed_data/" + rawname + ".csv",
                result);
        cout << "Done!" << endl;
    }
}

int main() {
    // vector<Enhancer> enhancers = read_csv();
    // map<string, vector<Enhancer> > enhancer_map = splitEnhancers(enhancers);
    // vector<float> result =
    //         processFile("bed_data/GSM759864_E10_5_forelimb_H3K27ac_ChIP_seq_rep1_index_6_1_all_RPM.bedGraph",
    //         enhancer_map, enhancers.size());
    // writeToFile("processed_data/GSM759864_E10_5_forelimb_H3K27ac_ChIP_seq_rep1_index_6_1_all_RPM.csv",
    //         result);
    processAllFiles("bedGraph_data");
    return 0;
}
