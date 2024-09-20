#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

const int LABEL_INDEX = 3; // only for this exact data

void read_data(string path, vector<vector<int>> &result) {
  ifstream file(path);
  string line;
  vector<int> d1, d2, d3, d4;
  while (getline(file, line)) {
    if (line[0] == 'G') {
      continue;
    }
    vector<int> data;
    stringstream ss(line);
    string cell;
    while (getline(ss, cell, ',')) {
      data.push_back(stoi(cell));
    }
    d1.push_back(data[0]);
    d2.push_back(data[1]);
    d3.push_back(data[2]);
    d4.push_back(data[3]);
  }
  file.close();
  result.push_back(d1);
  result.push_back(d2);
  result.push_back(d3);
  result.push_back(d4);
}

float calculate_entropy(vector<int> feature_data) {
  unordered_map<int, int> cnt;
  for (int i = 0; i < feature_data.size(); i++) {
    cnt[feature_data[i]]++;
  }

  int size = feature_data.size();

  float entropy = 0;
  for (auto it : cnt) {
    int val = it.first;
    float p = (float)cnt[val] / size;
    entropy += p * log2(p + 1e-6);
  }

  return -entropy;
}

class Node {
public:
  vector<vector<int>> part_data;
  bool is_leaf;
  int pred_class;
  int split_index;
  Node(){};
  Node(vector<vector<int>> part_data) {
    this->part_data = part_data;
    this->is_leaf = false;
    this->pred_class = -1;
  };
  unordered_map<int, Node *> child_nodes;
  ~Node() {
    for (auto it : child_nodes) {
      delete it.second;
    }
  }
};

vector<vector<int>> get_data_by_value(vector<vector<int>> input_data,
                                      int target_value, int index) {
  vector<vector<int>> result;
  int n_cols = input_data.size();
  int n_rows = input_data[0].size();
  for (int i = 0; i < n_cols; i++) {
    result.push_back(vector<int>());
  }

  for (int i = 0; i < n_rows; i++) {
    if (input_data[index][i] == target_value) {
      for (int j = 0; j < n_cols; j++) {
        result[j].push_back(input_data[j][i]);
      }
    }
  }
  return result;
}

pair<unordered_map<int, Node *>, float> split(vector<vector<int>> data,
                                              int feat_index, int label_index) {
  unordered_set<int> unique_values;
  for (int i : data[feat_index]) {
    unique_values.insert(i);
  }

  float weighted_entropy = 0;
  int n = data[0].size();
  unordered_map<int, Node *> split_nodes;

  for (int unique_value : unique_values) {
    vector<vector<int>> partition =
        get_data_by_value(data, unique_value, feat_index);

    int n_cols = partition.size();
    int n_rows = partition[0].size();

    vector<int> partition_label = partition[label_index];
    float node_entropy = calculate_entropy(partition_label);
    weighted_entropy += ((float)n_rows / n) * node_entropy;

    Node *node = new Node(partition);
    split_nodes[unique_value] = node;
  }
  return make_pair(split_nodes, weighted_entropy);
}

bool meet_criteria(Node *node, int label_index) {
  float entropy = calculate_entropy(node->part_data[label_index]);
  bool ans = abs(entropy) < 1e-5;
  // cout << "meet criteria entropy " << entropy << " ans " << ans << endl;
  return ans;
}

unordered_map<int, int> count_values(vector<int> input) {
  unordered_map<int, int> cnt;
  for (int i = 0; i < input.size(); i++) {
    int value = input[i];
    cnt[value]++;
  }
  return cnt;
}

void get_pred_class(Node *node, int label_index) {
  int n = node->part_data[label_index].size();
  unordered_map<int, int> cnt = count_values(node->part_data[label_index]);
  int result = -1;
  int max_cnt = -1;
  for (auto it : cnt) {
    if (it.second > max_cnt) {
      max_cnt = it.second;
      result = it.first;
    }
  }
  // cout << "get pred class " << result << endl;
  node->pred_class = result;
}

void best_split(Node *node, int depth, const int max_depth) {
  if (depth >= max_depth || meet_criteria(node, LABEL_INDEX)) {
    node->is_leaf = true;
    get_pred_class(node, LABEL_INDEX);
    return;
  }

  int index_feature_split = -1;
  float min_entropy = 1e6;

  int n_cols = node->part_data.size();
  int n_rows = node->part_data[0].size();

  unordered_map<int, Node *> child_nodes;

  for (int i = 0; i < n_cols; i++) {
    if (i == LABEL_INDEX) {
      continue;
    }
    pair<unordered_map<int, Node *>, float> split_nodes =
        split(node->part_data, i, LABEL_INDEX);
    if (split_nodes.second < min_entropy) {
      min_entropy = split_nodes.second;
      index_feature_split = i;
      child_nodes = split_nodes.first;
    }
  }
  node->child_nodes = child_nodes;
  node->split_index = index_feature_split;

  for (auto it : node->child_nodes) {
    best_split(it.second, depth + 1, max_depth);
  }
}

int predict(Node *node, vector<int> data) {
  //   if (node == nullptr) {
  //     return 0;
  //   }
  if (node->is_leaf) {
    return node->pred_class;
  }

  int feat_val = data[node->split_index];

  int best_node_val = 100000;
  Node *best_node;
  for (pair<int, Node *> it : node->child_nodes) {
    if (abs(it.first - feat_val) < best_node_val) {
      best_node_val = abs(it.first - feat_val);
      best_node = it.second;
    }
  }

  //   return predict(node->child_nodes[feat_val], data);
  return predict(best_node, data);
}

void print_tree(Node *node, int split_value) {
  if (node->is_leaf) {
    cout << "leaf with value " << node->pred_class << endl;
  } else {
    cout << "split on feature " << node->split_index << " split value "
         << split_value << endl;
    for (auto it : node->child_nodes) {
      print_tree(it.second, it.first);
    }
  }
}

class Boosting {
public:
  int max_depth;
  float learning_rate;
  int n_estimators;
  int random_state;

  Boosting(int max_depth = 3, float learning_rate = 0.1, int n_estimators = 100,
           int random_state = 42) {
    this->max_depth = max_depth;
    this->learning_rate = learning_rate;
    this->n_estimators = n_estimators;
    this->max_depth = max_depth;
    this->random_state = random_state;
  }
};

pair<vector<vector<int>>, vector<vector<int>>> read_train_test() {
  vector<vector<int>> data;
  read_data("input.csv", data);
  cout << data.size() << " " << data[0].size() << endl;

  vector<vector<int>> train;
  vector<vector<int>> test;

  size_t n_data = data[0].size();
  size_t n_dim = data.size();

  for (int i = 0; i < n_dim; i++) {
    train.push_back({});
    test.push_back({});
  }

  int n_test = n_data / 10;
  int n_train = n_data - n_test;

  for (int i = 0; i < n_data; i++) {
    for (int j = 0; j < n_dim; j++) {
      if (i < n_train) {
        train[j].push_back(data[j][i]);
      } else {
        test[j].push_back(data[j][i]);
      }
    }
  }

  return make_pair(train, test);
}

int test_tree() {
  auto d = read_train_test();
  vector<vector<int>> train = d.first;
  vector<vector<int>> test = d.second;
  int n_test = test[0].size();
  int n_train = train[0].size();
  int n_dim = train.size();

  cout << "split done, test size " << n_test << " train size " << n_train
       << endl;
  cout << "train dim " << train.size() << " " << train[0].size() << endl;
  cout << "test dim " << test.size() << " " << test[0].size() << endl;

  Node *root = new Node(train);
  best_split(root, 0, 4);
  //   print_tree(root, -1);

  cout << "split ok " << endl;

  float acc = 0;
  for (int i = 0; i < n_test; i++) {
    vector<int> row;
    for (int j = 0; j < n_dim; j++) {
      row.push_back(test[j][i]);
    }
    int cur_pred = predict(root, row);
    acc += (cur_pred == test[LABEL_INDEX][i]);
  }

  cout << "test accuracy " << acc / n_test << endl;

  return 0;
}

vector<vector<int>> make_one_hot(vector<int> y) {
  size_t n_dim = y.size();

  int n_class = *max_element(y.begin(), y.end()) + 1;

  vector<vector<int>> one_hot(n_class, vector<int>(n_dim, 0));
  for (int i = 0; i < n_dim; i++) {
    one_hot[y[i]][i] = 1;
  }
  return one_hot;
}

vector<float> softmax(vector<float> logits) {
  vector<float> exp_logits;
  for (float logit : logits) {
    exp_logits.push_back(expf(logit));
  }
  float s = accumulate(exp_logits.begin(), exp_logits.end(), 0);
  vector<float> result;
  for (float exp_logit : exp_logits) {
    result.push_back(exp_logit / s);
  }
  return result;
}

void test_boosting() {
  auto d = read_train_test();
  vector<vector<int>> train = d.first;
  vector<vector<int>> test = d.second;
  int n_test = test[0].size();
  int n_train = train[0].size();
  int n_dim = train.size();

  vector<vector<int>> one_hot_train = make_one_hot(train[LABEL_INDEX]);
  vector<vector<int>> one_hot_test = make_one_hot(test[LABEL_INDEX]);

  
}

int main() { test_boosting(); }