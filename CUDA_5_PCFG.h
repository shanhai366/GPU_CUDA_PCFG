#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h> 
#include <omp.h>
// #include <chrono>   
// using namespace chrono;
// 添加以下常量定义
#define MAX_BATCH_SIZE 8      // 最大批量大小
#define MAX_SEGMENTS 10            // 每个PT的最大segment数量
#define MAX_GUESS_LENGTH 64        // 猜测口令的最大长度
#define MAX_VALUES_PER_SEG 1000    // 每个segment的最大值数量
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:




 struct GPUSegmentInfo {
        int type;
        int length;
        int value_index;
    };
    
    // 获取GPU所需的信息
    void getGPUData(GPUSegmentInfo* d_segments, char** d_values, int* d_max_indices, int seg_idx, cudaStream_t stream);
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
 
 // 添加GPU设备端数据结构
    struct GPUModelData {
        char** letter_values;
        int* letter_value_lengths;
        int* letter_value_counts;
        int letter_seg_count;
        
        // 类似添加digits和symbols
    };
    
    void copyToGPU(GPUModelData* d_model);
    void freeGPUData(GPUModelData* d_model);




// 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue {
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;

    // GPU相关功能
    void initGPU();
    model& getModel() { return m; }

    // 批量处理函数
    void PopNextBatch(int batch_size);
    
    // 批量生成函数
    void GenerateBatch(std::vector<PT>& pts);
    
    // GPU生成函数
    void GenerateGPU(std::vector<PT>& pts);
    
//进阶问问题二

 std::vector<PT> currentBatch;  // 当前处理批次
    std::vector<PT> nextBatch;     // 下一批次准备区
    
    cudaStream_t computeStream;   // 计算流
    cudaStream_t dataStream;       // 数据传输流
    
    // 设备端缓冲区
    PT::GPUSegmentInfo* d_pt_segments[2];
    int* d_pt_sizes[2];
    char* d_output[2];
    
    // 主机端结果缓冲区
    std::vector<char> h_output[2];
    int activeBuffer = 0;  // 当前活动缓冲区索引 (0或1)

  void InitPipeline();

   void PrepareNextBatch(int bufferIndex, int batch_size) ;
void LaunchGPUComputation(int bufferIndex);
void ProcessResults(int bufferIndex);
private:
    model::GPUModelData d_model; // 使用model类中定义的GPUModelData
    bool gpu_initialized = false;
};