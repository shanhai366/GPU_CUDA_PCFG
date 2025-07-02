#include "CUDA_3_PCFG.h"
using namespace std;
#include <cuda_runtime.h>
#include <vector>
#include <string>


void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>

// CUDA核函数：生成猜测口令
__global__ void generateGuessesKernel(char* d_output, const int* d_output_offsets, 
                                      const char* d_prefix, int prefix_len, 
                                      const char* d_values, const int* d_value_offsets,
                                      const int* d_value_lengths, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 获取当前value在d_values中的位置
    int value_offset = d_value_offsets[idx];
    int value_len = d_value_lengths[idx];
    
    // 获取当前输出位置
    int output_offset = d_output_offsets[idx];
    char* output = d_output + output_offset;
    
    // 复制前缀
    for (int i = 0; i < prefix_len; i++) {
        output[i] = d_prefix[i];
    }
    
    // 复制当前value
    const char* value_ptr = d_values + value_offset;
    for (int i = 0; i < value_len; i++) {
        output[prefix_len + i] = value_ptr[i];
    }
    
    // 添加字符串结束符
    output[prefix_len + value_len] = '\0';
}

void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    
    segment* a = nullptr;
    int n = 0;
    std::string prefix;
    bool is_single_segment = (pt.content.size() == 1);

    if (is_single_segment) {
        // 单segment处理
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        n = pt.max_indices[0];
    } else {
        // 多segment处理：构建前缀
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx >= pt.content.size() - 1) break;
        }
        
        // 获取最后一个segment
        segment& last_seg = pt.content.back();
        if (last_seg.type == 1) {
            a = &m.letters[m.FindLetter(last_seg)];
        } else if (last_seg.type == 2) {
            a = &m.digits[m.FindDigit(last_seg)];
        } else if (last_seg.type == 3) {
            a = &m.symbols[m.FindSymbol(last_seg)];
        }
        n = pt.max_indices.back();
    }

    // 准备GPU数据
    std::vector<std::string> values;
    std::vector<int> value_lengths;
    std::vector<int> value_offsets;
    
    // 填充value数据
    int total_value_size = 0;
    for (int i = 0; i < n; i++) {
        values.push_back(a->ordered_values[i]);
        value_lengths.push_back(a->ordered_values[i].size());
        value_offsets.push_back(total_value_size);
        total_value_size += value_lengths.back();
    }
    
    // 计算输出偏移和总大小
    std::vector<int> output_offsets;
    int total_output_size = 0;
    for (int i = 0; i < n; i++) {
        output_offsets.push_back(total_output_size);
        total_output_size += prefix.size() + value_lengths[i] + 1; // +1 for null terminator
    }
    
    // 扁平化values数据
    std::string flat_values;
    for (const auto& val : values) {
        flat_values += val;
    }
    
    // 分配主机输出缓冲区
    std::vector<char> h_output(total_output_size);
    
    // GPU内存分配
    char *d_output = nullptr;
    char *d_prefix = nullptr;
    char *d_values = nullptr;
    int *d_value_offsets = nullptr;
    int *d_value_lengths = nullptr;
    int *d_output_offsets = nullptr;
    
    // 分配设备内存
    cudaMalloc(&d_output, total_output_size);
    cudaMalloc(&d_prefix, prefix.size() + 1);
    cudaMalloc(&d_values, flat_values.size());
    cudaMalloc(&d_value_offsets, n * sizeof(int));
    cudaMalloc(&d_value_lengths, n * sizeof(int));
    cudaMalloc(&d_output_offsets, n * sizeof(int));
    
    // 复制数据到设备
    cudaMemcpy(d_prefix, prefix.c_str(), prefix.size() + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, flat_values.c_str(), flat_values.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_offsets, value_offsets.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lengths, value_lengths.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_offsets, output_offsets.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动核函数
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    generateGuessesKernel<<<gridSize, blockSize>>>(d_output, d_output_offsets, 
                                                  d_prefix, prefix.size(),
                                                  d_values, d_value_offsets,
                                                  d_value_lengths, n);
    
    // 等待核函数完成
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_output.data(), d_output, total_output_size, cudaMemcpyDeviceToHost);
    
    // 添加口令到队列
    for (int i = 0; i < n; i++) {
        guesses.push_back(std::string(h_output.data() + output_offsets[i]));
        total_guesses++;
    }
    
    // 释放设备内存
    cudaFree(d_output);
    cudaFree(d_prefix);
    cudaFree(d_values);
    cudaFree(d_value_offsets);
    cudaFree(d_value_lengths);
    cudaFree(d_output_offsets);
}