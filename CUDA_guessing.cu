#include "CUDA_PCFG.h"
using namespace std;
// 在CUDA_guessing.cpp和CUDA_main.cu文件开头添加
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// __device__ void my_strcpy(char* dest, const char* src) {
//     int i = 0;
//     while (src[i]) {
//         dest[i] = src[i];
//         i++;
//     }
//     dest[i] = '\0';
// }

// __device__ void my_strcat(char* dest, const char* src) {
//     int i = 0;
//     while (dest[i]) i++;
//     int j = 0;
//     while (src[j]) {
//         dest[i + j] = src[j];
//         j++;
//     }
//     dest[i + j] = '\0';
// }

// __global__
// void generateSingleSegmentKernel(const char** values, int count, char** results) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
//         my_strcpy(results[idx], values[idx]);
//     }
// }

// __global__
// void generateMultiSegmentKernel(const char* prefix, const char** suffixes, int count, char** results) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
//         my_strcpy(results[idx], prefix);
//         my_strcat(results[idx], suffixes[idx]);
//     }
// }


__global__
void generateSingleSegmentKernel(const char** values, int count, char** results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 内联 my_strcpy
        const char* src = values[idx];
        char* dest = results[idx];
        int i = 0;
        while (src[i]) {
            dest[i] = src[i];
            i++;
        }
        dest[i] = '\0';
    }
}

__global__
void generateMultiSegmentKernel(const char* prefix, const char** suffixes, int count, char** results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 内联 my_strcpy
        char* dest = results[idx];
        int i = 0;
        while (prefix[i]) {
            dest[i] = prefix[i];
            i++;
        }
        dest[i] = '\0';

        // 内联 my_strcat
        int j = 0;
        while (dest[j]) j++;
        const char* src = suffixes[idx];
        int k = 0;
        while (src[k]) {
            dest[j + k] = src[k];
            k++;
        }
        dest[j + k] = '\0';
    }
}

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
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment* a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        else
            return;

        int count = pt.max_indices[0];
        const int MAX_LEN = 128;

        AllocDeviceMemSingle(count);

        // Host准备数据
        char* h_pool = new char[count * MAX_LEN];
        char** h_ptrs = new char*[count];
        for (int i = 0; i < count; ++i) {
            strncpy(h_pool + i * MAX_LEN, a->ordered_values[i].c_str(), MAX_LEN - 1);
            h_pool[i * MAX_LEN + MAX_LEN - 1] = '\0';
            h_ptrs[i] = h_pool + i * MAX_LEN;
        }

        cudaMemcpy(d_pool, h_pool, count * MAX_LEN, cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_ptrs, count * sizeof(char*), cudaMemcpyHostToDevice);

        // 重新构造设备结果指针数组
        std::vector<char*> d_res_ptrs(count);
        for (int i = 0; i < count; ++i)
            d_res_ptrs[i] = d_result_pool + i * MAX_LEN;
        cudaMemcpy(d_results, d_res_ptrs.data(), count * sizeof(char*), cudaMemcpyHostToDevice);

        int bs = 128, gs = (count + bs - 1) / bs;
        generateSingleSegmentKernel<<<gs, bs>>>((const char**)d_values, count, d_results);
        cudaDeviceSynchronize();

        char* h_res = new char[count * MAX_LEN];
        cudaMemcpy(h_res, d_result_pool, count * MAX_LEN, cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; ++i)
            guesses.emplace_back(h_res + i * MAX_LEN);
        total_guesses += count;

        delete[] h_pool;
        delete[] h_ptrs;
        delete[] h_res;
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }

        segment* a = nullptr;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else if (pt.content.back().type == 3)
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        else
            return;

        int count = pt.max_indices.back();
        const int MAX_LEN = 256;

        AllocDeviceMemMulti(count, guess.size() + 1);

        char* h_suffix = new char[count * MAX_LEN];
        char** h_suf_ptrs = new char*[count];
        for (int i = 0; i < count; ++i) {
            strncpy(h_suffix + i * MAX_LEN, a->ordered_values[i].c_str(), MAX_LEN - 1);
            h_suffix[i * MAX_LEN + MAX_LEN - 1] = '\0';
            h_suf_ptrs[i] = h_suffix + i * MAX_LEN;
        }

        cudaMemcpy(d_prefix, guess.c_str(), guess.size() + 1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_suffix_pool, h_suffix, count * MAX_LEN, cudaMemcpyHostToDevice);

        for (int i = 0; i < count; ++i)
            h_suf_ptrs[i] = d_suffix_pool + i * MAX_LEN;
        cudaMemcpy(d_suf_ptrs, h_suf_ptrs, count * sizeof(char*), cudaMemcpyHostToDevice);

        for (int i = 0; i < count; ++i)
            h_suf_ptrs[i] = d_result_pool_multi + i * MAX_LEN;
        cudaMemcpy(d_result_ptrs, h_suf_ptrs, count * sizeof(char*), cudaMemcpyHostToDevice);

        int bs = 128, gs = (count + bs - 1) / bs;
        generateMultiSegmentKernel<<<gs, bs>>>(d_prefix, (const char**)d_suf_ptrs, count, d_result_ptrs);
        cudaDeviceSynchronize();

        char* h_res = new char[count * MAX_LEN];
        cudaMemcpy(h_res, d_result_pool_multi, count * MAX_LEN, cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; ++i)
            guesses.emplace_back(h_res + i * MAX_LEN);
        total_guesses += count;

        delete[] h_suffix;
        delete[] h_suf_ptrs;
        delete[] h_res;
    }
}
// 在PriorityQueue类里增加成员变量，记录预分配的设备内存及大小

char* d_pool = nullptr;
char* d_result_pool = nullptr;
char** d_values = nullptr;
char** d_results = nullptr;
int d_count_allocated = 0;

char* d_prefix = nullptr;
char* d_suffix_pool = nullptr;
char* d_result_pool_multi = nullptr;
char** d_suf_ptrs = nullptr;
char** d_result_ptrs = nullptr;
int d_count_allocated_multi = 0;
int d_prefix_size = 0;

// 预分配函数（单段）
void PriorityQueue::AllocDeviceMemSingle(int count) {
    if (count <= d_count_allocated) return;
    if (d_pool) cudaFree(d_pool);
    if (d_result_pool) cudaFree(d_result_pool);
    if (d_values) cudaFree(d_values);
    if (d_results) cudaFree(d_results);

    d_count_allocated = count;
    cudaMalloc(&d_pool, count * 128);
    cudaMalloc(&d_result_pool, count * 128);
    cudaMalloc(&d_values, count * sizeof(char*));
    cudaMalloc(&d_results, count * sizeof(char*));
}

// 预分配函数（多段）
void PriorityQueue::AllocDeviceMemMulti(int count, int prefix_size) {
    if (count <= d_count_allocated_multi && prefix_size <= d_prefix_size) return;
    if (d_prefix) cudaFree(d_prefix);
    if (d_suffix_pool) cudaFree(d_suffix_pool);
    if (d_result_pool_multi) cudaFree(d_result_pool_multi);
    if (d_suf_ptrs) cudaFree(d_suf_ptrs);
    if (d_result_ptrs) cudaFree(d_result_ptrs);

    d_count_allocated_multi = count;
    d_prefix_size = prefix_size;
    cudaMalloc(&d_prefix, prefix_size);
    cudaMalloc(&d_suffix_pool, count * 256);
    cudaMalloc(&d_result_pool_multi, count * 256);
    cudaMalloc(&d_suf_ptrs, count * sizeof(char*));
    cudaMalloc(&d_result_ptrs, count * sizeof(char*));
}

void PriorityQueue::FreeDeviceMem() {
    if (d_pool) cudaFree(d_pool);
    if (d_result_pool) cudaFree(d_result_pool);
    if (d_values) cudaFree(d_values);
    if (d_results) cudaFree(d_results);

    if (d_prefix) cudaFree(d_prefix);
    if (d_suffix_pool) cudaFree(d_suffix_pool);
    if (d_result_pool_multi) cudaFree(d_result_pool_multi);
    if (d_suf_ptrs) cudaFree(d_suf_ptrs);
    if (d_result_ptrs) cudaFree(d_result_ptrs);

    d_pool = nullptr; d_result_pool = nullptr; d_values = nullptr; d_results = nullptr;
    d_prefix = nullptr; d_suffix_pool = nullptr; d_result_pool_multi = nullptr; d_suf_ptrs = nullptr; d_result_ptrs = nullptr;

    d_count_allocated = 0; d_count_allocated_multi = 0; d_prefix_size = 0;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// void PriorityQueue::Generate(PT pt)
// {
//     // 计算PT的概率，这里主要是给PT的概率进行初始化
//     CalProb(pt);

//     // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
//    if (pt.content.size() == 1)
// {
//     segment* a;
//     if (pt.content[0].type == 1)
//         a = &m.letters[m.FindLetter(pt.content[0])];
//     if (pt.content[0].type == 2)
//         a = &m.digits[m.FindDigit(pt.content[0])];
//     if (pt.content[0].type == 3)
//         a = &m.symbols[m.FindSymbol(pt.content[0])];

//  int count = pt.max_indices[0];
//     const int MAX_LEN = 128;

//     // Host 泳池 & 指针
//     char* h_pool = new char[count * MAX_LEN];
//     char** h_ptrs = new char*[count];
//     for (int i = 0; i < count; ++i) {
//         strncpy(h_pool + i*MAX_LEN, a->ordered_values[i].c_str(), MAX_LEN-1);
//         h_pool[i*MAX_LEN + MAX_LEN -1] = '\0';
//         h_ptrs[i] = h_pool + i*MAX_LEN;
//     }

//     // Device 分配
//     char *d_pool, *d_result_pool;
//     char **d_values, **d_results;
//     cudaMalloc(&d_pool, count * MAX_LEN);
//     cudaMalloc(&d_result_pool, count * MAX_LEN);
//     cudaMalloc(&d_values, count * sizeof(char*));
//     cudaMalloc(&d_results, count * sizeof(char*));

//     // 拷贝数据
//     cudaMemcpy(d_pool, h_pool, count*MAX_LEN, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values, h_ptrs, count*sizeof(char*), cudaMemcpyHostToDevice);

//     // 构造 results 指针
//     for (int i = 0; i < count; i++) h_ptrs[i] = d_pool + i*MAX_LEN;
//     cudaMemcpy(d_results, h_ptrs, count*sizeof(char*), cudaMemcpyHostToDevice);

//     // Launch kernel
//     int bs = 128, gs = (count + bs - 1)/bs;
//     generateSingleSegmentKernel<<<gs, bs>>>((const char**)d_values, count, d_results);
//     cudaDeviceSynchronize();

//     // 获取结果
//     char* h_res = new char[count * MAX_LEN];
//     cudaMemcpy(h_res, d_pool, count*MAX_LEN, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < count; ++i) guesses.emplace_back(h_res + i*MAX_LEN);
//     total_guesses += count;

//     // Cleanup
//     delete[] h_pool; delete[] h_ptrs; delete[] h_res;
//     cudaFree(d_pool); cudaFree(d_result_pool);
//     cudaFree(d_values); cudaFree(d_results);
// } 
//   else
// {
//     string guess;
//     int seg_idx = 0;
//     for (int idx : pt.curr_indices)
//     {
//         if (pt.content[seg_idx].type == 1)
//             guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//         if (pt.content[seg_idx].type == 2)
//             guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//         if (pt.content[seg_idx].type == 3)
//             guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//         seg_idx += 1;
//         if (seg_idx == pt.content.size() - 1)
//             break;
//     }

//     segment *a;
//     if (pt.content.back().type == 1)
//         a = &m.letters[m.FindLetter(pt.content.back())];
//     if (pt.content.back().type == 2)
//         a = &m.digits[m.FindDigit(pt.content.back())];
//     if (pt.content.back().type == 3)
//         a = &m.symbols[m.FindSymbol(pt.content.back())];

//     int count = pt.max_indices.back();
//     const int MAX_LEN = 256; // prefix + suffix
//     char* h_suffix = new char[count * MAX_LEN];
//     char** h_suf_ptrs = new char*[count];
//     for (int i = 0; i < count; ++i) {
//         strncpy(h_suffix + i*MAX_LEN, a->ordered_values[i].c_str(), MAX_LEN-1);
//         h_suffix[i*MAX_LEN + MAX_LEN -1] = '\0';
//         h_suf_ptrs[i] = h_suffix + i*MAX_LEN;
//     }

//     // Device alloc
//     char *d_prefix, *d_suffix_pool, *d_result_pool;
//     char **d_suf_ptrs, **d_result_ptrs;
//     cudaMalloc(&d_prefix, guess.size()+1);
//     cudaMalloc(&d_suffix_pool, count * MAX_LEN);
//     cudaMalloc(&d_result_pool, count * MAX_LEN);
//     cudaMalloc(&d_suf_ptrs, count * sizeof(char*));
//     cudaMalloc(&d_result_ptrs, count * sizeof(char*));

//     // Copy host → device
//     cudaMemcpy((void*)d_prefix, (const void*)guess.c_str(), guess.size()+1, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_suffix_pool, h_suffix, count*MAX_LEN, cudaMemcpyHostToDevice);
//     for (int i = 0; i < count; ++i) h_suf_ptrs[i] = d_suffix_pool + i*MAX_LEN;
//     cudaMemcpy(d_suf_ptrs, h_suf_ptrs, count*sizeof(char*), cudaMemcpyHostToDevice);
//     for (int i = 0; i < count; ++i) h_suf_ptrs[i] = d_result_pool + i*MAX_LEN;
//     cudaMemcpy(d_result_ptrs, h_suf_ptrs, count*sizeof(char*), cudaMemcpyHostToDevice);

//     // Launch kernel
//     int bs = 128, gs = (count + bs - 1)/bs;
//     generateMultiSegmentKernel<<<gs, bs>>>(d_prefix, (const char**)d_suf_ptrs, count, d_result_ptrs);
//     cudaDeviceSynchronize();

//     // Copy back
//     char* h_res = new char[count * MAX_LEN];
//     cudaMemcpy(h_res, d_result_pool, count*MAX_LEN, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < count; ++i) guesses.emplace_back(h_res + i*MAX_LEN);
//     total_guesses += count;

//     // Cleanup
//     delete[] h_suffix; delete[] h_suf_ptrs; delete[] h_res;
//     cudaFree(d_prefix); cudaFree(d_suffix_pool); cudaFree(d_result_pool);
//     cudaFree(d_suf_ptrs); cudaFree(d_result_ptrs);
// }
//     }
