#include "CUDA_6_PCFG.h"
#include <algorithm>
using namespace std;
// CUDA 内核函数
__global__ void generateGuessesKernel(/* 参数 */) {
    // 实际的CUDA内核实现
}

void PriorityQueue::GenerateWithCUDA(const PT& pt) {
    // 1. 准备设备内存
    segment* d_segment;
    cudaMalloc(&d_segment, sizeof(segment));
    
    // 2. 拷贝数据到设备
    cudaMemcpy(d_segment, &pt.content.back(), 
              sizeof(segment), cudaMemcpyHostToDevice);

    // 3. 启动内核
    int blockSize = 256;
    int gridSize = (pt.max_indices.back() + blockSize - 1) / blockSize;
    generateGuessesKernel<<<gridSize, blockSize>>>(/* 参数 */);
char* d_results = nullptr; 
    // 4. 拷贝结果回主机
    std::vector<std::string> hostResults(pt.max_indices.back());
    cudaMemcpy(hostResults.data(), d_results,
              pt.max_indices.back() * sizeof(std::string),
              cudaMemcpyDeviceToHost);

    // 5. 添加到猜测列表
    guesses.insert(guesses.end(), hostResults.begin(), hostResults.end());
    total_guesses += hostResults.size();

    // 6. 清理设备内存
    cudaFree(d_segment);
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

// void PriorityQueue::PopNext()
// {

//     // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
//     Generate(priority.front());

//     // 然后需要根据即将出队的PT，生成一系列新的PT
//     vector<PT> new_pts = priority.front().NewPTs();
//     for (PT pt : new_pts)
//     {
//         // 计算概率
//         CalProb(pt);
//         // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
//         for (auto iter = priority.begin(); iter != priority.end(); iter++)
//         {
//             // 对于非队首和队尾的特殊情况
//             if (iter != priority.end() - 1 && iter != priority.begin())
//             {
//                 // 判定概率
//                 if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
//                 {
//                     priority.emplace(iter + 1, pt);
//                     break;
//                 }
//             }
//             if (iter == priority.end() - 1)
//             {
//                 priority.emplace_back(pt);
//                 break;
//             }
//             if (iter == priority.begin() && iter->prob < pt.prob)
//             {
//                 priority.emplace(iter, pt);
//                 break;
//             }
//         }
//     }

//     // 现在队首的PT善后工作已经结束，将其出队（删除）
//     priority.erase(priority.begin());
// }

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
void PriorityQueue::GenerateOnGPU(const PT& pt) {
    // 这里应该是实际的GPU内核调用
    // 以下是模拟实现，实际应替换为CUDA/OpenCL代码
    
    if (pt.content.size() == 1) {
        segment* a = nullptr;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        #pragma omp parallel for
        for (int i = 0; i < pt.max_indices[0]; ++i) {
            #pragma omp critical
            {
                guesses.emplace_back(a->ordered_values[i]);
                ++total_guesses;
            }
        }
    } else {
        std::string prefix;
        for (size_t i = 0; i < pt.curr_indices.size(); ++i) {
            const segment& seg = pt.content[i];
            if (seg.type == 1) {
                prefix += m.letters[m.FindLetter(seg)].ordered_values[pt.curr_indices[i]];
            } else if (seg.type == 2) {
                prefix += m.digits[m.FindDigit(seg)].ordered_values[pt.curr_indices[i]];
            } else if (seg.type == 3) {
                prefix += m.symbols[m.FindSymbol(seg)].ordered_values[pt.curr_indices[i]];
            }
        }

        segment* last_seg = nullptr;
        const segment& seg = pt.content.back();
        if (seg.type == 1) {
            last_seg = &m.letters[m.FindLetter(seg)];
        } else if (seg.type == 2) {
            last_seg = &m.digits[m.FindDigit(seg)];
        } else if (seg.type == 3) {
            last_seg = &m.symbols[m.FindSymbol(seg)];
        }

        #pragma omp parallel for
        for (int i = 0; i < pt.max_indices.back(); ++i) {
            #pragma omp critical
            {
                guesses.emplace_back(prefix + last_seg->ordered_values[i]);
                ++total_guesses;
            }
        }
    }
}

void PriorityQueue::ProcessBatch(const std::vector<PT>& pts, size_t batch_size) {
    std::vector<std::pair<PT, bool>> workloads; // PT + use_gpu
    
    // 评估工作负载并分类
    for (const auto& pt : pts) {
        size_t wl = pt.EstimateWorkload();
        workloads.emplace_back(pt, wl > 150000); // 使用GPU的阈值
    }

    // 分批处理
    for (size_t i = 0; i < workloads.size(); i += batch_size) {
        auto start = workloads.begin() + i;
        auto end = (i + batch_size) < workloads.size() ? start + batch_size : workloads.end();
        
        #pragma omp parallel for
        for (auto it = start; it != end; ++it) {
            if (it->second) {
                GenerateOnGPU(it->first);
            } else {
                Generate(it->first);
            }
        }
    }
}

void PriorityQueue::PopNext() {
    if (priority.empty()) return;

    // 生成当前PT的猜测
    Generate(priority.front());

    // 生成新PT
    std::vector<PT> new_pts = priority.front().NewPTs();
    
    // 使用批量处理接口
    ProcessBatch(new_pts);

    // 从队列中移除
    priority.erase(priority.begin());
}
#include <algorithm>  // 用于std::find_if等

void PriorityQueue::Generate(const PT& pt) {
    // 1. 计算概率
    CalProb(const_cast<PT&>(pt));  // 需要去除const限定
    
    // 2. 根据工作量决定使用CPU还是GPU
    const size_t workload = pt.EstimateWorkload();
    const size_t GPU_THRESHOLD = 150000;
    
    if (workload > GPU_THRESHOLD) {
        GenerateWithCUDA(pt);
        return;
    }

    // 3. CPU实现
    if (pt.content.size() == 1) {
        segment* seg = nullptr;
        if (pt.content[0].type == 1) {
            seg = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            seg = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            seg = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        if (seg) {
            for (int i = 0; i < pt.max_indices[0]; ++i) {
                guesses.emplace_back(seg->ordered_values[i]);
                ++total_guesses;
            }
        }
    } else {
        std::string prefix;
        for (size_t i = 0; i < pt.curr_indices.size(); ++i) {
            const segment& seg = pt.content[i];
            if (seg.type == 1) {
                prefix += m.letters[m.FindLetter(seg)].ordered_values[pt.curr_indices[i]];
            } else if (seg.type == 2) {
                prefix += m.digits[m.FindDigit(seg)].ordered_values[pt.curr_indices[i]];
            } else if (seg.type == 3) {
                prefix += m.symbols[m.FindSymbol(seg)].ordered_values[pt.curr_indices[i]];
            }
        }

        const segment& last_seg = pt.content.back();
        segment* seg_ptr = nullptr;
        if (last_seg.type == 1) {
            seg_ptr = &m.letters[m.FindLetter(last_seg)];
        } else if (last_seg.type == 2) {
            seg_ptr = &m.digits[m.FindDigit(last_seg)];
        } else if (last_seg.type == 3) {
            seg_ptr = &m.symbols[m.FindSymbol(last_seg)];
        }

        if (seg_ptr) {
            for (int i = 0; i < pt.max_indices.back(); ++i) {
                guesses.emplace_back(prefix + seg_ptr->ordered_values[i]);
                ++total_guesses;
            }
        }
    }
}