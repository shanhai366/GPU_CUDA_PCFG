#include "CUDA_5_PCFG.h"
#include <omp.h>
#include <cuda_runtime.h>
using namespace std;
// GPU核函数声明
__global__ void generateKernel(PT::GPUSegmentInfo* d_pt_segments, int* d_pt_sizes, 
                               char** d_letter_values, int* d_letter_counts,
                               char* d_output, int max_guess_length, int batch_size);
void PriorityQueue::InitPipeline() {
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&dataStream);
    
    // 初始化双缓冲
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_pt_segments[i], MAX_BATCH_SIZE * MAX_SEGMENTS * sizeof(PT::GPUSegmentInfo));
        cudaMalloc(&d_pt_sizes[i], MAX_BATCH_SIZE * sizeof(int));
        cudaMalloc(&d_output[i], MAX_BATCH_SIZE * MAX_GUESS_LENGTH);
        h_output[i].resize(MAX_BATCH_SIZE * MAX_GUESS_LENGTH);
    }
}

void PriorityQueue::PrepareNextBatch(int bufferIndex, int batch_size) {
    nextBatch.clear();
    
    // 从优先队列获取下一批PT
    int actual_size = std::min(batch_size, static_cast<int>(priority.size()));
    for (int i = 0; i < actual_size; i++) {
        nextBatch.push_back(priority[i]);
    }
    
    // 准备主机端数据结构
    std::vector<PT::GPUSegmentInfo> h_pt_segments;
    std::vector<int> h_pt_sizes;
    
    for (PT& pt : nextBatch) {
        h_pt_sizes.push_back(pt.content.size());
        for (int i = 0; i < pt.content.size(); i++) {
            PT::GPUSegmentInfo seg_info;
            seg_info.type = pt.content[i].type;
            seg_info.length = pt.content[i].length;
            seg_info.value_index = (i < pt.curr_indices.size()) ? pt.curr_indices[i] : 0;
            h_pt_segments.push_back(seg_info);
        }
    }
    
    // 异步传输数据到设备
    cudaMemcpyAsync(d_pt_segments[bufferIndex], h_pt_segments.data(), 
                   h_pt_segments.size() * sizeof(PT::GPUSegmentInfo), 
                   cudaMemcpyHostToDevice, dataStream);
    
    cudaMemcpyAsync(d_pt_sizes[bufferIndex], h_pt_sizes.data(),
                   batch_size * sizeof(int), 
                   cudaMemcpyHostToDevice, dataStream);
}


void PriorityQueue::LaunchGPUComputation(int bufferIndex) {
    // 计算网格和块大小
    int block_size = 256;
    int grid_size = (currentBatch.size() + block_size - 1) / block_size;
    
    // 启动核函数
    generateKernel<<<grid_size, block_size, 0, computeStream>>>(
        d_pt_segments[bufferIndex], 
        d_pt_sizes[bufferIndex],
        d_model.letter_values, 
        d_model.letter_value_counts,
        d_output[bufferIndex], 
        MAX_GUESS_LENGTH, 
        currentBatch.size()
    );
}


void PriorityQueue::ProcessResults(int bufferIndex) {
    // 等待数据传输完成
    cudaStreamSynchronize(dataStream);
    
    // 处理GPU计算结果
    #pragma omp parallel for
    for (int i = 0; i < currentBatch.size(); i++) {
        std::string guess(&h_output[bufferIndex][i * MAX_GUESS_LENGTH]);
        #pragma omp critical
        {
            guesses.push_back(guess);
            total_guesses++;
        }
    }
    
    // 生成新的PT并插入优先队列
    #pragma omp parallel for
    for (int i = 0; i < currentBatch.size(); i++) {
        PT& pt = currentBatch[i];
        std::vector<PT> new_pts = pt.NewPTs();
        #pragma omp critical
        {
            for (PT& new_pt : new_pts) {
                CalProb(new_pt);
                auto it = std::lower_bound(priority.begin(), priority.end(), new_pt, 
                    [](const PT& a, const PT& b) { return a.prob > b.prob; });
                priority.insert(it, new_pt);
            }
        }
    }
    
    // 从队列中移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + currentBatch.size());
    
    // 清空当前批次
    currentBatch.clear();
}
void PriorityQueue::PopNextBatch(int batch_size) {
    // 1. 准备下一批数据到非活动缓冲区 (CPU工作)
    int nextBuffer = 1 - activeBuffer;
    if (!priority.empty()) {
        // 准备下一批PT数据
        nextBatch.clear();
        
        // 从优先队列获取下一批PT
        int actual_size = std::min(batch_size, static_cast<int>(priority.size()));
        for (int i = 0; i < actual_size; i++) {
            nextBatch.push_back(priority[i]);
        }
        
        // 准备主机端数据结构
        std::vector<PT::GPUSegmentInfo> h_pt_segments;
        std::vector<int> h_pt_sizes;
        
        for (PT& pt : nextBatch) {
            h_pt_sizes.push_back(pt.content.size());
            for (int i = 0; i < pt.content.size(); i++) {
                PT::GPUSegmentInfo seg_info;
                seg_info.type = pt.content[i].type;
                seg_info.length = pt.content[i].length;
                seg_info.value_index = (i < pt.curr_indices.size()) ? pt.curr_indices[i] : 0;
                h_pt_segments.push_back(seg_info);
            }
        }
        
        // 异步传输数据到设备
        cudaMemcpyAsync(d_pt_segments[nextBuffer], h_pt_segments.data(), 
                       h_pt_segments.size() * sizeof(PT::GPUSegmentInfo), 
                       cudaMemcpyHostToDevice, dataStream);
        
        cudaMemcpyAsync(d_pt_sizes[nextBuffer], h_pt_sizes.data(),
                       actual_size * sizeof(int), 
                       cudaMemcpyHostToDevice, dataStream);
    }
    
    // 2. 等待当前GPU计算完成
    cudaStreamSynchronize(computeStream);
    
    // 3. 处理上一批结果 (CPU工作)
    if (!currentBatch.empty()) {
        // 等待数据传输完成
        cudaStreamSynchronize(dataStream);
        
        // 处理GPU计算结果
        #pragma omp parallel for
        for (int i = 0; i < currentBatch.size(); i++) {
            std::string guess(&h_output[activeBuffer][i * MAX_GUESS_LENGTH]);
            #pragma omp critical
            {
                guesses.push_back(guess);
                total_guesses++;
            }
        }
        
        // 生成新的PT并插入优先队列
        #pragma omp parallel for
        for (int i = 0; i < currentBatch.size(); i++) {
            PT& pt = currentBatch[i];
            std::vector<PT> new_pts = pt.NewPTs();
            #pragma omp critical
            {
                for (PT& new_pt : new_pts) {
                    CalProb(new_pt);
                    auto it = std::lower_bound(priority.begin(), priority.end(), new_pt, 
                        [](const PT& a, const PT& b) { return a.prob > b.prob; });
                    priority.insert(it, new_pt);
                }
            }
        }
        
        // 从队列中移除已处理的PT
        priority.erase(priority.begin(), priority.begin() + currentBatch.size());
    }
    
    // 4. 将准备好的批次设为当前批次
    if (!nextBatch.empty()) {
        currentBatch = std::move(nextBatch);
        activeBuffer = nextBuffer;
    }
    
    // 5. 启动当前批次的GPU计算
    if (!currentBatch.empty()) {
        // 计算网格和块大小
        int block_size = 256;
        int grid_size = (currentBatch.size() + block_size - 1) / block_size;
        
        // 启动核函数
        generateKernel<<<grid_size, block_size, 0, computeStream>>>(
            d_pt_segments[activeBuffer], 
            d_pt_sizes[activeBuffer],
            d_model.letter_values, 
            d_model.letter_value_counts,
            d_output[activeBuffer], 
            MAX_GUESS_LENGTH, 
            currentBatch.size()
        );
        
        // 6. 异步传输结果 (与CPU工作并行)
        cudaMemcpyAsync(h_output[activeBuffer].data(), d_output[activeBuffer], 
                       currentBatch.size() * MAX_GUESS_LENGTH, 
                       cudaMemcpyDeviceToHost, dataStream);
    }
}
void PriorityQueue::GenerateBatch(std::vector<PT>& pts)
{
    #pragma omp parallel for
    for (int i = 0; i < pts.size(); i++) {
        PT& pt = pts[i];
        CalProb(pt);
        
        if (pt.content.size() == 1) {
            segment* seg = nullptr;
            if (pt.content[0].type == 1) seg = &m.letters[m.FindLetter(pt.content[0])];
            else if (pt.content[0].type == 2) seg = &m.digits[m.FindDigit(pt.content[0])];
            else if (pt.content[0].type == 3) seg = &m.symbols[m.FindSymbol(pt.content[0])];
            
            #pragma omp critical
            {
                for (int j = 0; j < pt.max_indices[0]; j++) {
                    guesses.push_back(seg->ordered_values[j]);
                    total_guesses++;
                }
            }
        }
        else {
            std::string prefix;
            for (int seg_idx = 0; seg_idx < pt.content.size() - 1; seg_idx++) {
                segment& s = pt.content[seg_idx];
                if (s.type == 1) prefix += m.letters[m.FindLetter(s)].ordered_values[pt.curr_indices[seg_idx]];
                else if (s.type == 2) prefix += m.digits[m.FindDigit(s)].ordered_values[pt.curr_indices[seg_idx]];
                else if (s.type == 3) prefix += m.symbols[m.FindSymbol(s)].ordered_values[pt.curr_indices[seg_idx]];
            }
            
            segment* seg = nullptr;
            if (pt.content.back().type == 1) seg = &m.letters[m.FindLetter(pt.content.back())];
            else if (pt.content.back().type == 2) seg = &m.digits[m.FindDigit(pt.content.back())];
            else if (pt.content.back().type == 3) seg = &m.symbols[m.FindSymbol(pt.content.back())];
            
            #pragma omp critical
            {
                for (int j = 0; j < pt.max_indices.back(); j++) {
                    guesses.push_back(prefix + seg->ordered_values[j]);
                    total_guesses++;
                }
            }
        }
    }
}

// 初始化GPU模型数据
void PriorityQueue::initGPU()
{
    if (gpu_initialized) return;
    
    // 字母段
    d_model.letter_seg_count = m.letters.size();
    cudaMalloc(&d_model.letter_values, d_model.letter_seg_count * sizeof(char*));
    cudaMalloc(&d_model.letter_value_counts, d_model.letter_seg_count * sizeof(int));
    
    for (int i = 0; i < d_model.letter_seg_count; i++) {
        int count = m.letters[i].ordered_values.size();
        d_model.letter_value_counts[i] = count;
        
        // 计算所有值字符串的总长度
        size_t total_length = 0;
        for (const auto& val : m.letters[i].ordered_values) {
            total_length += val.length() + 1; // +1 for null terminator
        }
        
        // 分配设备内存并复制数据
        char* d_values;
        cudaMalloc(&d_values, total_length);
        char* current = d_values;
        for (const auto& val : m.letters[i].ordered_values) {
            cudaMemcpy(current, val.c_str(), val.length() + 1, cudaMemcpyHostToDevice);
            current += val.length() + 1;
        }
        cudaMemcpy(&d_model.letter_values[i], &d_values, sizeof(char*), cudaMemcpyHostToDevice);
    }
    
    // 类似处理数字和符号段...
    
    gpu_initialized = true;
}



void PriorityQueue::GenerateGPU(std::vector<PT>& pts)
{
    if (!gpu_initialized) {
        initGPU();
    }
    
    // 准备批量PT数据
    int batch_size = pts.size();
    std::vector<PT::GPUSegmentInfo> h_pt_segments;
    std::vector<int> h_pt_sizes;
    
    for (PT& pt : pts) {
        h_pt_sizes.push_back(pt.content.size());
        for (int i = 0; i < pt.content.size(); i++) {
            PT::GPUSegmentInfo seg_info;
            seg_info.type = pt.content[i].type;
            seg_info.length = pt.content[i].length;
            seg_info.value_index = (i < pt.curr_indices.size()) ? pt.curr_indices[i] : 0;
            h_pt_segments.push_back(seg_info);
        }
    }
    
    // 分配设备内存
    PT::GPUSegmentInfo* d_pt_segments;
    int* d_pt_sizes;
    cudaMalloc(&d_pt_segments, h_pt_segments.size() * sizeof(PT::GPUSegmentInfo));
    cudaMalloc(&d_pt_sizes, batch_size * sizeof(int));
    
    // 复制数据到设备
    cudaMemcpy(d_pt_segments, h_pt_segments.data(), 
               h_pt_segments.size() * sizeof(PT::GPUSegmentInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_sizes, h_pt_sizes.data(), 
               batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // 分配输出内存
    int max_guess_length = 64; // 假设最大密码长度为64
    char* d_output;
    cudaMalloc(&d_output, batch_size * max_guess_length);
    
    // 计算网格和块大小
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    // 启动核函数
    generateKernel<<<grid_size, block_size>>>(d_pt_segments, d_pt_sizes,
                                              d_model.letter_values, d_model.letter_value_counts,
                                              d_output, max_guess_length, batch_size);
    
    // 将结果复制回主机
    std::vector<char> h_output(batch_size * max_guess_length);
    cudaMemcpy(h_output.data(), d_output, batch_size * max_guess_length, cudaMemcpyDeviceToHost);
    
    // 处理结果
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        std::string guess(&h_output[i * max_guess_length]);
        #pragma omp critical
        {
            guesses.push_back(guess);
            total_guesses++;
        }
    }
    
    // 释放设备内存
    cudaFree(d_pt_segments);
    cudaFree(d_pt_sizes);
    cudaFree(d_output);
}

// GPU核函数实现
__global__ void generateKernel(PT::GPUSegmentInfo* d_pt_segments, int* d_pt_sizes, 
                               char** d_letter_values, int* d_letter_counts,
                               char* d_output, int max_guess_length, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int seg_start = 0;
    for (int i = 0; i < idx; i++) {
        seg_start += d_pt_sizes[i];
    }
    
    int seg_count = d_pt_sizes[idx];
    PT::GPUSegmentInfo* segs = &d_pt_segments[seg_start];
    
    char* output = &d_output[idx * max_guess_length];
    int pos = 0;
    
    for (int i = 0; i < seg_count; i++) {
        PT::GPUSegmentInfo seg = segs[i];
        
        if (seg.type == 1) { // Letter
            int seg_index = seg.length; // 简化处理，实际需要更复杂的映射
            char* values = d_letter_values[seg_index];
            int value_index = seg.value_index;
            
            // 跳过前面的值
            char* current = values;
            for (int j = 0; j < value_index; j++) {
                while (*current++); // 移动到下一个字符串
            }
            
            // 复制当前值
            char* val_ptr = current;
            while (*val_ptr && pos < max_guess_length - 1) {
                output[pos++] = *val_ptr++;
            }
        }
        // 类似处理其他类型...
    }
    output[pos] = '\0'; // Null-terminate the string
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


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}